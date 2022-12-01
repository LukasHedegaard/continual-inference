from collections import OrderedDict
from enum import Enum
from functools import partial, reduce
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from continual.delay import State as DelayState

from .closure import Lambda, Unity
from .container import BroadcastReduce, Residual, Sequential
from .delay import Delay, PaddingMode
from .linear import Linear
from .multihead_attention import (
    RetroactiveMultiheadAttention,
    SingleOutputMultiheadAttention,
)

__all__ = ["TransformerEncoder", "TransformerEncoderLayerFactory"]


class MhaType(Enum):
    """Type of Multi-head Attention
    Supported tupes are:
    - RETROACTIVE:      RetroactiveMultiheadAttention
    - SINGLE_OUTPUT:    SingleOutputMultiheadAttention
    - REGULAR:          nn.MultiheadAttention
    """

    RETROACTIVE = "retroactive"
    SINGLE_OUTPUT = "single_output"
    SINGLE_OUTPUT_FORWARD = "single_output_forward"
    REGULAR = "regular"


class SelectOrDelay(Delay):
    """Select a temporal index during forward, or delay correspondingly during forward_step(s)"""

    def forward(self, x: Tensor) -> Optional[Tensor]:
        assert len(x.shape) >= 3  # N, C, T
        return x[:, :, -1 - self._delay].unsqueeze(2)


class RetroactiveUnity(Delay):
    """Unity mapping during forward. During forward_step(s), a single-to-many mapping is assumed,
    and all cached values are output."""

    def __init__(
        self,
        delay: int,
        temporal_fill: PaddingMode = "zeros",
        auto_shrink: bool = False,
        time_dim=-1,
    ):
        """Initialise Delay block

        Args:
            delay (int): the number of steps to delay an output.
            temporal_fill (PaddingMode, optional): Temporal state initialisation mode ("zeros" or "replicate"). Defaults to "zeros".
            auto_shrink (int, optional): Whether to shrink the temporal dimension of the feature map during forward.
                This is handy for residuals that are parallel to modules which reduce the number of temporal steps. Defaults to False.
            time_dim (int, optional): Which dimension to concatenate step outputs along
        """
        self.time_dim = time_dim
        Delay.__init__(self, delay, temporal_fill, auto_shrink)

    def init_state(
        self,
        first_output: Tensor,
    ) -> DelayState:
        padding = self.make_padding(first_output)
        state_buffer = torch.stack([padding for _ in range(self.delay + 1)], dim=0)
        state_index = torch.tensor(-self.delay)
        return state_buffer, state_index

    def _forward_step(
        self, input: Tensor, prev_state: DelayState
    ) -> Tuple[Tensor, DelayState]:
        if prev_state is None:
            buffer, index = self.init_state(input)
        else:
            buffer, index = prev_state

        # Update state
        buffer[index % (self.delay + 1)] = input
        new_index = index + 1
        if new_index > 0:
            new_index = new_index % self.delay

        # Get output
        output = None
        if index >= 0:
            output = buffer.clone().roll(shifts=int(-index - 1), dims=0)
            idx = (
                self.time_dim + len(output.shape)
                if self.time_dim < 0
                else self.time_dim
            )
            output = output.permute(
                list(range(1, idx + 1)) + [0] + list(range(idx + 1, len(output.shape)))
            )

        return output, (buffer, new_index)


class RetroactiveLambda(Lambda):
    """
    Lambda wrapper for functions that are applied after retroactive modules.
    """

    def forward(self, input: Tensor) -> Tensor:
        return Lambda.forward(self, input)

    def forward_step(self, input: Tensor, *args, **kwargs) -> Tensor:
        return self.forward(input)

    def _forward_step(self, input: Tensor, prev_state=None, *args, **kwargs) -> Tensor:
        return self.forward(input), prev_state

    def forward_steps(self, input: Tensor, *args, **kwargs) -> Tensor:
        return torch.stack(
            [self.forward(input[:, :, t]) for t in range(input.shape[2])], dim=2
        )

    @staticmethod
    def build_from(
        fn: Callable[[Tensor], Tensor], takes_time=False
    ) -> "RetroactiveLambda":
        return RetroactiveLambda(fn, takes_time)


class NaiveResidual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def sum_last_pairs(inputs: Sequence[Tensor]) -> Tensor:
    if inputs[0].shape != inputs[1].shape:
        T_min = min(inputs[i].shape[2] for i in range(len(inputs)))
        inputs = [inp[:, :, -T_min:] for inp in inputs]
    return reduce(torch.Tensor.add, inputs[1:], inputs[0])


def SingleOutputTransformerEncoderLayer(
    d_model: int,
    nhead: int,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation: Union[nn.Module, Callable[[Tensor], Tensor]] = nn.functional.relu,
    layer_norm_eps: float = 1e-5,
    # batch_first: bool = True,
    # norm_first: bool = False,
    device=None,
    dtype=None,
    sequence_len: int = None,
    single_output_forward=False,
):

    factory_kwargs = {"device": device, "dtype": dtype}
    norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

    mha = SingleOutputMultiheadAttention(
        embed_dim=d_model,
        num_heads=nhead,
        dropout=dropout,
        bias=True,
        batch_first=True,
        embed_dim_second=True,
        query_index=-1,
        device=device,
        dtype=dtype,
        sequence_len=sequence_len,
        forward_returns_attn_mask=False,
        single_output_forward=single_output_forward,
    )

    ff = Sequential(
        OrderedDict(
            [
                (
                    "linear1",
                    Linear(d_model, dim_feedforward, channel_dim=1, **factory_kwargs),
                ),
                (
                    "activation",
                    activation,
                ),
                (
                    "dropout",
                    nn.Dropout(dropout),
                ),
                (
                    "linear2",
                    Linear(dim_feedforward, d_model, channel_dim=1, **factory_kwargs),
                ),
                (
                    "dropout2",
                    nn.Dropout(dropout),
                ),
            ]
        )
    )

    return Sequential(
        BroadcastReduce(
            OrderedDict(
                [
                    (
                        "residual",
                        SelectOrDelay(mha.delay) if single_output_forward else Unity(),
                    ),
                    (
                        "self_attn",
                        mha,
                    ),
                ]
            ),
            reduce=sum_last_pairs,
            auto_delay=False,
        ),
        Sequential(
            OrderedDict(
                [
                    ("norm1", Lambda(norm1, takes_time=False)),
                    ("_ff_block", Residual(ff)),
                    ("norm2", Lambda(norm2, takes_time=False)),
                ]
            )
        ),
    )


def RetroactiveTransformerEncoderLayer(
    d_model: int,
    nhead: int,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation: Union[nn.Module, Callable[[Tensor], Tensor]] = nn.functional.relu,
    layer_norm_eps: float = 1e-5,
    # batch_first: bool = True,
    # norm_first: bool = False,
    device=None,
    dtype=None,
    sequence_len: int = None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

    mha = RetroactiveMultiheadAttention(
        embed_dim=d_model,
        num_heads=nhead,
        dropout=dropout,
        bias=True,
        batch_first=True,
        embed_dim_second=True,
        device=device,
        dtype=dtype,
        sequence_len=sequence_len,
        forward_returns_attn_mask=False,
    )

    ff = Sequential(
        OrderedDict(
            [
                ("linear1", Linear(d_model, dim_feedforward, **factory_kwargs)),
                ("activation", Lambda(activation, takes_time=True)),
                ("dropout", nn.Dropout(dropout)),
                ("linear2", Linear(dim_feedforward, d_model, **factory_kwargs)),
                ("dropout2", nn.Dropout(dropout)),
            ]
        )
    )

    return Sequential(
        BroadcastReduce(
            OrderedDict(
                [
                    ("residual", RetroactiveUnity(mha.delay)),
                    ("self_attn", mha),
                ]
            ),
            reduce="sum",
            auto_delay=False,
        ),
        RetroactiveLambda(
            nn.Sequential(
                OrderedDict(
                    [
                        ("norm1", norm1),
                        ("_ff_block", NaiveResidual(ff)),
                        ("norm2", norm2),
                    ]
                )
            )
        ),
    )


# TODO: impl
def StepLocalTransformerEncoderLayer(
    d_model: int,
    nhead: int,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation: Union[nn.Module, Callable[[Tensor], Tensor]] = nn.functional.relu,
    layer_norm_eps: float = 1e-5,
    # batch_first: bool = True,
    # norm_first: bool = False,
    device=None,
    dtype=None,
    sequence_len: int = None,
):
    ...


def TransformerEncoderLayerFactory(
    d_model: int,
    nhead: int,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation: Union[nn.Module, Callable[[Tensor], Tensor]] = nn.functional.relu,
    layer_norm_eps: float = 1e-5,
    # batch_first: bool = True,
    # norm_first: bool = False,
    device=None,
    dtype=None,
    sequence_len: int = None,
) -> Callable[[MhaType], Sequential]:
    def TransformerEncoderLayer(mha_type: MhaType):

        factory_fn = {
            MhaType.RETROACTIVE: RetroactiveTransformerEncoderLayer,
            MhaType.SINGLE_OUTPUT: SingleOutputTransformerEncoderLayer,
            MhaType.SINGLE_OUTPUT_FORWARD: partial(
                SingleOutputTransformerEncoderLayer, single_output_forward=True
            ),
            MhaType.REGULAR: StepLocalTransformerEncoderLayer,
        }[MhaType(mha_type)]

        return factory_fn(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            # batch_first
            # norm_first
            device,
            dtype,
            sequence_len,
        )

    return TransformerEncoderLayer


class TransformerEncoder(Sequential):
    """Continual Transformer Encoder as proposed by Hedegaard et al. in
    "Continual Transformers: Redundancy-Free Attention for Online Inference"
    https://arxiv.org/abs/2201.06268TransformerEncoder

    This class deviates from the Pytorch implementation in the following ways:
    - `encoder_layer` parameter takes a factory functor, TransformerEncoderLayerFactory
    - `mask` and `src_key_padding_mask` are not supported currently.

    Args:
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples:
        encoder_layer = co.TransformerEncoderLayerFactory(d_model=512, nhead=8)
        transformer_encoder = co.TransformerEncoder(encoder_layer, num_layers=2)
        src = torch.rand(10, 512, 32)
        out = transformer_encoder(src)
    """

    def __init__(
        self,
        encoder_layer: Callable[[MhaType, Optional[bool]], Sequential],
        num_layers: int,
        norm: nn.Module = None,
    ):

        layers = []
        if num_layers == 1:
            layers.append(encoder_layer(MhaType.SINGLE_OUTPUT))
        else:
            layers.append(encoder_layer(MhaType.RETROACTIVE))
            for _ in range(2, num_layers - 1):
                layers.append(
                    RetroactiveLambda(encoder_layer(MhaType.REGULAR), takes_time=True)
                )
            layers.append(
                RetroactiveLambda(
                    encoder_layer(MhaType.SINGLE_OUTPUT_FORWARD), takes_time=True
                )
            )

            def unity(x):
                return x

            def squeeze_last(x):
                return x.squeeze(-1)

            layers.append(
                Lambda(unity, None, squeeze_last, squeeze_last, takes_time=True)
            )

        Sequential.__init__(self, OrderedDict([("layers", Sequential(*layers))]))
        if norm is not None:
            self.add_module("norm", Lambda(norm, takes_time=False))

    @staticmethod
    def build_from(
        trans_enc: nn.TransformerEncoder, sequence_len: int
    ) -> "TransformerEncoder":
        assert isinstance(trans_enc, nn.TransformerEncoder)

        # Create model
        tel = trans_enc.layers[0]
        layer_factory = TransformerEncoderLayerFactory(
            d_model=tel.self_attn.embed_dim,
            nhead=tel.self_attn.num_heads,
            dim_feedforward=tel.linear1.out_features,
            dropout=tel.dropout.p,
            activation=tel.activation,
            layer_norm_eps=tel.norm1.eps,
            device=tel.linear1.weight.device,
            dtype=tel.linear1.weight.dtype,
            sequence_len=sequence_len,
        )
        net = TransformerEncoder(
            layer_factory, num_layers=trans_enc.num_layers, norm=trans_enc.norm
        )

        # Transfer weights
        new_sd = {}

        net_keys = list(net.state_dict().keys())
        match_keys, key_inds = zip(
            *sorted(
                [
                    (
                        k.replace("fn.", "")
                        .replace("_ff_block.", "")
                        .replace(".0.self_attn", ".self_attn"),
                        i,
                    )
                    for i, k in enumerate(net_keys)
                ],
                key=lambda x: x[0],
            )
        )

        reg_keys, weights = zip(
            *sorted(
                [item for item in trans_enc.state_dict().items()], key=lambda x: x[0]
            )
        )

        assert all(
            [
                ".".join(k1.split(".")[-2:]) == ".".join(k2.split(".")[-2:])
                for k1, k2 in zip(match_keys, reg_keys)
            ]
        )

        new_sd = {net_keys[key_inds[i]]: weights[i] for i in range(len(net_keys))}

        net.load_state_dict(new_sd)

        return net
