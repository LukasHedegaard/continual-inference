from collections import OrderedDict
from enum import Enum
from functools import reduce
from typing import Callable, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from continual.delay import State as DelayState
from continual.module import TensorPlaceholder

from .closure import Lambda
from .container import BroadcastReduce, Residual, Sequential
from .delay import Delay, PaddingMode
from .linear import Linear
from .multihead_attention import (
    RetroactiveMultiheadAttention,
    SingleOutputMultiheadAttention,
)

__all__ = ["TransformerEncoder", "TransformerEncoderLayerFactory", "MhaType"]


class MhaType(Enum):
    """Type of Multi-head Attention
    Supported tupes are:
    - RETROACTIVE:      RetroactiveMultiheadAttention
    - SINGLE_OUTPUT:    SingleOutputMultiheadAttention
    - REGULAR:          nn.MultiheadAttention
    """

    RETROACTIVE = "retroactive"
    SINGLE_OUTPUT = "single_output"
    REGULAR = "regular"


class SelectOrDelay(Delay):
    """Select a temporal index during forward, or delay correspondingly during forward_step(s)"""

    def forward(self, x: Tensor) -> Union[Tensor, TensorPlaceholder]:
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
        state_index = -self.delay
        if not hasattr(self, "state_buffer"):
            self.register_buffer("state_buffer", state_buffer, persistent=False)
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
        if index >= 0:
            output = buffer.clone().roll(shifts=-index - 1, dims=0)
            idx = (
                self.time_dim + len(output.shape)
                if self.time_dim < 0
                else self.time_dim
            )
            output = output.permute(
                list(range(1, idx + 1)) + [0] + list(range(idx + 1, len(output.shape)))
            )
        else:
            output = TensorPlaceholder(buffer[0].shape)

        return output, (buffer, new_index)


class RetroactiveLambda(Lambda):
    """
    Lambda wrapper for functions that are applied after retroactive modules.
    """

    def forward(self, input: Tensor) -> Tensor:
        return Lambda.forward(self, input)

    def forward_step(self, input: Tensor, *args, **kwargs) -> Tensor:
        return self.forward(input)

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


# TODO: Inherit from Sequential to add attributed and methods such as build_from?
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
    )

    ff = Sequential(
        OrderedDict(
            [
                ("linear1", Linear(d_model, dim_feedforward, **factory_kwargs)),
                ("activation", activation),
                # Bad name, kept for weight-compat with torch impl:
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
                    ("residual", SelectOrDelay(mha.delay)),
                    ("self_atn", mha),
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


# TODO: Inherit from Sequential to add attributed and methods such as build_from?
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
                ("activation", activation),
                # Bad name, kept for weight-compat with torch impl:
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
                    ("self_atn", mha),
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
) -> Callable[[str], Sequential]:
    def TransformerEncoderLayer(mha_type: MhaType):

        factory_fn = {
            MhaType.RETROACTIVE: RetroactiveTransformerEncoderLayer,
            MhaType.SINGLE_OUTPUT: SingleOutputTransformerEncoderLayer,
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


# TODO: Inherit from Sequential to add attributed and methods such as build_from?
def TransformerEncoder(encoder_layer, num_layers, norm: nn.Module = None):
    """Continual Transformer Encoder as proposed by Hedegaard et al. in
    "Continual Transformers: Redundancy-Free Attention for Online Inference"
    https://arxiv.org/abs/2201.06268TransformerEncoder

    This class deviates from the Pytorch implementation in the following ways:
    - `encoder_layer` parameter takes a factory functor, TransformerEncoderLayerFactory
    - `mask` and `src_key_padding_mask` are not supported currently.

    Args:
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> transformer_encoder = co.TransformerEncoder(num_layers=2)
        >>> src = torch.rand(10, 512, 32)
        >>> out = transformer_encoder(src)
    """

    layers = []
    if num_layers == 1:
        layers.append(("block1", encoder_layer(MhaType.SINGLE_OUTPUT)))
    else:
        layers.append(("block1", encoder_layer(MhaType.RETROACTIVE)))
        for i in range(2, num_layers - 1):
            layers.append((f"block{i}", encoder_layer(MhaType.REGULAR)))
        layers.append((f"block{num_layers}", encoder_layer(MhaType.SINGLE_OUTPUT)))

    if norm is not None:
        layers.append(("norm", norm))

    return Sequential(OrderedDict([layers]))


# TODO: impl and merge with TransformerEncoder
def build_transformer_encoder_from(
    trans_enc: nn.TransformerEncoder, sequence_len: int
) -> Sequential:
    ...
