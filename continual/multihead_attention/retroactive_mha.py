import math
from functools import partial
from typing import Optional, Tuple

import torch
from torch import Tensor

from continual.logging import getLogger
from continual.module import _callmode

from .mha_base import MultiheadAttentionBase, scaled_dot_prod_attn_flops

logger = getLogger(__name__)
logger_once = getLogger(__name__, log_once=True)

State = Tuple[
    Tensor,  # d_mem, (B, Nt-1)
    Tensor,  # AV_mem, (B, Ns-1, E)
    Tensor,  # Q_mem, (B, Nt-1, E)
    Tensor,  # K_T_mem, (B, E, Ns)
    Tensor,  # V_mem, (B, Ns, E)
    Tensor,  # state_index
]


def _scaled_dot_product_attention_default_state(
    batch_size: int,
    sequence_len: int,
    embed_dim: int,
    num_heads: int,
    init_fn=torch.zeros,
    dtype=None,
    device=None,
):
    init_fn = partial(init_fn, dtype=dtype, device=device)
    E = embed_dim // num_heads
    B = batch_size * num_heads
    N = sequence_len
    d_mem = init_fn((B, N - 1, 1))
    AV_mem = init_fn((B, N - 1, E))
    Q_mem = init_fn((B, N - 1, E))
    K_T_mem = init_fn((B, E, N))
    V_mem = init_fn((B, N, E))
    return (d_mem, AV_mem, Q_mem, K_T_mem, V_mem)


def _scaled_dot_product_attention_step(
    prev_state: State,
    q_step: Tensor,  # step input (B, E)
    k_step: Tensor,  # step input (B, E)
    v_step: Tensor,  # step input (B, E)
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, State]:
    """
    Computes the Continual Retroactive Scaled Dot-Product Attention on query, key and value tensors.
    Returns attended values and updated states.

    Args:
        q_step, k_step, v_step: query, key and value tensors for a step. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q_step: :math:`(B, E)` where B is batch size and E is embedding dimension.
        - k_step: :math:`(B, E)` where B is batch size and E is embedding dimension.
        - v_step: :math:`(B, E)` where B is batch size and E is embedding dimension.

        - Output: attention values have shape :math:`(B, Nt, E)`; new state
    """
    if attn_mask is not None:  # pragma: no cover
        logger_once.warning("attn_mask is not supported yet and will be skipped")
    if dropout_p != 0.0:  # pragma: no cover
        logger_once.warning("dropout_p is not supported yet and will be skipped")

    (
        d_mem,  # (B, Nt-1)
        AV_mem,  # (B, Ns-1, E)
        Q_mem,  # (B, Nt-1, E)
        K_T_mem,  # (B, E, Ns)
        V_mem,  # (B, Ns, E)
    ) = prev_state

    B, E = q_step.shape
    q_step = q_step / math.sqrt(E)

    # Compute oldest and newest entries in attention matrix A:
    # L . . . R
    # L . . . R
    # L . . . R
    #   B B B B

    # Left column attention values
    A_left = torch.exp(torch.bmm(Q_mem, K_T_mem[:, :, 0].unsqueeze(-1)))

    # Right column attention values
    A_right = torch.exp(torch.bmm(Q_mem, k_step.unsqueeze(-1)))

    # Update Q_mem and K_mem
    Q_mem_new = torch.roll(Q_mem, shifts=-1, dims=(1,))
    Q_mem_new[:, -1] = q_step

    K_T_mem_new = torch.roll(K_T_mem, shifts=-1, dims=(2,))
    K_T_mem_new[:, :, -1] = k_step

    # Bottom row attention values
    A_bottom = torch.exp(torch.bmm(q_step.unsqueeze(1), K_T_mem_new))

    # Compute normalisation
    d = torch.cat(
        (
            d_mem - A_left + A_right,
            (A_bottom.sum(-1)).unsqueeze(-1),
        ),
        dim=1,
    )

    # Compute AV matrix top
    AV_sub = torch.bmm(A_left, V_mem[:, 0].unsqueeze(1))
    AV_add = torch.bmm(A_right, v_step.unsqueeze(1))
    AV_top = AV_mem - AV_sub + AV_add

    # Update V_mem
    V_mem_new = torch.roll(V_mem, shifts=-1, dims=(1,))
    V_mem_new[:, -1] = v_step

    # Compute AV_bottom
    AV_bottom = torch.bmm(A_bottom, V_mem_new)

    AV_new = torch.cat((AV_top, AV_bottom), dim=1)

    # Compute final output
    output = AV_new / d

    new_states = (
        d[:, 1:],  # (B, Nt-1)
        AV_new[:, 1:],  # (B, Ns-1, E)
        Q_mem_new,
        K_T_mem_new,
        V_mem_new,
    )

    return output, new_states


class RetroactiveMultiheadAttention(MultiheadAttentionBase):
    """
    MultiHeadAttention with retroactively updated attention outputs during continual inference.

    Continual MHAs were proposed by Hedegaard et al. in
    "Continual Transformers: Redundancy-Free Attention for Online Inference"
    https://arxiv.org/abs/2201.06268 (paper) https://www.youtube.com/watch?v=gy802Tlp-eQ (video).

    This module augments the MultiHeadAttention in PyTorch with
    `forward_step` / `forward_steps` functions, in which one / more
    query, key, and value tokens are passed to yield the multihead attentions, and
    updated outputs are computed for each token input.

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        device: torch device to initialize layer on. Defaults to None.
        dtype: datatype of layer parameters. Defaults to None.
        sequence_len: Length of token sequence.
        forward_returns_attn_mask: Whether forward should return attention mask.
        embed_dim_second: Whether the embed dimension should be second.

    .. note::
        If :attr:`kdim` and :attr:`vdim` are None, they will be set
        to :attr:`embed_dim` such that query, key, and value have the same
        number of features.

    Examples::

        mha = co.RetroactiveMultiheadAttention(
            embed_dim=512,
            num_heads=8,
            sequence_len=32,
            dropout=0.0,
            batch_first=True,
            embed_dim_second=True,
        )
        x = torch.rand(10, 512, 32)

        out, attn_mask = mha.forward(x)

        # continual inference API
        firsts = mha.forward_steps(x[:,:,:-1])
        last = mha.forward_step(x[:,:,-1])

        assert firsts is None  # The module first needs to observe ``sequence_len`` values
        assert torch.allclose(out, last, atol=1e-6)
    """

    _state_shape = 6
    #                     d_mem, AV_mem, Q_mem, K_T_mem, V_mem ,state_index
    _dynamic_state_inds = [True, True, True, True, True, False]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        sequence_len=None,
        forward_returns_attn_mask=True,
        embed_dim_second=False,
    ) -> None:
        MultiheadAttentionBase.__init__(
            self,
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            device,
            dtype,
            sequence_len,
            partial(
                _scaled_dot_product_attention_default_state,
                sequence_len=sequence_len,
                embed_dim=embed_dim,
                num_heads=num_heads,
            ),
            _scaled_dot_product_attention_step,
            forward_returns_attn_mask,
            embed_dim_second,
        )
        self.register_buffer("d_mem", torch.tensor([]), persistent=False)
        self.register_buffer("AV_mem", torch.tensor([]), persistent=False)
        self.register_buffer("Q_mem", torch.tensor([]), persistent=False)
        self.register_buffer("K_T_mem", torch.tensor([]), persistent=False)
        self.register_buffer("V_mem", torch.tensor([]), persistent=False)
        self.register_buffer("stride_index", torch.tensor(0), persistent=False)

    def get_state(self) -> Optional[State]:
        if len(self.d_mem) > 0:
            return (
                self.d_mem,
                self.AV_mem,
                self.Q_mem,
                self.K_T_mem,
                self.V_mem,
                self.stride_index,
            )

    def set_state(self, state: State):
        (
            self.d_mem,
            self.AV_mem,
            self.Q_mem,
            self.K_T_mem,
            self.V_mem,
            self.stride_index,
        ) = state

    def clean_state(self):
        self.d_mem = torch.tensor([])
        self.AV_mem = torch.tensor([])
        self.Q_mem = torch.tensor([])
        self.K_T_mem = torch.tensor([])
        self.V_mem = torch.tensor([])
        self.stride_index = torch.tensor(0)

    def _forward_step(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        prev_state: Optional[State] = None,
    ) -> Tuple[Optional[Tensor], State]:
        """
        Args:
            query, key, value: step_inputs for mapping a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.

        Shapes for inputs:
            - query: :math:`(N, E)` where N is the batch size, E is the embedding dimension.
            - key: :math:`(N, E)`, where N is the batch size, E is the embedding dimension.
            - value: :math:`(N, E)` where N is the batch size, E is the embedding dimension.

        Shapes for outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
              :math:`(N, E, L)` if ``batch_first`` and ``embed_dim_second ``True``.
        """
        if key is None:
            key = query
        if value is None:
            value = query

        o, next_state = MultiheadAttentionBase._forward_step(
            self, query, key, value, prev_state
        )

        if o is not None:
            if self.batch_first:
                o = o.transpose(1, 0)
            if self.embed_dim_second:
                o = o.transpose(1, 2)

        return o, next_state

    def forward_step(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        update_state=True,
        *args,
        **kwargs,
    ) -> Optional[Tensor]:
        """
        Args:
            query, key, value: step_inputs for mapping a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.

        Shapes for inputs:
            - query: :math:`(N, E)` N is the batch size, E is the embedding dimension.
            - key: :math:`(N, E)`, where N is the batch size, E is the embedding dimension.
            - value: :math:`(N, E)` where N is the batch size, E is the embedding dimension.

        Shapes for outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
              :math:`(N, E, L)` if ``batch_first`` and ``embed_dim_second ``True``.
        """
        o = MultiheadAttentionBase.forward_step(
            self, query, key, value, update_state, *args, **kwargs
        )

        if isinstance(o, Tensor) and self.embed_dim_second:
            o = o.transpose(1, 2)

        return o

    def forward_steps(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        update_state=True,
        *args,
        **kwargs,
    ) -> Optional[Tensor]:
        o = MultiheadAttentionBase.forward_steps(
            self, query, key, value, update_state, *args, **kwargs
        )

        if isinstance(o, Tensor) and self.embed_dim_second:
            o = o.permute(0, 3, 1, 2)  # N T T' E -> N E T T'

        return o

    def flops(self, include_muls=True, include_adds=False, include_exps=False):
        f = 0

        # Linear projection
        steps_taken = {
            _callmode("forward"): self.sequence_len,
            _callmode("forward_step"): 1,
        }[self.call_mode]

        f += (
            steps_taken
            * self.embed_dim
            * self.embed_dim
            * 3  # Assuming equal len for Q, K, and V
        )

        if include_adds:
            f += 3 * steps_taken * self.embed_dim * (self.embed_dim - 1)

        if self.in_proj_bias is not None:
            f += 3 * steps_taken * self.embed_dim

            if include_adds:
                f += 3 * steps_taken * self.embed_dim

        # Multi-head Scaled Dot-Product Attention
        f += self.num_heads * {
            _callmode("forward"): scaled_dot_prod_attn_flops,
            _callmode("forward_step"): retractive_scaled_dot_prod_attn_step_flops,
        }[self.call_mode](
            self.sequence_len,
            self.embed_dim // self.num_heads,
            include_muls,
            include_adds,
            include_exps,
        )

        # Linear projection
        f += self.sequence_len * self.embed_dim * (self.embed_dim + 1)

        return f


def retractive_scaled_dot_prod_attn_step_flops(
    sequence_len, embed_dim, include_muls=True, include_adds=False, include_exps=False
):
    n = sequence_len
    d = embed_dim

    flops = 0

    if include_muls:
        flops += 7 * n * d + 2 * n - 3 * d
    if include_adds:
        flops += 6 * n * d + 3 * n - 6 * d - 3
    if include_exps:
        flops += 3 * n - 2

    return flops
