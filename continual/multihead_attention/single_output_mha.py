import math
from functools import partial
from logging import getLogger
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from continual.module import _callmode

from .mha_base import MultiheadAttentionBase, scaled_dot_prod_attn_flops

logger = getLogger(__name__)

State = Tuple[
    Tensor,  # Q_mem, (B, Nt-1, E)
    Tensor,  # K_T_mem, (B, E, Ns)
    Tensor,  # V_mem, (B, Ns, E)
    Tensor,  # stride_index
]


def _scaled_dot_product_attention_default_state(
    batch_size: int,
    sequence_len: int,
    embed_dim: int,
    num_heads: int,
    query_index=-1,
    init_fn=torch.zeros,
    dtype=None,
    device=None,
):
    init_fn = partial(init_fn, dtype=dtype, device=device)
    E = embed_dim // num_heads
    B = batch_size * num_heads
    N = sequence_len
    Nq = sequence_len - query_index - 1 if query_index >= 0 else -query_index - 1
    Q_mem = init_fn((B, Nq, E))
    K_T_mem = init_fn((B, E, N))
    V_mem = init_fn((B, N, E))
    return (Q_mem, K_T_mem, V_mem)


def _scaled_dot_product_attention_step(
    prev_state: State,
    q_step: Tensor,  # step input (B, E)
    k_step: Tensor,  # step input (B, E)
    v_step: Tensor,  # step input (B, E)
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, State]:
    """
    Computes the Continual Singe-output Scaled Dot-Product Attention on query, key and value tensors.
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
        logger.warning("attn_mask is not supported yet and will be skipped")
    if dropout_p != 0.0:  # pragma: no cover
        logger.warning("dropout_p is not supported yet and will be skipped")

    (
        Q_mem,  # (B, Nq, E)
        K_T_mem,  # (B, E, Ns)
        V_mem,  # (B, Ns, E)
    ) = prev_state

    B, E = q_step.shape
    q_step = q_step / math.sqrt(E)
    q_sel = (Q_mem[:, 0] if Q_mem.shape[1] > 0 else q_step).unsqueeze(1)

    # Update states
    # Note: We're allowing the K and V mem to have one more entry than
    # strictly necessary to simplify computatations.

    K_T_new = torch.roll(K_T_mem, shifts=-1, dims=(2,))
    K_T_new[:, :, -1] = k_step

    V_new = torch.roll(V_mem, shifts=-1, dims=(1,))
    V_new[:, -1] = v_step

    attn = torch.bmm(q_sel, K_T_new)
    attn_sm = F.softmax(attn, dim=-1)

    if dropout_p > 0.0:  # pragma: no cover
        attn_sm = F.dropout(attn_sm, p=dropout_p)

    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn_sm, V_new)

    if Q_mem.shape[1] > 0:
        Q_new = torch.roll(Q_mem, shifts=-1, dims=(1,))
        Q_new[:, -1] = q_step
    else:
        Q_new = Q_mem

    new_states = (Q_new, K_T_new, V_new)

    return output, new_states


class SingleOutputMultiheadAttention(MultiheadAttentionBase):
    """
    Continual Single-output MultiHeadAttention as proposed by Hedegaard et al. in
    "Continual Transformers: Redundancy-Free Attention for Online Inference"
    https://arxiv.org/abs/2201.06268

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
        sequence_len: Length of token sequence

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.
    """

    _state_shape = 4
    #                      Q_mem, K_T_mem, V_mem, stride_index
    _dynamic_state_inds = [True, True, True, False]

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
        batch_first=True,
        device=None,
        dtype=None,
        sequence_len=None,
        query_index=-1,
        forward_returns_attn_mask=True,
        embed_dim_second=False,
        single_output_forward=True,
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
                query_index=query_index,
            ),
            _scaled_dot_product_attention_step,
            forward_returns_attn_mask,
            embed_dim_second,
        )
        assert query_index < sequence_len
        self.query_index = query_index
        self.single_output_forward = single_output_forward
        self.register_buffer("Q_mem", torch.tensor([]), persistent=False)
        self.register_buffer("K_T_mem", torch.tensor([]), persistent=False)
        self.register_buffer("V_mem", torch.tensor([]), persistent=False)
        self.register_buffer("stride_index", torch.tensor(0), persistent=False)

    def get_state(self) -> Optional[State]:
        """Get model state

        Returns:
            Optional[State]: A State tuple if the model has been initialised and otherwise None.
        """
        if len(self.V_mem) > 0:
            return (
                self.Q_mem,
                self.K_T_mem,
                self.V_mem,
                self.stride_index,
            )
        return None

    def set_state(self, state: State):
        """Set model state

        Args:
            state (State): State tuple to set as new internal internal state
        """
        (
            self.Q_mem,
            self.K_T_mem,
            self.V_mem,
            self.stride_index,
        ) = state

    def clean_state(self):
        """Clean model state"""
        self.Q_mem = torch.tensor([])
        self.K_T_mem = torch.tensor([])
        self.V_mem = torch.tensor([])
        self.stride_index = torch.tensor(0)

    @property
    def delay(self) -> int:
        return (
            self.sequence_len - self.query_index - 1
            if self.query_index >= 0
            else -self.query_index - 1
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if not self.single_output_forward:
            return MultiheadAttentionBase.forward(
                self, query, key, value, key_padding_mask, need_weights, attn_mask
            )

        if key is None:
            key = query
        if value is None:
            value = query

        # Select a single query entry
        if self.batch_first:
            if self.embed_dim_second:
                query = query[:, :, self.query_index].unsqueeze(2)
            else:
                query = query[:, self.query_index].unsqueeze(1)
        else:
            query = query[self.query_index].unsqueeze(0)

        o = MultiheadAttentionBase.forward(
            self, query, key, value, key_padding_mask, need_weights, attn_mask
        )

        return o

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
            - query: :math:`(N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.

        Shapes for outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        """
        if key is None:
            key = query
        if value is None:
            value = query

        o, next_state = MultiheadAttentionBase._forward_step(
            self, query, key, value, prev_state
        )

        if o is not None:
            o = o.squeeze(0)

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
            - query: :math:`(N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.

        Shapes for outputs:
            - attn_output: :math:`(N, E)` where N is the batch size and E is the embedding dimension.
        """
        o = MultiheadAttentionBase.forward_step(
            self, query, key, value, update_state, *args, **kwargs
        )
        return o.squeeze(1 if self.batch_first else 0) if isinstance(o, Tensor) else o

    def forward_steps(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        update_state=True,
        *args,
        **kwargs,
    ) -> Optional[Tensor]:
        """Forward computation for multiple steps with state initialisation

        Args:
            query (Tensor): query.
            key (Tensor): key.
            value (Tensor): value.
            update_state (bool): Whether internal state should be updated during this operation.

        Returns:
            Tensor: Stepwise layer outputs
        """
        o = MultiheadAttentionBase.forward_steps(
            self, query, key, value, update_state, *args, **kwargs
        )

        if isinstance(o, Tensor):
            o = o.squeeze(2)
            if self.embed_dim_second:
                o = o.transpose(1, 2)  # N T E -> N E T

        return o

    def flops(self, include_muls=True, include_adds=False, include_exps=False):
        f = 0

        # Linear projection
        steps_taken = {
            _callmode("forward"): 1
            if self.single_output_forward
            else self.sequence_len,
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
            _callmode("forward"): single_output_scaled_dot_prod_attn_flops
            if self.single_output_forward
            else scaled_dot_prod_attn_flops,
            _callmode("forward_step"): single_output_scaled_dot_prod_attn_flops,
        }[self.call_mode](
            self.sequence_len,
            self.embed_dim // self.num_heads,
            include_muls,
            include_adds,
            include_exps,
        )

        # Linear projection
        f += 1 * self.embed_dim * (self.embed_dim + 1)

        return f


def single_output_scaled_dot_prod_attn_flops(
    sequence_len, embed_dim, include_muls=True, include_adds=False, include_exps=False
):
    n = sequence_len
    d = embed_dim

    flops = 0

    if include_muls:
        flops += 2 * n * d + 2 * d
    if include_adds:
        flops += 2 * n * d - d - 1
    if include_exps:
        flops += n

    return flops
