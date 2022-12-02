from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.activation import MultiheadAttention

from continual.module import CoModule

State = List[Tensor]


def _clone_state(state):
    return [s.clone() if isinstance(s, torch.Tensor) else s for s in state]


# Copy of torch.nn impl
def multi_head_attention_forward_step(  # noqa: C901
    scaled_dot_product_attention_step: Callable,
    prev_state: Any,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Any]:  # pragma: no cover
    """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.

    Shape:
        Inputs:
        - query: :math:`(N, E)` where N is the batch size and E is the embedding dimension.
        - key: :math:`(N, E)`, where N is the batch size and E is the embedding dimension.
        - value: :math:`(N, E)` where N is the batch size and E is the embedding dimension.
        - key_padding_mask: :math:`(N)` where N is the batch size.
        If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
        will be unchanged. If a BoolTensor is provided, the positions with the
        value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length.
        3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
        S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
        positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
        while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
        are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
        is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
        N is the batch size and E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
        N is the batch size and E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(N, E)` where N is the batch size and E is the embedding dimension.
        - state: Internal state for continual computataion.

    """
    assert add_zero_attn is False, "add_zero_attn is not supported"
    assert key_padding_mask is None, "key_padding_mask is not supported"
    assert attn_mask is None, "attn_mask is not supported"
    assert static_k is None, "static_k is not supported"
    assert static_v is None, "static_v is not supported"

    # set up shape vars
    assert len(query.shape) == 2, "query should have shape (N, E)"
    assert len(key.shape) == 2, "key should have shape (N, E)"
    assert len(value.shape) == 2, "value should have shape (N, E)"
    query = query.unsqueeze(0)  # shape = (1, N, E)
    key = key.unsqueeze(0)  # shape = (1, N, E)
    value = value.unsqueeze(0)  # shape = (1, N, E)

    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert (
        head_dim * num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

    # compute in-projection
    if not use_separate_proj_weight:
        # Note: Also works for single step (unqueeze dim 0)
        q, k, v = F._in_projection_packed(
            query, key, value, in_proj_weight, in_proj_bias
        )
    else:
        assert (
            q_proj_weight is not None
        ), "use_separate_proj_weight is True but q_proj_weight is None"
        assert (
            k_proj_weight is not None
        ), "use_separate_proj_weight is True but k_proj_weight is None"
        assert (
            v_proj_weight is not None
        ), "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = F._in_projection(
            query,
            key,
            value,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            b_q,
            b_k,
            b_v,
        )

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        # if attn_mask is not None: TODO: Handle these branches
        #     attn_mask = F.pad(attn_mask, (0, 1))
        # if key_padding_mask is not None:
        #     key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # reshape q, k, v for multihead attention and make em batch first
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)

    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    # update source sequence length after adjustments
    # src_len = k.size(1)

    # merge key padding and attention masks
    # if key_padding_mask is not None:  # TODO: Handle this branch
    #     assert key_padding_mask.shape == (
    #         bsz,
    #         src_len,
    #     ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
    #     key_padding_mask = (
    #         key_padding_mask.view(bsz, 1, 1, src_len)
    #         .expand(-1, num_heads, -1, -1)
    #         .reshape(bsz * num_heads, 1, src_len)
    #     )
    #     if attn_mask is None:
    #         attn_mask = key_padding_mask
    #     elif attn_mask.dtype == torch.bool:
    #         attn_mask = attn_mask.logical_or(key_padding_mask)
    #     else:
    #         attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float  # TODO: Handle this branch
    # if attn_mask is not None and attn_mask.dtype == torch.bool:
    #     new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
    #     new_attn_mask.masked_fill_(attn_mask, float("-inf"))
    #     attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # (deep breath) calculate attention and out projection
    q, k, v = q.squeeze(1), k.squeeze(1), v.squeeze(1)
    attn_output, new_state = scaled_dot_product_attention_step(
        prev_state, q, k, v, attn_mask, dropout_p
    )
    attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    return attn_output, new_state


# Corresponds to MultiheadAttention in Pytorch v1.9
class MultiheadAttentionBase(CoModule, MultiheadAttention):
    """
    Continual MultiHeadAttention Base.


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
        scaled_dot_product_attention_default_state_fn: Callable = None,
        scaled_dot_product_attention_step_fn: Callable = None,
        forward_returns_attn_mask=True,
        embed_dim_second=False,
    ) -> None:
        MultiheadAttention.__init__(
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
        )
        self.sequence_len = sequence_len
        self._sdpa_default_state = scaled_dot_product_attention_default_state_fn
        self._sdpa_step = scaled_dot_product_attention_step_fn
        self.forward_returns_attn_mask = forward_returns_attn_mask
        self.embed_dim_second = embed_dim_second

    @abstractmethod
    def get_state(self) -> Optional[Any]:
        ...  # pragma: no cover

    @abstractmethod
    def set_state(self, state: Any):
        ...  # pragma: no cover

    @abstractmethod
    def clean_state(self):
        ...  # pragma: no cover

    def forward(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shapes for inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a ByteTensor is provided, the non-zero positions will be ignored while the position
              with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
              source sequence length.

              If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
              length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
              the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
              while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
              is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.

        Shapes for outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """
        if key is None:
            key = query
        if value is None:
            value = query

        if self.embed_dim_second:
            # N E T -> N T E
            query = query.permute(0, 2, 1)
            key = key.permute(0, 2, 1)
            value = value.permute(0, 2, 1)

        o, attn_output_weights = MultiheadAttention.forward(
            self, query, key, value, key_padding_mask, need_weights, attn_mask
        )

        if self.embed_dim_second:
            o = o.permute(0, 2, 1)  # N T E -> N E T

        if self.forward_returns_attn_mask:
            return o, attn_output_weights
        else:
            return o

    def _forward_step(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        prev_state: State = None,
        *args,
        **kwargs,
    ) -> Tuple[Optional[Tensor], State]:
        """Forward computation for a single step with state initialisation

        Args:
            query, key, value: step inputs of shape `(B, E)` where B is the batch size and E is the embedding dimension.

        Returns:
            Tuple[Optional[Tensor], State]: Step output and new state.
        """
        if prev_state is None:
            prev_state = (
                *self._sdpa_default_state(
                    batch_size=query.shape[0],
                    dtype=query.dtype,
                    device=query.device,
                ),
                torch.tensor(-self.sequence_len),
            )

        o, new_state = multi_head_attention_forward_step(
            self._sdpa_step,
            prev_state[:-1],
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
        )
        stride_index = prev_state[-1]
        if stride_index < 0:
            stride_index += 1

        new_state = (*new_state, stride_index)

        return (
            None if stride_index < 0 else o,
            new_state,
        )

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
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        """
        if key is None:
            key = query
        if value is None:
            value = query

        tmp_state = self.get_state()

        if not update_state and tmp_state:
            backup_state = _clone_state(tmp_state)

        o, tmp_state = MultiheadAttentionBase._forward_step(
            self, query, key, value, tmp_state
        )

        if o is not None and self.batch_first:
            o = o.transpose(1, 0)

        if update_state:
            self.set_state(tmp_state)
        elif tmp_state is not None:
            self.set_state(backup_state)

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
        """Forward computation for multiple steps with state initialisation

        Args:
            query (Tensor): query.
            key (Tensor): key.
            value (Tensor): value.
            update_state (bool): Whether internal state should be updated during this operation.

        Returns:
            Tensor: Stepwise layer outputs
        """
        if key is None:
            key = query
        if value is None:
            value = query

        if self.embed_dim_second:
            # N E T -> N T E
            query = query.permute(0, 2, 1)
            key = key.permute(0, 2, 1)
            value = value.permute(0, 2, 1)

        if self.batch_first:
            # N T E -> T N E
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tmp_state = self.get_state()

        if not update_state and tmp_state:
            backup_state = _clone_state(tmp_state)

        T = query.shape[0]
        assert T == key.shape[0]
        assert T == value.shape[0]
        outs = []
        for t in range(T):
            o, tmp_state = MultiheadAttentionBase._forward_step(
                self, query[t], key[t], value[t], tmp_state
            )

            if isinstance(o, Tensor):
                if self.batch_first:
                    o = o.transpose(0, 1)
                outs.append(o)

        if update_state:
            self.set_state(tmp_state)
        elif backup_state is not None:
            self.set_state(backup_state)

        if len(outs) == 0:
            return o

        o = torch.stack(outs, dim=int(self.batch_first))

        return o

    @property
    def receptive_field(self) -> int:
        return self.sequence_len

    @classmethod
    def build_from(cls, module: MultiheadAttention, sequence_len: int, **kwargs):
        comodule = cls(
            **{
                **dict(
                    embed_dim=module.embed_dim,
                    num_heads=module.num_heads,
                    dropout=module.dropout,
                    bias=module.in_proj_bias is not None,
                    add_bias_kv=module.bias_k is not None,
                    add_zero_attn=module.add_zero_attn,
                    kdim=module.kdim,
                    vdim=module.vdim,
                    batch_first=module.batch_first,
                    device=module.out_proj.weight.device,
                    dtype=module.out_proj.weight.dtype,
                    sequence_len=sequence_len,
                ),
                **kwargs,
            }
        )
        with torch.no_grad():
            if module.in_proj_weight is not None:
                comodule.in_proj_weight.copy_(module.in_proj_weight)

            if module.q_proj_weight is not None:
                comodule.q_proj_weight.copy_(module.q_proj_weight)
            if module.k_proj_weight is not None:
                comodule.k_proj_weight.copy_(module.k_proj_weight)
            if module.v_proj_weight is not None:
                comodule.v_proj_weight.copy_(module.v_proj_weight)

            if module.in_proj_bias is not None:
                comodule.in_proj_bias.copy_(module.in_proj_bias)
            if module.out_proj is not None:
                comodule.out_proj.weight.copy_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    comodule.out_proj.bias.copy_(module.out_proj.bias)
            if module.bias_k is not None:
                comodule.bias_k.copy_(module.bias_k)
            if module.bias_v is not None:
                comodule.bias_v.copy_(module.bias_v)
        return comodule


def scaled_dot_prod_attn_flops(
    sequence_len, embed_dim, include_muls=True, include_adds=False, include_exps=False
):
    n = sequence_len
    d = embed_dim

    flops = 0

    if include_muls:
        flops += 2 * n * n * d + 2 * n * d
    if include_adds:
        flops += 2 * n * n - n * d - n
    if include_exps:
        flops += n * n

    return flops
