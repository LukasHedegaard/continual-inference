import torch
from ptflops import get_model_complexity_info
from torch.nn.modules.activation import MultiheadAttention

from continual.multihead_attention.single_output_mha import (
    SingleOutputMultiheadAttention,
)

torch.manual_seed(42)


def test_multi_head_attention_default_query_index():
    L = 10  # target sequence length
    S = L  # source sequence length
    E = 4  # embedding dimension
    N = 4  # batch size
    H = 2  # num heads
    mha = MultiheadAttention(
        embed_dim=E,
        num_heads=H,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    )
    query_index = -1
    comha = SingleOutputMultiheadAttention.build_from(
        mha, sequence_len=L, query_index=query_index
    )

    query = torch.randn((L, N, E))
    key = torch.randn((S, N, E))
    value = torch.randn((S, N, E))

    # forward description
    # query: (L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension. (N,L,E) if batch_first is True.
    # key: (S,N,E), where S is the source sequence length, N is the batch size, E is the embedding dimension. (N,S,E) if batch_first is True.
    # value: (S,N,E) where S is the source sequence length, N is the batch size, E is the embedding dimension. (N,S,E) if batch_first is True.
    attn_output, attn_weight = mha.forward(query, key, value)

    # Forward is identical
    attn_output2, attn_weight2 = comha.forward(query, key, value)
    assert torch.equal(attn_output[query_index].unsqueeze(0), attn_output2)
    assert torch.equal(attn_weight[:, query_index].unsqueeze(1), attn_weight2)

    # Forward steps gives same output for full sequence
    attn_output3 = comha.forward_steps(query, key, value)
    assert torch.allclose(attn_output[query_index], attn_output3)

    # Initialise, then use forward_step
    comha.clean_state()
    attn_output_dummy = comha.forward_steps(query[:-1], key[:-1], value[:-1])
    assert attn_output_dummy is None
    attn_output4 = comha.forward_step(query[-1], key[-1], value[-1])
    assert torch.allclose(attn_output[query_index], attn_output4)

    # Shift query, key and value by a time-step
    query_step = torch.randn((N, E))
    key_step = torch.randn((N, E))
    value_step = torch.randn((N, E))

    query2 = torch.cat((query[1:], query_step.unsqueeze(0)), dim=0)
    key2 = torch.cat((key[1:], key_step.unsqueeze(0)), dim=0)
    value2 = torch.cat((value[1:], value_step.unsqueeze(0)), dim=0)

    assert torch.equal(query[1:], query2[:-1])
    assert torch.equal(key[1:], key2[:-1])
    assert torch.equal(value[1:], value2[:-1])

    attn_output_next, _ = mha.forward(query2, key2, value2)

    # Continual MHA should yield the same result by using query_step, key_step, and value_step
    attn_output_next2 = comha.forward_step(query_step, key_step, value_step)
    assert torch.allclose(attn_output_next[query_index], attn_output_next2)


def test_multi_head_attention_nondefault_query_index():
    L = 10  # target sequence length
    S = L  # source sequence length
    E = 4  # embedding dimension
    N = 4  # batch size
    H = 2  # num heads
    mha = MultiheadAttention(
        embed_dim=E,
        num_heads=H,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    )
    query_index = 2
    comha = SingleOutputMultiheadAttention.build_from(
        mha, sequence_len=L, query_index=query_index
    )

    query = torch.randn((L, N, E))
    key = torch.randn((S, N, E))
    value = torch.randn((S, N, E))

    # forward description
    # query: (L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension. (N,L,E) if batch_first is True.
    # key: (S,N,E), where S is the source sequence length, N is the batch size, E is the embedding dimension. (N,S,E) if batch_first is True.
    # value: (S,N,E) where S is the source sequence length, N is the batch size, E is the embedding dimension. (N,S,E) if batch_first is True.
    attn_output, attn_weight = mha.forward(query, key, value)

    # Forward is identical
    attn_output2, attn_weight2 = comha.forward(query, key, value)
    assert torch.equal(attn_output[query_index].unsqueeze(0), attn_output2)
    assert torch.equal(attn_weight[:, query_index].unsqueeze(1), attn_weight2)

    # Forward steps gives same output for full sequence
    attn_output3 = comha.forward_steps(query, key, value)
    assert torch.allclose(attn_output[query_index], attn_output3)

    # Initialise, then use forward_step
    comha.clean_state()
    attn_output_dummy = comha.forward_steps(query[:-1], key[:-1], value[:-1])
    assert attn_output_dummy is None
    attn_output4 = comha.forward_step(query[-1], key[-1], value[-1])
    assert torch.allclose(attn_output[query_index], attn_output4)

    # Shift query, key and value by a time-step
    query_step = torch.randn((N, E))
    key_step = torch.randn((N, E))
    value_step = torch.randn((N, E))

    query2 = torch.cat((query[1:], query_step.unsqueeze(0)), dim=0)
    key2 = torch.cat((key[1:], key_step.unsqueeze(0)), dim=0)
    value2 = torch.cat((value[1:], value_step.unsqueeze(0)), dim=0)

    assert torch.equal(query[1:], query2[:-1])
    assert torch.equal(key[1:], key2[:-1])
    assert torch.equal(value[1:], value2[:-1])

    attn_output_next, _ = mha.forward(query2, key2, value2)

    # Continual MHA should yield the same result by using query_step, key_step, and value_step
    attn_output_next2 = comha.forward_step(query_step, key_step, value_step)
    assert torch.allclose(attn_output_next[query_index], attn_output_next2)


def test_flops():
    L = 10  # target sequence length
    E = 4  # embedding dimension
    H = 2  # num heads

    # Regular net
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mha = MultiheadAttention(
                embed_dim=E,
                num_heads=H,
                dropout=0.0,
                bias=True,
                add_bias_kv=False,
                add_zero_attn=False,
                kdim=None,
                vdim=None,
                batch_first=True,
                device=None,
                dtype=None,
            )

        def forward(self, x):
            y, _ = self.mha(x, x, x)
            return y

    net = Net()

    flops, params = get_model_complexity_info(
        net,
        (L, E),
        as_strings=False,  # input_constructor=input_constructor,
    )

    # Continual net
    co_net = SingleOutputMultiheadAttention(
        embed_dim=E,
        num_heads=H,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=True,
        device=None,
        dtype=None,
        sequence_len=L,
        forward_returns_attn_mask=False,
    )

    co_flops, co_params = get_model_complexity_info(
        co_net,
        (L, E),
        as_strings=False,  # input_constructor=input_constructor,
    )

    assert 0.5 * flops > co_flops  # Within 10%
    assert co_params == params

    co_net.call_mode = "forward_step"
    co_step_flops, co_step_params = get_model_complexity_info(
        co_net,
        (E,),
        as_strings=False,  # input_constructor=input_constructor,
    )

    assert 0.1 * flops > co_step_flops
    assert co_step_flops > 0
    assert co_step_params == params

    # Including adds and exps in flops
    co_step_flops_all = co_net.flops(
        include_muls=True, include_adds=True, include_exps=True
    )
    assert co_step_flops_all > co_step_flops
