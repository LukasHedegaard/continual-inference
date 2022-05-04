import torch
from torch.nn.modules.activation import MultiheadAttention

from continual.multihead_attention import (
    RetroactiveMultiheadAttention,
    SingleOutputMultiheadAttention,
)


def test_stacked_mha():
    L = 10  # target sequence length
    E = 4  # embedding dimension
    N = 5  # batch size
    H = 2  # num heads

    # Regular net
    r1 = MultiheadAttention(
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
    r2 = MultiheadAttention(
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

    class Reg(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = r1
            self.l2 = r2

        def forward(self, x):
            x, _ = self.l1(x, x, x)
            x, _ = self.l2(x, x, x)
            return x

    reg = Reg()

    # Continal net
    c1 = RetroactiveMultiheadAttention.build_from(
        r1, sequence_len=L, forward_returns_attn_mask=False
    )
    c2 = SingleOutputMultiheadAttention.build_from(
        r2, sequence_len=L, query_index=-1, forward_returns_attn_mask=False
    )

    class Con(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = c1
            self.l2 = c2

        def forward(self, x):
            x = self.l1(x, x, x)
            x = self.l2(x, x, x)
            return x

        def forward_step(self, x):
            x = self.l1.forward_step(x, x, x)
            if not isinstance(x, torch.Tensor):
                return x
            x = self.l2.forward(x, x, x)
            return x

        def forward_steps(self, x):
            outs = []
            for t in range(x.shape[1]):
                o = self.forward_step(x[:, t])
                if isinstance(o, torch.Tensor):
                    outs.append(o)

            if len(outs) > 0:
                return torch.stack(outs, dim=1)
            return o

    con = Con()

    # Test
    x = torch.randn((N, L, E))

    o_reg = reg.forward(x)  # Baseline

    # Forward
    o_con = con.forward(x)

    assert torch.allclose(o_reg[:, -1], o_con[:, 0])

    # Forward step
    _ = con.forward_steps(x[:, :-1])  # initialise
    o_con_step = con.forward_step(x[:, -1])

    assert torch.allclose(o_reg[:, -1], o_con_step[:, 0])
