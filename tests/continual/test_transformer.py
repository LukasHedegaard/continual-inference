import torch
from ptflops import get_model_complexity_info
from torch import nn

import continual as co


def test_trans_enc_b1():
    T = 10  # temporal sequence length
    E = 4  # embedding dimension
    N = 1  # batch size
    H = 2  # num heads

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=E, nhead=H, dim_feedforward=E * 2, dropout=0.0, batch_first=True
    )
    regenc = nn.TransformerEncoder(
        encoder_layer,
        num_layers=1,
        norm=nn.LayerNorm(E),
    )

    enc = co.TransformerEncoder.build_from(regenc, sequence_len=T)

    # NB: regular and continual transformers expect different input formats
    query = torch.randn((N, E, T))
    query_reg_format = query.permute(0, 2, 1)  # N, T, E

    # Baseline
    oreg = regenc.forward(query_reg_format)
    o = enc.forward(query)

    assert torch.allclose(oreg.permute(0, 2, 1), o)

    # Forward step
    o_step = enc.forward_steps(query[:, :, :-1])  # init

    o_step = enc.forward_step(query[:, :, -1])

    assert torch.allclose(o[:, :, -1], o_step)

    # FLOPs
    flops, _ = get_model_complexity_info(
        enc,
        (E, T),
        as_strings=False,
    )

    enc.call_mode = "forward_step"
    step_flops, _ = get_model_complexity_info(
        enc,
        (E,),
        as_strings=False,
    )

    assert step_flops <= flops / T
