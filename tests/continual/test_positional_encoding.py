import torch

from continual.positional_encoding import RecyclingPositionalEncoding


def test_RecyclingPositionalEncoding_learned():
    B, C, T = 2, 3, 4

    x = torch.zeros((B, C, T))

    cpe = RecyclingPositionalEncoding(
        embed_dim=C, num_embeds=T, forward_update_index_steps=0
    )

    o_forward = cpe.forward(x)

    o_forward_steps = cpe.forward_steps(x[:, :, :-1])
    assert torch.equal(o_forward[:, :, :-1], o_forward_steps)

    o_forward_step = cpe.forward_step(x[:, :, -1])
    assert torch.equal(o_forward[:, :, -1], o_forward_step)

    cpe.clean_state()
    assert torch.equal(cpe.forward(x), o_forward)


def test_RecyclingPositionalEncoding_static():
    B, C, T = 2, 13, 4

    x = torch.zeros((B, C, T))

    cpe = RecyclingPositionalEncoding(
        embed_dim=C, num_embeds=T, forward_update_index_steps=0, learned=False
    )

    o_forward = cpe.forward(x)

    o_forward_steps = cpe.forward_steps(x[:, :, :-1])
    assert torch.equal(o_forward[:, :, :-1], o_forward_steps)

    o_forward_step = cpe.forward_step(x[:, :, -1])
    assert torch.equal(o_forward[:, :, -1], o_forward_step)
