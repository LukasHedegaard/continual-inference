import torch
from torch import nn

import continual as co


def test_linear():
    mod = nn.Linear(2, 3, bias=True)
    comod = co.Linear.build_from(mod)

    assert comod.delay == 0
    assert (
        comod.__repr__()
        == "Linear(in_features=2, out_features=3, bias=True, channel_dim=-1)"
    )

    #                     N, T, C
    sample = torch.randn((1, 3, 2))

    target = mod.forward(sample)

    # forward
    o = comod.forward(sample)
    assert torch.allclose(target, o)

    # forward_steps
    o_steps = comod.forward_steps(sample[:, :2, :])
    assert torch.allclose(target[:, :2, :], o_steps)

    # forward_step
    o_step = comod.forward_step(sample[:, 2, :])
    assert torch.allclose(target[:, 2, :], o_step)
    assert torch.allclose(target[:, 2, :], o_step)

    # Alternative channel_dim:
    comod_alt = co.Linear.build_from(mod, channel_dim=1)
    sample_alt = sample.swapaxes(1, 2)
    o_step_alt = comod_alt.forward_step(sample_alt[:, :, 2])
    assert torch.allclose(target[:, 2, :], o_step_alt)
