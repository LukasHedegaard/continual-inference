import torch
from torch import nn

import continual as co


def test_nn_norms():
    S = 3

    long_example_clip = torch.normal(mean=torch.zeros(10 * 3 * 3)).reshape(
        (1, 1, 10, 3, 3)
    )

    b_norm = nn.BatchNorm3d(1)
    b_norm.weight = nn.Parameter(3 * torch.ones_like(b_norm.weight))
    b_norm.bias = nn.Parameter(1 * torch.ones_like(b_norm.bias))

    i_norm = nn.InstanceNorm3d(2, affine=True, track_running_stats=True)
    i_norm.weight = nn.Parameter(4 * torch.ones_like(i_norm.weight))
    i_norm.bias = nn.Parameter(2 * torch.ones_like(i_norm.bias))

    l_norm = nn.LayerNorm([S, S])  # NB: Doesn't work over temporal axis
    l_norm.weight = nn.Parameter(5 * torch.ones_like(l_norm.weight))
    l_norm.bias = nn.Parameter(3 * torch.ones_like(l_norm.bias))

    g_norm = nn.GroupNorm(2, 2)
    g_norm.weight = nn.Parameter(6 * torch.ones_like(g_norm.weight))
    g_norm.bias = nn.Parameter(4 * torch.ones_like(g_norm.bias))

    seq = nn.Sequential(
        b_norm,
        nn.Conv3d(
            in_channels=1,
            out_channels=2,
            kernel_size=(5, S, S),
            bias=True,
            padding=(0, 1, 1),
            padding_mode="zeros",
        ),
        i_norm,
        l_norm,
        g_norm,
        nn.Conv3d(
            in_channels=2,
            out_channels=1,
            kernel_size=(3, S, S),
            bias=True,
            padding=(0, 1, 1),
            padding_mode="zeros",
        ),
        nn.MaxPool3d(kernel_size=(1, 2, 2)),
    )
    seq.eval()

    coseq = co.Sequential.build_from(seq)
    coseq.eval()

    assert coseq.delay == (5 - 1) + (3 - 1)

    # forward
    output = seq.forward(long_example_clip)
    co_output = coseq.forward(long_example_clip)
    assert torch.allclose(output, co_output)

    # forward_steps
    co_output_firsts_0 = coseq.forward_steps(
        long_example_clip[:, :, :-1], update_state=False
    )
    co_output_firsts = coseq.forward_steps(long_example_clip[:, :, :-1])
    assert torch.allclose(co_output_firsts, co_output_firsts_0, atol=1e-7)
    assert torch.allclose(co_output_firsts, output[:, :, :-1], atol=1e-7)

    # forward_step
    co_output_last_0 = coseq.forward_step(
        long_example_clip[:, :, -1], update_state=False
    )
    co_output_last = coseq.forward_step(long_example_clip[:, :, -1])
    assert torch.allclose(co_output_last, co_output_last_0, atol=1e-7)
    assert torch.allclose(co_output_last, output[:, :, -1], atol=1e-7)

    # Clean state can be used to restart seq computation
    coseq.clean_state()
    co_output_firsts = coseq.forward_steps(long_example_clip[:, :, :-1])
    assert torch.allclose(co_output_firsts, output[:, :, :-1], atol=1e-7)
