import torch
from torch import nn

import continual as co

torch.manual_seed(42)


def test_sequential():
    S = 3

    long_example_clip = torch.normal(mean=torch.zeros(10 * 3 * 3)).reshape(
        (1, 1, 10, 3, 3)
    )

    seq = nn.Sequential(
        nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(5, S, S),
            bias=True,
            padding=(0, 1, 1),
            padding_mode="zeros",
        ),
        nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, S, S),
            bias=True,
            padding=(0, 1, 1),
            padding_mode="zeros",
        ),
        nn.MaxPool3d(kernel_size=(1, 2, 2)),
    )

    coseq = co.Sequential.build_from(seq)

    # Forward results are identical
    output = seq.forward(long_example_clip)
    co_output = coseq.forward(long_example_clip)
    assert torch.allclose(output, co_output)

    # Alternative forwards
    co_output_firsts = coseq.forward_steps(long_example_clip[:, :, :-1])
    assert torch.allclose(output[:, :, :-1], co_output_firsts)

    co_output_last = coseq.forward_step(long_example_clip[:, :, -1])
    assert torch.allclose(output[:, :, -1], co_output_last)
