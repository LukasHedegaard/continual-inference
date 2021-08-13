import torch
from torch import nn

import continual as co

torch.manual_seed(42)


def test_sequential():
    S = 3

    long_example_clip = torch.normal(mean=torch.zeros(8 * 3 * 3)).reshape(
        (1, 1, 8, 3, 3)
    )
    next_example_frame = torch.normal(mean=torch.zeros(3 * 3)).reshape((1, 1, 3, 3))
    long_next_example_clip = torch.stack(
        [
            *[long_example_clip[:, :, i] for i in range(1, 8)],
            next_example_frame,
        ],
        dim=2,
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
        nn.MaxPool3d(kernel_size=(2, 2, 2)),
    )

    coseq = co.Sequential.build_from(seq)

    # Forward results are identical
    output = seq.forward(long_example_clip)
    co_output = coseq.forward(long_example_clip)

    assert torch.allclose(output, co_output)
