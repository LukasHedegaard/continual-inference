import torch
from torch import nn

from continual import (
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AdaptiveMaxPool3d,
    AvgPool1d,
    AvgPool3d,
    MaxPool3d,
)

torch.manual_seed(42)


def test_AvgPool1d():
    C = 2
    T = 3
    L = 5
    sample = torch.normal(mean=torch.zeros(L * C)).reshape((1, C, L))

    # Regular
    pool = nn.AvgPool1d(T, stride=1)
    target = pool(sample)

    # Continual
    co_pool = AvgPool1d(T)
    output = []

    # Frame by frame
    for i in range(sample.shape[2]):
        output.append(co_pool.forward_step(sample[:, :, i]))

    # Match after delay of T - 1
    for t in range(sample.shape[2] - (T - 1)):
        assert torch.allclose(target[:, :, t], output[t + (T - 1)])

    # Whole time-series
    output = co_pool.forward_steps(sample)
    assert torch.allclose(target, output)

    # Exact
    output2 = co_pool.forward(sample)
    assert torch.equal(target, output2)


def test_AdaptiveAvgPool2d():
    C = 2
    L = 5
    S = 3
    sample = torch.normal(mean=torch.zeros(L * C * S)).reshape((1, C, L, S))

    # Regular
    pool = nn.AdaptiveAvgPool2d((1, 1))
    target = pool(sample)

    # Continual
    co_pool = AdaptiveAvgPool2d(window_size=L, output_size=(1,))

    # Whole time-series
    output = co_pool.forward_steps(sample)
    assert torch.allclose(target, output)

    output2 = co_pool.forward(sample)
    assert torch.allclose(target, output2)


example_clip = torch.normal(mean=torch.zeros(2 * 4 * 4 * 4)).reshape((1, 2, 4, 4, 4))
example_long = torch.normal(mean=torch.zeros(2 * 8 * 4 * 4)).reshape((1, 2, 8, 4, 4))
next_example_frame = torch.normal(mean=torch.zeros(2 * 1 * 4 * 4)).reshape((1, 2, 4, 4))
next_example_clip = torch.stack(
    [
        example_clip[:, :, 1],
        example_clip[:, :, 2],
        example_clip[:, :, 3],
        next_example_frame,
    ],
    dim=2,
)


def test_AvgPool3d():
    target = nn.AvgPool3d((2, 2, 2))(example_clip)
    output = AvgPool3d(window_size=2, kernel_size=(2, 2)).forward_steps(example_clip)
    sub_output = torch.stack(
        [
            output[:, :, 0],
            output[:, :, 2],
        ],
        dim=2,
    )
    assert torch.allclose(sub_output, target)


def test_AdaptiveAvgPool3d():
    pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    copool = AdaptiveAvgPool3d(window_size=4, output_size=(1, 1))

    target = pool(example_clip)
    output = copool.forward_steps(example_clip)
    assert torch.allclose(output, target)

    # Now that memory is full (via `forward_steps`), pooling works as expected for subsequent frames
    target_next = pool(next_example_clip).squeeze(2)
    output_frame_next = copool.forward_step(next_example_frame)
    assert torch.allclose(target_next, output_frame_next)


def test_MaxPool3d():
    target = nn.MaxPool3d((2, 2, 2))(example_clip)
    output = MaxPool3d(window_size=2, kernel_size=(2, 2)).forward_steps(example_clip)
    sub_output = torch.stack(
        [
            output[:, :, 0],
            output[:, :, 2],
        ],
        dim=2,
    )
    assert torch.allclose(sub_output, target)


def test_AdaptiveMaxPool3d():
    target = nn.AdaptiveMaxPool3d((1, 1, 1))(example_clip)
    output = AdaptiveMaxPool3d(window_size=4, output_size=(1, 1)).forward_steps(
        example_clip
    )
    assert torch.allclose(output, target)


def test_MaxPool3d_dilation():
    target = nn.MaxPool3d((2, 2, 2), dilation=(2, 1, 1))(example_long)
    output = MaxPool3d(
        window_size=4, kernel_size=(2, 2), temporal_dilation=2
    ).forward_steps(example_long)
    assert torch.allclose(target, output.index_select(2, torch.tensor([0, 2, 4])))
