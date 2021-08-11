import torch
from torch.nn.modules.pooling import (
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AdaptiveMaxPool3d,
    AvgPool1d,
    AvgPool3d,
    MaxPool3d,
)

from continual import (
    AdaptiveAvgPoolCo2d,
    AdaptiveAvgPoolCo3d,
    AdaptiveMaxPoolCo3d,
    AvgPoolCo1d,
    AvgPoolCo3d,
    MaxPoolCo3d,
)

torch.manual_seed(42)


def test_AvgPoolCo1d():
    C = 2
    T = 3
    L = 5
    sample = torch.normal(mean=torch.zeros(L * C)).reshape((1, C, L))

    # Regular
    pool = AvgPool1d(T, stride=1)
    target = pool(sample)

    # Continual
    co_pool = AvgPoolCo1d(T)
    output = []

    # Frame by frame
    for i in range(sample.shape[2]):
        output.append(co_pool(sample[:, :, i]))

    # Match after delay of T - 1
    for t in range(sample.shape[2] - (T - 1)):
        assert torch.allclose(target[:, :, t], output[t + (T - 1)])

    # Whole time-series
    output = co_pool.forward_regular(sample)
    assert torch.allclose(target, output)

    # Exact
    output2 = co_pool.forward_regular_unrolled(sample)
    assert torch.equal(target, output2)


def test_AdaptiveAvgPoolCo2d():
    C = 2
    L = 5
    S = 3
    sample = torch.normal(mean=torch.zeros(L * C * S)).reshape((1, C, L, S))

    # Regular
    pool = AdaptiveAvgPool2d((1, 1))
    target = pool(sample)

    # Continual
    co_pool = AdaptiveAvgPoolCo2d(window_size=L, output_size=(1,))

    # Whole time-series
    output = co_pool.forward_regular(sample)
    assert torch.allclose(target, output)

    output2 = co_pool.forward_regular_unrolled(sample)
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


def test_AvgPoolCo3d():
    target = AvgPool3d((2, 2, 2))(example_clip)
    output = AvgPoolCo3d(window_size=2, kernel_size=(2, 2)).forward_regular(
        example_clip
    )
    sub_output = torch.stack(
        [
            output[:, :, 0],
            output[:, :, 2],
        ],
        dim=2,
    )
    assert torch.allclose(sub_output, target)


def test_AdaptiveAvgPoolCo3d():
    pool = AdaptiveAvgPool3d((1, 1, 1))
    rpool = AdaptiveAvgPoolCo3d(window_size=4, output_size=(1, 1))

    target = pool(example_clip)
    output = rpool.forward_regular(example_clip)
    assert torch.allclose(output, target)

    # Now that memory is full (via `forward_regular`), pooling works as expected for subsequent frames
    target_next = pool(next_example_clip).squeeze(2)
    output_frame_next = rpool(next_example_frame)
    assert torch.allclose(target_next, output_frame_next)


def test_MaxPoolCo3d():
    target = MaxPool3d((2, 2, 2))(example_clip)
    output = MaxPoolCo3d(window_size=2, kernel_size=(2, 2)).forward_regular(
        example_clip
    )
    sub_output = torch.stack(
        [
            output[:, :, 0],
            output[:, :, 2],
        ],
        dim=2,
    )
    assert torch.allclose(sub_output, target)


def test_AdaptiveMaxPoolCo3d():
    target = AdaptiveMaxPool3d((1, 1, 1))(example_clip)
    output = AdaptiveMaxPoolCo3d(window_size=4, output_size=(1, 1)).forward_regular(
        example_clip
    )
    assert torch.allclose(output, target)


def test_MaxPoolCo3d_dilation():
    target = MaxPool3d((2, 2, 2), dilation=(2, 1, 1))(example_long)
    output = MaxPoolCo3d(
        window_size=4, kernel_size=(2, 2), temporal_dilation=2
    ).forward_regular(example_long)
    assert torch.allclose(target, output.index_select(2, torch.tensor([0, 2, 4])))
