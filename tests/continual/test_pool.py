import torch
from torch import nn

from continual import (
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
)

torch.manual_seed(42)

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


def test_AvgPool1d_padded():
    C = 1
    T = 3
    L = 5
    sample = torch.arange(0, L * C, dtype=torch.float).reshape((1, C, L))

    pool = nn.AvgPool1d(T, stride=1, padding=1)
    target = pool(sample)

    co_pool = AvgPool1d.build_from(pool)
    co_pool.clean_state()  # Nothing should happen

    # forward
    output2 = co_pool.forward(sample)
    assert torch.equal(target, output2)

    # forward_steps
    output = co_pool.forward_steps(sample, pad_end=True)
    assert torch.allclose(target, output)

    # broken-up
    co_pool.clean_state()
    firsts = co_pool.forward_steps(sample[:, :, :3], pad_end=False)
    assert torch.allclose(firsts, target[:, :, :2])

    mid = co_pool.forward_step(sample[:, :, 3])
    assert torch.allclose(mid, target[:, :, 2])

    lasts = co_pool.forward_steps(sample[:, :, 4:], pad_end=True)
    assert torch.allclose(lasts, target[:, :, 3:])


def test_AvgPool2d_stride():
    C = 1
    T = 3
    L = 7
    S = 3
    # sample = torch.randn((1, C, L))
    sample = torch.arange(0, L * C * S, dtype=torch.float).reshape((1, C, L, S))

    # Regular
    pool = nn.AvgPool2d(T, stride=(2, 1))
    target = pool(sample)

    # Continual
    co_pool = AvgPool2d.build_from(pool)

    output = co_pool.forward_steps(sample)
    assert torch.allclose(target, output)

    # Exact
    output2 = co_pool.forward(sample)
    assert torch.equal(target, output2)


def test_AvgPool3d():
    pool = nn.AvgPool3d((2, 2, 2))
    target = pool.forward(example_clip)

    co_pool = AvgPool3d.build_from(pool)
    output = co_pool.forward_steps(example_clip)
    assert torch.allclose(output, target)


def test_MaxPool1d_dilation():
    C = 1
    T = 3
    L = 7
    sample = torch.arange(0, L * C, dtype=torch.float).reshape((1, C, L))

    # Regular
    pool = nn.MaxPool1d(T, dilation=2)
    target = pool(sample)

    # Continual
    co_pool = MaxPool1d.build_from(pool)
    output = co_pool.forward_steps(sample)

    assert torch.allclose(target, output)


def test_MaxPool2d():
    C = 2
    T = 3
    L = 7
    S = 5
    sample = torch.arange(0, L * C * S, dtype=torch.float).reshape((1, C, L, S))

    # Regular
    pool = nn.MaxPool2d(T)
    target = pool(sample)

    # Continual
    co_pool = MaxPool2d.build_from(pool)
    output = co_pool.forward_steps(sample)

    assert torch.allclose(target, output)


def test_MaxPool3d():
    # Regular
    pool = nn.MaxPool3d(2)
    target = pool(example_clip)

    # Continual
    co_pool = MaxPool3d.build_from(pool)
    output = co_pool.forward_steps(example_clip)
    assert torch.allclose(output, target)


def test_AdaptiveAvgPool2d():
    C = 2
    L = 5
    S = 3
    sample = torch.normal(mean=torch.zeros(L * C * S)).reshape((1, C, L, S))

    # Regular
    pool = nn.AdaptiveAvgPool2d((1, 1))
    target = pool(sample)

    # Continual
    co_pool = AdaptiveAvgPool2d.build_from(pool, kernel_size=L)

    # Whole time-series
    output = co_pool.forward_steps(sample)
    assert torch.allclose(target, output)

    output2 = co_pool.forward(sample)
    assert torch.allclose(target, output2)


def test_AdaptiveAvgPool3d():
    sample = example_long[:, :, :4]
    next_sample = example_long[:, :, 1:5]
    pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    copool = AdaptiveAvgPool3d.build_from(pool, kernel_size=4)

    target = pool.forward(sample)

    # forward
    output = copool.forward_steps(sample)
    assert torch.allclose(output, target)

    # forward_steps
    output = copool.forward_steps(sample)
    assert torch.allclose(output, target)

    # Now that memory is full (via `forward_steps`), pooling works as expected for subsequent frames
    target_next = pool(next_sample).squeeze(2)
    output_frame_next = copool.forward_step(next_sample[:, :, -1])
    assert torch.allclose(target_next, output_frame_next)

    # pad_end: Here, empty the delay memory
    copool.clean_state()


def test_AdaptiveMaxPool2d():
    C = 2
    L = 5
    S = 3
    sample = torch.normal(mean=torch.zeros(L * C * S)).reshape((1, C, L, S))

    # Regular
    pool = nn.AdaptiveMaxPool2d((1, 1))
    target = pool(sample)

    # Continual
    co_pool = AdaptiveMaxPool2d.build_from(pool, kernel_size=L)

    # Whole time-series
    output = co_pool.forward_steps(sample)
    assert torch.allclose(target, output)

    output2 = co_pool.forward(sample)
    assert torch.allclose(target, output2)


def test_AdaptiveMaxPool3d():
    pool = nn.AdaptiveMaxPool3d((1, 1, 1))
    copool = AdaptiveMaxPool3d.build_from(pool, kernel_size=4)

    target = pool(example_clip)
    output = copool.forward_steps(example_clip)
    assert torch.allclose(output, target)

    # Now that memory is full (via `forward_steps`), pooling works as expected for subsequent frames
    target_next = pool(next_example_clip).squeeze(2)
    output_frame_next = copool.forward_step(next_example_frame)
    assert torch.allclose(target_next, output_frame_next)


def xtest_Pool1d_understanding():
    example = torch.tensor([[1, 2, 3, 10, 20, 30, 100, 200, 300]], dtype=torch.float)

    k2s2d1 = nn.MaxPool1d(kernel_size=2, stride=2, dilation=1)(example)
    assert torch.equal(
        k2s2d1,
        torch.tensor(
            [
                [
                    # 1,
                    2,  # max(2,1)
                    # 3,
                    10,  # max(10,3)
                    # 20,
                    30,
                    # 100,
                    200,
                    # 300,
                ]
            ],
            dtype=torch.float,
        ),
    )

    k2s1d2 = nn.MaxPool1d(kernel_size=2, stride=1, dilation=2)(example)
    assert torch.equal(
        k2s1d2,
        torch.tensor(
            [
                [
                    # 1,
                    # 2,
                    3,  # max(3, 1)
                    10,  # max(10, 2)
                    20,  # max(20, 3)
                    30,  # ...
                    100,
                    200,
                    300,
                ]
            ],
            dtype=torch.float,
        ),
    )

    k2s2d2 = nn.MaxPool1d(kernel_size=2, stride=2, dilation=2)(example)
    assert torch.equal(
        k2s2d2,
        torch.tensor(
            [
                [
                    # 1,
                    # 2,
                    3,  # max(3,1)
                    # 10,
                    20,  # max(20, 3)
                    # 30,
                    100,  # max(100, 20)
                    # 200,
                    300,  # max(300, 100)
                ]
            ],
            dtype=torch.float,
        ),
    )
