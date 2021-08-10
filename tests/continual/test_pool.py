import torch
from torch.nn import AdaptiveAvgPool2d, AvgPool1d

from continual import AdaptiveAvgPoolCo2d, AvgPoolCo1d


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
