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

    assert coseq.delay == (5 - 1) + (3 - 1)

    # Forward results are identical
    output = seq.forward(long_example_clip)
    co_output = coseq.forward(long_example_clip)
    assert torch.allclose(output, co_output)

    # Alternative forwards
    co_output_firsts = coseq.forward_steps(long_example_clip[:, :, :-1])
    assert torch.allclose(output[:, :, :-1], co_output_firsts)

    co_output_last = coseq.forward_step(long_example_clip[:, :, -1])
    assert torch.allclose(output[:, :, -1], co_output_last)

    # Clean state can be used to restart seq computation
    coseq.clean_state()
    co_output_firsts = coseq.forward_steps(long_example_clip[:, :, :-1])
    assert torch.allclose(output[:, :, :-1], co_output_firsts)


def test_sum_aggregation():
    ones = torch.ones((1, 2, 4, 3, 3))
    twos = torch.ones((1, 2, 4, 3, 3)) * 2
    res = co.container.parallel_sum([ones, ones])
    assert torch.allclose(res, twos)


def test_concat_aggregation():
    ones = torch.ones((1, 2, 4, 3, 3))
    twos = torch.ones((1, 2, 4, 3, 3)) * 2
    res = co.container.parallel_concat([ones, twos])
    assert res.shape == (1, 4, 4, 3, 3)
    assert torch.allclose(res[:, :2], ones)
    assert torch.allclose(res[:, 2:], twos)


def test_residual():
    input = torch.normal(mean=torch.zeros(6 * 3 * 3)).reshape((1, 1, 6, 3, 3))

    t_pad = 1
    conv = nn.Conv3d(1, 1, (3, 3, 3), padding=(t_pad, 1, 1))

    co_conv = co.Conv3d(1, 1, (3, 3, 3), padding=(t_pad, 1, 1))
    co_conv.load_state_dict(conv.state_dict())

    co_res = co.Residual(co_conv)

    # Target behavior: Discard outputs from temporal padding
    target = (conv(input) + input)[:, :, 1:-1]

    # forward
    out_manual_res = co_conv.forward(input) + input[:, :, 1:-1]
    assert torch.allclose(out_manual_res, target)

    out_res = co_res.forward(input)
    assert torch.allclose(out_res, target)

    # forward_steps
    out_firsts = co_res.forward_steps(input[:, :, :-1])
    assert torch.allclose(
        out_firsts[:, :, co_res.delay + t_pad : -1], target[:, :, :-2]
    )

    # forward_step
    out_last = co_res.forward_step(input[:, :, -1])
    assert torch.allclose(out_last, target[:, :, -1])


def test_parallel():
    input = torch.normal(mean=torch.zeros(7 * 5 * 5)).reshape((1, 1, 7, 5, 5))

    c5 = co.Conv3d(1, 1, (5, 5, 5), padding=(2, 2, 2))
    c3 = co.Conv3d(1, 1, (3, 3, 3), padding=(1, 1, 1))
    c1 = co.Conv3d(1, 1, (1, 1, 1), padding=(0, 0, 0))
    par = co.Parallel(c5, c3, c1)

    # forward
    out_all = par(input)
    assert out_all.shape == (1, 1, 3, 5, 5)

    # forward_steps
    out_firsts = par.forward_steps(input[:, :, :-1])
    assert torch.allclose(out_firsts[:, :, 2 * par.delay : -1], out_all[:, :, :-2])

    # forward_step
    out_last = par.forward_step(input[:, :, -1])
    assert torch.allclose(out_last, out_all[:, :, -1])

    par.clean_state()
