from collections import OrderedDict

import torch
from torch import nn

import continual as co
from continual.module import TensorPlaceholder

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

    # forward
    output = seq.forward(long_example_clip)
    co_output = coseq.forward(long_example_clip)
    assert torch.allclose(output, co_output)

    # forward_steps
    co_output_firsts_0 = coseq.forward_steps(
        long_example_clip[:, :, :-1], update_state=False
    )
    co_output_firsts = coseq.forward_steps(long_example_clip[:, :, :-1])
    assert torch.allclose(co_output_firsts, co_output_firsts_0)
    assert torch.allclose(co_output_firsts, output[:, :, :-1])

    # forward_step
    co_output_last_0 = coseq.forward_step(
        long_example_clip[:, :, -1], update_state=False
    )
    co_output_last = coseq.forward_step(long_example_clip[:, :, -1])
    assert torch.allclose(co_output_last, co_output_last_0)
    assert torch.allclose(co_output_last, output[:, :, -1])

    # Clean state can be used to restart seq computation
    coseq.clean_state()
    co_output_firsts = coseq.forward_steps(long_example_clip[:, :, :-1])
    assert torch.allclose(co_output_firsts, output[:, :, :-1])


def test_sequential_with_TensorPlaceholder():
    sample = torch.arange(32, dtype=torch.float).reshape((1, 1, 32))

    seq = nn.Sequential(
        nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            bias=False,
            padding=1,
            padding_mode="zeros",
        ),
        nn.MaxPool1d(
            kernel_size=2,
            stride=2,  # Has temporal skips
            padding=0,
        ),
        nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            bias=False,
            stride=2,  # Has temporal skips
            padding=1,
            padding_mode="zeros",
        ),
    )
    torch.nn.init.ones_(seq[0].weight)
    torch.nn.init.ones_(seq[2].weight)

    coseq = co.Sequential.build_from(seq)
    assert coseq.stride == 4
    assert coseq.padding == 3
    assert coseq.receptive_field == 8
    assert coseq.delay == 4

    target = seq.forward(sample)

    # forward_steps with padding
    output = coseq.forward_steps(sample, pad_end=True)

    assert torch.allclose(target, output)

    coseq.clean_state()

    out_stepwise = []

    for i in range(sample.shape[2]):
        out_stepwise.append(coseq.forward_step(sample[:, :, i]))

    out_cleaned = torch.stack(
        [o for o in out_stepwise if isinstance(o, torch.Tensor)], dim=2
    )

    assert torch.allclose(target[:, :, :-1], out_cleaned)


def test_sum_reduce():
    ones = torch.ones((1, 2, 4, 3, 3))
    twos = torch.ones((1, 2, 4, 3, 3)) * 2
    res = co.container.reduce_sum([ones, ones])
    assert torch.allclose(res, twos)


def test_concat_reduce():
    ones = torch.ones((1, 2, 4, 3, 3))
    twos = torch.ones((1, 2, 4, 3, 3)) * 2
    res = co.container.reduce_concat([ones, twos])
    assert res.shape == (1, 4, 4, 3, 3)
    assert torch.allclose(res[:, :2], ones)
    assert torch.allclose(res[:, 2:], twos)


def test_residual():
    input = torch.arange(6, dtype=torch.float).reshape((1, 1, 6))

    t_pad = 1
    conv = nn.Conv1d(1, 1, kernel_size=3, padding=t_pad, bias=False)
    torch.nn.init.ones_(conv.weight)

    co_conv = co.Conv1d.build_from(conv)

    co_res = co.Residual(co_conv)

    # Target behavior: Discard outputs from temporal padding
    target = conv(input) + input

    # forward
    out_manual_res = co_conv.forward(input) + input
    assert torch.allclose(out_manual_res, target)

    out_res = co_res.forward(input)
    assert torch.allclose(out_res, target)

    # forward_steps
    out_firsts = co_res.forward_steps(input[:, :, :-1], pad_end=False)
    assert torch.allclose(out_firsts, target[:, :, :4])

    # forward_step
    out_last = co_res.forward_step(input[:, :, -1])
    assert torch.allclose(out_last, target[:, :, -2])


def test_broadcast_reduce():
    input = torch.arange(7, dtype=torch.float).reshape((1, 1, 7))

    c5 = co.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
    c3 = co.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
    c1 = co.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)
    torch.nn.init.ones_(c5.weight)
    torch.nn.init.ones_(c3.weight)
    torch.nn.init.ones_(c1.weight)
    par = co.BroadcastReduce(OrderedDict([("c5", c5), ("c3", c3), ("c1", c1)]))

    assert par.stride == 1
    assert par.delay == 2
    assert par.padding == 2
    assert "BroadcastReduce(" in par.__repr__() and "reduce=" in par.__repr__()

    # forward
    out_all = par.forward(input)
    assert torch.allclose(
        out_all, torch.tensor([[[4.0, 10.0, 18.0, 27.0, 36.0, 38.0, 32.0]]])
    )

    # forward_step
    out_steps = [par.forward_step(input[:, :, i]) for i in range(input.shape[2])]
    assert all(isinstance(o, TensorPlaceholder) for o in out_steps[: par.delay])

    out_steps = torch.stack(out_steps[par.delay :], dim=2)
    assert torch.allclose(out_steps, out_all[:, :, : -par.delay])

    # forward_steps
    par.clean_state()
    out_steps_0 = par.forward_steps(input[:, :, :-1], pad_end=False, update_state=False)
    out_steps = par.forward_steps(input[:, :, :-1], pad_end=False)
    assert torch.allclose(out_steps, out_steps_0)
    assert torch.allclose(out_steps, out_all[:, :, : -par.delay - 1])

    out_step_0 = par.forward_step(input[:, :, -1], update_state=False)  # continuation
    out_step = par.forward_step(input[:, :, -1])  # continuation
    assert torch.allclose(out_step, out_step_0)
    assert torch.allclose(out_step, out_all[:, :, -par.delay - 1])

    # with pad_end
    par.clean_state()
    out_steps = par.forward_steps(input, pad_end=True)
    assert torch.allclose(out_steps, out_all)


def test_flat_state_dict():
    # >> Part 1: Save both flat and original state dicts

    # If modules are not named, it can be flattened
    seq_to_flatten = co.Sequential(nn.Conv1d(1, 1, 3))

    sd = seq_to_flatten.state_dict()
    assert set(sd) == {"0.weight", "0.bias"}

    sd_flat = seq_to_flatten.state_dict(flatten=True)
    assert set(sd_flat) == {"weight", "bias"}

    seq_not_to_flatten = co.Sequential(OrderedDict([("c1", nn.Conv1d(1, 1, 3))]))
    sd_no_flat = seq_not_to_flatten.state_dict(flatten=True)
    assert set(sd_no_flat) == {"c1.weight", "c1.bias"}

    # A nested example:
    nested = co.BroadcastReduce(seq_to_flatten, seq_not_to_flatten)
    sd = nested.state_dict()
    assert set(sd) == {"0.0.weight", "0.0.bias", "1.c1.weight", "1.c1.bias"}

    sd_flat = nested.state_dict(flatten=True)
    assert set(sd_flat) == {"weight", "bias", "c1.weight", "c1.bias"}

    # >> Part 2: Load flat state dict
    nested_new = co.BroadcastReduce(
        co.Sequential(nn.Conv1d(1, 1, 3)),
        co.Sequential(OrderedDict([("c1", nn.Conv1d(1, 1, 3))])),
    )

    assert not torch.equal(nested[0][0].weight, nested_new[0][0].weight)
    assert not torch.equal(nested[0][0].bias, nested_new[0][0].bias)
    assert not torch.equal(nested[1].c1.weight, nested_new[1].c1.weight)
    assert not torch.equal(nested[1].c1.bias, nested_new[1].c1.bias)

    nested_new.load_state_dict(sd_flat, flatten=True)

    assert torch.equal(nested[0][0].weight, nested_new[0][0].weight)
    assert torch.equal(nested[0][0].bias, nested_new[0][0].bias)
    assert torch.equal(nested[1].c1.weight, nested_new[1].c1.weight)
    assert torch.equal(nested[1].c1.bias, nested_new[1].c1.bias)

    # >> Part 3: Test context manager
    with co.utils.flat_state_dict:
        # Export works as above despite `flatten=False`
        sd_flat2 = nested.state_dict(flatten=False)
        assert sd_flat.keys() == sd_flat2.keys()
        assert all(torch.equal(sd_flat[key], sd_flat2[key]) for key in sd_flat.keys())

        # Loading works as above despite `flatten=False`
        nested_new.load_state_dict(sd_flat, flatten=False)

        assert torch.equal(nested[0][0].weight, nested_new[0][0].weight)
        assert torch.equal(nested[0][0].bias, nested_new[0][0].bias)
        assert torch.equal(nested[1].c1.weight, nested_new[1].c1.weight)
        assert torch.equal(nested[1].c1.bias, nested_new[1].c1.bias)

    assert True  # Need to step down here to trigger context manager __exit__


def test_conditional_only_first():
    x = torch.ones((1, 1, 3))

    def is_training(module, *args):
        return module.training

    mod = co.Conditional(is_training, co.Multiply(2))

    mod.train()
    assert torch.equal(mod.forward(x), x * 2)
    assert torch.equal(mod.forward_steps(x), x * 2)
    assert torch.equal(mod.forward_step(x[:, :, 0]), x[:, :, 0] * 2)

    mod.eval()
    assert torch.equal(mod.forward(x), x)
    assert torch.equal(mod.forward_steps(x), x)
    assert torch.equal(mod.forward_step(x[:, :, 0]), x[:, :, 0])


def test_conditional_both_cases():
    x = torch.ones((1, 1, 3))

    def is_training(module, *args):
        return module.training

    mod = co.Conditional(is_training, co.Multiply(2), co.Multiply(3))

    mod.train()
    assert torch.equal(mod.forward(x), x * 2)
    assert torch.equal(mod.forward_steps(x), x * 2)
    assert torch.equal(mod.forward_step(x[:, :, 0]), x[:, :, 0] * 2)

    mod.eval()
    assert torch.equal(mod.forward(x), x * 3)
    assert torch.equal(mod.forward_steps(x), x * 3)
    assert torch.equal(mod.forward_step(x[:, :, 0]), x[:, :, 0] * 3)


def test_conditional_delay():
    # if_true.delay < if_false.delay
    mod = co.Conditional(lambda a, b: True, co.Delay(2), co.Delay(3))
    assert mod.delay == 3
    assert mod._modules["0"].delay == 3
    assert mod._modules["1"].delay == 3

    # if_true.delay > if_false.delay
    mod = co.Conditional(lambda a, b: True, co.Delay(3), co.Delay(2))
    assert mod.delay == 3
    assert mod._modules["0"].delay == 3
    assert mod._modules["1"].delay == 3


def test_broadcast():
    x = 42

    mod = co.Broadcast(2)
    assert mod.delay == 0
    assert mod.forward(x) == [x, x]
    assert mod.forward_step(x) == [x, x]
    assert mod.forward_steps(x) == [x, x]


def test_parallel():
    x = torch.randn((1, 1, 3))
    xx = [x, x]

    c3 = co.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
    c1 = co.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)

    par = co.Parallel(OrderedDict([("c3", c3), ("c1", c1)]))
    assert par.delay == 1
    assert par.padding == 1
    assert par.stride == 1

    o1 = par.forward(xx)
    assert torch.equal(c3.forward(x), o1[0])
    assert torch.equal(c1.forward(x), o1[1])

    o2 = par.forward_steps(xx, pad_end=True, update_state=False)
    assert torch.equal(c3.forward_steps(x, pad_end=True), o2[0])
    assert torch.equal(c1.forward_steps(x, pad_end=True), o2[1])

    par.clean_state()
    par.forward_step([x[:, :, 0], x[:, :, 0]], update_state=True)
    o3 = par.forward_step([x[:, :, 1], x[:, :, 1]], update_state=False)
    assert torch.equal(c3.forward_step(x[:, :, 1]), o3[0])
    assert torch.equal(c1.forward_step(x[:, :, 0]), o3[1])  # x[:,:,0] due to auto delay


def test_reduce():
    x = torch.tensor([[[1.0, 2.0]]])
    xx = [x, x]

    mod = co.Reduce("sum")
    assert mod.delay == 0
    assert torch.equal(mod.forward(xx), torch.tensor([[[2.0, 4.0]]]))
    assert torch.equal(mod.forward_steps(xx), torch.tensor([[[2.0, 4.0]]]))
    assert torch.equal(
        mod.forward_step([x[:, :, 0], x[:, :, 0]]), torch.tensor([[2.0]])
    )


def test_parallel_sequential():
    x = torch.arange(7, dtype=torch.float).reshape((1, 1, 7))

    # Test two equivalent implementations
    # First
    c5 = co.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
    c3 = co.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
    c1 = co.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)
    torch.nn.init.ones_(c5.weight)
    torch.nn.init.ones_(c3.weight)
    torch.nn.init.ones_(c1.weight)

    mod1 = co.BroadcastReduce(c5, c3, c1, reduce="sum")

    # Second
    c5 = co.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
    c3 = co.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
    c1 = co.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)
    torch.nn.init.ones_(c5.weight)
    torch.nn.init.ones_(c3.weight)
    torch.nn.init.ones_(c1.weight)

    mod2 = co.Sequential(
        co.Broadcast(),  # Sequential can infer broadcast dimensions
        co.Parallel(c5, c3, c1),
        co.Reduce("sum"),
    )

    # Compare
    o1 = mod1.forward(x)
    o2 = mod2.forward(x)

    assert torch.equal(o1, o2)
