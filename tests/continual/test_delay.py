import torch

from continual.delay import Delay

torch.manual_seed(42)


def test_delay_3d():
    sample = torch.normal(mean=torch.zeros(4 * 3 * 3)).reshape((1, 1, 4, 3, 3))
    delay = Delay(delay=2, temporal_fill="zeros")
    delay.clean_state()  # Nothing should happen

    ones = torch.ones_like(sample[:, :, 0])

    assert delay.forward_step(sample[:, :, 0]) is None
    assert delay.forward_step(sample[:, :, 1]) is None

    assert torch.equal(delay.forward_step(sample[:, :, 2]), sample[:, :, 0])

    assert torch.equal(delay.forward_step(sample[:, :, 3]), sample[:, :, 1])

    assert torch.equal(delay.forward_step(ones), sample[:, :, 2])

    assert torch.equal(delay.forward_step(ones), sample[:, :, 3])

    assert torch.equal(delay.forward_step(ones), ones)


def test_delay_2d():
    sample = torch.rand((2, 2, 4, 3))
    delay = Delay(delay=2, temporal_fill="zeros")

    ones = torch.ones_like(sample[:, :, 0])

    assert delay.forward_step(sample[:, :, 0]) is None
    assert delay.forward_step(sample[:, :, 1]) is None

    assert torch.equal(delay.forward_step(sample[:, :, 2]), sample[:, :, 0])

    assert torch.equal(delay.forward_step(sample[:, :, 3]), sample[:, :, 1])

    assert torch.equal(delay.forward_step(ones), sample[:, :, 2])

    assert torch.equal(delay.forward_step(ones), sample[:, :, 3])

    assert torch.equal(delay.forward_step(ones), ones)


def test_delay_forward():
    sample = torch.rand((2, 2, 4, 3))
    delay = Delay(delay=2, temporal_fill="zeros")

    assert torch.equal(sample, delay.forward(sample))
    assert torch.equal(sample, delay.forward_steps(sample, pad_end=True))


def test_state():
    sample = torch.rand((2, 2, 4, 3))
    delay = Delay(delay=2, temporal_fill="zeros")

    zeros = torch.zeros_like(sample[:, :, 0])

    # State stays clean
    assert len(getattr(delay, "state_buffer", torch.tensor([]))) == 0
    assert getattr(delay, "state_index", torch.tensor(0)) == 0

    delay.forward(sample)

    assert len(getattr(delay, "state_buffer", torch.tensor([]))) == 0
    assert getattr(delay, "state_index", torch.tensor(0)) == 0

    # State is populated
    delay.forward_step(sample[:, :, 0])

    assert torch.equal(delay.state_buffer[0], sample[:, :, 0])
    assert torch.equal(delay.state_buffer[1], zeros)
    assert delay.state_index == -1  # still initialising

    # State can be cleaned
    delay.clean_state()
    assert len(getattr(delay, "state_buffer", torch.tensor([]))) == 0
    assert getattr(delay, "state_index", torch.tensor(0)) == 0

    assert torch.equal(delay.forward_steps(sample, pad_end=True), sample)
    # state has not been flushed
    assert torch.equal(delay.state_buffer[0], sample[:, :, 2])
    assert torch.equal(delay.state_buffer[1], sample[:, :, 3])


def test_zero_delay():
    sample = torch.rand((2, 2, 4, 3))

    no_delay = Delay(delay=0)
    assert torch.equal(no_delay.forward(sample), sample)
    assert torch.equal(no_delay.forward_steps(sample), sample)
    assert torch.equal(no_delay.forward_step(sample[:, :, -1]), sample[:, :, -1])


def test_repr():
    delay = Delay(delay=2)
    assert delay.__repr__() == "Delay(2)"

    delay = Delay(delay=2, auto_shrink=True)
    assert delay.__repr__() == "Delay(2, auto_shrink=True)"


def test_auto_shrink():
    sample = torch.rand((2, 2, 5, 3))
    delay = Delay(delay=1, auto_shrink=True)

    # forward
    output = delay.forward(sample)
    assert torch.equal(sample[:, :, 1:-1], output)

    # forward_steps
    output = delay.forward_steps(sample)
    assert torch.equal(sample[:, :, 1:-1], output)
