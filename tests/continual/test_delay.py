import torch

from continual.delay import Delay
from continual.interface import TensorPlaceholder

torch.manual_seed(42)


def test_delay_3d():
    example_input = torch.normal(mean=torch.zeros(4 * 3 * 3)).reshape((1, 1, 4, 3, 3))
    delay = Delay(delay=2, temporal_fill="zeros")

    ones = torch.ones_like(example_input[:, :, 0])

    assert isinstance(delay.forward_step(example_input[:, :, 0]), TensorPlaceholder)
    assert isinstance(delay.forward_step(example_input[:, :, 1]), TensorPlaceholder)

    assert torch.equal(
        delay.forward_step(example_input[:, :, 2]), example_input[:, :, 0]
    )

    assert torch.equal(
        delay.forward_step(example_input[:, :, 3]), example_input[:, :, 1]
    )

    assert torch.equal(delay.forward_step(ones), example_input[:, :, 2])

    assert torch.equal(delay.forward_step(ones), example_input[:, :, 3])

    assert torch.equal(delay.forward_step(ones), ones)


def test_delay_2d():
    example_input = torch.rand((2, 2, 4, 3))
    delay = Delay(delay=2, temporal_fill="zeros")

    ones = torch.ones_like(example_input[:, :, 0])

    assert isinstance(delay.forward_step(example_input[:, :, 0]), TensorPlaceholder)
    assert isinstance(delay.forward_step(example_input[:, :, 1]), TensorPlaceholder)

    assert torch.equal(
        delay.forward_step(example_input[:, :, 2]), example_input[:, :, 0]
    )

    assert torch.equal(
        delay.forward_step(example_input[:, :, 3]), example_input[:, :, 1]
    )

    assert torch.equal(delay.forward_step(ones), example_input[:, :, 2])

    assert torch.equal(delay.forward_step(ones), example_input[:, :, 3])

    assert torch.equal(delay.forward_step(ones), ones)


def test_delay_forward():
    example_input = torch.rand((2, 2, 4, 3))
    delay = Delay(delay=2, temporal_fill="zeros")

    assert torch.equal(example_input, delay.forward(example_input))
    assert torch.equal(example_input, delay.forward_steps(example_input))


def test_state():
    example_input = torch.rand((2, 2, 4, 3))
    delay = Delay(delay=2, temporal_fill="zeros")

    zeros = torch.zeros_like(example_input[:, :, 0])

    # State stays clean
    assert getattr(delay, "state_buffer", None) is None
    assert getattr(delay, "state_index", None) is None

    delay.forward(example_input)

    assert getattr(delay, "state_buffer", None) is None
    assert getattr(delay, "state_index", None) is None

    # State is populated
    delay.forward_step(example_input[:, :, 0])

    assert torch.equal(delay.state_buffer[0], example_input[:, :, 0])
    assert torch.equal(delay.state_buffer[1], zeros)
    assert delay.state_index == -1  # still initialising

    # State can be cleaned
    delay.clean_state()
    assert getattr(delay, "state_buffer", None) is None
    assert getattr(delay, "state_index", None) is None

    assert torch.equal(delay.forward_steps(example_input, pad_end=True), example_input)
    # state has not been flushed
    assert torch.equal(delay.state_buffer[0], example_input[:, :, 2])
    assert torch.equal(delay.state_buffer[1], example_input[:, :, 3])
