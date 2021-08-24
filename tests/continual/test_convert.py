import continual as co
from torch import nn
import torch


def test_forward_stepping():
    sample = torch.randn((1, 1, 4, 1, 1))

    regular = nn.Conv3d(1, 1, kernel_size=1, bias=True)
    target = regular.forward(sample)

    co3 = co.forward_stepping(regular)

    # forward
    output = co3.forward(sample)
    assert torch.allclose(output, target)

    # forward_steps
    output = co3.forward_steps(sample)
    assert torch.allclose(output, target)

    # forward_step
    output = co3.forward_step(sample[:, :, 0])
    assert torch.allclose(output, target[:, :, 0])

    # The original function remains unchanged
    assert torch.allclose(regular.forward(sample), target)
