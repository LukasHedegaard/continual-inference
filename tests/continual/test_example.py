import torch
from torch import nn

import continual as co


def test_readme_example():  # B, C, T, H, W
    example = torch.normal(mean=torch.zeros(5 * 3 * 3)).reshape((1, 1, 5, 3, 3))

    # Acts as a drop-in replacement for torch.nn modules ✅
    co_conv = co.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3))
    nn_conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3))
    co_conv.load_state_dict(nn_conv.state_dict())

    co_output = co_conv(example)  # Same exact computation
    nn_output = nn_conv(example)  # Same exact computation
    assert torch.equal(co_output, nn_output)

    # But can also perform online inference efficiently 🚀
    firsts = co_conv.forward_steps(example[:, :, :4])
    last = co_conv.forward_step(example[:, :, 4])

    assert torch.allclose(nn_output[:, :, : co_conv.delay], firsts)
    assert torch.allclose(nn_output[:, :, co_conv.delay], last)
