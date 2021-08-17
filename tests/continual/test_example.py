import torch
from torch import nn

import continual as co


def test_readme_example():
    #                      B, C, T, H, W
    example = torch.randn((1, 1, 5, 3, 3))

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


def test_mb_conv():
    #                      B,  C, T, H, W
    example = torch.randn((1, 32, 7, 5, 5))

    mb_conv = co.Residual(
        co.Sequential(
            co.Conv3d(32, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU6(),
            co.Conv3d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.ReLU6(),
            co.Conv3d(64, 32, kernel_size=1),
            nn.BatchNorm3d(32),
        )
    )

    output = mb_conv.forward(example)
    assert output.shape == (1, 32, 7, 5, 5)

    output_steps = mb_conv.forward_steps(example)
    assert output_steps.shape == output.shape


def test_inception_module():
    #                      B,  C, T, H, W
    example = torch.randn((1, 192, 7, 5, 5))

    def norm_relu(module, channels):
        return co.Sequential(
            module,
            nn.BatchNorm3d(channels),
            nn.ReLU(),
        )

    inception_module = co.Parallel(
        co.Conv3d(192, 64, kernel_size=1),
        co.Sequential(
            norm_relu(co.Conv3d(192, 96, kernel_size=1), 96),
            norm_relu(co.Conv3d(96, 128, kernel_size=3, padding=1), 128),
        ),
        co.Sequential(
            norm_relu(co.Conv3d(192, 16, kernel_size=1), 16),
            norm_relu(co.Conv3d(16, 32, kernel_size=3, padding=1), 32),
        ),
        co.Sequential(
            co.MaxPool3d(kernel_size=3, padding=1, stride=1),
            norm_relu(co.Conv3d(192, 32, kernel_size=1), 32),
        ),
        aggregation_fn="concat",
    )

    output = inception_module.forward(example)
    assert output.shape == (1, 64 + 128 + 32 + 32, 7, 5, 5)

    output_steps = inception_module.forward_steps(example)
    assert output_steps.shape == output.shape
