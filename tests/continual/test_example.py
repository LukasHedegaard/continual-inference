from collections import OrderedDict

import torch
from torch import nn

import continual as co


def test_paper_example():
    net = co.Sequential(
        co.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=2),
        co.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0),
    )
    assert net.receptive_field == 7
    assert net.padding[0] == 2
    assert net.delay == 4


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

    # Temporal properties
    assert co_conv.receptive_field == 3
    assert co_conv.delay == 2


def test_residual():
    x = torch.randn((1, 2, 5, 3, 3))
    conv = co.Conv3d(2, 2, kernel_size=3, padding=1)
    residual1 = co.Sequential(
        co.Broadcast(2),
        co.Parallel(conv, co.Delay(1)),
        co.Reduce("sum"),
    )

    residual2 = co.BroadcastReduce(conv, co.Delay(1), reduce="sum")

    residual3 = co.Residual(conv, reduce="sum")

    assert len(set([residual1.delay, residual2.delay, residual3.delay])) == 1
    o1 = residual1.forward(x)
    o2 = residual2.forward(x)
    o3 = residual3.forward(x)
    assert torch.equal(o1, o2)
    assert torch.equal(o1, o3)


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

    # Temporal properties
    assert mb_conv.receptive_field == 3
    assert mb_conv.delay == 1

    mb_conv.eval()
    output = mb_conv.forward(example)
    assert output.shape == (1, 32, 7, 5, 5)

    firsts = mb_conv.forward_steps(example[:, :, :6])
    last = mb_conv.forward_step(example[:, :, 6])

    assert torch.allclose(
        output[:, :, : -mb_conv.padding[0] - mb_conv.delay], firsts, atol=1e-7
    )
    assert torch.allclose(
        output[:, :, -mb_conv.padding[0] - mb_conv.delay], last, atol=1e-7
    )


def test_inception_module():
    #                      B,  C, T, H, W
    example = torch.randn((1, 192, 7, 5, 5))

    def norm_relu(module):
        return co.Sequential(
            module,
            nn.BatchNorm3d(module.out_channels),
            nn.ReLU(),
        )

    inception_module = co.BroadcastReduce(
        co.Conv3d(192, 64, kernel_size=1),
        co.Sequential(
            norm_relu(co.Conv3d(192, 96, kernel_size=1)),
            norm_relu(co.Conv3d(96, 128, kernel_size=3, padding=1)),
        ),
        co.Sequential(
            norm_relu(co.Conv3d(192, 16, kernel_size=1)),
            norm_relu(co.Conv3d(16, 32, kernel_size=5, padding=2)),
        ),
        co.Sequential(
            co.MaxPool3d(kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
            norm_relu(co.Conv3d(192, 32, kernel_size=1)),
        ),
        reduce="concat",
    )

    output = inception_module.forward(example)
    assert output.shape == (1, 64 + 128 + 32 + 32, 7, 5, 5)

    output_steps = inception_module.forward_steps(example, pad_end=True)
    assert output_steps.shape == output.shape


def test_inception_module_alt():
    #                      B,  C, T, H, W
    example = torch.randn((1, 192, 7, 5, 5))

    def norm_relu(module, channels):
        return co.Sequential(
            module,
            nn.BatchNorm3d(channels),
            nn.ReLU(),
        )

    inception_module = co.Sequential(
        co.Broadcast(4),
        co.Parallel(
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
                co.MaxPool3d(kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
                norm_relu(co.Conv3d(192, 32, kernel_size=1), 32),
            ),
        ),
        co.Reduce("concat"),
    )

    output = inception_module.forward(example)
    assert output.shape == (1, 64 + 128 + 32 + 32, 7, 5, 5)

    output_steps = inception_module.forward_steps(example, pad_end=True)
    assert output_steps.shape == output.shape


def test_se():
    #                      B,  C, T, H, W
    example = torch.randn((1, 256, 7, 5, 5))

    se = co.Residual(
        co.Sequential(
            OrderedDict(
                [
                    ("pool", co.AdaptiveAvgPool3d((1, 1, 1), kernel_size=7)),
                    ("down", co.Conv3d(256, 16, kernel_size=1)),
                    ("act1", nn.ReLU()),
                    ("up", co.Conv3d(16, 256, kernel_size=1)),
                    ("act2", nn.Sigmoid()),
                ]
            )
        ),
        reduce="mul",
    )

    output = se.forward(example)
    assert output.shape == example.shape

    output_steps = se.forward_steps(example, pad_end=True)
    assert torch.allclose(output_steps, output)


def test_linear():
    #                   B  C  T   H    W
    input = torch.randn(1, 3, 16, 128, 128)
    net = co.Sequential(
        co.Conv3d(3, 32, 3),
        co.AdaptiveAvgPool3d((1, 1, 1), 32),
        co.Linear(32, 10, channel_dim=1),
    )
    output = net(input)
    assert output.size() == torch.Size([1, 10, 1, 1, 1])
