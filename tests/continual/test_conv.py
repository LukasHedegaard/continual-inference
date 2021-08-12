import torch
from torch import nn

import continual as co
from continual.utils import TensorPlaceholder

torch.manual_seed(42)


def test_Conv1d():
    C = 2
    T = 3
    L = 5
    sample = torch.normal(mean=torch.zeros(L * C)).reshape((1, C, L))

    # Regular
    conv = nn.Conv1d(in_channels=C, out_channels=1, kernel_size=T, bias=True)
    target = conv(sample)

    # Continual
    co_conv = co.Conv1d.from_regular(conv, "zeros")
    output = []

    # Frame by frame
    for i in range(sample.shape[2]):
        output.append(co_conv.forward_step(sample[:, :, i]))

    # Match after delay of T - 1
    for t in range(sample.shape[2] - (T - 1)):
        assert torch.allclose(target[:, :, t], output[t + (T - 1)])

    # Whole time-series
    output = co_conv.forward_steps(sample)
    assert torch.allclose(target, output)

    # Exact computation
    output2 = co_conv.forward(sample)
    assert torch.equal(target, output2)


def test_Conv1d_stride():
    C = 2
    T = 3
    L = 5
    S = 2
    sample = torch.normal(mean=torch.zeros(L * C)).reshape((1, C, L))

    # Regular
    conv = nn.Conv1d(in_channels=C, out_channels=1, kernel_size=T, bias=True, stride=S)
    target = conv(sample)

    # Continual
    co_conv = co.Conv1d.from_regular(conv, "zeros")
    output = []

    # Frame by frame
    for i in range(sample.shape[2]):
        output.append(co_conv.forward_step(sample[:, :, i]))

    # Match after delay of T - 1
    for t in range(sample.shape[2] - (T - 1)):
        if t % S == 0:
            assert torch.allclose(target[:, :, t // S], output[t + (T - 1)])
        else:
            assert type(output[t + (T - 1)]) is TensorPlaceholder

    # Whole time-series
    output = co_conv.forward_steps(sample)
    assert torch.allclose(target, output)


def test_Conv2d():
    C = 2
    T = 3
    S = 2
    L = 5
    H = 3
    sample = torch.normal(mean=torch.zeros(L * C * H)).reshape((1, C, L, H))

    # Regular
    conv = nn.Conv2d(in_channels=C, out_channels=1, kernel_size=(T, S), bias=True)
    target = conv(sample)

    # Continual
    co_conv = co.Conv2d.from_regular(conv, "zeros")
    output = []

    # Frame by frame
    for i in range(sample.shape[2]):
        output.append(co_conv.forward_step(sample[:, :, i]))

    # Match after delay of T - 1
    for t in range(sample.shape[2] - (T - 1)):
        assert torch.allclose(target[:, :, t], output[t + (T - 1)])

    # Whole time-series
    output = co_conv.forward_steps(sample)
    assert torch.allclose(target, output)

    # Exact computation
    output2 = co_conv.forward(sample)
    assert torch.equal(target, output2)


def test_Conv2d_stride():
    C = 2
    T = 3
    S = 2
    L = 5
    H = 3
    stride = 2
    sample = torch.normal(mean=torch.zeros(L * C * H)).reshape((1, C, L, H))

    # Regular
    conv = nn.Conv2d(
        in_channels=C, out_channels=1, kernel_size=(T, S), bias=True, stride=stride
    )
    target = conv(sample)

    # Continual
    co_conv = co.Conv2d.from_regular(conv, "zeros")
    output = []

    # Frame by frame
    for i in range(sample.shape[2]):
        output.append(co_conv.forward_step(sample[:, :, i]))

    # Match after delay of T - 1
    for t in range(sample.shape[2] - (T - 1)):
        if t % S == 0:
            assert torch.allclose(target[:, :, t // stride], output[t + (T - 1)])
        else:
            assert type(output[t + (T - 1)]) is TensorPlaceholder

    # Whole time-series
    output = co_conv.forward_steps(sample)
    assert torch.allclose(target, output)


T = S = 3
example_clip = torch.normal(mean=torch.zeros(4 * 3 * 3)).reshape((1, 1, 4, 3, 3))
next_example_frame = torch.normal(mean=torch.zeros(3 * 3)).reshape((1, 1, 3, 3))
next_example_clip = torch.stack(
    [
        example_clip[:, :, 1],
        example_clip[:, :, 2],
        example_clip[:, :, 3],
        next_example_frame,
    ],
    dim=2,
)
# Long example clip
long_example_clip = torch.normal(mean=torch.zeros(8 * 3 * 3)).reshape((1, 1, 8, 3, 3))
long_next_example_clip = torch.stack(
    [
        *[long_example_clip[:, :, i] for i in range(1, 8)],
        next_example_frame,
    ],
    dim=2,
)


def test_seperability():
    # Checks that the basic idea is sound

    # Take an example input and pass it thorugh a co.Conv3D the traditional way
    regular = torch.nn.Conv3d(
        in_channels=1, out_channels=1, kernel_size=(T, S, S), bias=True
    )

    regular_output = regular(example_clip).detach()

    # Take an example input and pass it thorugh a co.Conv3D the seperated way
    seperated = torch.nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(T, S, S),
        bias=False,
        padding=(T - 1, 0, 0),
    )
    seperated.weight = regular.weight

    a = seperated(example_clip[:, :, 0, :, :].unsqueeze(2))[0, 0, :, 0, 0]
    b = seperated(example_clip[:, :, 1, :, :].unsqueeze(2))[0, 0, :, 0, 0]
    c = seperated(example_clip[:, :, 2, :, :].unsqueeze(2))[0, 0, :, 0, 0]
    d = seperated(example_clip[:, :, 3, :, :].unsqueeze(2))[0, 0, :, 0, 0]

    seperated_output = torch.tensor(
        [
            [
                [
                    [[c[0] + b[1] + a[2] + regular.bias]],
                    [[d[0] + c[1] + b[2] + regular.bias]],
                ]
            ]
        ]
    )

    assert torch.allclose(regular_output, seperated_output)


def test_basic_forward():
    conv = torch.nn.Conv3d(
        in_channels=1, out_channels=1, kernel_size=(T, S, S), bias=True
    )
    target = conv(example_clip)

    coconv = co.Conv3d.from_regular(conv)
    # coconv = co.Conv3d(
    #     in_channels=1,
    #     out_channels=1,
    #     kernel_size=(S, T, T),
    #     bias=True,
    #     temporal_fill="zeros",
    # )
    # coconv.conv.weight = conv.weight
    # coconv.bias = conv.bias

    _ = coconv.forward_step(example_clip[:, :, 0])
    _ = coconv.forward_step(example_clip[:, :, 1])
    x1 = coconv.forward_step(example_clip[:, :, 2])
    x2 = coconv.forward_step(example_clip[:, :, 3])
    output = torch.tensor([[[[[x1]], [[x2]]]]])

    output_alternative = coconv.forward_steps(example_clip)

    assert torch.allclose(output, output_alternative)
    assert torch.allclose(output, target)


def test_forward_long_kernel():
    conv = torch.nn.Conv3d(
        in_channels=1, out_channels=1, kernel_size=(T, S, S), bias=True
    )
    target = conv(example_clip)

    coconv = co.Conv3d.from_regular(conv)
    # coconv = co.Conv3d(
    #     in_channels=1,
    #     out_channels=1,
    #     kernel_size=(T, S, S),
    #     bias=True,
    #     temporal_fill="zeros",
    # )
    # coconv.weight = conv.weight
    # coconv.bias = conv.bias

    _ = coconv.forward_step(example_clip[:, :, 0])
    _ = coconv.forward_step(example_clip[:, :, 1])
    x1 = coconv.forward_step(example_clip[:, :, 2])
    x2 = coconv.forward_step(example_clip[:, :, 3])
    output = torch.tensor([[[[[x1]], [[x2]]]]])

    output_alternative = coconv.forward_steps(example_clip)

    assert torch.allclose(output, output_alternative)
    assert torch.allclose(output, target)


def test_from_conv3d():
    regular = torch.nn.Conv3d(
        in_channels=1, out_channels=1, kernel_size=(T, S, S), bias=True
    )
    target = regular(example_clip)

    co3 = co.Conv3d.from_regular(regular)

    output = []
    for i in range(example_clip.shape[2]):
        output.append(co3.forward_step(example_clip[:, :, i]))

    for t in range(example_clip.shape[2] - (T - 1)):
        assert torch.allclose(target[:, :, t], output[t + (T - 1)])

    # Alternative: gives same output as regular version
    output3 = co3.forward_steps(example_clip)
    assert torch.allclose(output3, target)


def test_from_conv3d_bad_shape():
    regular = torch.nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(T, S, S),
        bias=True,
        dilation=(2, 2, 2),
        padding=(1, 1, 1),
        stride=(2, 2, 2),
    )

    # Also warns
    co3 = co.Conv3d.from_regular(regular)

    # Changed               V
    assert co3.dilation == (1, 2, 2)
    assert co3.stride == (1, 2, 2)

    # Not changed
    assert co3.padding == (1, 1, 1)


example_clip_large = torch.normal(mean=torch.zeros(2 * 2 * 4 * 8 * 8)).reshape(
    (2, 2, 4, 8, 8)
)


def test_complex():
    # Take an example input and pass it thorugh a co.Conv3D the traditional way
    regular = torch.nn.Conv3d(
        in_channels=2,
        out_channels=4,
        kernel_size=(T, S, S),
        bias=True,
        groups=2,
        dilation=(1, 2, 2),
        stride=(1, 2, 2),
        padding=(2, 1, 1),
    )
    regular_output = regular(example_clip_large).detach()

    co3 = co.Conv3d.from_regular(regular, temporal_fill="zeros")
    co3_output = co3.forward_steps(example_clip_large)

    assert torch.allclose(regular_output, co3_output, atol=5e-8)


def test_forward_continuation():
    conv = torch.nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(T, S, S),
        bias=True,
        padding=(1, 1, 1),
        padding_mode="zeros",
    )
    coconv = co.Conv3d.from_regular(conv, temporal_fill="zeros")

    # Run batch inference and fill memory
    target1 = conv(example_clip)
    output1 = coconv.forward_steps(example_clip)
    assert torch.allclose(target1, output1)

    # Next forward
    target2 = conv(next_example_clip)
    output2 = coconv.forward_step(next_example_frame)

    # Next-to-last frame matches
    assert torch.allclose(target2[:, :, -2], output2, atol=5e-8)

    # Passing in zeros gives same output
    output3 = coconv.forward_step(
        torch.zeros_like(next_example_frame), update_state=False
    )
    assert torch.allclose(target2[:, :, -1], output3, atol=5e-8)


def test_stacked_impulse_response():
    # An input has effect corresponding to the receptive field
    zeros = torch.zeros_like(next_example_frame)
    ones = torch.ones_like(next_example_frame)

    # Init regular
    conv1 = torch.nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(5, S, S),
        bias=True,
        padding=(0, 1, 1),
        padding_mode="zeros",
    )
    conv2 = torch.nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(3, S, S),
        bias=True,
        padding=(0, 1, 1),
        padding_mode="zeros",
    )

    # Init continual
    cnn = [
        co.Conv3d.from_regular(conv1, temporal_fill="zeros"),
        co.Conv3d.from_regular(conv2, temporal_fill="zeros"),
        co.Conv3d.from_regular(conv2, temporal_fill="zeros"),
    ]

    # Impulse
    for mod in cnn:
        mod.forward_step(ones)

    outputs = []
    for _ in range(15):
        for mod in cnn:
            mod.forward_step(zeros)

    same_as_last = [
        torch.equal(outputs[i], outputs[i - 1]) for i in range(1, len(outputs))
    ]

    # Correct result is output
    for i in range(len(same_as_last)):
        if i >= (5 - 1) + (3 - 1) + (3 - 1):
            assert same_as_last[i]
        else:
            assert not same_as_last[i]


def test_stacked_no_pad():
    # Without initialisation using forward_steps, the output has no delay

    # Init regular
    conv1 = torch.nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(5, S, S),
        bias=True,
        padding=(0, 1, 1),
        padding_mode="zeros",
    )
    conv2 = torch.nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(3, S, S),
        bias=True,
        padding=(0, 1, 1),
        padding_mode="zeros",
    )

    # Init continual
    coconv1 = co.Conv3d.from_regular(conv1, temporal_fill="zeros")
    coconv2 = co.Conv3d.from_regular(conv2, temporal_fill="zeros")

    # Targets
    target11 = conv1(long_example_clip)
    target12 = conv2(target11)

    target21 = conv1(long_next_example_clip)
    target22 = conv2(target21)

    # Test 3D mode
    output11 = coconv1.forward_steps(long_example_clip)
    output12 = coconv2.forward_steps(output11)
    torch.allclose(target12, output12, atol=5e-8)

    # Next 2D forward
    output21 = coconv1.forward_step(next_example_frame)
    output22 = coconv2.forward_step(output21)

    # Correct result is output
    assert torch.allclose(target22[:, :, -1], output22, atol=5e-8)
