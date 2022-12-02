import pytest
import torch
from torch import nn

import continual as co

torch.manual_seed(42)


def test_cpp_impl_1d():
    C = 2
    T = 5
    L = 10
    sample = torch.normal(mean=torch.zeros(L * C)).reshape((1, C, L))

    # Regular
    conv = nn.Conv1d(in_channels=C, out_channels=1, kernel_size=T, bias=True)
    target = conv(sample)

    # Continual
    co_conv = co.Conv1d.build_from(conv, "zeros")
    co_conv.train()

    # Py impl is used when training
    output = []

    for i in range(sample.shape[2]):
        output.append(co_conv.forward_step(sample[:, :, i]))

    assert all(output[i] is None for i in range(co_conv.delay))

    outputs = torch.stack(output[co_conv.delay :], dim=2)
    assert torch.allclose(outputs, target)

    # Cpp impl used when not training
    co_conv.clean_state()
    co_conv.eval()

    output_cpp = []

    for i in range(sample.shape[2]):
        output_cpp.append(co_conv.forward_step(sample[:, :, i]))

    assert all(output_cpp[i] is None for i in range(co_conv.delay))
    outputs_cpp = torch.stack(output_cpp[co_conv.delay :], dim=2)
    assert torch.allclose(outputs_cpp, target)


def test_cpp_impl_2d():
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
    co_conv = co.Conv2d.build_from(conv, "zeros")
    co_conv.train()

    # Py impl is used when training
    output = []

    for i in range(sample.shape[2]):
        output.append(co_conv.forward_step(sample[:, :, i]))

    assert all(output[i] is None for i in range(co_conv.delay))

    output = torch.stack(output[co_conv.delay :], dim=2)
    assert torch.allclose(output, target)

    # Cpp impl used when not training
    co_conv.clean_state()
    co_conv.eval()

    output_cpp = []

    for i in range(sample.shape[2]):
        output_cpp.append(co_conv.forward_step(sample[:, :, i]))

    assert all(output_cpp[i] is None for i in range(co_conv.delay))
    output_cpp = torch.stack(output_cpp[co_conv.delay :], dim=2)
    assert torch.allclose(output_cpp, target)


def test_Conv1d():
    C = 2
    T = 5
    L = 10
    sample = torch.normal(mean=torch.zeros(L * C)).reshape((1, C, L))

    # Regular
    conv = nn.Conv1d(in_channels=C, out_channels=1, kernel_size=T, bias=True)
    target = conv(sample)

    # Continual
    co_conv = co.Conv1d.build_from(conv, "zeros")
    output = []

    # Frame by frame
    for i in range(sample.shape[2]):
        output.append(co_conv.forward_step(sample[:, :, i]))

    # First outputs are invalid
    assert all(output[i] is None for i in range(co_conv.delay))

    # The rest match
    output = torch.stack(output[co_conv.delay :], dim=2)
    assert torch.allclose(output, target)

    # Whole time-series
    co_conv.clean_state()
    output = co_conv.forward_steps(sample)
    assert torch.allclose(target, output)

    # Exact computation
    output2 = co_conv.forward(sample)
    assert torch.equal(target, output2)


def test_Conv1d_stride():
    C = 1
    T = 3
    L = 5
    sample = torch.arange(L * C, dtype=torch.float).reshape((1, C, L))

    # Regular
    conv = nn.Conv1d(
        in_channels=C, out_channels=1, kernel_size=T, bias=False, padding=1, stride=2
    )
    torch.nn.init.ones_(conv.weight)
    target = conv(sample)

    # Continual
    co_conv = co.Conv1d.build_from(conv, "zeros")
    output = []

    # Frame by frame
    for i in range(sample.shape[2]):
        output.append(co_conv.forward_step(sample[:, :, i]))

    output = torch.stack([o for o in output if isinstance(o, torch.Tensor)], dim=2)
    assert torch.allclose(output, target[:, :, :-1])

    # Whole time-series
    co_conv.clean_state()
    output = co_conv.forward_steps(sample, pad_end=True)
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
    co_conv = co.Conv2d.build_from(conv, "zeros")
    output = []

    # Frame by frame
    for i in range(sample.shape[2]):
        output.append(co_conv.forward_step(sample[:, :, i]))

    # Match after delay of T - 1
    for t in range(sample.shape[2] - (T - 1)):
        assert torch.allclose(target[:, :, t], output[t + (T - 1)], atol=1e-7)

    # Whole time-series
    co_conv.clean_state()
    output = co_conv.forward_steps(sample)
    assert torch.allclose(target, output, atol=1e-7)

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
    co_conv = co.Conv2d.build_from(conv, "zeros")
    output = []

    # Frame by frame
    for i in range(sample.shape[2]):
        output.append(co_conv.forward_step(sample[:, :, i]))

    # Match after delay of T - 1
    for t in range(sample.shape[2] - (T - 1)):
        if t % S == 0:
            assert torch.allclose(
                target[:, :, t // stride], output[t + (T - 1)], atol=1e-7
            )
        else:
            assert output[t + (T - 1)] is None

    # Whole time-series
    co_conv.clean_state()
    output = co_conv.forward_steps(sample)
    assert torch.allclose(target, output, atol=1e-7)


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


def xtest_seperability():
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

    coconv = co.Conv3d.build_from(conv)

    _ = coconv.forward_step(example_clip[:, :, 0])
    _ = coconv.forward_step(example_clip[:, :, 1])
    x1 = coconv.forward_step(example_clip[:, :, 2])
    x2 = coconv.forward_step(example_clip[:, :, 3])
    output = torch.tensor([[[[[x1]], [[x2]]]]])

    coconv.clean_state()
    output_alternative = coconv.forward_steps(example_clip)

    assert torch.allclose(output, output_alternative)
    assert torch.allclose(output, target)


def test_forward_long_kernel():
    conv = torch.nn.Conv3d(
        in_channels=1, out_channels=1, kernel_size=(T, S, S), bias=True
    )
    target = conv(example_clip)

    coconv = co.Conv3d.build_from(conv)

    _ = coconv.forward_step(example_clip[:, :, 0])
    _ = coconv.forward_step(example_clip[:, :, 1])
    x1 = coconv.forward_step(example_clip[:, :, 2])
    x2 = coconv.forward_step(example_clip[:, :, 3])
    output = torch.tensor([[[[[x1]], [[x2]]]]])

    coconv.clean_state()
    output_alternative = coconv.forward_steps(example_clip)

    assert torch.allclose(output, output_alternative)
    assert torch.allclose(output, target)


def test_from_conv3d():
    regular = torch.nn.Conv3d(
        in_channels=1, out_channels=1, kernel_size=(T, S, S), bias=True
    )
    target = regular(example_clip)

    co3 = co.Conv3d.build_from(regular)

    # forward
    output = co3.forward(example_clip)
    assert torch.allclose(target, output)

    # forward_step
    output = []
    for i in range(example_clip.shape[2]):
        output.append(co3.forward_step(example_clip[:, :, i]))

    for t in range(example_clip.shape[2] - (T - 1)):
        assert torch.allclose(target[:, :, t], output[t + (T - 1)])

    # forward_steps
    co3.clean_state()
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

    with pytest.raises(AssertionError):
        co.Conv3d.build_from(regular)


def test_complex():
    sample = torch.randn((1, 5, 4, 8, 8))

    # Take an example input and pass it thorugh a co.Conv3D the traditional way
    regular = torch.nn.Conv3d(
        in_channels=5,
        out_channels=5,
        kernel_size=(T, S, S),
        bias=False,
        dilation=(1, 2, 2),
        stride=(1, 2, 2),
        padding=(1, 1, 1),
        groups=5,
        padding_mode="replicate",
    )

    target = regular(sample)

    co3 = co.Conv3d.build_from(regular)

    output = co3.forward(sample)
    assert torch.allclose(target, output)

    output = co3.forward_steps(sample, pad_end=True)
    assert torch.allclose(target, output, atol=5e-6)


def test_forward_continuation():
    conv = torch.nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(T, S, S),
        bias=True,
        padding=(1, 1, 1),
        padding_mode="zeros",
    )
    coconv = co.Conv3d.build_from(conv, temporal_fill="zeros")

    # Run batch inference and fill memory
    target1 = conv(example_clip)
    output1 = coconv.forward_steps(example_clip, pad_end=True)
    assert torch.allclose(target1, output1, atol=1e-7)

    # Next forward
    target2 = conv(next_example_clip)
    output2 = coconv.forward_step(next_example_frame)

    # Next-to-last frame matches
    assert torch.allclose(target2[:, :, -2], output2, atol=1e-7)

    # Passing in zeros gives same output
    output3 = coconv.forward_step(
        torch.zeros_like(next_example_frame), update_state=False
    )
    assert torch.allclose(target2[:, :, -1], output3, atol=1e-7)


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
        co.Conv3d.build_from(conv1, temporal_fill="zeros"),
        co.Conv3d.build_from(conv2, temporal_fill="zeros"),
        co.Conv3d.build_from(conv2, temporal_fill="zeros"),
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
    for i, s in enumerate(same_as_last):
        if i >= (5 - 1) + (3 - 1) + (3 - 1):
            assert s[i]
        else:
            assert not s[i]


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
    coconv1 = co.Conv3d.build_from(conv1, temporal_fill="zeros")
    coconv2 = co.Conv3d.build_from(conv2, temporal_fill="zeros")

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
    assert torch.allclose(target22[:, :, -1], output22, atol=1e-7)


def test_update_state_false():
    sample = torch.randn((1, 1, 5, 3, 3))
    conv = torch.nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        bias=True,
        padding=1,
        padding_mode="zeros",
    )
    coconv = co.Conv3d.build_from(conv)
    coconv.clean_state()  # Nothing should happen

    target = conv.forward(sample)

    # forward_steps
    firsts_0 = coconv.forward_steps(
        sample[:, :, :-1], pad_end=False, update_state=False
    )
    firsts_1 = coconv.forward_steps(sample[:, :, :-1], pad_end=False, update_state=True)
    assert torch.allclose(firsts_0, firsts_1, atol=1e-7)
    assert torch.allclose(firsts_1, target[:, :, :-2], atol=1e-7)

    # forward_step
    last_0 = coconv.forward_step(sample[:, :, -1], update_state=False)
    last_1 = coconv.forward_step(sample[:, :, -1], update_state=True)
    assert torch.allclose(last_0, last_1, atol=1e-7)
    assert torch.allclose(last_1, target[:, :, -2], atol=1e-7)
