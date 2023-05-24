import shutil
from collections import OrderedDict
from pathlib import Path
from timeit import default_timer as timer

import onnxruntime as ort
import pytest
import torch
from torch import nn, onnx

import continual as co
from continual.onnx import OnnxWrapper
from continual.utils import flatten


@pytest.fixture
def tmp_path():
    p = Path("tmp")
    p.mkdir(exist_ok=True)
    yield p
    shutil.rmtree(p)


def test_conv_forward_step(tmp_path):
    batch_size = 1
    in_channels = 3
    out_channels = 4
    receptive_field = 5
    model_path = tmp_path / "conv.onnx"

    net = co.Conv1d(in_channels, out_channels, receptive_field)
    net.call_mode = "forward_step"
    net.train()
    with torch.no_grad():
        net.weight.fill_(1.0)
        net.bias.fill_(1.0)

    firsts = torch.arange(
        batch_size * in_channels * receptive_field, dtype=torch.float
    ).reshape(batch_size, in_channels, receptive_field)
    last = torch.ones(batch_size, in_channels)

    # Baseline
    state0 = None
    with torch.no_grad():
        for i in range(receptive_field):
            _, state0 = net._forward_step(firsts[:, :, i], state0)

    target, target_state = net._forward_step(last, state0)

    # Export to TorchScript
    net.call_mode = "forward_step"
    ts_net = torch.jit.script(net)
    ts_output, ts_state = ts_net._forward_step(last, state0)

    assert torch.allclose(ts_output, target)
    for i in range(len(target_state)):
        assert torch.allclose(target_state[i], ts_state[i])

    # Export to ONNX
    onnx.export(
        OnnxWrapper(net),
        (last, *[s.clone() for s in state0]),
        model_path,
        input_names=["input", *[f"i{i}" for i in range(3)]],
        output_names=["output", *[f"o{i}" for i in range(3)]],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
            "i0": {0: "batch"},
            "o0": {0: "batch"},
        },
        do_constant_folding=False,
        verbose=True,
        opset_version=11,
    )

    ort_session = ort.InferenceSession(str(model_path))

    # Run for whole input
    inputs = {
        **{"input": last.numpy()},
        **{f"i{i}": state0[i].detach().numpy() for i in range(3)},
    }
    onnx_output, *onnx_state = ort_session.run(None, inputs)

    assert torch.allclose(torch.tensor(onnx_output), target)
    for i in range(len(target_state)):
        assert torch.allclose(torch.tensor(onnx_state[i]), target_state[i])

    # Time the models
    num_runs = 100

    # Regular
    net.eval()
    start = timer()
    with torch.no_grad():
        for _ in range(num_runs):
            net._forward_step(last, state0)
    reg_time = timer() - start

    # TorchScript
    start = timer()
    with torch.no_grad():
        for _ in range(num_runs):
            ts_net._forward_step(last, state0)
    ts_time = timer() - start

    # ONNX
    start = timer()
    for _ in range(num_runs):
        ort_session.run(None, inputs)
    onnx_time = timer() - start

    assert reg_time > onnx_time
    assert ts_time > onnx_time


def test_sequential_pure(tmp_path):
    batch_size = 1
    in_channels = 3
    hi_channels = 6
    out_channels = 4
    kernel_size = 3
    receptive_field = 3
    model_path = tmp_path / "seq_conv.onnx"

    net = co.Sequential(
        co.Conv1d(in_channels, hi_channels, kernel_size),
        co.Conv1d(hi_channels, out_channels, kernel_size=1),
        # co.forward_stepping(nn.Conv1d(hi_channels, out_channels, kernel_size=1)),
    )
    net.eval()
    with torch.no_grad():
        net[0].weight.fill_(1.0)
        net[0].bias.fill_(1.0)
        net[1].weight.fill_(1.0)
        net[1].bias.fill_(1.0)

    firsts = torch.arange(
        batch_size * in_channels * receptive_field, dtype=torch.float
    ).reshape(batch_size, in_channels, receptive_field)
    last = torch.ones(batch_size, in_channels)

    # Baseline
    state0 = None
    with torch.no_grad():
        for i in range(receptive_field):
            _, state0 = net._forward_step(firsts[:, :, i], state0)

    # Export to ONNX
    o_net = OnnxWrapper(net)
    flat_state = [s.clone() for s in flatten(state0)]
    onnx.export(
        o_net,
        # Since a CoModule may choose to modify parts of its state rather than to
        # clone and modify, we need to pass a clone to ensure fair comparison later
        (last, *[s.clone() for s in flat_state]),
        model_path,
        input_names=["input", *o_net.state_input_names],
        output_names=["output", *o_net.state_output_names],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
            **o_net.state_dynamic_axes,
        },
        do_constant_folding=False,
        verbose=True,
        opset_version=11,
    )

    ort_session = ort.InferenceSession(str(model_path))

    # Run for whole input
    inputs = {
        "input": last.numpy(),
        **{k: v.detach().numpy() for k, v in zip(o_net.state_input_names, flat_state)},
    }
    onnx_output, *onnx_state = ort_session.run(None, inputs)

    target, target_state = net._forward_step(last, state0)

    for os, ts in zip(onnx_state, flatten(target_state)):
        assert torch.allclose(torch.tensor(os), ts)
    assert torch.allclose(torch.tensor(onnx_output), target)


def test_sequential_mixed(tmp_path):
    batch_size = 1
    in_channels = 3
    hi_channels = 6
    out_channels = 4
    kernel_size = 3
    receptive_field = 5
    model_path = tmp_path / "seq_conv_mixed.onnx"

    net = co.Sequential(
        co.Conv1d(in_channels, hi_channels, kernel_size),
        nn.BatchNorm1d(hi_channels),
        nn.ReLU(),
        co.Conv1d(hi_channels, out_channels, kernel_size),
    )
    net.eval()
    with torch.no_grad():
        net[0].weight.fill_(1.0)
        net[0].bias.fill_(1.0)
        net[3].weight.fill_(1.0)
        net[3].bias.fill_(1.0)

    firsts = torch.arange(
        batch_size * in_channels * receptive_field, dtype=torch.float
    ).reshape(batch_size, in_channels, receptive_field)
    last = torch.ones(batch_size, in_channels)

    # Baseline
    state0 = None
    with torch.no_grad():
        for i in range(receptive_field):
            _, state0 = net._forward_step(firsts[:, :, i], state0)

    # Export to ONNX
    flat_state = [s.clone() for s in flatten(state0)]
    co.onnx.export(
        net,
        # Since a CoModule may choose to modify parts of its state rather than to
        # clone and update, we need to pass a clone to ensure fair comparison later
        (last, *[s.clone() for s in flat_state]),
        model_path,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        verbose=True,
        opset_version=11,
    )

    ort_session = ort.InferenceSession(str(model_path))

    # Run for whole input
    inputs = {
        "input": last.numpy(),
        **{
            k: v.detach().numpy()
            for k, v in zip(OnnxWrapper(net).state_input_names, flat_state)
        },
    }
    onnx_output, *onnx_state = ort_session.run(None, inputs)

    net.eval()
    target, target_state = net._forward_step(last, state0)

    for os, ts in zip(onnx_state, flatten(target_state)):
        assert torch.allclose(torch.tensor(os), ts)
    assert torch.allclose(torch.tensor(onnx_output), target)


def test_residual(tmp_path):
    batch_size = 1
    in_channels = 3
    out_channels = 3
    kernel_size = 5
    receptive_field = 5
    model_path = tmp_path / "residual.onnx"

    conv = co.Conv1d(in_channels, out_channels, kernel_size)
    with torch.no_grad():
        conv.weight.fill_(1.0)
        conv.bias.fill_(1.0)

    net = co.Residual(conv)
    net.eval()

    firsts = torch.arange(
        batch_size * in_channels * receptive_field, dtype=torch.float
    ).reshape(batch_size, in_channels, receptive_field)
    last = torch.ones(batch_size, in_channels)

    # Baseline
    state0 = None
    with torch.no_grad():
        for i in range(receptive_field):
            _, state0 = net._forward_step(firsts[:, :, i], state0)

    # Export to ONNX
    flat_state = [s.clone() for s in flatten(state0)]
    co.onnx.export(
        net,
        # Since a CoModule may choose to modify parts of its state rather than to
        # clone and update, we need to pass a clone to ensure fair comparison later
        (last, *[s.clone() for s in flat_state]),
        model_path,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        verbose=True,
        opset_version=11,
    )

    ort_session = ort.InferenceSession(str(model_path))

    # Run for whole input
    inputs = {
        "input": last.numpy(),
        **{
            k: v.detach().numpy()
            for k, v in zip(OnnxWrapper(net).state_input_names, flat_state)
        },
    }
    onnx_output, *onnx_state = ort_session.run(None, inputs)

    net.eval()
    target, target_state = net._forward_step(last, state0)

    for os, ts in zip(onnx_state, flatten(target_state)):
        assert torch.allclose(torch.tensor(os), ts)
    assert torch.allclose(torch.tensor(onnx_output), target)


def test_advanced_routing(tmp_path):
    batch_size = 1
    in_channels = 3
    out_channels = 3
    kernel_size = 3
    receptive_field = kernel_size
    model_path = tmp_path / "advanced_routing.onnx"

    net = co.Sequential(
        co.Conditional(lambda a, b: True, co.Add(1), co.Add(-1)),
        co.Conditional(lambda a, b: False, co.Add(1), co.Add(-1)),
        co.RecyclingPositionalEncoding(3, 3),
        co.Broadcast(2),
        co.ParallelDispatch([1, 0]),  # Swap positions
        co.Parallel(
            co.AvgPool1d(kernel_size, stride=1),
            co.Sequential(
                co.Add(1),
                co.Conv1d(in_channels, out_channels, kernel_size),
            ),
        ),
        co.Reduce("sum"),
        co.Linear(out_channels, out_channels),
    )
    net.eval()

    firsts = torch.arange(
        batch_size * in_channels * receptive_field, dtype=torch.float
    ).reshape(batch_size, in_channels, receptive_field)
    last = torch.ones(batch_size, in_channels)

    # Prep state
    state0 = None
    with torch.no_grad():
        for i in range(receptive_field):
            _, state0 = net._forward_step(firsts[:, :, i], state0)

    # Export to ONNX
    flat_state = [s.clone() for s in flatten(state0)]
    co.onnx.export(
        net,
        # Since a CoModule may choose to modify parts of its state rather than to
        # clone and update, we need to pass a clone to ensure fair comparison later
        (last, *[s.clone() for s in flat_state]),
        model_path,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        verbose=True,
        opset_version=11,
    )

    ort_session = ort.InferenceSession(str(model_path))

    # Run for whole input
    inputs = {
        "input": last.numpy(),
        **{
            k: v.detach().numpy()
            for k, v in zip(OnnxWrapper(net).state_input_names, flat_state)
        },
    }
    onnx_output, *onnx_state = ort_session.run(None, inputs)

    net.eval()
    target, target_state = net._forward_step(last, state0)

    for os, ts in zip(onnx_state, flatten(target_state)):
        assert torch.allclose(torch.tensor(os), ts)
    assert torch.allclose(torch.tensor(onnx_output), target)


def test_trans_enc_b1(tmp_path):
    B = 1
    T = 10  # temporal sequence length
    E = 4  # embedding dimension
    H = 2  # num heads

    model_path = tmp_path / "transformer_encoder_b1.onnx"

    net = co.TransformerEncoder(
        co.TransformerEncoderLayerFactory(
            d_model=E,
            nhead=H,
            dim_feedforward=E * 2,
            dropout=0.0,
            sequence_len=T,
            activation="relu",
        ),
        num_layers=1,
    )

    net.eval()

    firsts = torch.rand(B, E, T)
    last = torch.ones(B, E)

    # Baseline
    state0 = None
    with torch.no_grad():
        for i in range(T - 1):
            _, state0 = net._forward_step(firsts[:, :, i], state0)
        o, state0 = net._forward_step(firsts[:, :, -1], state0)

    # Check agains baseline
    o_baseline = net.forward(firsts)
    assert torch.allclose(o, o_baseline[:, :, -1])

    # Export to ONNX
    flat_state = [s.clone() for s in flatten(state0)]
    co.onnx.export(
        net,
        # Since a CoModule may choose to modify parts of its state rather than to
        # clone and update, we need to pass a clone to ensure fair comparison later
        (last, *[s.clone() for s in flat_state]),
        model_path,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        verbose=True,
        opset_version=14,
    )

    ort_session = ort.InferenceSession(str(model_path))

    # Run for whole input
    inputs = {
        "input": last.numpy(),
        **{
            k: v.detach().numpy()
            for k, v in zip(OnnxWrapper(net).state_input_names, flat_state)
        },
    }
    onnx_output, *onnx_state = ort_session.run(None, inputs)

    net.eval()
    target, target_state = net._forward_step(last, state0)

    for os, ts in zip(onnx_state, flatten(target_state)):
        assert torch.allclose(torch.tensor(os), ts)
    assert torch.allclose(torch.tensor(onnx_output), target)

    # # Check timing
    # num_runs = 100

    # # Regular
    # net.eval()
    # start = timer()
    # with torch.no_grad():
    #     for _ in range(num_runs):
    #         net._forward_step(last, state0)
    # reg_time = timer() - start

    # # ONNX
    # start = timer()
    # for _ in range(num_runs):
    #     ort_session.run(None, inputs)
    # onnx_time = timer() - start

    # assert reg_time > onnx_time


def test_trans_enc_b2(tmp_path):
    B = 1
    T = 10  # temporal sequence length
    E = 4  # embedding dimension
    H = 2  # num heads

    model_path = tmp_path / "transformer_encoder_b2.onnx"

    net = co.TransformerEncoder(
        co.TransformerEncoderLayerFactory(
            d_model=E,
            nhead=H,
            dim_feedforward=E * 2,
            dropout=0.0,
            sequence_len=T,
        ),
        num_layers=2,
    )

    net.eval()

    firsts = torch.rand(B, E, T)
    last = torch.ones(B, E)

    # Baseline
    state0 = None
    with torch.no_grad():
        for i in range(T - 1):
            _, state0 = net._forward_step(firsts[:, :, i], state0)
        o, state0 = net._forward_step(firsts[:, :, -1], state0)

    # Check against baseline
    o_baseline = net.forward(firsts)
    assert torch.allclose(o.squeeze(-1), o_baseline[:, :, -1])

    # Export to ONNX
    flat_state = [s.clone() for s in flatten(state0)]
    co.onnx.export(
        net,
        # Since a CoModule may choose to modify parts of its state rather than to
        # clone and update, we need to pass a clone to ensure fair comparison later
        (last, *[s.clone() for s in flat_state]),
        model_path,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        verbose=True,
        opset_version=14,
    )

    ort_session = ort.InferenceSession(str(model_path))

    # Run for whole input
    inputs = {
        "input": last.numpy(),
        **{
            k: v.detach().numpy()
            for k, v in zip(OnnxWrapper(net).state_input_names, flat_state)
        },
    }
    onnx_output, *onnx_state = ort_session.run(None, inputs)

    net.eval()
    target, target_state = net._forward_step(last, state0)

    for os, ts in zip(onnx_state, flatten(target_state)):
        assert torch.allclose(torch.tensor(os), ts)
    assert torch.allclose(torch.tensor(onnx_output), target)


def test_trans_enc_full(tmp_path):
    B = 1
    T = 10  # temporal sequence length
    E = 4  # embedding dimension
    H = 2  # num heads

    model_path = tmp_path / "transformer_encoder_full.onnx"

    pos_enc = co.RecyclingPositionalEncoding(
        embed_dim=E,
        num_embeds=T * 2 - 1,
        learned=False,
    )
    trans_enc = co.TransformerEncoder(
        co.TransformerEncoderLayerFactory(
            d_model=E,
            nhead=H,
            dim_feedforward=E * 2,
            dropout=0.0,
            sequence_len=T,
        ),
        num_layers=2,
    )
    lin = co.Linear(E, E, channel_dim=-1)

    net = co.Sequential(
        OrderedDict(
            [
                ("pos_enc", pos_enc),
                ("trans_enc", trans_enc),
                (
                    "select",
                    co.Lambda(
                        fn=lambda x: x[:, :, -1],
                        forward_step_only_fn=lambda x: x,
                        takes_time=True,
                    ),
                ),
                ("lin", lin),
            ]
        )
    )

    net.eval()

    firsts = torch.rand(B, E, T)
    last = torch.ones(B, E)

    # Baseline
    state0 = None
    with torch.no_grad():
        for i in range(T - 1):
            _, state0 = net._forward_step(firsts[:, :, i], state0)
        o, state0 = net._forward_step(firsts[:, :, -1], state0)

    # Check against baseline
    o_baseline = net.forward(firsts)
    assert torch.allclose(o, o_baseline, atol=1e-7)

    # Export to ONNX
    flat_state = [s.clone() for s in flatten(state0)]
    co.onnx.export(
        net,
        # Since a CoModule may choose to modify parts of its state rather than to
        # clone and update, we need to pass a clone to ensure fair comparison later
        (last, *[s.clone() for s in flat_state]),
        model_path,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        verbose=True,
        opset_version=14,
    )

    ort_session = ort.InferenceSession(str(model_path))

    # Run for whole input
    inputs = {
        "input": last.numpy(),
        **{
            k: v.detach().numpy()
            for k, v in zip(OnnxWrapper(net).state_input_names, flat_state)
        },
    }
    onnx_output, *onnx_state = ort_session.run(None, inputs)

    net.eval()
    target, target_state = net._forward_step(last, state0)

    for os, ts in zip(onnx_state, flatten(target_state)):
        assert torch.allclose(torch.tensor(os), ts, atol=1e-7)
    assert torch.allclose(torch.tensor(onnx_output), target, atol=1e-7)


def test_rnn_mix(tmp_path):
    batch_size = 1
    in_channels = 3
    hidden_channels = 3
    receptive_field = 1
    model_path = tmp_path / "rnn_mix.onnx"

    net = co.Sequential(
        co.RNN(in_channels, hidden_channels),
        co.GRU(in_channels, hidden_channels),
        co.LSTM(in_channels, hidden_channels),
    )
    net.eval()

    firsts = torch.arange(
        batch_size * in_channels * receptive_field, dtype=torch.float
    ).reshape(batch_size, in_channels, receptive_field)
    last = torch.ones(batch_size, in_channels)

    # Prep state
    state0 = None
    with torch.no_grad():
        for i in range(receptive_field):
            _, state0 = net._forward_step(firsts[:, :, i], state0)

    # Export to ONNX
    flat_state = [s.clone() for s in flatten(state0)]
    co.onnx.export(
        net,
        # Since a CoModule may choose to modify parts of its state rather than to
        # clone and update, we need to pass a clone to ensure fair comparison later
        (last, *[s.clone() for s in flat_state]),
        model_path,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        verbose=True,
        opset_version=11,
    )

    ort_session = ort.InferenceSession(str(model_path))

    # Run for whole input
    inputs = {
        "input": last.numpy(),
        **{
            k: v.detach().numpy()
            for k, v in zip(OnnxWrapper(net).state_input_names, flat_state)
        },
    }
    onnx_output, *onnx_state = ort_session.run(None, inputs)

    net.eval()
    target, target_state = net._forward_step(last, state0)

    for os, ts in zip(onnx_state, flatten(target_state)):
        assert torch.allclose(torch.tensor(os), ts)
    assert torch.allclose(torch.tensor(onnx_output), target)
