import shutil
from pathlib import Path
from timeit import default_timer as timer

import onnxruntime as ort
import pytest
import torch
from torch import onnx

import continual as co


@pytest.fixture
def tmp_path():
    p = Path("tmp")
    p.mkdir(exist_ok=True)
    yield p
    shutil.rmtree(p)


class CollapseArgs(torch.nn.Module):
    """Collapses input args and flattens output args.
    This is necessary as the ``dynamic_axes`` arg for
    :py:meth:`torch.onnx.export` doesn't accept nested Tuples.
    Args:
        model: A :py:class:`DeepSpeech1`.
    """

    def __init__(self, model: co.Conv1d):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, *states: torch.Tensor):
        out, next_states = self.model._forward_step(x, states)
        return out, *next_states


def test_conv_forward_step(tmp_path):
    batch_size = 1
    in_channels = 3
    out_channels = 4
    receptive_field = 5
    model_path = tmp_path / "conv.onnx"

    net = co.Conv1d(in_channels, out_channels, receptive_field)
    net.call_mode = "forward_step"
    net.train()  # For testing purposes, we force the network to clone its buffer
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
        CollapseArgs(net),
        (last, *[s.clone() for s in state0]),
        model_path,
        input_names=["input", *[f"i{i}" for i in range(3)]],
        output_names=["output", *[f"o{i}" for i in range(3)]],
        dynamic_axes={"input": [0], "output": [0], "i0": [0], "o0": [0]},
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

    assert reg_time > 2 * onnx_time
    assert ts_time > 2 * onnx_time


def xtest_sequential_pure():
    pass


def xtest_sequential_mixed():
    pass


def xtest_residual():
    pass


def xtest_trans():
    pass
