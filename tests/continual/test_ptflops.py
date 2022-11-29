import torch
from ptflops import get_model_complexity_info
from torch import nn

import continual as co


def test_conv():
    #            C  T  H  W
    T = 4
    input_res = (3, T, 3, 3)

    mod = nn.Conv3d(3, 3, 3, padding=(1, 0, 0))
    macs, params = get_model_complexity_info(mod, input_res, as_strings=False)

    co_mod = co.Conv3d(3, 3, 3, padding=(1, 0, 0))
    co_macs, co_params = get_model_complexity_info(co_mod, input_res, as_strings=False)

    assert macs == co_macs
    assert params == co_params

    co_mod = co.Conv3d(3, 3, 3)  # Padding not necessary
    co_mod.forward_steps(torch.randn(1, 3, T - 1, 3, 3))  # warm up
    with co.call_mode("forward_step"):
        co_step_macs, co_step_params = get_model_complexity_info(
            co_mod, (3, 3, 3), as_strings=False
        )

    assert macs / T == co_step_macs
    assert params == co_step_params


def test_linear():
    input_res = (3, 3)

    mod = nn.Linear(3, 3)
    macs, params = get_model_complexity_info(mod, input_res, as_strings=False)

    co_mod = co.Linear(3, 3)
    co_macs, co_params = get_model_complexity_info(co_mod, input_res, as_strings=False)

    assert macs == co_macs
    assert params == co_params

    with co.call_mode("forward_step"):
        co_step_macs, co_step_params = get_model_complexity_info(
            co_mod, (3, 3), as_strings=False
        )

    assert macs == co_step_macs
    assert params == co_step_params


def test_sequential():
    #            C  T  H  W
    T = 4
    input_res = (3, T)
    seq = [
        co.Conv1d(3, 3, 3, padding=(1,)),
        co.Conv1d(3, 3, 3, padding=(1,)),
    ]
    mod = nn.Sequential(*seq)
    macs, params = get_model_complexity_info(mod, input_res, as_strings=False)

    co_mod = co.Sequential(*seq)
    co_macs, co_params = get_model_complexity_info(co_mod, input_res, as_strings=False)

    assert macs == co_macs
    assert params == co_params

    co_mod.call_mode = "forward_step"
    co_mod.warm_up(step_shape=(1, 3))
    co_step_macs, co_step_params = get_model_complexity_info(
        co_mod, (3,), as_strings=False
    )

    assert macs / T == co_step_macs
    assert params == co_step_params


def test_broadcast_reduce():
    #            C  T  H  W
    T = 4
    input_res = (3, T)

    seq = co.Sequential(
        co.Conv1d(3, 3, 3, padding=(1,)), co.Conv1d(3, 3, 3, padding=(1,))
    )

    br = co.BroadcastReduce(
        co.Conv1d(3, 3, 3, padding=(1,)), co.Conv1d(3, 3, 3, padding=(1,))
    )

    # forward
    seq_macs, seq_params = get_model_complexity_info(seq, input_res, as_strings=False)
    br_macs, br_params = get_model_complexity_info(br, input_res, as_strings=False)

    assert seq_macs == br_macs
    assert seq_params == br_params

    # warm up
    seq.forward_steps(torch.randn((1, *input_res)))
    br.forward_steps(torch.randn((1, *input_res)))

    # forward_step
    with co.call_mode("forward_step"):
        seq_macs_step, _ = get_model_complexity_info(seq, (3,), as_strings=False)
        br_macs_step, _ = get_model_complexity_info(br, (3,), as_strings=False)

    assert seq_macs_step == br_macs_step
    assert seq_macs / T == seq_macs_step
    assert br_macs / T == br_macs_step


def test_converted():
    T = 4
    input_res = (3, T)

    mod = co.continual(nn.BatchNorm1d(3))
    seq_macs, params = get_model_complexity_info(mod, input_res, as_strings=False)

    assert params == 6

    with co.call_mode("forward_step"):
        step_macs, _ = get_model_complexity_info(mod, (3,), as_strings=False)

    assert seq_macs / T == step_macs
