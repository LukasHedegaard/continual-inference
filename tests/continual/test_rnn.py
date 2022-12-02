import torch
from torch import nn

import continual as co


def test_rnn():
    C = 2
    T = 10
    sample = torch.normal(mean=torch.zeros(C * T)).reshape((1, C, T))

    # Regular
    reg = nn.RNN(
        input_size=C, hidden_size=3, num_layers=4, batch_first=True, bidirectional=False
    )
    _, hidden = reg.forward(sample.swapaxes(1, 2))  # (B, T, C) required
    target, _ = reg.forward(sample.swapaxes(1, 2), hidden)
    target = target.swapaxes(1, 2)  # B, T, C -> B, C, T

    # Continual
    con = co.RNN.build_from(reg)
    assert con.delay == 0

    # forward
    out1, _ = con.forward(sample, hidden)
    assert torch.allclose(out1, target)

    con.set_state(hidden)
    # forward_steps
    out2 = con.forward_steps(sample[:, :, :-1])
    assert torch.allclose(out2, target[:, :, :-1])

    # forward_step
    out3 = con.forward_step(sample[:, :, -1])
    assert torch.allclose(out3, target[:, :, -1])

    # Check that state initialisation is the same
    con.clean_state()
    con.forward_steps(sample)
    assert torch.allclose(con.get_state()[0], hidden)


def test_gru():
    C = 2
    T = 10
    sample = torch.normal(mean=torch.zeros(C * T)).reshape((1, C, T))

    # Regular
    reg = nn.GRU(
        input_size=C, hidden_size=3, num_layers=4, batch_first=True, bidirectional=False
    )
    _, hidden = reg.forward(sample.swapaxes(1, 2))  # (B, T, C) required
    target, _ = reg.forward(sample.swapaxes(1, 2), hidden)
    target = target.swapaxes(1, 2)  # B, T, C -> B, C, T

    # Continual
    con = co.GRU.build_from(reg)
    assert con.delay == 0

    # forward
    out1, _ = con.forward(sample, hidden)
    assert torch.allclose(out1, target)

    con.set_state(hidden)
    # forward_steps
    out2 = con.forward_steps(sample[:, :, :-1])
    assert torch.allclose(out2, target[:, :, :-1])

    # forward_step
    out3 = con.forward_step(sample[:, :, -1])
    assert torch.allclose(out3, target[:, :, -1])

    # Check that state initialisation is the same
    con.clean_state()
    con.forward_steps(sample)
    assert torch.allclose(con.get_state()[0], hidden)


def test_lstm():
    C = 2
    T = 10
    sample = torch.normal(mean=torch.zeros(C * T)).reshape((1, C, T))

    # Regular
    reg = nn.LSTM(
        input_size=C,
        hidden_size=3,
        num_layers=4,
        batch_first=True,
        bidirectional=False,
        proj_size=2,
    )
    _, state = reg.forward(sample.swapaxes(1, 2))  # (B, T, C) required
    target, _ = reg.forward(sample.swapaxes(1, 2), state)
    target = target.swapaxes(1, 2)  # B, T, C -> B, C, T

    # Continual
    con = co.LSTM.build_from(reg)
    assert con.delay == 0

    # forward
    out1, _ = con.forward(sample, state)
    assert torch.allclose(out1, target)

    con.set_state(state)
    # forward_steps
    out2 = con.forward_steps(sample[:, :, :-1])
    assert torch.allclose(out2, target[:, :, :-1])

    # forward_step
    out3 = con.forward_step(sample[:, :, -1])
    assert torch.allclose(out3, target[:, :, -1])

    # Check that state initialisation is the same
    con.clean_state()
    con.forward_steps(sample)
    assert torch.allclose(con.get_state()[0], state[0])
    assert torch.allclose(con.get_state()[1], state[1])
