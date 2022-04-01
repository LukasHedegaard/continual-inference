from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence

from .module import CoModule, TensorPlaceholder

State = Tensor
LSTMState = Tuple[Tensor, Tensor]


class RNN(CoModule, nn.RNN):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity="tanh",
        bias: bool = True,
        batch_first: bool = True,  # NB: differs from torch.nn version!
        dropout: float = 0.0,
        bidirectional: bool = False,  # NB: differs from torch.nn version!
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ):
        assert (
            batch_first
        ), "`batch_first == False` is not supported for a Continual module"
        assert (
            not bidirectional
        ), "`bidirectional == True` is not supported for a Continual module"

        nn.RNN.__init__(
            self,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def build_from(module: nn.RNN, **kwargs) -> "RNN":
        comodule = RNN(
            **{
                **dict(
                    input_size=module.input_size,
                    hidden_size=module.hidden_size,
                    num_layers=module.num_layers,
                    nonlinearity=module.nonlinearity,
                    bias=module.bias,
                    batch_first=True,
                    dropout=module.dropout,
                    bidirectional=False,
                    device=module._flat_weights[0].device,
                    dtype=module._flat_weights[0].dtype,
                ),
                **kwargs,
            }
        )
        with torch.no_grad():
            for (ours, theirs) in zip(comodule._flat_weights, module._flat_weights):
                ours.copy_(theirs)
        return comodule

    @property
    def delay(self) -> int:
        return 0

    def clean_state(self):
        if hasattr(self, "_hidden_state"):
            del self._hidden_state

    def get_state(self) -> Optional[State]:
        if hasattr(self, "_hidden_state"):
            return self._hidden_state

    def set_state(self, state: State):
        self._hidden_state = state

    def forward(
        self, input: Union[Tensor, PackedSequence], hx: Optional[Tensor] = None
    ) -> Tuple[Union[Tensor, PackedSequence], Tensor]:
        """forward function.
        NB: input format is (B, C, T)

        Args:
            input (Union[Tensor, PackedSequence]): input tensot
            hx (Optional[Tensor], optional): optional previous hidden state. Defaults to None.

        Returns:
            Tuple[Union[Tensor, PackedSequence], Tensor]: Output tensor and output hidden state.
        """
        input = input.swapaxes(1, 2)  # B, C, T -> B, T, C
        output, hidden = nn.RNN.forward(self, input, hx)
        output = output.swapaxes(1, 2)  # B, T, C -> B, C, T
        return (output, hidden)

    def forward_step(
        self, input: Tensor, update_state=True
    ) -> Union[Tensor, TensorPlaceholder]:
        input = input.unsqueeze(1)  # B, C -> B, T, C
        output, new_state = nn.RNN.forward(self, input, self.get_state())
        output = output.squeeze(1)  # B, T, C -> B, C
        if update_state:
            self.set_state(new_state)
        return output

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True):
        output, new_state = self.forward(input, self.get_state())
        if update_state:
            self.set_state(new_state)
        return output


class GRU(CoModule, nn.GRU):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,  # NB: differs from torch.nn version!
        dropout: float = 0.0,
        bidirectional: bool = False,  # NB: differs from torch.nn version!
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ):
        assert (
            batch_first
        ), "`batch_first == False` is not supported for a Continual module"
        assert (
            not bidirectional
        ), "`bidirectional == True` is not supported for a Continual module"

        nn.GRU.__init__(
            self,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def build_from(module: nn.GRU, **kwargs) -> "GRU":
        comodule = GRU(
            **{
                **dict(
                    input_size=module.input_size,
                    hidden_size=module.hidden_size,
                    num_layers=module.num_layers,
                    bias=module.bias,
                    batch_first=True,
                    dropout=module.dropout,
                    bidirectional=False,
                    device=module._flat_weights[0].device,
                    dtype=module._flat_weights[0].dtype,
                ),
                **kwargs,
            }
        )
        with torch.no_grad():
            for (ours, theirs) in zip(comodule._flat_weights, module._flat_weights):
                ours.copy_(theirs)
        return comodule

    @property
    def delay(self) -> int:
        return 0

    def clean_state(self):
        if hasattr(self, "_hidden_state"):
            del self._hidden_state

    def get_state(self) -> Optional[State]:
        if hasattr(self, "_hidden_state"):
            return self._hidden_state

    def set_state(self, state: State):
        self._hidden_state = state

    def forward(
        self, input: Union[Tensor, PackedSequence], hx: Optional[Tensor] = None
    ) -> Tuple[Union[Tensor, PackedSequence], Tensor]:
        """forward function.
        NB: input format is (B, C, T)

        Args:
            input (Union[Tensor, PackedSequence]): input tensot
            hx (Optional[Tensor], optional): optional previous hidden state. Defaults to None.

        Returns:
            Tuple[Union[Tensor, PackedSequence], Tensor]: Output tensor and output hidden state.
        """
        input = input.swapaxes(1, 2)  # B, C, T -> B, T, C
        output, hidden = nn.GRU.forward(self, input, hx)
        output = output.swapaxes(1, 2)  # B, T, C -> B, C, T
        return (output, hidden)

    def forward_step(
        self, input: Tensor, update_state=True
    ) -> Union[Tensor, TensorPlaceholder]:
        input = input.unsqueeze(1)  # B, C -> B, T, C
        output, new_state = nn.GRU.forward(self, input, self.get_state())
        output = output.squeeze(1)  # B, T, C -> B, C
        if update_state:
            self.set_state(new_state)
        return output

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True):
        output, new_state = self.forward(input, self.get_state())
        if update_state:
            self.set_state(new_state)
        return output


class LSTM(CoModule, nn.LSTM):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,  # NB: differs from torch.nn version!
        dropout: float = 0.0,
        bidirectional: bool = False,  # NB: differs from torch.nn version!
        proj_size=0,
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ):
        assert (
            batch_first
        ), "`batch_first == False` is not supported for a Continual module"
        assert (
            not bidirectional
        ), "`bidirectional == True` is not supported for a Continual module"

        nn.LSTM.__init__(
            self,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def build_from(module: nn.LSTM, **kwargs) -> "LSTM":
        comodule = LSTM(
            **{
                **dict(
                    input_size=module.input_size,
                    hidden_size=module.hidden_size,
                    num_layers=module.num_layers,
                    bias=module.bias,
                    batch_first=True,
                    dropout=module.dropout,
                    bidirectional=False,
                    proj_size=module.proj_size,
                    device=module._flat_weights[0].device,
                    dtype=module._flat_weights[0].dtype,
                ),
                **kwargs,
            }
        )
        with torch.no_grad():
            for (ours, theirs) in zip(comodule._flat_weights, module._flat_weights):
                ours.copy_(theirs)
        return comodule

    @property
    def delay(self) -> int:
        return 0

    def clean_state(self):
        if hasattr(self, "_hidden_state"):
            del self._hidden_state
        if hasattr(self, "_cell_state"):
            del self._cell_state

    def get_state(self) -> Optional[LSTMState]:
        if hasattr(self, "_hidden_state") and hasattr(self, "_cell_state"):
            return (self._hidden_state, self._cell_state)

    def set_state(self, state: LSTMState):
        self._hidden_state, self._cell_state = state

    def forward(
        self,
        input: Union[Tensor, PackedSequence],
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Union[Tensor, PackedSequence], Tensor]:
        """forward function.
        NB: input format is (B, C, T)

        Args:
            input (Union[Tensor, PackedSequence]): input tensot
            hx (Optional[Tuple[Tensor, Tensor]], optional): optional previous hidden state and cell state. Defaults to None.

        Returns:
            Tuple[Union[Tensor, PackedSequence], Tensor]: Output tensor and output hidden state.
        """
        input = input.swapaxes(1, 2)  # B, C, T -> B, T, C
        output, hidden = nn.LSTM.forward(self, input, hx)
        output = output.swapaxes(1, 2)  # B, T, C -> B, C, T
        return (output, hidden)

    def forward_step(
        self, input: Tensor, update_state=True
    ) -> Union[Tensor, TensorPlaceholder]:
        input = input.unsqueeze(1)  # B, C -> B, T, C
        output, new_state = nn.LSTM.forward(self, input, self.get_state())
        output = output.squeeze(1)  # B, T, C -> B, C
        if update_state:
            self.set_state(new_state)
        return output

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True):
        output, new_state = self.forward(input, self.get_state())
        if update_state:
            self.set_state(new_state)
        return output
