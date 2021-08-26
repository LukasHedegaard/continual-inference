from typing import Tuple

import torch
from torch import Tensor

from .module import CoModule, PaddingMode, TensorPlaceholder
from .utils import temporary_parameter

State = Tuple[Tensor, int]

__all__ = ["Delay"]


class Delay(torch.nn.Module, CoModule):
    """Continual delay modules

    This module only introduces a delay in the continual modes, i.e. on `forward_step` and `forward_steps`.
    This corresponds to the equvalent computations when delay is used to align continual computations.
    """

    def __init__(
        self,
        delay: int,
        temporal_fill: PaddingMode = "zeros",
    ):
        assert temporal_fill in {"zeros", "replicate"}
        self._delay = delay
        self.make_padding = {"zeros": torch.zeros_like, "replicate": torch.clone}[
            temporal_fill
        ]

        super(Delay, self).__init__()
        # state is initialised in self.forward

    def init_state(
        self,
        first_output: Tensor,
    ) -> State:
        padding = self.make_padding(first_output)
        state_buffer = torch.stack([padding for _ in range(self.delay)], dim=0)
        state_index = -self.delay
        if not hasattr(self, "state_buffer"):
            self.register_buffer("state_buffer", state_buffer, persistent=False)
        return state_buffer, state_index

    def clean_state(self):
        if hasattr(self, "state_buffer"):
            del self.state_buffer
        if hasattr(self, "state_index"):
            del self.state_index

    def get_state(self):
        if (
            hasattr(self, "state_buffer")
            and self.state_buffer is not None
            and hasattr(self, "state_index")
            and self.state_buffer is not None
        ):
            return (self.state_buffer, self.state_index)

    def set_state(self, state: State):
        self.state_buffer, self.state_index = state

    def _forward_step(self, input: Tensor, prev_state: State) -> Tuple[Tensor, State]:
        if prev_state is None:
            buffer, index = self.init_state(input)
        else:
            buffer, index = prev_state

        # Get output
        if index >= 0:
            output = buffer[index].clone()
        else:
            output = TensorPlaceholder(buffer[0].shape)

        # Update state
        buffer[index % self.delay] = input
        new_index = index + 1
        if new_index > 0:
            new_index = new_index % self.delay

        return output, (buffer, new_index)

    def forward_step(self, input: Tensor, update_state=True) -> Tensor:
        if self._delay == 0:
            return input

        return CoModule.forward_step(self, input, update_state)

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True) -> Tensor:
        if self._delay == 0:
            return input

        with temporary_parameter(self, "padding", (self.delay,)):
            output = CoModule.forward_steps(self, input, pad_end, update_state)

        return output

    def forward(self, input: Tensor) -> Tensor:
        # No delay during regular forward
        return input

    @property
    def delay(self) -> int:
        return self._delay

    def extra_repr(self):
        return f"{self.delay}"
