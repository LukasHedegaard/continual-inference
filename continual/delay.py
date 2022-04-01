from typing import Tuple, Union

import torch
from torch import Tensor

from .module import CoModule, PaddingMode, TensorPlaceholder
from .utils import temporary_parameter

State = Tuple[Tensor, int]

__all__ = ["Delay"]


class Delay(CoModule, torch.nn.Module):
    """Continual delay modules

    This module only introduces a delay in the continual modes, i.e. on `forward_step` and `forward_steps`.
    This corresponds to the equvalent computations when delay is used to align continual computations.
    """

    def __init__(
        self,
        delay: int,
        temporal_fill: PaddingMode = "zeros",
        auto_shrink: Union[bool, str] = False,
    ):
        """Initialise Delay block

        Args:
            delay (int): the number of steps to delay an output.
            temporal_fill (PaddingMode, optional): Temporal state initialisation mode ("zeros" or "replicate"). Defaults to "zeros".
            auto_shrink (bool | str, optional): Whether to shrink the temporal dimension of the feature map during forward.
                This module is handy for residuals that are parallel to modules which reduce the number of temporal steps.
                Options: "centered" or True: Centered residual shrink; "lagging": lagging shrink. Defaults to False.
        """
        assert delay >= 0
        self._delay = delay
        assert auto_shrink in {True, False, "centered", "lagging"}
        self.auto_shrink = auto_shrink
        assert temporal_fill in {"zeros", "replicate"}
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
        first_run = self.get_state() is None
        if self._delay == 0:
            return input

        with temporary_parameter(self, "padding", (self.delay,)):
            output = CoModule.forward_steps(self, input, pad_end, update_state)

        if first_run and self.auto_shrink in {True, "centered"}:
            output = output[:, :, self.delay :]
        return output

    def forward(self, input: Tensor) -> Tensor:
        # No delay during regular forward
        if not self.auto_shrink or self.delay == 0:
            return input
        if self.auto_shrink == "lagging":
            return input[:, :, : -self.delay]
        return input[:, :, self.delay : -self.delay]

    @property
    def receptive_field(self) -> int:
        return self.delay + 1

    @property
    def delay(self) -> int:
        return self._delay

    @property
    def stride(self) -> int:
        return 1

    def extra_repr(self):
        shrink_str = ", auto_shrink=True" if self.auto_shrink else ""
        return f"{self.delay}" + shrink_str
