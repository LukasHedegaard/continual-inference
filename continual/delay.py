from typing import Tuple, Union

import torch
from torch import Tensor

from .module import CoModule, PaddingMode
from .utils import temporary_parameter

State = Tuple[Tensor, Tensor]

__all__ = ["Delay"]


class Delay(CoModule, torch.nn.Module):
    """Delay an input by a number of steps.

    This module only introduces a delay in the continual modes, i.e. on ``forward_step`` and ``forward_steps``.
    In essence it caches the input for ``delay`` steps before outputting it again.

    The ``Delay`` modules is used extensively in various container modules to align delays of
    different computational branches. For instance, it is used to align the :class:`Residual` module
    as shown in the example below.

    Arguments:
        delay: The number of steps to delay an output.
        temporal_fill: Temporal state initialisation mode ("zeros" or "replicate")
        auto_shrink: Whether to shrink the temporal dimension of the feature map during forward.
            This module is handy for residuals that are parallel to modules which reduce the number of temporal steps.
            Options: "centered" or True: Centered residual shrink; "lagging": lagging shrink.

    Examples::

        conv = co.Conv3d(32, 32, kernel_size=3, padding=1)
        residual = co.BroadcastReduce(conv, co.Delay(2), reduce="sum")

    """

    @property
    def _state_shape(self):
        return 2 if self.delay > 0 else 0

    @property
    def _dynamic_state_inds(self):
        return [True, False] if self.delay > 0 else []

    def __init__(
        self,
        delay: int,
        temporal_fill: PaddingMode = "zeros",
        auto_shrink: Union[bool, str] = False,
    ):
        assert delay >= 0
        self._delay = delay
        assert auto_shrink in {True, False, "centered", "lagging"}
        self.auto_shrink = auto_shrink
        assert temporal_fill in {"zeros", "replicate"}
        self._make_padding = {"zeros": torch.zeros_like, "replicate": torch.clone}[
            temporal_fill
        ]

        super(Delay, self).__init__()
        self.register_buffer("state_buffer", torch.tensor([]), persistent=False)
        self.register_buffer("state_index", torch.tensor(0), persistent=False)

    def init_state(
        self,
        first_output: Tensor,
    ) -> State:
        padding = self._make_padding(first_output)
        state_buffer = torch.stack([padding for _ in range(self.delay)], dim=0)
        state_index = torch.tensor(
            -2 * self.delay
            if self.auto_shrink and isinstance(self.auto_shrink, bool)
            else -self.delay
        )
        return state_buffer, state_index

    def clean_state(self):
        self.state_buffer = torch.tensor([])
        self.state_index = torch.tensor(0)

    def get_state(self):
        if len(self.state_buffer) > 0:
            return (self.state_buffer, self.state_index)
        return None

    def set_state(self, state: State):
        self.state_buffer, self.state_index = state

    def _forward_step(self, input: Tensor, prev_state: State) -> Tuple[Tensor, State]:
        if self._delay == 0:
            return input, prev_state

        if prev_state is None:
            buffer, index = self.init_state(input)
        else:
            buffer, index = prev_state

        # Get output
        if index >= 0:
            output = buffer[index].clone()
        else:
            output = None

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
    def stride(self) -> Tuple[int]:
        return (1,)

    def extra_repr(self):
        shrink_str = ", auto_shrink=True" if self.auto_shrink else ""
        return f"{self.delay}" + shrink_str
