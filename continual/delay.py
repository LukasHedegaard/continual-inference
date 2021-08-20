from typing import Tuple

import torch
from torch import Tensor

from .interface import CoModule, Padded, PaddingMode, TensorPlaceholder

State = Tuple[Tensor, int]

__all__ = ["Delay"]


class Delay(torch.nn.Module, Padded, CoModule):
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
        self.state_buffer = None
        self.state_index = None

    def get_state(self):
        if (
            hasattr(self, "state_buffer")
            and self.state_buffer is not None
            and hasattr(self, "state_index")
            and self.state_buffer is not None
        ):
            return (self.state_buffer, self.state_index)
        else:
            return None

    def forward_step(self, input: Tensor) -> Tensor:
        if self._delay == 0:
            return input

        output, (self.state_buffer, self.state_index) = self._forward_step(
            input, self.get_state()
        )
        return output

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

    def forward_steps(self, input: Tensor, pad_end=True) -> Tensor:
        outs = []
        for t in range(input.shape[2]):
            o = self.forward_step(input[:, :, t])
            if isinstance(o, Tensor):
                outs.append(o)

        if pad_end:
            # Empty out delay values, but don't save state for the end-padding
            (tmp_buffer, tmp_index) = self.get_state()
            tmp_buffer = tmp_buffer.clone()
            for t, i in enumerate(
                [self.make_padding(input[:, :, -1]) for _ in range(self.delay)]
            ):
                o, (tmp_buffer, tmp_index) = self._forward_step(
                    i, (tmp_buffer, tmp_index)
                )
                if isinstance(o, Tensor):
                    outs.append(o)

        if len(outs) > 0:
            outs = torch.stack(outs, dim=2)
        else:
            outs = torch.tensor([])  # pragma: no cover

        return outs

    def forward(self, input: Tensor) -> Tensor:
        # No delay during regular forward
        # nan = torch.tensor(float('nan')).repeat(3,2)
        return input

    @property
    def delay(self) -> int:
        return self._delay

    def extra_repr(self):
        return f"{self.delay}"
