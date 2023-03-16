from typing import Tuple

import torch
from torch import Tensor

from .module import CoModule

State = Tuple[Tensor]

__all__ = ["Skip"]


class Skip(CoModule, torch.nn.Module):
    """Skip a number of input steps.

    Arguments:
        steps: The number of steps to skip.
    """

    @property
    def _state_shape(self):
        return 1

    @property
    def _dynamic_state_inds(self):
        return [False]

    def __init__(self, steps: int):
        assert steps > 0
        self.steps = steps

        super(Skip, self).__init__()
        self.register_buffer("state_index", torch.tensor(-self.steps), persistent=False)

    def clean_state(self):
        self.state_index = torch.tensor(-self.steps)

    def get_state(self):
        return (self.state_index,)

    def set_state(self, state: State):
        self.state_index = state[0]

    def _forward_step(self, input: Tensor, prev_state: State) -> Tuple[Tensor, State]:
        if prev_state is None:
            index = torch.tensor(-self.steps)
        else:
            index = prev_state[0]

        if index >= 0:
            output = input
            new_index = index
        else:
            output = None
            new_index = index + 1

        return output, (new_index,)

    def forward(self, input: Tensor) -> Tensor:
        return input[:, :, self.steps :]

    @property
    def receptive_field(self) -> int:
        return 1

    @property
    def delay(self) -> int:
        return 0

    @property
    def stride(self) -> Tuple[int]:
        return (1,)

    def extra_repr(self):
        return f"{self.steps}"
