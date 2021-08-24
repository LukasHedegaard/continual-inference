from abc import ABC
from enum import Enum
from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor


class TensorPlaceholder:
    shape: Tuple[int]

    def __init__(self, shape: Tuple[int] = tuple()):
        self.shape = shape

    def size(self):
        return self.shape

    def __len__(self):
        return 0


# First element must be a Tensor
State = Union[
    Tuple[Tensor, int],
    Tuple[Tensor, int, int],
]


def _clone_first(state: State) -> State:
    return (state[0].clone(), *state[1:])


class CoModule(ABC):
    """Base class for continual modules.
    Deriving from this class enforces that neccessary class methods are implemented
    """

    def __init_subclass__(cls) -> None:
        CoModule._validate_class(cls)

    @staticmethod
    def _validate_class(cls):
        for fn, description in [
            ("forward_step", "forward computation for a single temporal step"),
            (
                "forward_steps",
                "forward computation for multiple temporal step",
            ),
            (
                "forward",
                "a forward computation which is identical to a regular non-continual forward.",
            ),
            ("get_state", "a retrieval of the internal state."),
            ("set_state", "an update of the internal state."),
            ("clean_state", "an internal state clean-up."),
        ]:
            assert callable(
                getattr(cls, fn, None)
            ), f"{cls} should implement a `{fn}` function which performs {description} to satisfy the CoModule interface."

        assert hasattr(cls, "delay") and type(cls.delay) in {
            int,
            property,
        }, f"{cls} should implement a `delay` property to satisfy the CoModule interface."

    @staticmethod
    def is_valid(module):
        try:
            CoModule._validate_class(module)
        except AssertionError:
            return False
        return True

    def get_state(self) -> Optional[State]:
        """Get model state

        Returns:
            Optional[State]: A State tuple if the model has been initialised and otherwise None.
        """
        ...  # pragma: no cover

    def set_state(self, state: State):
        """Set model state

        Args:
            state (State): State tuple to set as new internal internal state
        """
        ...  # pragma: no cover

    def clean_state(self):
        """Clean model state"""
        ...  # pragma: no cover

    make_padding = torch.zeros_like

    def forward_step(
        self, input: Tensor, update_state=True
    ) -> Union[Tensor, TensorPlaceholder]:
        """Forward computation for a single step with state initialisation

        Args:
            input (Tensor): Layer input.
            update_state (bool): Whether internal state should be updated during this operation.

        Returns:
            Union[Tensor, TensorPlaceholder]: Step output. This will be a placeholder while the module initialises and every (stride - 1) : stride.
        """
        state = self.get_state()
        if not update_state and state:
            state = _clone_first(state)
        output, state = self._forward_step(input, state)
        if update_state:
            self.set_state(state)
        return output

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True) -> Tensor:
        """Forward computation for multiple steps with state initialisation

        Args:
            input (Tensor): Layer input.
            pad_end (bool): Whether results for temporal padding at sequence end should be included.
            update_state (bool): Whether internal state should be updated during this operation.

        Returns:
            Tensor: Layer output
        """
        outs = []
        tmp_state = self.get_state()

        if not update_state and tmp_state:
            tmp_state = _clone_first(tmp_state)

        for t in range(input.shape[2]):
            o, tmp_state = self._forward_step(input[:, :, t], tmp_state)
            if isinstance(o, Tensor):
                outs.append(o)

        if update_state:
            self.set_state(tmp_state)

        if pad_end:
            # Don't save state for the end-padding
            tmp_state = _clone_first(self.get_state())
            for t, i in enumerate(
                [self.make_padding(input[:, :, -1]) for _ in range(self.padding[0])]
            ):
                o, tmp_state = self._forward_step(i, tmp_state)
                if isinstance(o, Tensor):
                    outs.append(o)

        if len(outs) == 0:
            return torch.tensor([])  # pragma: no cover

        return torch.stack(outs, dim=2)

    def forward(self, input: Tensor) -> Any:
        """Forward computation for multiple steps without state initialisation.
        This function is identical to the non-continual module found `torch.nn`

        Args:
            input (Tensor): Layer input.
        """
        ...  # pragma: no cover


class PaddingMode(Enum):
    REPLICATE = "replicate"
    ZEROS = "zeros"
