import functools
import itertools
from abc import ABC
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.utils.hooks as hooks
from torch import Tensor
from torch.nn.modules.module import (
    _global_backward_hooks,
    _global_forward_hooks,
    _global_forward_pre_hooks,
)

# First element is a value buffer, while others are indexes
# State = Tuple[Tensor, Optional[int], Optional[int]]
State = Tuple[Tensor, Tensor, Tensor]


class PaddingMode(Enum):
    REPLICATE = "replicate"
    ZEROS = "zeros"


# Not compatible with torch.jit:
# class CallMode(Enum):
#     FORWARD = "forward"
#     FORWARD_STEPS = "forward_steps"
#     FORWARD_STEP = "forward_step"

CALL_MODES = {
    "forward": torch.tensor(0),
    "forward_steps": torch.tensor(1),
    "forward_step": torch.tensor(2),
}


def _callmode(cm) -> torch.Tensor:
    """_summary_

    Args:
        cm (Union[str, int, torch.Tensor]): Identifier for call mode

    Returns:
        torch.Tensor: validated call_mode
    """
    if isinstance(cm, str):
        cm = CALL_MODES[cm.lower()]
    elif isinstance(cm, int):
        cm = torch.tensor(int)
    return cm


class _CallModeContext(object):
    """Context-manager state holder."""

    def __init__(self):
        self.cur = _callmode("forward")
        self.prev = None

    def __call__(self, value: Union[str, int, torch.Tensor]) -> "_CallModeContext":
        self.prev = self.cur
        self.cur = _callmode(value)
        return self

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        self.cur = self.prev
        self.prev = None


call_mode = _CallModeContext()
"""Context-manager that holds a call_mode

When the call_mode context is used, the __call__ function of continual modules with be set accordingly

Example:
    >>> forward_output = module(forward_input)
    >>>
    >>> with co.call_mode("forward_step"):
    >>>     forward_step_output = module(forward_step_input)
"""


def _clone_first(state: State) -> State:
    # return (state[0].clone(), state[1], state[2])
    return (state[0].clone(),) + state[1:]
    # return (state[0].clone(),) + state[1:]


class CoModule(ABC):
    """Base class for continual modules.
    Deriving from this class enforces that neccessary class methods are implemented
    """

    receptive_field: int = 1
    stride: Tuple[int, ...] = (1,)
    padding: Tuple[int, ...] = (0,)
    make_padding = torch.zeros_like
    _state_shape: int = 0
    _dynamic_state_inds: List[bool] = []
    _call_mode = _callmode("forward")

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

        for prop in {"delay", "receptive_field"}:
            assert type(getattr(cls, prop, None)) in {
                int,
                torch.Tensor,
                property,
            }, f"{cls} should implement a `{prop}` property to satisfy the CoModule interface."

        for prop in {"stride", "padding", "_state_shape", "_dynamic_state_inds"}:
            assert type(getattr(cls, prop, None)) in {
                int,
                property,
                tuple,
                list,
            }, f"{cls} should implement a `{prop}` property to satisfy the CoModule interface."

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

    @property
    def delay(self) -> int:
        return self.receptive_field - 1 - self.padding[0]

    def forward_step(
        self, input: Tensor, update_state: bool = True
    ) -> Optional[Tensor]:
        """Forward computation for a single step with state initialisation

        Args:
            input (Tensor): Layer input.
            update_state (bool): Whether internal state should be updated during this operation.

        Returns:
            Optional[Tensor]: Step output. This will be a placeholder while the module initialises and every (stride - 1) : stride.
        """
        state = self.get_state()
        if not update_state and state:
            state = _clone_first(state)
        output, state = self._forward_step(input, state)
        if update_state and state:
            self.set_state(state)
        return output

    def forward_steps(
        self, input: Tensor, pad_end: bool = False, update_state=True
    ) -> Optional[Tensor]:
        """Forward computation for multiple steps with state initialisation

        Args:
            input (Tensor): Layer input.
            pad_end (bool): Whether results for temporal padding at sequence end should be included.
            update_state (bool): Whether internal state should be updated during this operation.

        Returns:
            Tensor: Layer output
        """
        return self._forward_steps_impl(input, pad_end, update_state)

    def _forward_steps_impl(
        self, input: Tensor, pad_end: bool = False, update_state: bool = True
    ) -> Optional[Tensor]:
        """Forward computation for multiple steps with state initialisation

        Args:
            module (CoModule): Continual module.
            input (Tensor): Layer input.
            pad_end (bool): Whether results for temporal padding at sequence end should be included.
            update_state (bool): Whether internal state should be updated during this operation.

        Returns:
            Tensor: Layer output
        """
        outs = []
        state: Optional[State] = self.get_state()

        if not update_state and state is not None:
            state = _clone_first(state)

        for t in range(input.shape[2]):
            o, state = self._forward_step(input[:, :, t], state)
            if isinstance(o, Tensor):
                outs.append(o)

        if update_state and state is not None:
            self.set_state(state)

        if pad_end:
            # Don't save state for the end-padding
            opt_state = self.get_state()
            if opt_state is not None:
                state = _clone_first(opt_state)

            for t, i in enumerate(
                [self.make_padding(input[:, :, -1]) for _ in range(self.padding[0])]
            ):
                o, state = self._forward_step(i, state)
                if isinstance(o, Tensor):
                    outs.append(o)

        if len(outs) == 0:
            return None  # pragma: no cover

        return torch.stack(outs, dim=2)

    def forward(self, input: Tensor) -> Any:
        """Forward computation for multiple steps without state initialisation.
        This function is identical to the non-continual module found `torch.nn`

        Args:
            input (Tensor): Layer input.
        """
        ...  # pragma: no cover

    def warm_up(self, step_shape: List[int]):
        """Warms up the model state with a dummy input.
        The initial `self.delay` steps will still produce inexact results.

        Args:
            step_shape (Sequence[int]): input shape with which to warm the model up, including batch size.
        """
        step_shape = (*step_shape[:2], self.delay, *step_shape[2:])
        dummy = self.make_padding(torch.zeros(step_shape, dtype=torch.float))
        self.forward_steps(dummy)

    @property
    def call_mode(self) -> torch.Tensor:
        return self._call_mode

    @call_mode.setter
    def call_mode(self, value):
        self._call_mode = _callmode(value)
        if hasattr(self, "__len__"):
            for m in self:
                if hasattr(m, "call_mode"):
                    m.call_mode = self._call_mode

    def _call_impl(self, *input, **kwargs):  # noqa: C901  # pragma: no cover
        """Modified version torch.nn.Module._call_impl

        Returns:
            [type]: [description]
        """
        _call_mode = call_mode.cur if call_mode.prev is not None else self.call_mode
        forward_call = {
            (True, _callmode("forward")): self._slow_forward,
            (False, _callmode("forward")): self.forward,
            (False, _callmode("forward_steps")): self.forward_steps,
            (False, _callmode("forward_step")): self.forward_step,
        }[(bool(torch._C._get_tracing_state()), _call_mode)]

        # If we don't have any hooks, we want to skip the rest of the logic in
        # this function, and just call forward.
        if not (
            self._backward_hooks
            or self._forward_hooks
            or self._forward_pre_hooks
            or _global_backward_hooks
            or _global_forward_hooks
            or _global_forward_pre_hooks
        ):
            return forward_call(*input, **kwargs)
        # Do not call functions when jit is used
        full_backward_hooks, non_full_backward_hooks = [], []
        if self._backward_hooks or _global_backward_hooks:
            full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()
        if _global_forward_pre_hooks or self._forward_pre_hooks:
            for hook in itertools.chain(
                _global_forward_pre_hooks.values(), self._forward_pre_hooks.values()
            ):
                result = hook(self, input)
                if result is not None:
                    if not isinstance(result, tuple):
                        result = (result,)
                    input = result

        bw_hook = None
        if full_backward_hooks:
            bw_hook = hooks.BackwardHook(self, full_backward_hooks)
            input = bw_hook.setup_input_hook(input)

        result = forward_call(*input, **kwargs)
        if _global_forward_hooks or self._forward_hooks:
            for hook in itertools.chain(
                _global_forward_hooks.values(), self._forward_hooks.values()
            ):
                hook_result = hook(self, input, result)
                if hook_result is not None:
                    result = hook_result

        if bw_hook:
            result = bw_hook.setup_output_hook(result)

        # Handle the non-full backward hooks
        if non_full_backward_hooks:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in non_full_backward_hooks:
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
                self._maybe_warn_non_full_backward_hook(input, result, grad_fn)

        return result

    __call__ = _call_impl
