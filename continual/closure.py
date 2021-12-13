from functools import partial
from typing import Callable, Union

import torch
from torch import Tensor, nn

from .module import CoModule
from .utils import function_repr


class Lambda(CoModule, nn.Module):
    """Module wrapper for stateless functions.

    NB: Operations performed in a Lambda are not counted in `ptflops`

    Args:
        fn (Callable[[Tensor], Tensor]): Function to be called during forward.
        forward_only_fn (Callable[[Tensor], Tensor]): Function to be called only during `forward`. `fn` is used for the other call modes.
        forward_step_only_fn (Callable[[Tensor], Tensor]): Function to be called only during `forward_step`. `fn` is used for the other call modes.
        forward_steps_only_fn (Callable[[Tensor], Tensor]): Function to be called only during `forward_steps`. `fn` is used for the other call modes.
        forward_only_fn (Callable[[Tensor], Tensor]): Function to be called only during forward. `fn` is used for the other call modes.
        takes_time (bool, optional): If True, `fn` recieves all steps, if False, it received one step and no time dimension. Defaults to False.
    """

    def __init__(
        self,
        fn: Callable[[Tensor], Tensor] = None,
        forward_only_fn=None,
        forward_step_only_fn=None,
        forward_steps_only_fn=None,
        takes_time=False,
    ):
        nn.Module.__init__(self)
        assert callable(fn) or all(
            callable(forward_only_fn),
            callable(forward_step_only_fn),
            callable(forward_steps_only_fn),
        ), "Either fn or all of forward_only_fn, forward_step_only_fn, and forward_steps_only_fn should be callable."

        self.fn = fn
        self.forward_only_fn = forward_only_fn
        self.forward_step_only_fn = forward_step_only_fn
        self.forward_steps_only_fn = forward_steps_only_fn
        self.takes_time = takes_time

    @staticmethod
    def build_from(
        fn: Callable[[Tensor], Tensor],
        forward_only_fn=None,
        forward_step_only_fn=None,
        forward_steps_only_fn=None,
        takes_time=False,
    ) -> "Lambda":
        return Lambda(
            fn, forward_only_fn, forward_step_only_fn, forward_steps_only_fn, takes_time
        )

    def __repr__(self) -> str:
        s = "Lambda("
        if callable(self.fn):
            s += f"{function_repr(self.fn)}"
        if callable(self.forward_only_fn):
            if callable(self.fn):
                s += ", "
            s += f"{function_repr(self.forward_only_fn)}"
        if callable(self.forward_step_only_fn):
            s += f", {function_repr(self.forward_step_only_fn)}"
        if callable(self.forward_steps_only_fn):
            s += f", {function_repr(self.forward_steps_only_fn)}"
        if self.takes_time:
            s += ", takes_time=True"
        s += ")"
        return s

    def forward(self, input: Tensor) -> Tensor:
        if self.forward_only_fn is not None:
            return self.forward_only_fn(input)

        if self.takes_time:
            return self.fn(input)

        return torch.stack(
            [self.fn(input[:, :, t]) for t in range(input.shape[2])], dim=2
        )

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True) -> Tensor:
        if self.forward_steps_only_fn is not None:
            return self.forward_steps_only_fn(input)

        if self.takes_time:
            return self.fn(input)

        return torch.stack(
            [self.fn(input[:, :, t]) for t in range(input.shape[2])], dim=2
        )

    def forward_step(self, input: Tensor, update_state=True) -> Tensor:
        if self.forward_step_only_fn is not None:
            return self.forward_step_only_fn(input)

        if self.takes_time:
            input = input.unsqueeze(dim=2)
        output = self.fn(input)
        if self.takes_time:
            output = output.squeeze(dim=2)
        return output


def _multiply(x: Tensor, factor: Union[float, int, Tensor]):
    return x * factor


def Multiply(factor) -> Lambda:
    """Create Lambda with multiplication function"""
    fn = partial(_multiply, factor=factor)
    return Lambda(fn, takes_time=True)


def _add(x: Tensor, constant: Union[float, int, Tensor]):
    return x + constant


def Add(constant) -> Lambda:
    """Create Lambda with addition function"""
    fn = partial(_add, constant=constant)
    return Lambda(fn, takes_time=True)


def _unity(x: Tensor):
    return x


def Unity() -> Lambda:
    """Create Lambda with addition function"""
    return Lambda(_unity, takes_time=True)


def Constant(constant: float):
    return Lambda(lambda x: constant * torch.ones_like(x), takes_time=True)


def Zero() -> Lambda:
    """Create Lambda with zero output"""
    return Lambda(torch.zeros_like, takes_time=True)


def One() -> Lambda:
    """Create Lambda with zero output"""
    return Lambda(torch.ones_like, takes_time=True)
