from functools import partial
from typing import Callable, Union

import torch
from torch import Tensor, nn

from .module import CoModule
from .utils import function_repr

__all__ = [
    "Lambda",
    "Multiply",
    "Add",
    "Identity",
    "Constant",
    "Zero",
    "One",
]


class Lambda(CoModule, nn.Module):
    """Module wrapper for stateless functions.

    .. note::
        Operations performed in a Lambda are not counted in `ptflops`

    Args:
        fn: Function to be called during forward.
        forward_only_fn: Function to be called only during ``forward``. ``fn`` is used for the other call modes.
        forward_step_only_fn: Function to be called only during ``forward_step``. ``fn`` is used for the other call modes.
        forward_steps_only_fn: Function to be called only during ``forward_steps``. ``fn`` is used for the other call modes.
        forward_only_fn: Function to be called only during forward. ``fn`` is used for the other call modes.
        takes_time: If True, ``fn`` receives all steps, if False, it received one step and no time dimension. Defaults to False.

    Examples::

        x = torch.arange(90).reshape(1,3,30) * 1.0

        # Using named function
        def same_stats_different_values(x):
            return torch.randn_like(x) * x.std() + x.mean()

        same_stats_layer = co.Lambda(same_stats_different_values)
        same_stats_layer(x)

        # Using unnamed function
        mean_layer = co.Lambda(lambda x: torch.ones_like(x) * x.mean())
        mean_layer(x)

        # Using functor
        sigmoid = co.Lambda(torch.nn.Sigmoid())
        sigmoid(x)

    """

    _state_shape = 0
    _dynamic_state_inds = []

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
            [
                callable(forward_only_fn),
                callable(forward_step_only_fn),
                callable(forward_steps_only_fn),
            ]
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
        s = self.__class__.__name__ + "("
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
        return self._forward_step(input)[0]

    def _forward_step(self, input: Tensor, prev_state=None) -> Tensor:
        if self.forward_step_only_fn is not None:
            return self.forward_step_only_fn(input), prev_state

        if self.takes_time:
            input = input.unsqueeze(dim=2)
        output = self.fn(input)
        if self.takes_time:
            output = output.squeeze(dim=2)

        return output, prev_state


def _multiply(x: Tensor, factor: Union[float, int, Tensor]):
    return x * factor


def Multiply(factor: float) -> Lambda:
    r"""Applies an scaling transformation to the incoming data: :math:`y = ax`.

    Args:
        factor (float): Number to multiply with.
    """
    fn = partial(_multiply, factor=factor)
    return Lambda(fn, takes_time=True)


def _add(x: Tensor, constant: Union[float, int, Tensor]):
    return x + constant


def Add(constant: float) -> Lambda:
    r"""Applies an additive translation to the incoming data: :math:`y = x + a`.

    Args:
        constant (float): Number to add.
    """
    """Create Lambda with addition function"""
    fn = partial(_add, constant=constant)
    return Lambda(fn, takes_time=True)


def _unity(x: Tensor):
    return x


def Identity(*args, **kwargs) -> Lambda:
    """A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        m = co.Identity(54, unused_argument1=0.1, unused_argument2=False)
        input = torch.randn(128, 20)
        output = m(input)
        assert output.size() == torch.Size([128, 20])

    """
    return Lambda(_unity, takes_time=True)


def Constant(constant: float) -> Lambda:
    """Returns ``constant * torch.ones_like(input)``.

    Arguments:
        constant: Constant value to return.
    """
    return Lambda(lambda x: constant * torch.ones_like(x), takes_time=True)


def Zero() -> Lambda:
    """Returns ``torch.zeros_like(input)``."""
    return Lambda(torch.zeros_like, takes_time=True)


def One() -> Lambda:
    """Returns ``torch.ones_like(input)``."""
    return Lambda(torch.ones_like, takes_time=True)
