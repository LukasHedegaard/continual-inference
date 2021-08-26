from functools import partial
from typing import Callable, Union

from torch import Tensor, nn

from .module import CoModule


class Lambda(CoModule, nn.Module):
    """Module wrapper for stateless functions"""

    def __init__(self, fn: Callable[[Tensor], Tensor]):
        nn.Module.__init__(self)
        assert callable(fn), "The pased function should be callable."
        self.fn = fn

    def forward(self, input: Tensor) -> Tensor:
        return self.fn(input)

    def forward_step(self, input: Tensor, update_state=True) -> Tensor:
        x = input.unsqueeze(dim=2)
        x = self.fn(x)
        x = x.squeeze(dim=2)
        return x

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True) -> Tensor:
        return self.fn(input)

    @property
    def delay(self) -> int:
        return 0

    @staticmethod
    def build_from(fn: Callable[[Tensor], Tensor]) -> "Lambda":
        return Lambda(fn)


def _multiply(x: Tensor, factor: Union[float, int, Tensor]):
    return x * factor


def Multiply(factor) -> Lambda:
    """Create Lambda with multiplication function"""
    fn = partial(_multiply, factor=factor)
    return Lambda(fn)


def _add(x: Tensor, constant: Union[float, int, Tensor]):
    return x + constant


def Add(constant) -> Lambda:
    """Create Lambda with addition function"""
    fn = partial(_add, constant=constant)
    return Lambda(fn)


def _unity(x: Tensor):
    return x


def Unity() -> Lambda:
    """Create Lambda with addition function"""
    return Lambda(_unity)
