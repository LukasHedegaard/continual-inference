from functools import partial
from inspect import getsource
from typing import Callable, Union

from torch import Tensor, nn

from .module import CoModule


class Lambda(CoModule, nn.Module):
    """Module wrapper for stateless functions

    NB: Operations performed in a Lambda are not counted in `ptflops`
    """

    def __init__(self, fn: Callable[[Tensor], Tensor], unsqueeze_step=True):
        nn.Module.__init__(self)
        assert callable(fn), "The pased function should be callable."
        self.fn = fn
        self.unsqueeze_step = unsqueeze_step

    def __repr__(self) -> str:
        s = self.fn.__name__
        if s == "<lambda>":
            s = getsource(self.fn)
            s = s[s.find("lambda") :]

            take_right_parens = 0
            stack = 0
            for c in s:
                if c == "(":
                    stack += 1
                    take_right_parens += 1
                elif c == ")":
                    stack -= 1
                if stack == -1:
                    break

            if take_right_parens > 0:
                i = -1
                for _ in range(take_right_parens + 1):
                    i = s.find(")", i + 1)
                s = s[:i]

            s = s.rstrip()
        return f"Lambda({s})"

    def forward(self, input: Tensor) -> Tensor:
        return self.fn(input)

    def forward_step(self, input: Tensor, update_state=True) -> Tensor:
        if self.unsqueeze_step:
            input = input.unsqueeze(dim=2)
        output = self.fn(input)
        if self.unsqueeze_step:
            output = output.squeeze(dim=2)
        return output

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
