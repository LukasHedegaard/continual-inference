from contextlib import contextmanager
from enum import Enum
from functools import reduce, wraps
from typing import Callable, Tuple

from torch import Tensor
from torch.nn import Module

from .interface import _CoModule


class TensorPlaceholder:
    shape: Tuple[int]

    def __init__(self, shape: Tuple[int]):
        self.shape = shape

    def size(self):
        return self.shape


class FillMode(Enum):
    REPLICATE = "replicate"
    ZEROS = "zeros"


class Zero(Module, _CoModule):
    def forward(self, input: Tensor) -> Tensor:
        return 0

    def forward_regular(self, input: Tensor) -> Tensor:
        return 0

    def forward_regular_unrolled(self, input: Tensor) -> Tensor:
        return 0


def Continual(instance: Module, dim: int = 2):
    def decorator(func: Callable[[Tensor], Tensor]):
        @wraps(func)
        def call(x: Tensor) -> Tensor:
            x = x.unsqueeze(dim)
            x = func(x)
            x = x.squeeze(dim)
            return x

        return call

    instance.forward_regular_unrolled = instance.forward
    instance.forward_regular = instance.forward
    instance.forward = decorator(instance.forward)

    return instance


@contextmanager
def temporary_parameter(obj, attr, val):
    prev_val = rgetattr(obj, attr)
    rsetattr(obj, attr, val)
    yield obj
    rsetattr(obj, attr, prev_val)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split("."))
