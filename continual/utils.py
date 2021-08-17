from contextlib import contextmanager
from functools import reduce, wraps
from typing import Callable

from torch import Tensor, nn

from .interface import CoModule


class Zero(nn.Module, CoModule):
    def forward_step(self, input: Tensor) -> Tensor:
        return 0

    def forward_steps(self, input: Tensor) -> Tensor:
        return 0

    def forward(self, input: Tensor) -> Tensor:
        return 0

    @property
    def delay(self) -> int:
        return 0

    def clean_state(self):
        ...


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
