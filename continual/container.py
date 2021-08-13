from collections import OrderedDict
from typing import Optional

from torch import nn

from .interface import CoModule

__all__ = ["Sequential"]


class Sequential(nn.Sequential, CoModule):
    """Continual Sequential module

    This module is a drop-in replacement for `torch.nn.Sequential`
    which adds the `forward_step`, `forward_steps`, and `like` methods
    as well as a `delay` property
    """

    def add_module(self, name: str, module: Optional["nn.Module"]) -> None:
        CoModule._validate_class(module)
        nn.Module.add_module(self, name, module)

    def forward_step(self, input):
        for module in self:
            input = module.forward_step(input)
        return input

    def forward_steps(self, input):
        for module in self:
            input = module.forward_steps(input)
        return input

    @property
    def delay(self):
        return sum(m.delay for m in self)

    @staticmethod
    def build_from(module: nn.Sequential) -> "Sequential":
        from .convert import continual  # import here due to circular import

        return Sequential(
            OrderedDict([(k, continual(m)) for k, m in module._modules.items()])
        )
