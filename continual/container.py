from collections import OrderedDict
from enum import Enum
from functools import reduce
from typing import Callable, Optional, Sequence, Union, overload

import torch
from torch import Tensor, nn

from .delay import Delay
from .interface import CoModule, FillMode

__all__ = ["Sequential", "Parallel", "Residual"]


class Sequential(nn.Sequential, CoModule):
    """Continual Sequential module

    This module is a drop-in replacement for `torch.nn.Sequential`
    which adds the `forward_step`, `forward_steps`, and `like` methods
    as well as a `delay` property
    """

    def add_module(self, name: str, module: Optional["nn.Module"]) -> None:
        if not CoModule.is_valid(module):
            # Attempt automatic conversion
            from continual.convert import continual  # break cyclical import

            module = continual(module)

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
        return sum(getattr(m, "delay", 0) for m in self)

    @staticmethod
    def build_from(module: nn.Sequential) -> "Sequential":
        from .convert import continual  # import here due to circular import

        return Sequential(
            OrderedDict([(k, continual(m)) for k, m in module._modules.items()])
        )

    def clean_state(self):
        for m in self:
            if hasattr(m, "clean_state"):
                m.clean_state()


class Aggregation(Enum):
    SUM = "sum"
    CONCAT = "concat"


def parallel_sum(modules: Sequence[Tensor]) -> Tensor:
    if len(modules) == 0:  # pragma: no cover
        return torch.tensor([])
    return reduce(torch.Tensor.add_, modules, torch.zeros_like(modules[0]))


def parallel_concat(modules: Sequence[Tensor]) -> Tensor:
    return torch.cat(modules, dim=1)  # channel dim for inputs of shape (B, C, T, H, W)


AggregationFunc = Union[Aggregation, Callable[[Sequence[Tensor]], Tensor]]


class Parallel(nn.Sequential, CoModule):
    """Continual parallel container.

    Args:
        *args: Either vargs of modules or an OrderedDict.
        aggregation_fn (AggregationFunc, optional):
            Function used to aggregate the parallel outputs.
            Sum or concatenation can be specified by passing Aggregation.SUM or Aggregation.CONCAT respectively.
            Custom aggregation functions can also be passed.
            Defaults to Aggregation.SUM.
        auto_delay (bool, optional):
            Automatically add delay to modules in order to match the longest delay.
            Defaults to True.

    """

    @overload
    def __init__(
        self,
        *args: CoModule,
        aggregation_fn: AggregationFunc = Aggregation.SUM,
        auto_delay=True,
    ) -> None:
        ...  # pragma: no cover

    @overload
    def __init__(
        self,
        arg: "OrderedDict[str, CoModule]",
        aggregation_fn: AggregationFunc = Aggregation.SUM,
        auto_delay=True,
    ) -> None:
        ...  # pragma: no cover

    def __init__(
        self,
        *args,
        aggregation_fn: AggregationFunc = Aggregation.SUM,
        auto_delay=True,
    ):
        super(Parallel, self).__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            modules = [(key, module) for key, module in args[0].items()]
        else:
            modules = [(str(idx), module) for idx, module in enumerate(args)]

        if auto_delay:
            # If there is a delay mismatch, automatically add delay to match the longest
            max_delay = max([m.delay for _, m in modules])
            modules = [
                (
                    key,
                    (
                        Sequential(module, Delay(max_delay - module.delay))
                        if module.delay < max_delay
                        else module
                    ),
                )
                for key, module in modules
            ]

        for key, module in modules:
            self.add_module(key, module)

        self.aggregation_fn = (
            aggregation_fn
            if callable(aggregation_fn)
            else {
                Aggregation.SUM: parallel_sum,
                Aggregation.CONCAT: parallel_concat,
            }[Aggregation(aggregation_fn)]
        )

        delays = set(m.delay for m in self)
        assert (
            len(delays) == 1
        ), f"Parallel modules should have the same delay, but found delays {delays}."
        self._delay = delays.pop()

    def forward_step(self, input: Tensor) -> Tensor:
        return self.aggregation_fn([m.forward_step(input) for m in self])

    def forward_steps(self, input: Tensor) -> Tensor:
        outs = [self.forward_step(input[:, :, t]) for t in range(input.shape[2])]

        if len(outs) > 0:
            outs = torch.stack(outs, dim=2)
        else:
            outs = torch.tensor([])  # pragma: no cover

        return outs

    def forward(self, input: Tensor) -> Tensor:
        outs = [m.forward(input) for m in self]

        ts = [x.shape[2] for x in outs]
        min_t = min(ts)
        if min_t == max(ts):
            return self.aggregation_fn(outs)

        # Modules may shrink the output map differently.
        # If an "even" shrink is detected, attempt to automatically
        # shrink the longest values to fit as if padding was used.
        assert all(
            [(t - min_t) % 2 == 0 for t in ts]
        ), f"Found incompatible temporal output-shapes {ts}"
        shrink = [(t - min_t) // 2 for t in ts]

        return self.aggregation_fn(
            [o[:, :, slice(s, -s or None)] for (s, o) in zip(shrink, outs)]
        )

    @property
    def delay(self) -> int:
        return self._delay

    def clean_state(self):
        for m in self:
            if hasattr(m, "clean_state"):
                m.clean_state()


def Residual(module: CoModule, temporal_fill: FillMode = None):
    return Parallel(
        OrderedDict(
            [
                ("module", module),
                (
                    "residual",
                    Delay(
                        delay=module.delay,
                        temporal_fill=temporal_fill
                        or getattr(module, "temporal_fill", FillMode.REPLICATE.value),
                    ),
                ),
            ]
        ),
        aggregation_fn=Aggregation.SUM,
        auto_delay=False,
    )
