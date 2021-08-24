from collections import OrderedDict
from enum import Enum
from functools import reduce, wraps
from typing import Callable, Optional, Sequence, Tuple, Union, overload

import torch
from torch import Tensor, nn

from .delay import Delay
from .interface import CoModule, Padded, PaddingMode, TensorPlaceholder
from .utils import load_state_dict, state_dict

__all__ = ["Sequential", "Parallel", "Residual"]


def int_from(tuple_or_int: Union[int, Tuple[int, ...]], dim=0) -> int:
    if isinstance(tuple_or_int, int):
        return tuple_or_int
    else:
        return tuple_or_int[dim]


class FlattenableStateDict:
    """Mixes in the ability to flatten state dicts.
    It is assumed that classes that inherit this modlue also inherit from nn.Module
    """

    def state_dict(
        self, destination=None, prefix="", keep_vars=False, flatten=False
    ) -> "OrderedDict[str, Tensor]":
        return state_dict(self, destination, prefix, keep_vars, flatten)

    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, Tensor]",
        strict: bool = True,
        flatten=False,
    ):
        return load_state_dict(self, state_dict, strict, flatten)


class Sequential(FlattenableStateDict, nn.Sequential, Padded, CoModule):
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

    def forward(self, input):
        for module in self:
            input = module.forward(input)
        return input

    def forward_step(self, input, update_state=True):
        for module in self:
            input = module.forward_step(input, update_state=update_state)
            if not isinstance(input, Tensor):
                return TensorPlaceholder()  # We can't infer output shape
        return input

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True):
        for module in self:
            if isinstance(module, Padded):
                input = module.forward_steps(
                    input, pad_end=pad_end, update_state=update_state
                )
            else:
                input = module.forward_steps(input, update_state=update_state)

        return input

    @property
    def delay(self):
        return sum(getattr(m, "delay", 0) for m in self)

    @property
    def stride(self) -> int:
        tot = 1
        for m in self:
            tot *= int_from(getattr(m, "stride", 1))
        return tot

    @property
    def padding(self) -> int:
        return max(int_from(getattr(m, "padding", 0)) for m in self)

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
    """Types of parallel tensor aggregation.
    Supported tupes are:
    - SUM:    Element-wise summation
    - CONCAT: Channel-wise concatenation
    - MUL:    Hadamark product
    """

    SUM = "sum"
    CONCAT = "concat"
    MUL = "mul"


AggregationFunc = Callable[[Sequence[Tensor]], Tensor]
AggregationFuncOrEnum = Union[Aggregation, AggregationFunc]


def parallel_sum(inputs: Sequence[Tensor]) -> Tensor:
    assert len(inputs) >= 2
    return reduce(torch.Tensor.add, inputs[1:], inputs[0])


def parallel_concat(inputs: Sequence[Tensor]) -> Tensor:
    """Channel-wise concatenation of input

    Args:
        inputs (Sequence[Tensor]): Inputs with broadcastable shapes.

    Returns:
        Tensor: Inputs concatenated in the channel dimension
    """
    return torch.cat(inputs, dim=1)  # channel dim for inputs of shape (B, C, T, H, W)


def parallel_mul(inputs: Sequence[Tensor]) -> Tensor:
    """Hadamard product between inputs

    Args:
        inputs (Sequence[Tensor]): Inputs with broadcastable shapes.

    Returns:
        Tensor: Haramard product of inputs
    """
    assert len(inputs) >= 2
    return reduce(torch.Tensor.mul, inputs[1:], inputs[0])


def nonempty(fn: AggregationFunc) -> AggregationFunc:
    @wraps(fn)
    def wrapped(inputs: Sequence[Tensor]) -> Tensor:
        if any(len(inp) == 0 for inp in inputs):
            return TensorPlaceholder(inputs[0].shape)
        return fn(inputs)

    return wrapped


class Parallel(FlattenableStateDict, nn.Sequential, Padded, CoModule):
    """Continual parallel container.

    Args:
        *args: Either vargs of modules or an OrderedDict.
        aggregation_fn (AggregationFuncOrEnum, optional):
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
        aggregation_fn: AggregationFuncOrEnum = Aggregation.SUM,
        auto_delay=True,
    ) -> None:
        ...  # pragma: no cover

    @overload
    def __init__(
        self,
        arg: "OrderedDict[str, CoModule]",
        aggregation_fn: AggregationFuncOrEnum = Aggregation.SUM,
        auto_delay=True,
    ) -> None:
        ...  # pragma: no cover

    def __init__(
        self,
        *args,
        aggregation_fn: AggregationFuncOrEnum = Aggregation.SUM,
        auto_delay=True,
    ):
        super(Parallel, self).__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            modules = [(key, module) for key, module in args[0].items()]
        else:
            modules = [(str(idx), module) for idx, module in enumerate(args)]

        assert (
            len(modules) > 1
        ), "You should pass at least two modules for the parallel operation to make sense."

        if auto_delay:
            # If there is a delay mismatch, automatically add delay to match the longest
            max_delay = max([m.delay for _, m in modules])
            modules = [
                (
                    key,
                    (
                        Sequential(Delay(max_delay - module.delay), module)
                        if module.delay < max_delay
                        else module
                    ),
                )
                for key, module in modules
            ]

        assert (
            len(set(int_from(getattr(m, "stride", 1)) for _, m in modules)) == 1
        ), f"Expected all parallel modules to have the same stride, but got strides {[(int_from(getattr(m, 'stride', 1))) for _, m in modules]}"

        for key, module in modules:
            self.add_module(key, module)

        self.aggregation_fn = nonempty(
            aggregation_fn
            if callable(aggregation_fn)
            else {
                Aggregation.SUM: parallel_sum,
                Aggregation.CONCAT: parallel_concat,
                Aggregation.MUL: parallel_mul,
            }[Aggregation(aggregation_fn)]
        )

        delays = set(m.delay for m in self)
        assert (
            len(delays) == 1
        ), f"Parallel modules should have the same delay, but found delays {delays}."
        self._delay = delays.pop()

    def forward_step(self, input: Tensor, update_state=True) -> Tensor:
        outs = [m.forward_step(input, update_state=update_state) for m in self]
        if all(isinstance(o, Tensor) for o in outs):
            return self.aggregation_fn(outs)
        else:
            # Try to infer shape
            shape = tuple()
            for o in outs:
                if isinstance(o, Tensor):
                    shape = o.shape
                    break
            return TensorPlaceholder(shape)

    # NB: There seems to be a bug hidden here
    def forward_steps(self, input: Tensor, pad_end=False, update_state=True) -> Tensor:
        outs = []
        for m in self:
            if isinstance(m, Padded):
                outs.append(
                    m.forward_steps(input, pad_end=pad_end, update_state=update_state)
                )
            else:
                outs.append(m.forward_steps(input, update_state=update_state))

        return self.aggregation_fn(outs)

    def forward(self, input: Tensor) -> Tensor:
        outs = [m.forward(input) for m in self]

        return self.aggregation_fn(outs)

        # Modules may shrink the output map differently.
        # If an "even" shrink is detected, attempt to automatically
        # shrink the longest values to fit as if padding was used.
        # assert all(
        #     [(t - min_t) % 2 == 0 for t in ts]
        # ), f"Found incompatible temporal output-shapes {ts}"
        # shrink = [(t - min_t) // 2 for t in ts]

        # return self.aggregation_fn(
        #     [o[:, :, slice(s, -s or None)] for (s, o) in zip(shrink, outs)]
        # )

    @property
    def delay(self) -> int:
        return self._delay

    @property
    def stride(self) -> int:
        return getattr(next(iter(self)), "stride", 1)

    @property
    def padding(self) -> int:
        return max(int_from(getattr(m, "padding", 0)) for m in self)

    def clean_state(self):
        for m in self:
            if hasattr(m, "clean_state"):
                m.clean_state()

    def extra_repr(self):
        return f"aggregation_fn={self.aggregation_fn.__name__}"


def Residual(
    module: CoModule,
    temporal_fill: PaddingMode = None,
    aggregation_fn: Aggregation = "sum",
):
    return Parallel(
        # Residual first yields easier broadcasting in aggregation functions
        Delay(
            delay=module.delay,
            temporal_fill=temporal_fill
            or getattr(module, "temporal_fill", PaddingMode.REPLICATE.value),
        ),
        module,
        aggregation_fn=aggregation_fn,
        auto_delay=False,
    )
