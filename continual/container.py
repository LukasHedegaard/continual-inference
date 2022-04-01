from collections import OrderedDict, abc
from enum import Enum
from functools import reduce, wraps
from typing import Callable, List, Optional, Sequence, TypeVar, Union, overload

import torch
from torch import Tensor, nn

from .delay import Delay
from .logging import getLogger
from .module import CallMode, CoModule, PaddingMode, TensorPlaceholder
from .utils import (
    function_repr,
    load_state_dict,
    num_from,
    state_dict,
    temporary_parameter,
)

logger = getLogger(__name__)

__all__ = [
    "Sequential",
    "BroadcastReduce",
    "Residual",
    "Broadcast",
    "Parallel",
    "Reduce",
]

T = TypeVar("T")


class Reduction(Enum):
    """Types of parallel tensor reduce operation.
    Supported tupes are:
    - SUM:    Element-wise summation
    - CONCAT: Channel-wise concatenation
    - MUL:    Hadamark product
    """

    SUM = "sum"
    CONCAT = "concat"
    MUL = "mul"


ReductionFunc = Callable[[Sequence[Tensor]], Tensor]
ReductionFuncOrEnum = Union[Reduction, ReductionFunc]


def apply_forward(module: CoModule, input: Tensor):
    if isinstance(module, nn.RNNBase):
        return module.forward(input)[0]
    return module(input)


def reduce_sum(inputs: Sequence[Tensor]) -> Tensor:
    assert len(inputs) >= 2
    return reduce(torch.Tensor.add, inputs[1:], inputs[0])


def reduce_concat(inputs: Sequence[Tensor]) -> Tensor:
    """Channel-wise concatenation of input

    Args:
        inputs (Sequence[Tensor]): Inputs with broadcastable shapes.

    Returns:
        Tensor: Inputs concatenated in the channel dimension
    """
    return torch.cat(inputs, dim=1)  # channel dim for inputs of shape (B, C, T, H, W)


def reduce_mul(inputs: Sequence[Tensor]) -> Tensor:
    """Hadamard product between inputs

    Args:
        inputs (Sequence[Tensor]): Inputs with broadcastable shapes.

    Returns:
        Tensor: Haramard product of inputs
    """
    assert len(inputs) >= 2
    return reduce(torch.Tensor.mul, inputs[1:], inputs[0])


def nonempty(fn: ReductionFunc) -> ReductionFunc:
    @wraps(fn)
    def wrapped(inputs: Sequence[Tensor]) -> Tensor:
        if any(len(inp) == 0 for inp in inputs):
            return TensorPlaceholder(inputs[0].shape)  # pragma: no cover
        return fn(inputs)

    return wrapped


class FlattenableStateDict:
    """Mixes in the ability to flatten state dicts.
    It is assumed that classes that inherit this modlue also inherit from nn.Module
    """

    flatten_state_dict = False

    def __init__(self, *args, **kwargs):
        ...  # pragma: no cover

    def state_dict(
        self, destination=None, prefix="", keep_vars=False, flatten=False
    ) -> "OrderedDict[str, Tensor]":
        flatten = flatten or self.flatten_state_dict
        return state_dict(self, destination, prefix, keep_vars, flatten)

    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, Tensor]",
        strict: bool = True,
        flatten=False,
    ):
        flatten = flatten or self.flatten_state_dict
        return load_state_dict(self, state_dict, strict, flatten)


def co_add_module(self, name: str, module: Optional["nn.Module"]) -> None:
    if not CoModule.is_valid(module):
        # Attempt automatic conversion
        from continual.convert import continual  # break cyclical import

        module = continual(module)

    nn.Module.add_module(self, name, module)


class Broadcast(CoModule, nn.Module):
    """Broadcast a single input to multiple streams"""

    def __init__(
        self,
        num_streams: int = None,
    ):
        nn.Module.__init__(self)
        self.num_streams = num_streams

    def forward(self, input: T) -> List[T]:
        assert isinstance(
            self.num_streams, int
        ), "Unknown number of target streams in Broadcast."
        return [input for _ in range(self.num_streams)]

    def forward_step(self, input: T, update_state=True) -> List[T]:
        return self.forward(input)

    def forward_steps(self, input: T, pad_end=False, update_state=True) -> List[T]:
        return self.forward(input)


class Parallel(FlattenableStateDict, CoModule, nn.Sequential):
    """Container for parallel modules.

    Args:
        *args: Either vargs of modules or an OrderedDict.
        auto_delay (bool, optional):
            Automatically add delay to modules in order to match the longest delay.
            Defaults to True.
    """

    @overload
    def __init__(
        self,
        *args: CoModule,
        auto_delay=True,
    ) -> None:
        ...  # pragma: no cover

    @overload
    def __init__(
        self,
        arg: "OrderedDict[str, CoModule]",
        auto_delay=True,
    ) -> None:
        ...  # pragma: no cover

    def __init__(
        self,
        *args,
        auto_delay=True,
    ):
        nn.Module.__init__(self)

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

        assert (
            len(set(num_from(getattr(m, "stride", 1)) for _, m in modules)) == 1
        ), f"Expected all modules to have the same stride, but got strides {[(num_from(getattr(m, 'stride', 1))) for _, m in modules]}"

        for key, module in modules:
            self.add_module(key, module)

        delays = set(m.delay for m in self)
        if len(delays) != 1:  # pragma: no cover
            logger.warning(
                f"It recommended that parallel modules have the same delay, but found delays {delays}. "
                "Temporal consistency cannot be guaranteed."
            )
        self._delay = max(delays)

        receptive_fields = set(m.receptive_field for m in self)
        self._receptive_field = max(receptive_fields)

    def add_module(self, name: str, module: Optional["nn.Module"]) -> None:
        co_add_module(self, name, module)

    def forward_step(self, inputs: List[T], update_state=True) -> List[T]:
        outs = []
        for i, m in enumerate(self):
            with temporary_parameter(m, "call_mode", CallMode.FORWARD_STEP):
                outs.append(m(inputs[i], update_state=update_state))
        return outs

    def forward_steps(
        self, inputs: List[T], pad_end=False, update_state=True
    ) -> List[T]:
        outs = []
        for i, m in enumerate(self):
            with temporary_parameter(m, "call_mode", CallMode.FORWARD_STEPS):
                outs.append(m(inputs[i], pad_end=pad_end, update_state=update_state))
        return outs

    def forward(self, inputs: List[T]) -> List[T]:
        outs = []
        for i, m in enumerate(self):
            with temporary_parameter(m, "call_mode", CallMode.FORWARD):
                outs.append(apply_forward(m, inputs[i]))
        return outs

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    @property
    def delay(self) -> int:
        return self._delay

    @property
    def stride(self) -> int:
        return num_from(getattr(next(iter(self)), "stride", 1))

    @property
    def padding(self) -> int:
        return max(num_from(getattr(m, "padding", 0)) for m in self)

    def clean_state(self):
        for m in self:
            if hasattr(m, "clean_state"):
                m.clean_state()


class ParallelDispatch(CoModule, nn.Module):
    """Parallel dispatch of many input streams to many output streams"""

    def __init__(
        self,
        dispatch_mapping: Sequence[Union[int, Sequence[int]]],
    ):
        """Initialise ParallelDispatch

        Args:
            dispatch_mapping (Sequence[Union[int, Sequence[int]]]):
                input-to-output mapping, where the integers signify the input stream ordering
                and the positions denote corresponding output ordering.
                Examples:
                    [1,0] to shufle order of streams.
                    [0,1,1] to copy stream 1 onto a new stream.
                    [[0,1],2] to collect stream 0 and 1 while keeping stream 2 separate.

        """
        nn.Module.__init__(self)

        def is_int_or_valid_list(x):
            if isinstance(x, int):
                return True
            elif isinstance(x, abc.Sequence):
                return all(is_int_or_valid_list(z) for z in x)
            else:
                return False

        assert isinstance(dispatch_mapping, abc.Sequence) and is_int_or_valid_list(
            dispatch_mapping
        ), "The dispatch_mapping should be of type Sequence[Union[StreamId, Sequence[StreamId]]]"

        self.dispatch_mapping = dispatch_mapping

    def forward(self, input: List[T]) -> List[Union[T, List[T]]]:
        def dispatch(mapping):
            nonlocal input
            if isinstance(mapping, abc.Sequence):
                return [dispatch(m) for m in mapping]
            else:
                return input[mapping]

        return dispatch(self.dispatch_mapping)

    def forward_step(
        self, input: List[T], update_state=True
    ) -> List[Union[T, List[T]]]:
        return self.forward(input)

    def forward_steps(
        self, input: List[T], pad_end=False, update_state=True
    ) -> List[Union[T, List[T]]]:
        return self.forward(input)


class Reduce(CoModule, nn.Module):
    """Reduce multiple input streams to a single output"""

    def __init__(
        self,
        reduce: ReductionFuncOrEnum = Reduction.SUM,
    ):
        nn.Module.__init__(self)
        self.reduce = nonempty(
            reduce
            if callable(reduce)
            else {
                Reduction.SUM: reduce_sum,
                Reduction.CONCAT: reduce_concat,
                Reduction.MUL: reduce_mul,
            }[Reduction(reduce)]
        )

    def forward(self, inputs: List[T]) -> T:
        return self.reduce(inputs)

    def forward_step(self, inputs: List[T], update_state=True) -> T:
        if all(isinstance(i, Tensor) for i in inputs):
            return self.reduce(inputs)
        return TensorPlaceholder()  # pragma: no cover

    def forward_steps(self, inputs: List[T], pad_end=False, update_state=True) -> T:
        return self.reduce(inputs)


class Sequential(FlattenableStateDict, CoModule, nn.Sequential):
    """Continual Sequential module

    This module is a drop-in replacement for `torch.nn.Sequential`
    which adds the `forward_step`, `forward_steps`, and `like` methods
    as well as a `delay` property
    """

    @overload
    def __init__(self, *args: nn.Module) -> None:
        ...  # pragma: no cover

    @overload
    def __init__(self, arg: "OrderedDict[str, nn.Module]") -> None:
        ...  # pragma: no cover

    def __init__(self, *args):
        nn.Module.__init__(self)
        modules = []
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                modules.append((key, module))
        else:
            for idx, module in enumerate(args):
                modules.append((str(idx), module))

        # If a co.Broadcast is followed by a co.Parallel, automatically infer num_streams
        for i in range(len(modules)):
            if isinstance(modules[i][1], Broadcast) and i < len(modules) - 1:
                if isinstance(modules[i + 1][1], Parallel):
                    modules[i][1].num_streams = modules[i][1].num_streams or len(
                        modules[i + 1][1]
                    )

        for n, m in modules:
            self.add_module(n, m)

    def add_module(self, name: str, module: Optional["nn.Module"]) -> None:
        co_add_module(self, name, module)

    def forward(self, input):
        for m in self:
            with temporary_parameter(m, "call_mode", CallMode.FORWARD):
                input = apply_forward(m, input)
        return input

    def forward_step(self, input, update_state=True):
        for module in self:
            # ptflops only works when __call__ is triggered
            with temporary_parameter(module, "call_mode", CallMode.FORWARD_STEP):
                input = module(  # == module.forward_step
                    input, update_state=update_state
                )
            if not type(input) in {Tensor, list}:
                return TensorPlaceholder()
        return input

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True):
        for m in self:
            if not type(input) in {Tensor, list} or len(input) == 0:
                return TensorPlaceholder()  # pragma: no cover
            # ptflops only works when __call__ is triggered
            with temporary_parameter(m, "call_mode", CallMode.FORWARD_STEPS):
                # == m.forward_steps
                input = m(input, pad_end=pad_end, update_state=update_state)

        return input

    @property
    def receptive_field(self) -> int:
        reverse_modules = [m for m in self][::-1]
        rf = reverse_modules[0].receptive_field
        for m in reverse_modules[1:]:
            s = num_from(getattr(m, "stride", 1))
            rf = s * rf + m.receptive_field - s
        return rf

    @property
    def stride(self) -> int:
        tot = 1
        for m in self:
            tot *= num_from(m.stride)
        return tot

    @property
    def padding(self) -> int:
        m = [m for m in self]
        p = num_from(m[0].padding)
        s = num_from(m[0].stride)
        for i in range(1, len(m)):
            p += num_from(m[i].padding) * s
            s = s * num_from(m[i].stride)
        return p

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


class BroadcastReduce(FlattenableStateDict, CoModule, nn.Sequential):
    """Broadcast an input to parallel modules and reduce.
    This module is a shorthand for

    >>> co.Sequential(co.Broadcast(), co.Parallel(*args), co.Reduce(reduce))

    Args:
        *args: Either vargs of modules or an OrderedDict.
        reduce (ReductionFuncOrEnum, optional):
            Function used to reduce the parallel outputs.
            Sum or concatenation can be specified by passing Reduction.SUM or Reduction.CONCAT respectively.
            Custom reduce functions can also be passed.
            Defaults to Reduction.SUM.
        auto_delay (bool, optional):
            Automatically add delay to modules in order to match the longest delay.
            Defaults to True.
    """

    @overload
    def __init__(
        self,
        *args: CoModule,
        reduce: ReductionFuncOrEnum = Reduction.SUM,
        auto_delay=True,
    ) -> None:
        ...  # pragma: no cover

    @overload
    def __init__(
        self,
        arg: "OrderedDict[str, CoModule]",
        reduce: ReductionFuncOrEnum = Reduction.SUM,
        auto_delay=True,
    ) -> None:
        ...  # pragma: no cover

    def __init__(
        self,
        *args,
        reduce: ReductionFuncOrEnum = Reduction.SUM,
        auto_delay=True,
    ):
        nn.Module.__init__(self)

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            modules = [(key, module) for key, module in args[0].items()]
        else:
            modules = [(str(idx), module) for idx, module in enumerate(args)]

        assert (
            len(modules) > 1
        ), "You should pass at least two modules for the map-reduce operation to make sense."

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

        assert (
            len(set(num_from(getattr(m, "stride", 1)) for _, m in modules)) == 1
        ), f"Expected all modules to have the same stride, but got strides {[(num_from(getattr(m, 'stride', 1))) for _, m in modules]}"

        for key, module in modules:
            self.add_module(key, module)

        self.reduce = nonempty(
            reduce
            if callable(reduce)
            else {
                Reduction.SUM: reduce_sum,
                Reduction.CONCAT: reduce_concat,
                Reduction.MUL: reduce_mul,
            }[Reduction(reduce)]
        )

        delays = set(m.delay for m in self)
        self._delay = max(delays)

        receptive_fields = set(m.receptive_field for m in self)
        self._receptive_field = max(receptive_fields)

    def add_module(self, name: str, module: Optional["nn.Module"]) -> None:
        co_add_module(self, name, module)

    def forward_step(self, input: Tensor, update_state=True) -> Tensor:
        outs = []
        for m in self:
            with temporary_parameter(m, "call_mode", CallMode.FORWARD_STEP):
                outs.append(m(input, update_state=update_state))  # == m.forward_step
        if all(isinstance(o, Tensor) for o in outs):
            return self.reduce(outs)

        # Try to infer shape
        shape = tuple()
        for o in outs:
            if isinstance(o, Tensor):  # pragma: no cover
                shape = o.shape
                break
        return TensorPlaceholder(shape)

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True) -> Tensor:
        outs = []
        for m in self:
            with temporary_parameter(m, "call_mode", CallMode.FORWARD_STEPS):
                # m.forward_steps
                outs.append(m(input, pad_end=pad_end, update_state=update_state))

        return self.reduce(outs)

    def forward(self, input: Tensor) -> Tensor:
        outs = []
        for m in self:
            with temporary_parameter(m, "call_mode", CallMode.FORWARD):
                outs.append(apply_forward(m, input))

        return self.reduce(outs)

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    @property
    def delay(self) -> int:
        return self._delay

    @property
    def stride(self) -> int:
        return num_from(getattr(next(iter(self)), "stride", 1))

    @property
    def padding(self) -> int:
        return max(num_from(getattr(m, "padding", 0)) for m in self)

    def clean_state(self):
        for m in self:
            if hasattr(m, "clean_state"):
                m.clean_state()

    def extra_repr(self):
        return f"reduce={self.reduce.__name__}"


def Residual(
    module: CoModule,
    temporal_fill: PaddingMode = None,
    reduce: Reduction = "sum",
    residual_shrink: Union[bool, str] = False,
) -> BroadcastReduce:
    """[summary]

    Args:
        module (CoModule): module to which a residual should be added.
        temporal_fill (PaddingMode, optional): temporal fill type in delay. Defaults to None.
        reduce (Reduction, optional): Reduction function. Defaults to "sum".
        residual_shrink (bool, optional):
            Set residual to shrink its forward to match the temporal dimension reduction of the wrapped module.
            Options: "centered" or True: Centered residual shrink; "lagging": lagging shrink. Defaults to False.

    Returns:
        BroadcastReduce: BroadcastReduce module with residual.
    """
    assert num_from(getattr(module, "stride", 1)) == 1, (
        "The simple `Residual` only works for modules with temporal stride=1. "
        "Complex residuals can be achieved using `BroadcastReduce` or the `Broadcast`, `Parallel`, and `Reduce` modules."
    )
    temporal_fill = temporal_fill or getattr(
        module, "temporal_fill", PaddingMode.REPLICATE.value
    )
    delay = module.delay
    equal_padding = module.receptive_field - num_from(module.padding) * 2 == 1
    if equal_padding:
        residual_shrink = False

    if residual_shrink in {True, "centered"}:
        assert delay % 2 == 0, "Auto-shrink only works for even-number delays."
        delay = delay // 2

    return BroadcastReduce(
        # Residual first yields easier broadcasting in reduce functions
        Delay(delay, temporal_fill, auto_shrink=residual_shrink),
        module,
        reduce=reduce,
        auto_delay=False,
    )


class Conditional(FlattenableStateDict, CoModule, nn.Module):
    """Module wrapper for conditional invocations at runtime"""

    def __init__(
        self,
        predicate: Callable[[CoModule, Tensor], bool],
        on_true: CoModule,
        on_false: CoModule = None,
    ):
        from continual.convert import continual  # Break cyclical import

        assert callable(predicate), "The pased function should be callable."
        if not isinstance(on_true, CoModule):
            on_true = continual(on_true)
        if not (isinstance(on_false, CoModule) or on_false is None):
            on_false = continual(on_false)

        nn.Module.__init__(self)

        self.predicate = predicate

        # Ensure modules have the same delay
        self._delay = max(on_true.delay, getattr(on_false, "delay", 0))
        self._receptive_field = max(
            on_true.receptive_field, getattr(on_false, "receptive_field", 1)
        )

        self.add_module(
            "0",
            on_true
            if on_true.delay == self._delay
            else Sequential(Delay(self._delay - on_true.delay), on_true),
        )

        if on_false is not None:
            self.add_module(
                "1",
                on_false
                if on_false.delay == self._delay
                else Sequential(Delay(self._delay - on_false.delay), on_false),
            )

    def forward(self, input: Tensor) -> Tensor:
        if self.predicate(self, input):
            return apply_forward(self._modules["0"], input)
        elif "1" in self._modules:
            return apply_forward(self._modules["1"], input)
        return input

    def forward_step(self, input: Tensor, update_state=True) -> Tensor:
        if self.predicate(self, input):
            return self._modules["0"].forward_step(input)
        elif "1" in self._modules:
            return self._modules["1"].forward_step(input)
        return input

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True) -> Tensor:
        if self.predicate(self, input):
            return self._modules["0"].forward_steps(input)
        elif "1" in self._modules:
            return self._modules["1"].forward_steps(input)
        return input

    @property
    def delay(self) -> int:
        return self._delay

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    def extra_repr(self):
        return f"predicate={function_repr(self.predicate)}"
