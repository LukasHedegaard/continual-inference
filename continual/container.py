from collections import OrderedDict, abc
from enum import Enum
from functools import reduce, wraps
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, Union, overload

import torch
from torch import Tensor, nn

from .delay import Delay
from .logging import getLogger
from .module import CoModule, PaddingMode, _callmode
from .skip import Skip
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
    "ParallelDispatch",
    "Reduce",
    "Conditional",
]

T = TypeVar("T")
S = TypeVar("S")

State = List[Optional[Tensor]]


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
    MAX = "max"


ReductionFunc = Callable[[Sequence[Tensor]], Tensor]
ReductionFuncOrEnum = Union[Reduction, ReductionFunc, str]


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


def reduce_max(inputs: Sequence[Tensor]) -> Tensor:
    assert len(inputs) >= 2
    return reduce(torch.max, inputs[1:], inputs[0])


def nonempty(fn: ReductionFunc) -> ReductionFunc:
    @wraps(fn)
    def wrapped(inputs: Sequence[Tensor]) -> Tensor:
        if any(inp is None or inp.shape[0] == 0 for inp in inputs):
            return None  # pragma: no cover
        return fn(inputs)

    return wrapped


class FlattenableStateDict:
    """Mixes in the ability to flatten state dicts.
    It is assumed that classes that inherit this module also inherit from nn.Module
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
    """Broadcast one input stream to multiple output streams.

    This is needed for handling parallel streams in subsequent modules.
    For instance, here is how it is used to create a residual connection::

        residual = co.Sequential(
            co.Broadcast(2),
            co.Parallel(
                co.Conv3d(32, 32, kernel_size=3, padding=1),
                co.Delay(2),
            ),
            co.Reduce("sum"),
        )

    Since the ``Broadcast`` -> ``Parallel`` -> ``Reduce`` sequence is so common,
    identical behavior can be achieved with ``BroadcastReduce`` ::

        residual = co.BroadcastReduce(
            co.Conv3d(32, 32, kernel_size=3, padding=1),
            co.Delay(2),
            reduce="sum"
        )

    Even shorter, the library features a residual connection, which automatically handles delays::

        residual = co.Residual(co.Conv3d(32, 32, kernel_size=3, padding=1))

    Args:
        num_streams (int):
            Number of streams to broadcast to. If none are given, a Sequential
            module may infer it automatically.
    """

    _state_shape = 0
    _dynamic_state_inds = []

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

    def _forward_step(self, input: T, prev_state=None):
        return self.forward(input), prev_state

    def forward_step(self, input: T, update_state=True) -> List[T]:
        return self.forward(input)

    def forward_steps(self, input: T, pad_end=False, update_state=True) -> List[T]:
        return self.forward(input)


class Parallel(FlattenableStateDict, CoModule, nn.Sequential):
    """Container for  executing modules in parallel.
    Modules will be added to it in the order they are passed in the
    constructor.

    For instance, here is how it is used to create a residual connection::

        residual = co.Sequential(
            co.Broadcast(2),
            co.Parallel(
                co.Conv3d(32, 32, kernel_size=3, padding=1),
                co.Delay(2),
            ),
            co.Reduce("sum"),
        )

    Since the ``Broadcast`` -> ``Parallel`` -> ``Reduce`` sequence is so common,
    identical behavior can be achieved with ``BroadcastReduce`` ::

        residual = co.BroadcastReduce(
            co.Conv3d(32, 32, kernel_size=3, padding=1),
            co.Delay(2),
            reduce="sum"
        )

    Even shorter, the library features a residual connection, which automatically handles delays::

        residual = co.Residual(co.Conv3d(32, 32, kernel_size=3, padding=1))


    Args:
        arg (OrderedDict[str, CoModule]): An OrderedDict of strings and modules.
        *args (CoModule): Comma-separated modules.
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

    @property
    def _state_shape(self):
        return [m._state_shape for m in self]

    @property
    def _dynamic_state_inds(self):
        return [m._dynamic_state_inds for m in self]

    def _forward_step(
        self, inputs: List[T], prev_state: Optional[List[Optional[S]]] = None
    ):
        prev_state = prev_state or [None for _ in range(len(self))]
        outs, next_state = [], []
        for i, module in enumerate(self):
            out, n_state = module._forward_step(inputs[i], prev_state=prev_state[i])
            outs.append(out)
            next_state.append(n_state)
        return outs, next_state

    def forward_step(self, inputs: List[T], update_state=True) -> List[T]:
        outs = []
        for i, m in enumerate(self):
            with temporary_parameter(m, "call_mode", _callmode("forward_step")):
                outs.append(m(inputs[i], update_state=update_state))
        return outs

    def forward_steps(
        self, inputs: List[T], pad_end=False, update_state=True
    ) -> List[T]:
        outs = []
        for i, m in enumerate(self):
            with temporary_parameter(m, "call_mode", _callmode("forward_steps")):
                outs.append(m(inputs[i], pad_end=pad_end, update_state=update_state))
        return outs

    def forward(self, inputs: List[T]) -> List[T]:
        outs = []
        for i, m in enumerate(self):
            with temporary_parameter(m, "call_mode", _callmode("forward")):
                outs.append(apply_forward(m, inputs[i]))
        return outs

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    @property
    def delay(self) -> int:
        return self._delay

    @property
    def stride(self) -> Tuple[int]:
        return getattr(next(iter(self)), "stride", (1,))

    @property
    def padding(self) -> Tuple[int]:
        return (max(getattr(m, "padding", (0,))[0] for m in self),)

    def clean_state(self):
        for m in self:
            if hasattr(m, "clean_state"):
                m.clean_state()


class ParallelDispatch(CoModule, nn.Module):
    """Reorder, copy, and group streams from parallel streams.

    Reorder example::

        net = co.Sequential(
            co.Broadcast(2),
            co.Parallel(co.Add(1), co.Identity()),
            co.ParallelDispatch([1,0]),  # Reorder stream 0 and 1
            co.Parallel(co.Identity(), co.Add(2)),
            co.Reduce("max"),
        )

        assert torch.equal(net(torch.tensor([0])), torch.tensor([3]))

    Depiction of the reorder example::

               | -> co.Add(1)     \\ / -> co.Identity() |
        [0] -> |                   X                   | -> max -> [3]
               | -> co.Identity() /  \\ -> co.Add(2)    |

    Copy example::

        net = co.Sequential(
            co.Broadcast(2),
            co.Parallel(co.Add(1), co.Identity()),
            co.ParallelDispatch([0, 0, 1]),  # Copy stream 0
            co.Parallel(co.Identity(), co.Add(2), co.Identity()),
            co.Reduce("max"),
        )

        assert torch.equal(net(torch.tensor([0])), torch.tensor([3]))

    Depiction of the copy example::

               | -> co.Add(1)  -> | -> co.Identity() -> |
        [0] -> |                  | -> co.Add(2)     -> | -> max -> [3]
               | -> co.Identity() ------> co.Add(1)  -> |

    Group example::

        net = co.Sequential(
            co.Broadcast(2),
            co.Parallel(co.Add(2), co.Identity()),
            co.ParallelDispatch([[0, 0], 1]),  # Copy and group stream 0
            co.Parallel(co.Reduce("sum"), co.Identity()),
            co.Reduce("max"),
        )

        assert torch.equal(net(torch.tensor([0])), torch.tensor([4]))

    Depiction of the group example::

                                 | -> |
               | -> co.Add(2) -> |    | ->    sum     -> |
        [0] -> |                 | -> |                  | -> max -> [4]
               | -> co.Identity() ----> co.Identity() -> |

    Args:
        dispatch_mapping (Sequence[Union[int, Sequence[int]]]):
            input-to-output mapping, where the integers signify the input stream ordering
            and the positions denote corresponding output ordering.
            Examples::
                [1,0] to shuffle order of streams.
                [0,1,1] to copy stream 1 onto a new stream.
                [[0,1],2] to group stream 0 and 1 while keeping stream 2 separate.

    """

    _state_shape = 0
    _dynamic_state_inds = []

    def __init__(
        self,
        dispatch_mapping: Sequence[Union[int, Sequence[int]]],
    ):
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

    def _forward_step(self, input: List[T], prev_state=None):
        return self.forward_step(input), prev_state

    def forward_step(
        self, input: List[T], update_state=True
    ) -> List[Union[T, List[T]]]:
        return self.forward(input)

    def forward_steps(
        self, input: List[T], pad_end=False, update_state=True
    ) -> List[Union[T, List[T]]]:
        return self.forward(input)


class Reduce(CoModule, nn.Module):
    """Reduce multiple input streams to a single using the selected function

    For instance, here is how it is used to sum streams in a residual connection::

        residual = co.Sequential(
            co.Broadcast(2),
            co.Parallel(
                co.Conv3d(32, 32, kernel_size=3, padding=1),
                co.Delay(2),
            ),
            co.Reduce("sum"),
        )

    A user-defined can be passed as well::

        from functools import reduce

        def my_sum(inputs):
            return reduce(torch.Tensor.add, inputs[1:], inputs[0])

        residual = co.Sequential(
            co.Broadcast(2),
            co.Parallel(
                co.Conv3d(32, 32, kernel_size=3, padding=1),
                co.Delay(2),
            ),
            co.Reduce(my_sum),
        )

    Args:
        reduce (Union[str, Callable[[Sequence[Tensor]], Tensor]]):
            Reduce function. Either one of ["sum", "channel", "mul", "max"] or
            user-defined function mapping a sequence of tensors to a single one.

    """

    _state_shape = 0
    _dynamic_state_inds = []

    def __init__(
        self,
        reduce: ReductionFuncOrEnum = "sum",
    ):
        nn.Module.__init__(self)
        self.reduce = nonempty(
            reduce
            if callable(reduce)
            else {
                Reduction.SUM: reduce_sum,
                Reduction.CONCAT: reduce_concat,
                Reduction.MUL: reduce_mul,
                Reduction.MAX: reduce_max,
            }[Reduction(reduce)]
        )

    def forward(self, inputs: List[T]) -> T:
        return self.reduce(inputs)

    def _forward_step(self, inputs: List[T], prev_state=None):
        if all(isinstance(i, Tensor) for i in inputs):
            return self.reduce(inputs), prev_state
        return None, prev_state  # pragma: no cover

    def forward_step(self, inputs: List[T], update_state=True) -> T:
        return self._forward_step(inputs)[0]

    def forward_steps(self, inputs: List[T], pad_end=False, update_state=True) -> T:
        return self.reduce(inputs)


class Sequential(FlattenableStateDict, CoModule, nn.Sequential):
    """A sequential container.
    This module is an augmentation of `torch.nn.Sequential`
    which adds continual inference methods

    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``OrderedDict`` of modules can be
    passed in. The ``forward()``, ``forward_step()`` and ``forward_steps()``
    methods of ``Sequential`` accept any input and forwards it to the first
    module it contains. It then "chains" outputs to inputs sequentially for
    each subsequent module, finally returning the output of the last module.

    The value a ``Sequential`` provides over manually calling a sequence
    of modules is that it allows treating the whole container as a
    single module, such that performing a transformation on the
    ``Sequential`` applies to each of the modules it stores (which are
    each a registered submodule of the ``Sequential``).

    Example::

        # Using Sequential to create a small model. When `model` is run,
        # input will first be passed to `Conv2d(1,20,5)`. The output of
        # `Conv2d(1,20,5)` will be used as the input to the first
        # `ReLU`; the output of the first `ReLU` will become the input
        # for `Conv2d(20,64,5)`. Finally, the output of
        # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
        model = co.Sequential(
            co.Conv2d(1,20,5),
            nn.ReLU(),
            co.Conv2d(20,64,5),
            nn.ReLU()
        )

        # Using Sequential with OrderedDict. This is functionally the
        # same as the above code
        model = co.Sequential(OrderedDict([
            ('conv1', co.Conv2d(1,20,5)),
            ('relu1', nn.ReLU()),
            ('conv2', co.Conv2d(20,64,5)),
            ('relu2', nn.ReLU())
        ]))
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
            with temporary_parameter(m, "call_mode", _callmode("forward")):
                input = apply_forward(m, input)
        return input

    def forward_step(self, input, update_state=True):
        for module in self:
            # ptflops only works when __call__ is triggered
            with temporary_parameter(module, "call_mode", _callmode("forward_step")):
                input = module(
                    input, update_state=update_state
                )  # == module.forward_step
            if not type(input) in {Tensor, list}:
                return None
        return input

    def _forward_step(self, input: torch.Tensor, prev_state: List[State]):
        prev_state = prev_state or [None for _ in range(len(self))]
        next_state = prev_state.copy()
        for i, module in enumerate(self):
            input, n_state = module._forward_step(input, prev_state=prev_state[i])
            next_state[i] = n_state
            if input is None:
                return None, next_state
        return input, next_state

    @property
    def _state_shape(self):
        return [m._state_shape for m in self]

    @property
    def _dynamic_state_inds(self):
        return [m._dynamic_state_inds for m in self]

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True):
        for m in self:
            if not type(input) in {Tensor, list} or len(input) == 0:
                return None  # pragma: no cover
            # ptflops only works when __call__ is triggered
            with temporary_parameter(m, "call_mode", _callmode("forward_steps")):
                # == m.forward_steps
                input = m(input, pad_end=pad_end, update_state=update_state)

        return input

    @property
    def receptive_field(self) -> int:
        reverse_modules = [m for m in self][::-1]
        rf = reverse_modules[0].receptive_field
        for m in reverse_modules[1:]:
            s = getattr(m, "stride", [1])
            s = s[0]
            rf = s * rf + m.receptive_field - s
        return rf

    @property
    def stride(self) -> Tuple[int]:
        tot = 1
        for m in self:
            tot *= m.stride[0]
        return (tot,)

    @property
    def padding(self) -> Tuple[int]:
        m = [m for m in self]
        p = m[0].padding[0]
        s = m[0].stride[0]
        for i in range(1, len(m)):
            p += m[i].padding[0] * s
            s = s * m[i].stride[0]
        return (p,)

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

    def append(self, module: nn.Module) -> "Sequential":
        r"""Appends a given module to the end.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self


class BroadcastReduce(Sequential):
    """Broadcast an input to parallel modules and reduce.
    This module is a shorthand for::

        co.Sequential(co.Broadcast(), co.Parallel(*args), co.Reduce(reduce))

    For instance, it can be used to succinctly create a continual 3D Inception Module::

        def norm_relu(module, channels):
            return co.Sequential(
                module,
                nn.BatchNorm3d(channels),
                nn.ReLU(),
            )

        inception_module = co.BroadcastReduce(
            co.Conv3d(192, 64, kernel_size=1),
            co.Sequential(
                norm_relu(co.Conv3d(192, 96, kernel_size=1), 96),
                norm_relu(co.Conv3d(96, 128, kernel_size=3, padding=1), 128),
            ),
            co.Sequential(
                norm_relu(co.Conv3d(192, 16, kernel_size=1), 16),
                norm_relu(co.Conv3d(16, 32, kernel_size=5, padding=2), 32),
            ),
            co.Sequential(
                co.MaxPool3d(kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
                norm_relu(co.Conv3d(192, 32, kernel_size=1), 32),
            ),
            reduce="concat",
        )

    Args:
        arg (OrderedDict[str, CoModule]): An OrderedDict or modules to be applied in parallel.
        *args (CoModule): Modules to be applied in parallel.
        reduce (ReductionFuncOrEnum, optional):
            Function used to reduce the parallel outputs.
            Sum or concatenation can be specified by passing "sum" or "concat" respectively.
            Custom reduce functions can also be passed.
            Defaults to "sum".
        auto_delay (bool, optional):
            Automatically add delay to modules in order to match the longest delay.
            Defaults to True.
    """

    @overload
    def __init__(
        self,
        *args: CoModule,
        reduce: ReductionFuncOrEnum = "sum",
        auto_delay=True,
    ) -> None:
        ...  # pragma: no cover

    @overload
    def __init__(
        self,
        arg: "OrderedDict[str, CoModule]",
        reduce: ReductionFuncOrEnum = "sum",
        auto_delay=True,
    ) -> None:
        ...  # pragma: no cover

    def __init__(
        self,
        *args,
        reduce: ReductionFuncOrEnum = "sum",
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
                Reduction.MAX: reduce_max,
            }[Reduction(reduce)]
        )

        delays = set(m.delay for m in self)
        self._delay = max(delays)

        receptive_fields = set(m.receptive_field for m in self)
        self._receptive_field = max(receptive_fields)

    def add_module(self, name: str, module: Optional["nn.Module"]) -> None:
        co_add_module(self, name, module)

    @property
    def _state_shape(self):
        return [m._state_shape for m in self]

    @property
    def _dynamic_state_inds(self):
        return [m._dynamic_state_inds for m in self]

    def _forward_step(self, input: torch.Tensor, prev_state: List[State] = None):
        prev_state = prev_state or [None for _ in range(len(self))]
        next_state = prev_state.copy()
        outs = []
        for i, module in enumerate(self):
            out, n_state = module._forward_step(input, prev_state=prev_state[i])
            next_state[i] = n_state
            outs.append(out)
        if all(isinstance(o, Tensor) for o in outs):
            return self.reduce(outs), next_state
        return None, next_state

    def forward_step(self, input: Tensor, update_state=True) -> Tensor:
        outs = []
        for m in self:
            with temporary_parameter(m, "call_mode", "forward_step"):
                outs.append(m(input, update_state=update_state))  # == m.forward_step
        if all(isinstance(o, Tensor) for o in outs):
            return self.reduce(outs)

        return None

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True) -> Tensor:
        outs = []
        for m in self:
            with temporary_parameter(m, "call_mode", "forward_steps"):
                # m.forward_steps
                outs.append(m(input, pad_end=pad_end, update_state=update_state))

        return self.reduce(outs)

    def forward(self, input: Tensor) -> Tensor:
        outs = []
        for m in self:
            with temporary_parameter(m, "call_mode", "forward"):
                outs.append(apply_forward(m, input))

        return self.reduce(outs)

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    @property
    def delay(self) -> int:
        return self._delay

    @property
    def stride(self) -> Tuple[int]:
        return getattr(next(iter(self)), "stride", (1,))

    @property
    def padding(self) -> Tuple[int]:
        return (max(getattr(m, "padding", (0,))[0] for m in self),)

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
    """Residual connection wrapper for input.

    This module produces a short form of BroadCast reduce with one delay stream::

        conv = co.Conv3d(32, 32, kernel_size=3, padding=1)
        res1 = co.BroadcastReduce(conv, co.Delay(2), reduce="sum")
        res2 = co.Residual(conv)

        x = torch.randn(1, 32, 5, 5, 5)
        assert torch.equal(res1(x), res2(x))

    Args:
        module (CoModule): module to which a residual should be added.
        temporal_fill (PaddingMode, optional): temporal fill type in delay. Defaults to None.
        reduce (Reduction, optional): Reduction function. Defaults to "sum".
        residual_shrink (bool, optional):
            Set residual to shrink its forward to match the temporal dimension reduction of the wrapped module.
            Options: "centered", "lagging" or True: Centered residual shrink;
                     "lagging": lagging shrink. Defaults to False.
                     "leading": leading shrink, i.e. no delay during forward_step(s).

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

    if residual_shrink == "leading":
        res = Skip(delay)
    else:
        res = Delay(delay, temporal_fill, auto_shrink=residual_shrink)

    return BroadcastReduce(
        res,  # Residual first yields easier broadcasting in reduce functions
        module,
        reduce=reduce,
        auto_delay=False,
    )


class Conditional(FlattenableStateDict, CoModule, nn.Module):
    """Module wrapper for conditional invocations at runtime.

    For instance, it can be used to apply a softmax if the module isn't training::

        net = co.Sequential()

        def not_training(module, x):
            return not net.training

        net.append(co.Conditional(not_training, torch.nn.Softmax(dim=1)))

    Args:
        predicate (Callable[[CoModule, Tensor], bool]):
            Function used to evaluate whether on module or the other should be invoked.
        on_true: CoModule: Module to invoke on True.
        on_false: Optional[CoModule]: Module to invoke on False. If no module is passed, execution is skipped.

    """

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
            return self._modules["0"].forward_step(input, update_state)
        elif "1" in self._modules:
            return self._modules["1"].forward_step(input, update_state)
        return input

    def _forward_step(
        self, input: Tensor, prev_state: Optional[State] = None
    ) -> Tuple[Tensor, Optional[State]]:
        prev_state = prev_state or [None, None]
        if self.predicate(self, input):
            return self._modules["0"]._forward_step(input, prev_state[0])
        elif "1" in self._modules:
            return self._modules["1"]._forward_step(input, prev_state[1])
        return input, prev_state

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True) -> Tensor:
        if self.predicate(self, input):
            return self._modules["0"].forward_steps(input)
        elif "1" in self._modules:
            return self._modules["1"].forward_steps(input)
        return input

    @property
    def _state_shape(self):
        return [m._state_shape for m in self._modules.values()]

    @property
    def _dynamic_state_inds(self):
        return [m._dynamic_state_inds for m in self._modules.values()]

    @property
    def delay(self) -> int:
        return self._delay

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    def extra_repr(self):
        return f"predicate={function_repr(self.predicate)}"
