""" Register modules with conversion system and 3rd-party libraries """

from functools import wraps
from types import FunctionType
from typing import Callable, Type, Union

from torch import Tensor, nn

from .closure import Lambda
from .container import Sequential
from .conv import Conv1d, Conv2d, Conv3d
from .linear import Linear
from .logging import getLogger
from .module import CoModule, _callmode, call_mode
from .pooling import (
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
)
from .rnn import GRU, LSTM, RNN
from .transformer import TransformerEncoder

logger = getLogger(__name__)

__all__ = [
    "continual",
    "forward_stepping",
]


def forward_stepping(module: nn.Module, dim: int = 2):
    """Enhances torch.nn.Module with `forward_step` and `forward_steps`

    .. note::
        The passed module must not have time-dependent operations!
        For instance, ``module = nn.Conv3d(1, 1, kernel_size=(1,1,1))`` is OK,
        but results for ``module = nn.Conv3d(1, 1, kernel_size=(3,3,3))`` would be invalid.

    Alternatively, one may attempt to automatically convert the module by using
    :class:`co.continual` instead.

    Args:
        module (nn.Module): the torch.nn.Module to enchance.
        dim (int, optional): The dimension to unsqueeze during `forward_step`. Defaults to 2.
    """

    def _forward_step(func: Callable[[Tensor], Tensor]):
        @wraps(func)
        def call(x: Tensor, prev_state=None) -> Tensor:
            return func(x.unsqueeze(dim)).squeeze(dim), prev_state

        return call

    def forward_step(func: Callable[[Tensor], Tensor]):
        @wraps(func)
        def call(x: Tensor, update_state=True) -> Tensor:
            return func(x.unsqueeze(dim)).squeeze(dim)

        return call

    def forward_steps(func: Callable[[Tensor], Tensor]):
        @wraps(func)
        def call(x: Tensor, pad_end=False, update_state=True) -> Tensor:
            x = func(x)
            return x

        return call

    def dummy(*args, **kwargs):
        ...  # pragma: no cover

    @staticmethod
    def build_from(mod):  # pragma: no cover
        return module.__class__()

    # module.__class__.__bases__ = (*module.__class__.__bases__, CoModule)
    orig_forward = module.forward
    module.forward_steps = forward_steps(module.forward)
    module.forward_step = forward_step(module.forward)
    module._forward_step = _forward_step(module.forward)
    module.delay = 0
    module.receptive_field = 1
    module.stride = tuple(getattr(module, "stride", [1]))
    module.padding = tuple(getattr(module, "padding", [0]))
    module.build_from = build_from
    module.get_state = dummy
    module.set_state = dummy
    module.clean_state = dummy
    module._state_shape = 0
    module._dynamic_state_inds = []

    # Call mode
    module.call_mode = _callmode("forward")

    def forward_with_callmode(*args, **kwargs):
        _call_mode = (
            call_mode.cur
            if call_mode.prev is not None
            else getattr(module, "call_mode", _callmode("forward"))
        )
        if _call_mode == _callmode("forward"):
            return orig_forward(*args, *kwargs)
        return CoModule._call_impl(module, *args, **kwargs)

    module.forward = forward_with_callmode

    return module


# A mapping from torch.nn modules to continual modules
MODULE_MAPPING = {}

NAIVE_MAPPING = {
    # >> Activations
    nn.Threshold,
    nn.ReLU,
    nn.RReLU,
    nn.Hardtanh,
    nn.ReLU6,
    nn.Sigmoid,
    nn.Hardsigmoid,
    nn.Tanh,
    nn.SiLU,
    nn.Hardswish,
    nn.ELU,
    nn.CELU,
    nn.SELU,
    nn.GLU,  # has dim
    nn.GELU,
    nn.Hardshrink,
    nn.LeakyReLU,
    nn.LogSigmoid,
    nn.Softplus,
    nn.Softshrink,
    nn.PReLU,
    nn.Softsign,
    nn.Tanhshrink,
    nn.Softmin,  # has dim
    nn.Softmax,  # has dim
    nn.Softmax2d,
    nn.LogSoftmax,
    # >> Norm modules
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
    # >> Dropout modules
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
}


_circumvent_message = " to work with automatic conversion. You can circumvent this by wrapping the module in `co.forward_stepping(your_module)`. Note however, that this may break correspondence between forward and forward_step."


def _instance_norm_condition(
    module: Union[nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
):
    assert module.affine, (
        f"{type(module)} must be specified with `affine==True`" + _circumvent_message
    )
    assert module.track_running_stats, (
        f"{type(module)} must be specified with `track_running_stats==True`"
        + _circumvent_message
    )


CONDITIONAL_MAPPING = {
    nn.InstanceNorm1d: _instance_norm_condition,
    nn.InstanceNorm2d: _instance_norm_condition,
    nn.InstanceNorm3d: _instance_norm_condition,
}


class ModuleNotRegisteredError(Exception):
    ...


def register(TorchNnModule: Type[nn.Module], CoClass: Type[CoModule]):
    CoModule._validate_class(CoClass)
    assert callable(getattr(CoClass, "build_from", None)), (
        f"To register {CoClass.__name__}, it should implement a `build_from` method:"
        """
        class MyContinualModule:

            @staticmethod
            def build_from(module: MyTorchModule) -> "MyContinualModule":
                ...
        """
    )
    MODULE_MAPPING[TorchNnModule] = CoClass
    return CoClass


def continual(module: nn.Module) -> CoModule:
    """Convert a ``torch.nn`` module to a Continual Inference Network enhanced with
    ``forward_step`` and ``forward_steps``.

    Modules may be either ``torch.nn`` Modules for which a corresponding module is
    implemented in this library (e.g. ``nn.Conv3d``), or a ``torch.nn`` which can be
    naively mapped (e.g. ``nn.ReLU``).

    Custom modules can also be made continual by means of the
    :class:`forward_stepping` function.

    Examples ::

        coconv = co.continual(nn.Conv3d(3, 3, 3))

        coseq = co.continual(nn.Sequential(
            nn.Conv3d(3, 3, 3),  # Automatically converted as well!
            nn.ReLU()
        ))

    """
    if isinstance(module, CoModule):
        return module

    if type(module) in NAIVE_MAPPING:
        return forward_stepping(module)

    if type(module) in CONDITIONAL_MAPPING:
        CONDITIONAL_MAPPING[type(module)](module)
        return forward_stepping(module)

    assert type(module) in MODULE_MAPPING, (
        f"A registered conversion for {module} was not found. "
        "You can register a custom conversion as follows:"
        """
        import continual as co

        co.convert.register(MyTorchModule, MyContinualModule)
        """
    )
    return MODULE_MAPPING[type(module)].build_from(module)


# Register modules with our conversion system

# Conv
register(nn.Conv1d, Conv1d)
register(nn.Conv2d, Conv2d)
register(nn.Conv3d, Conv3d)

# Pooling
register(nn.AvgPool1d, AvgPool1d)
register(nn.MaxPool1d, MaxPool1d)
register(nn.AvgPool2d, AvgPool2d)
register(nn.MaxPool2d, MaxPool2d)
register(nn.AdaptiveAvgPool2d, AdaptiveAvgPool2d)
register(nn.AdaptiveMaxPool2d, AdaptiveMaxPool2d)
register(nn.AvgPool3d, AvgPool3d)
register(nn.MaxPool3d, MaxPool3d)
register(nn.AdaptiveAvgPool3d, AdaptiveAvgPool3d)
register(nn.AdaptiveMaxPool3d, AdaptiveMaxPool3d)

# Linear
register(nn.Linear, Linear)

# Container
register(nn.Sequential, Sequential)

# Closure
register(FunctionType, Lambda)

# RNN
register(nn.RNN, RNN)
register(nn.LSTM, LSTM)
register(nn.GRU, GRU)

# Transformer
register(nn.TransformerEncoder, TransformerEncoder)
