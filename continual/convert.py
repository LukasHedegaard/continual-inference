""" Register modules with conversion system and 3rd-party libraries """

from functools import wraps
from typing import Type, Callable
from torch import nn, Tensor
from .batchnorm import BatchNorm2d
from .container import Sequential
from .conv import Conv1d, Conv2d, Conv3d
from .interface import CoModule
from .logging import getLogger
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

logger = getLogger(__name__)


def forward_stepping(module: nn.Module, dim: int = 2):
    """Enhances torch.nn.Module with `forward_step` and `forward_steps`
    by unsqueezing the temporal dimension.

    NB: The passed module must not have time-dependent operations!
    For instance, `module = nn.Conv3d(1, 1, kernel_size=(1,1,1))` is OK,
    but results for `module = nn.Conv3d(1, 1, kernel_size=(3,3,3))` would be invalid.

    Alternatively, one may attempt to automatically convert the module by using
    `co.continual(module)` instead.

    Args:
        module (nn.Module): the torch.nn.Module to enchange
        dim (int, optional): The dimension to unsqueeze during `forward_step`. Defaults to 2.
    """

    def decorator(func: Callable[[Tensor], Tensor]):
        @wraps(func)
        def call(x: Tensor) -> Tensor:
            x = x.unsqueeze(dim)
            x = func(x)
            x = x.squeeze(dim)
            return x

        return call

    module.forward = module.forward
    module.forward_steps = module.forward
    module.forward_step = decorator(module.forward)

    return module


# A mapping from torch.nn modules to continual modules
MODULE_MAPPING = {}


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
    assert type(module) in MODULE_MAPPING, (
        f"A registered conversion for {module.__name__} was not found. "
        "You can register a custom conversion as follows:"
        """
        from continual.utils import register

        register(MyTorchModule, MyContinualModule)
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

# BatchNorm
register(nn.BatchNorm2d, BatchNorm2d)

# Container
register(nn.Sequential, Sequential)


# Register modules in `ptflops`
try:
    from ptflops import flops_counter as fc

    # BatchNorm
    fc.MODULES_MAPPING[BatchNorm2d] = fc.bn_flops_counter_hook

    # Conv
    fc.MODULES_MAPPING[Conv1d] = fc.conv_flops_counter_hook
    fc.MODULES_MAPPING[Conv2d] = fc.conv_flops_counter_hook
    fc.MODULES_MAPPING[Conv3d] = fc.conv_flops_counter_hook

    # Pooling
    fc.MODULES_MAPPING[AvgPool1d] = fc.pool_flops_counter_hook
    fc.MODULES_MAPPING[MaxPool1d] = fc.pool_flops_counter_hook
    fc.MODULES_MAPPING[AvgPool2d] = fc.pool_flops_counter_hook
    fc.MODULES_MAPPING[MaxPool2d] = fc.pool_flops_counter_hook
    fc.MODULES_MAPPING[AdaptiveAvgPool2d] = fc.pool_flops_counter_hook
    fc.MODULES_MAPPING[AdaptiveMaxPool2d] = fc.pool_flops_counter_hook
    fc.MODULES_MAPPING[AvgPool3d] = fc.pool_flops_counter_hook
    fc.MODULES_MAPPING[MaxPool3d] = fc.pool_flops_counter_hook
    fc.MODULES_MAPPING[AdaptiveAvgPool3d] = fc.pool_flops_counter_hook
    fc.MODULES_MAPPING[AdaptiveMaxPool3d] = fc.pool_flops_counter_hook

except ModuleNotFoundError:  # pragma: no cover
    pass
except Exception as e:  # pragma: no cover
    logger.warning(f"Failed to add flops_counter_hook: {e}")
