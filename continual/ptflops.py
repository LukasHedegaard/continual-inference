""" Register modules with `ptflops` """

from .closure import Add, Multiply  # noqa: F401
from .container import (  # noqa: F401
    Broadcast,
    BroadcastReduce,
    Conditional,
    Parallel,
    Reduce,
    Residual,
    Sequential,
)
from .conv import Conv1d, Conv2d, Conv3d  # noqa: F401
from .linear import Linear  # noqa: F401
from .logging import getLogger
from .pooling import (  # noqa: F401
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


# Register modules in `ptflops`
def _register_ptflops():
    try:
        import ptflops

        if hasattr(ptflops, "pytorch_ops"):  # >= v0.6.8
            fc = ptflops.pytorch_ops
        else:  # < v0.6.7
            fc = ptflops.flops_counter

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

        # Linear
        fc.MODULES_MAPPING[Linear] = fc.linear_flops_counter_hook

    except ModuleNotFoundError:  # pragma: no cover
        pass
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to add flops_counter_hook: {e}")


_register_ptflops()
