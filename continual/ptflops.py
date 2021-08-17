""" Register modules with `ptflops` """

from .conv import Conv1d, Conv2d, Conv3d
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


# Register modules in `ptflops`
def _register_ptflops():
    try:
        from ptflops import flops_counter as fc

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


_register_ptflops()
