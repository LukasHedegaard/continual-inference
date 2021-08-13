""" Register modules with `ptflops` """

from .conv import Conv1d, Conv2d, Conv3d
from .pooling import (
    AvgPool1d,
    MaxPool1d,
    AvgPool2d,
    MaxPool2d,
    AdaptiveAvgPool2d,
    AdaptiveMaxPool2d,
    AvgPool3d,
    MaxPool3d,
    AdaptiveAvgPool3d,
    AdaptiveMaxPool3d,
)
from .batchnorm import BatchNorm2d
from .logging import getLogger

logger = getLogger(__name__)


# Register modules in `ptflops`
def register_ptflops():
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


register_ptflops()
