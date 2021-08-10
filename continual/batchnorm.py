from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from .interface import _CoModule
from .utils import temporary_parameter


def normalise_momentum(num_frames: int, base_mom=0.1):
    return 2 / (num_frames * (2 / base_mom - 1) + 1)


class BatchNormCo2d(_BatchNorm, _CoModule):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        window_size=1,
    ):
        super(BatchNormCo2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        # Normalise momentum w.r.t. the expected clip size
        self.momentum = normalise_momentum(window_size, momentum)
        self.unnormalised_momentum = momentum

    def _check_input_dim(self, input):
        # This function is called from _BatchNorm.forward to validate the input shape
        # In this module, we need to cover the same cases as both BatchNorm1d and BatchNorm2d
        if input.dim() not in {2, 3, 4}:
            raise ValueError(
                "expected 2D, 3D, or 4D input (got {}D input)".format(input.dim())
            )

    def forward(self, input: Tensor) -> Tensor:
        output = _BatchNorm.forward(self, input)
        return output

    def forward_regular(self, input: Tensor) -> Tensor:
        return self.forward_regular_unrolled(input)

    def forward_regular_unrolled(self, input: Tensor) -> Tensor:
        with temporary_parameter(self, "momentum", self.unnormalised_momentum):
            output = _BatchNorm.forward(self, input)
        return output

    @property
    def delay(self) -> int:
        return 0
