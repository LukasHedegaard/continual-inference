from enum import Enum
from functools import partial
from typing import Callable, Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from torch.nn.modules.utils import _ntuple, _pair, _single, _triple

from .interface import CoModule, Padded, PaddingMode, TensorPlaceholder

State = Tuple[Tensor, int]

T = TypeVar("T")
U = TypeVar("U")

__all__ = [
    "AvgPool1d",
    "MaxPool1d",
    "AvgPool2d",
    "MaxPool2d",
    "AdaptiveAvgPool2d",
    "AdaptiveMaxPool2d",
    "AvgPool3d",
    "MaxPool3d",
    "AdaptiveAvgPool3d",
    "AdaptiveMaxPool3d",
]

State = Tuple[Tensor, int]


def unity(x: Tensor) -> Tensor:
    return x


class PoolType(Enum):
    AVG = "avg"
    MAX = "max"


class _PoolNd(Padded, CoModule, nn.Module):
    """Base class for Continual Pooling modules
    This module implements a naive but flexible temporal pooling system.

    The approach is as follows:
    - For each step, the current frame is pooled in isolation.
    - The pooled frame is saved in a temporal buffer.
    - The appropriate frames (depending on kernel size, stride, and dilation)
      are selected from the temporal buffer and pooled in time.

    The implementation is naive in the sense, that there are possible optimisations
    that can be made for the important case of average pooling with stride = 1 and dilation = 1.
    In this case, we could keep a "running sum" from which we would
    subtract the oldest frame and add the newest frame at each step.
    """

    def __init__(
        self,
        temporal_pool: PoolType,
        spatial_pool_fn: Callable[[Tensor], Tensor],
        num_input_dims: int,
        kernel_size: _size_any_t = 1,
        stride: _size_any_t = 1,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        temporal_fill: PaddingMode = "zeros",
    ):
        nn.Module.__init__(self)
        assert num_input_dims in {1, 2, 3}, "Only 1d, 2d, and 3d pooling is supported."
        self.num_input_dims = num_input_dims

        _tuple = _ntuple(self.num_input_dims)
        self.kernel_size = _tuple(kernel_size)
        self.stride = _tuple(stride)
        self.padding = _tuple(padding)
        self.dilation = _tuple(dilation)
        assert (
            self.kernel_size[0] > 0
        ), "A pooling module should over at least on time step."

        self._spatial_pool_fn = spatial_pool_fn

        self.input_shape_desciption = ("batch_size", "channel", "time") + {
            1: tuple(),
            2: ("space",),
            3: ("height", "width"),
        }[num_input_dims]

        temporal_fill = PaddingMode(temporal_fill)
        self._make_padding = {
            temporal_fill.ZEROS: torch.zeros_like,
            temporal_fill.REPLICATE: torch.clone,
        }[temporal_fill]

        temporal_pool = PoolType(temporal_pool)
        self._temporal_pool_fn = {
            PoolType.AVG: nn.AdaptiveAvgPool1d,
            PoolType.MAX: nn.AdaptiveMaxPool1d,
        }[temporal_pool](1)

        # Select forward reshape mode depending on dimensionality
        def pooling_with_1d_reshape(frame_selection: Tensor) -> Tensor:
            _, B, C = frame_selection.shape
            x = self._temporal_pool_fn(
                frame_selection.permute(1, 2, 0)  # T, B, C -> B, C, T
            ).reshape(B, C)
            return x

        def pooling_with_2d_reshape(frame_selection: Tensor) -> Tensor:
            T, B, C, S = frame_selection.shape
            x = frame_selection.permute(1, 3, 2, 0)  # B, S, C, T
            x = x.reshape(B * S, C, T)
            x = self._temporal_pool_fn(x)
            x = x.reshape(B, S, C)
            x = x.permute(0, 2, 1)  # B, C, S
            return x

        def pooling_with_3d_reshape(frame_selection: Tensor) -> Tensor:
            T, B, C, H, W = frame_selection.shape
            x = frame_selection.permute(1, 3, 4, 2, 0)  # B, H, W, C, T
            x = x.reshape(B * H * W, C, T)
            x = self._temporal_pool_fn(x)
            x = x.reshape(B, H, W, C)
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            return x

        self._reshaped_temporal_pool_fn = {
            1: pooling_with_1d_reshape,
            2: pooling_with_2d_reshape,
            3: pooling_with_3d_reshape,
        }[num_input_dims]

        if self.dilation[0] > 1:
            self._frame_index_selection = torch.tensor(
                range(
                    0,
                    self.kernel_size[0] * self.dilation[0],
                    self.dilation[0],
                )
            )

    def init_state(
        self,
        first_output: Tensor,
    ) -> State:
        padding = self._make_padding(first_output)
        # buf_len = k + (d - 1) * (k - 1) =
        buf_len = self.dilation[0] * (self.kernel_size[0] - 1) + 1
        state_buffer = torch.stack(
            [padding for _ in range(buf_len)],
            dim=0,
        )
        state_index = 0
        stride_index = self.stride[0] - buf_len + self.padding[0]

        if not hasattr(self, "state_buffer"):
            self.register_buffer("state_buffer", state_buffer, persistent=False)
        return state_buffer, state_index, stride_index

    def clean_state(self):
        self.state_buffer = None
        self.state_index = None
        self.stride_index = None

    def get_state(self):
        if (
            hasattr(self, "state_buffer")
            and self.state_buffer is not None
            and hasattr(self, "state_index")
            and self.state_index is not None
            and hasattr(self, "stride_index")
            and self.stride_index is not None
        ):
            return (self.state_buffer, self.state_index, self.stride_index)

    def _forward_step(
        self,
        input: Tensor,
        prev_state: State,
    ) -> Tuple[Tensor, State]:
        assert (
            len(input.shape) == self.num_input_dims + 1
        ), f"A tensor of size {(*self.input_shape_desciption[:2], *self.input_shape_desciption[3:])} should be passed as input."

        pooled_frame = self._spatial_pool_fn(input)
        buffer, index, stride_index = prev_state or self.init_state(pooled_frame)
        buffer[index] = pooled_frame

        next_buffer = buffer  # .clone() if self.training else buffer.detach()
        next_index = (index + 1) % self.kernel_size[0]
        next_stride_index = stride_index + 1
        if next_stride_index > 0:
            next_stride_index = next_stride_index % self.stride[0]

        if stride_index == self.stride[0] - 1:
            if self.dilation[0] == 1:
                frame_selection = buffer
            else:
                frame_selection = buffer.index_select(
                    dim=0,
                    index=torch.remainder(
                        self._frame_index_selection + next_index, len(buffer)
                    ),
                )
            output = self._reshaped_temporal_pool_fn(frame_selection)
        else:
            output = TensorPlaceholder(pooled_frame.shape)

        return output, (next_buffer, next_index, next_stride_index)

    def forward_step(self, input: Tensor) -> Tensor:
        output, (
            self.state_buffer,
            self.state_index,
            self.stride_index,
        ) = self._forward_step(input, self.get_state())
        return output

    def forward_steps(self, input: Tensor, pad_end=True):
        assert (
            len(input.shape) == self.num_input_dims + 2
        ), f"A tensor of size {self.input_shape_desciption} should be passed as input."

        outs = []
        for t in range(input.shape[2]):
            o = self.forward_step(input[:, :, t])
            if isinstance(o, Tensor):
                outs.append(o)

        if pad_end:
            # Don't save state for the end-padding
            tmp_buffer, tmp_index, tmp_stride_index = self.get_state()
            for t, i in enumerate(
                [torch.zeros_like(input[:, :, -1]) for _ in range(self.padding[0])]
            ):
                o, (tmp_buffer, tmp_index, tmp_stride_index) = self._forward_step(
                    i, (tmp_buffer, tmp_index, tmp_stride_index)
                )
                if isinstance(o, Tensor):
                    outs.append(o)

        if len(outs) == 0:
            return torch.tensor([])  # pragma: no cover

        return torch.stack(outs, dim=2)

    @property
    def delay(self):
        return self.dilation[0] * (self.kernel_size[0] - 1)


class AvgPool1d(nn.AvgPool1d, _PoolNd):
    """
    Continual Average Pool in 1D

    This is the continual version of the regular :class:`torch.nn.AvgPool1d`
    """

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t = None,
        padding: _size_1_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        dilation: _size_1_t = 1,
        temporal_fill: PaddingMode = "zeros",
    ) -> None:
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if stride is not None else kernel_size)
        self.padding = _single(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dilation = _single(dilation)

        _PoolNd.__init__(
            self,
            temporal_pool=PoolType.AVG,
            spatial_pool_fn=unity,
            num_input_dims=1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            temporal_fill=temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.AvgPool1d,
        dilation: _size_1_t = 1,
        temporal_fill: PaddingMode = "zeros",
        **kwargs,
    ) -> "AvgPool1d":
        return AvgPool1d(
            **{
                **dict(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    ceil_mode=module.ceil_mode,
                    count_include_pad=module.count_include_pad,
                    dilation=dilation,
                    temporal_fill=temporal_fill,
                ),
                **kwargs,
            }
        )


class AvgPool2d(nn.AvgPool2d, _PoolNd):
    """
    Continual Average Pool in 2D

    This is the continual version of the regular :class:`torch.nn.AvgPool2d`
    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        dilation: _size_2_t = 1,
        temporal_fill: PaddingMode = "zeros",
    ) -> None:
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dilation = _pair(dilation)
        assert self.dilation[1] == 1, "Spatial dilation is not supported"

        _PoolNd.__init__(
            self,
            temporal_pool=PoolType.AVG,
            spatial_pool_fn=partial(
                F.avg_pool1d,
                kernel_size=self.kernel_size[1:],
                stride=self.stride[1:],
                padding=self.padding[1:],
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            ),
            num_input_dims=2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            temporal_fill=temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.AvgPool2d,
        dilation: _size_2_t = 1,
        temporal_fill: PaddingMode = "zeros",
        **kwargs,
    ) -> "AvgPool2d":
        return AvgPool2d(
            **{
                **dict(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    ceil_mode=module.ceil_mode,
                    count_include_pad=module.count_include_pad,
                    divisor_override=module.divisor_override,
                    dilation=dilation,
                    temporal_fill=temporal_fill,
                ),
                **kwargs,
            }
        )


class AvgPool3d(nn.AvgPool3d, _PoolNd):
    """
    Continual Average Pool in 3D

    This is the continual version of the regular :class:`torch.nn.AvgPool3d`
    """

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: _size_3_t = None,
        padding: _size_3_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        dilation: _size_3_t = 1,
        temporal_fill: PaddingMode = "zeros",
    ) -> None:
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride if stride is not None else kernel_size)
        self.padding = _triple(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dilation = _triple(dilation)
        assert self.dilation[1:] == (1, 1), "Spatial dilation is not supported"

        _PoolNd.__init__(
            self,
            temporal_pool=PoolType.AVG,
            spatial_pool_fn=partial(
                F.avg_pool2d,
                kernel_size=self.kernel_size[1:],
                stride=self.stride[1:],
                padding=self.padding[1:],
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                divisor_override=self.divisor_override,
            ),
            num_input_dims=3,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            temporal_fill=temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.AvgPool3d,
        dilation: _size_3_t = 1,
        temporal_fill: PaddingMode = "zeros",
        **kwargs,
    ) -> "AvgPool3d":
        return AvgPool3d(
            **{
                **dict(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    ceil_mode=module.ceil_mode,
                    count_include_pad=module.count_include_pad,
                    divisor_override=module.divisor_override,
                    dilation=dilation,
                    temporal_fill=temporal_fill,
                ),
                **kwargs,
            }
        )


class MaxPool1d(nn.MaxPool1d, _PoolNd):
    """
    Continual Max Pool in 1D

    This is the continual version of the regular :class:`torch.nn.MaxPool1d`
    """

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: Optional[_size_1_t] = None,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        # return_indices: bool = False, # Not supported
        ceil_mode: bool = False,
        temporal_fill: PaddingMode = "zeros",
    ) -> None:
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if (stride is not None) else kernel_size)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.return_indices = False
        self.ceil_mode = ceil_mode

        _PoolNd.__init__(
            self,
            temporal_pool=PoolType.MAX,
            spatial_pool_fn=unity,
            num_input_dims=1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            temporal_fill=temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.MaxPool1d,
        temporal_fill: PaddingMode = "zeros",
        **kwargs,
    ) -> "MaxPool1d":
        return MaxPool1d(
            **{
                **dict(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    ceil_mode=module.ceil_mode,
                    temporal_fill=temporal_fill,
                ),
                **kwargs,
            }
        )


class MaxPool2d(nn.MaxPool2d, _PoolNd):
    """
    Continual Max Pool in 2D

    This is the continual version of the regular :class:`torch.nn.MaxPool2d`
    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        # return_indices: bool = False, # Not supported
        ceil_mode: bool = False,
        temporal_fill: PaddingMode = "zeros",
    ) -> None:
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if (stride is not None) else kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.return_indices = False
        self.ceil_mode = ceil_mode

        _PoolNd.__init__(
            self,
            temporal_pool=PoolType.MAX,
            spatial_pool_fn=partial(
                F.max_pool1d,
                kernel_size=self.kernel_size[1:],
                stride=self.stride[1:],
                padding=self.padding[1:],
                dilation=self.dilation[1:],
                ceil_mode=self.ceil_mode,
                return_indices=False,
            ),
            num_input_dims=2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            temporal_fill=temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.MaxPool2d, temporal_fill: PaddingMode = "zeros", **kwargs
    ) -> "MaxPool2d":
        return MaxPool2d(
            **{
                **dict(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    ceil_mode=module.ceil_mode,
                    temporal_fill=temporal_fill,
                ),
                **kwargs,
            }
        )


class MaxPool3d(nn.MaxPool3d, _PoolNd):
    """
    Continual Max Pool in 3D

    This is the continual version of the regular :class:`torch.nn.MaxPool2d`
    """

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        # return_indices: bool = False, # Not supported
        ceil_mode: bool = False,
        temporal_fill: PaddingMode = "zeros",
    ) -> None:
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride if (stride is not None) else kernel_size)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.return_indices = False
        self.ceil_mode = ceil_mode

        _PoolNd.__init__(
            self,
            temporal_pool=PoolType.MAX,
            spatial_pool_fn=partial(
                F.max_pool2d,
                kernel_size=self.kernel_size[1:],
                stride=self.stride[1:],
                padding=self.padding[1:],
                dilation=self.dilation[1:],
                ceil_mode=self.ceil_mode,
                return_indices=False,
            ),
            num_input_dims=3,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            temporal_fill=temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.MaxPool3d, temporal_fill: PaddingMode = "zeros", **kwargs
    ) -> "MaxPool3d":
        return MaxPool3d(
            **{
                **dict(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    ceil_mode=module.ceil_mode,
                    temporal_fill=temporal_fill,
                ),
                **kwargs,
            }
        )


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, _PoolNd):
    """
    Continual Adaptive Average Pool in 2D

    This is the continual version of the regular :class:`torch.nn.AdaptiveAvgPool2d`
    """

    def __init__(
        self,
        output_size: _size_2_t,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        temporal_fill: PaddingMode = "zeros",
    ) -> None:

        self.output_size = _pair(output_size)
        assert self.output_size[0] == 1, "Only output_size[0] = 1 is supported"

        _PoolNd.__init__(
            self,
            temporal_pool=PoolType.AVG,
            spatial_pool_fn=partial(
                F.adaptive_avg_pool1d, output_size=self.output_size[1]
            ),
            num_input_dims=2,
            kernel_size=(kernel_size, None),
            stride=(stride, None),
            padding=(padding, None),
            dilation=(dilation, None),
            temporal_fill=temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.AdaptiveAvgPool2d,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        temporal_fill: PaddingMode = "zeros",
        output_size: _size_2_t = None,
    ) -> "AdaptiveAvgPool2d":
        return AdaptiveAvgPool2d(
            output_size=output_size or module.output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            temporal_fill=temporal_fill,
        )


class AdaptiveAvgPool3d(nn.AdaptiveAvgPool3d, _PoolNd):
    """
    Continual Adaptive Average Pool in 3D

    This is the continual version of the regular :class:`torch.nn.AdaptiveAvgPool3d`
    """

    def __init__(
        self,
        output_size: _size_3_t,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        temporal_fill: PaddingMode = "zeros",
    ) -> None:

        self.output_size = _triple(output_size)
        assert self.output_size[0] == 1, "Only output_size[0] = 1 is supported"

        _PoolNd.__init__(
            self,
            temporal_pool=PoolType.AVG,
            spatial_pool_fn=partial(
                F.adaptive_avg_pool2d, output_size=self.output_size[1:]
            ),
            num_input_dims=3,
            kernel_size=(kernel_size, None, None),
            stride=(stride, None, None),
            padding=(padding, None, None),
            dilation=(dilation, None, None),
            temporal_fill=temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.AdaptiveAvgPool3d,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        temporal_fill: PaddingMode = "zeros",
        output_size: _size_3_t = None,
    ) -> "AdaptiveAvgPool3d":
        return AdaptiveAvgPool3d(
            output_size=output_size or module.output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            temporal_fill=temporal_fill,
        )


class AdaptiveMaxPool2d(nn.AdaptiveMaxPool2d, _PoolNd):
    """
    Continual Adaptive Max Pool in 2D

    This is the continual version of the regular :class:`torch.nn.AdaptiveMaxPool2d`
    """

    def __init__(
        self,
        output_size: _size_2_t,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        temporal_fill: PaddingMode = "zeros",
    ) -> None:

        self.output_size = _pair(output_size)
        self.return_indices = False
        assert self.output_size[0] == 1, "Only output_size[0] = 1 is supported"

        _PoolNd.__init__(
            self,
            temporal_pool=PoolType.MAX,
            spatial_pool_fn=partial(
                F.adaptive_max_pool1d,
                output_size=self.output_size[1],
                return_indices=False,
            ),
            num_input_dims=2,
            kernel_size=(kernel_size, None),
            stride=(stride, None),
            padding=(padding, None),
            dilation=(dilation, None),
            temporal_fill=temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.AdaptiveMaxPool2d,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        temporal_fill: PaddingMode = "zeros",
        output_size: _size_2_t = None,
    ) -> "AdaptiveMaxPool2d":
        return AdaptiveMaxPool2d(
            output_size=output_size or module.output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            temporal_fill=temporal_fill,
        )


class AdaptiveMaxPool3d(nn.AdaptiveMaxPool3d, _PoolNd):
    """
    Continual Adaptive Max Pool in 3D

    This is the continual version of the regular :class:`torch.nn.AdaptiveMaxPool3d`
    """

    def __init__(
        self,
        output_size: _size_3_t,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        temporal_fill: PaddingMode = "zeros",
    ) -> None:

        self.output_size = _triple(output_size)
        self.return_indices = False
        assert self.output_size[0] == 1, "Only output_size[0] = 1 is supported"

        _PoolNd.__init__(
            self,
            temporal_pool=PoolType.MAX,
            spatial_pool_fn=partial(
                F.adaptive_max_pool2d,
                output_size=self.output_size[1:],
                return_indices=False,
            ),
            num_input_dims=3,
            kernel_size=(kernel_size, None, None),
            stride=(stride, None, None),
            padding=(padding, None, None),
            dilation=(dilation, None, None),
            temporal_fill=temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.AdaptiveMaxPool3d,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        temporal_fill: PaddingMode = "zeros",
        output_size: _size_3_t = None,
    ) -> "AdaptiveMaxPool3d":
        return AdaptiveMaxPool3d(
            output_size=output_size or module.output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            temporal_fill=temporal_fill,
        )
