from enum import Enum
from functools import partial
from typing import Callable, Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from torch.nn.modules.utils import _ntuple, _pair, _single, _triple

from continual.utils import neginf_like

from .module import CoModule, PaddingMode

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

T = TypeVar("T")
U = TypeVar("U")

State = Tuple[Tensor, Tensor, Tensor]


def unity(x: Tensor) -> Tensor:
    return x


class PoolType(Enum):
    AVG = "avg"
    MAX = "max"


class _PoolNd(CoModule, nn.Module):
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

    _state_shape = 3
    _dynamic_state_inds = [True, False, False]

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
        ), "A pooling module should span over at least one time step."

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
            temporal_fill.NEG_INF: neginf_like,
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

        self.register_buffer("state_buffer", torch.tensor([]), persistent=False)
        self.register_buffer("state_index", torch.tensor(0), persistent=False)
        self.register_buffer("stride_index", torch.tensor(0), persistent=False)

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
        state_index = torch.tensor(0)
        stride_index = torch.tensor(self.stride[0] - buf_len + self.padding[0])

        return state_buffer, state_index, stride_index

    def clean_state(self):
        self.state_buffer = torch.tensor([])
        self.state_index = torch.tensor(0)
        self.stride_index = torch.tensor(0)

    def get_state(self):
        if len(self.state_buffer) > 0:
            return (self.state_buffer, self.state_index, self.stride_index)

    def set_state(self, state: State):
        self.state_buffer, self.state_index, self.stride_index = state

    def _forward_step(
        self,
        input: Tensor,
        prev_state: Optional[State] = None,
    ) -> Tuple[Tensor, State]:
        assert (
            len(input.shape) == self.num_input_dims + 1
        ), f"A tensor of size {(*self.input_shape_desciption[:2], *self.input_shape_desciption[3:])} should be passed as input but got {input.shape}."

        pooled_frame = self._spatial_pool_fn(input)
        buffer, index, stride_index = prev_state or self.init_state(pooled_frame)
        buffer[index] = pooled_frame

        next_buffer = buffer  # .clone() if self.training else buffer.detach()
        next_index = (index + 1) % self.kernel_size[0]
        next_stride_index = stride_index + 1
        if next_stride_index > 0:
            next_stride_index = next_stride_index % self.stride[0]

        output = None
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

        return output, (next_buffer, next_index, next_stride_index)

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True):
        assert (
            len(input.shape) == self.num_input_dims + 2
        ), f"A tensor of size {self.input_shape_desciption} should be passed as input but got {input.shape}."

        return CoModule.forward_steps(self, input, pad_end, update_state)

    @property
    def receptive_field(self) -> int:
        return self.dilation[0] * self.kernel_size[0]


class AvgPool1d(nn.AvgPool1d, _PoolNd):
    r"""Applies a Continual 1D average pooling over an input signal.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`,
    output :math:`(N, C, L_{out})` and :attr:`kernel_size` :math:`k`
    can be precisely described as:

    .. math::

        \text{out}(N_i, C_j, l) = \frac{1}{k} \sum_{m=0}^{k-1}
                               \text{input}(N_i, C_j, \text{stride} \times l + m)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    .. note::
        When stride > 1, the forward_step will only produce non-None values every stride steps.

    .. note::
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can each be
    an ``int`` or a one-element tuple.

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation
        dilation: The stride between elements within a sliding window, must be > 0. Only temporal dimension is supported
        temporal_fill: How temporal states are initialized

    Shape:
        - Input: :math:`(N, C, L_{in})`.
        - Output: :math:`(N, C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} +
              2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} + 1\right\rfloor

    Examples::

        m = co.AvgPool1d(3, padding=1)
        x = torch.randn(20, 16, 50)
        assert torch.allclose(m.forward(x), m.forward_steps(x))
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
    r"""Applies a Continual 2D average pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, T, W)`,
    output :math:`(N, C, T_{out}, W_{out})` and :attr:`kernel_size` :math:`(kT, kW)`
    can be precisely described as:

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kT * kW} \sum_{m=0}^{kT-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    .. note::
        When stride > 1, the forward_step will only produce non-None values every stride steps.

    .. note::
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used
        dilation: The stride between elements within a sliding window, must be > 0. Only temporal dimension is supported
        temporal_fill: How temporal states are initialized


    Shape:
        - Input: :math:`(N, C, T_{in}, W_{in})``.
        - Output: :math:`(N, C, T_{out}, W_{out})``, where

          .. math::
              T_{out} = \left\lfloor\frac{T_{in}  + 2 \times \text{padding}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    Examples::

        m = co.AvgPool2d(3, stride=(2, 1))
        x = torch.randn(20, 16, 50, 32)
        assert torch.allclose(m.forward(x), m.forward_steps(x), atol=1e-7)
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
    r"""Applies a Continual 3D average pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, T, H, W)`,
    output :math:`(N, C, T_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kT, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \sum_{k=0}^{kT-1} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1} \\
                                              & \frac{\text{input}(N_i, C_j, \text{stride}[0] \times d + k,
                                                      \text{stride}[1] \times h + m, \text{stride}[2] \times w + n)}
                                                     {kT \times kH \times kW}
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on all three sides
    for :attr:`padding` number of points.

    .. note::

        When stride > 1, the forward_step will only produce non-None values every stride steps.

    .. note::

        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on all three sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise :attr:`kernel_size` will be used
        dilation: The stride between elements within a sliding window, must be > 0. Only temporal dimension is supported
        temporal_fill: How temporal states are initialized

    Shape:
        - Input: :math:`(N, C, T_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, T_{out}, H_{out}, W_{out})`, where

          .. math::
              T_{out} = \left\lfloor\frac{T_{in} + 2 \times \text{padding}[0] -
                    \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] -
                    \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] -
                    \text{kernel\_size}[2]}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        m = co.AvgPool3d((3, 3, 3))
        x = torch.randn(20, 16, 50, 44, 31)
        assert torch.allclose(m.forward(x), m.forward_steps(x), atol=1e-7)
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
    r"""Applies a Continual 1D max pooling over an input signal.

    In the simplest case, the output value of the layer with input size :math:`(N, C, T)`
    and output :math:`(N, C, T_{out})` can be precisely described as:

    .. math::
        out(N_i, C_j, k) = \max_{m=0, \ldots, \text{kernel\_size} - 1}
                input(N_i, C_j, stride \times k + m)

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` is the stride between the elements within the
    sliding window. This `link`_ has a nice visualization of the pooling parameters.

    .. note::

        When :attr:`stride` > 1, the forward_step will only produce non-None values every :attr:`stride` steps.

    .. note::

        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    Args:
        kernel_size: The size of the sliding window, must be > 0.
        stride: The stride of the sliding window, must be > 0. Default value is :attr:`kernel_size`.
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
                   ensures that every element in the input tensor is covered by a sliding window.
        temporal_fill: How temporal states are initialized.

    Shape:
        - Input: :math:`(N, C, T_{in})`.
        - Output: :math:`(N, C, T_{out})`, where

          .. math::
              T_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Examples::

        m = co.MaxPool1d(kernel_size=3, dilation=2)
        x = torch.randn(20, 16, 50)
        assert torch.allclose(m.forward(x), m.forward_steps(x))

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: Optional[_size_1_t] = None,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        # return_indices: bool = False, # Not supported
        ceil_mode: bool = False,
        temporal_fill: PaddingMode = "neg_inf",
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
    r"""Applies a Continual 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, T, S)`,
    output :math:`(N, C, T_{out}, S_{out})` and :attr:`kernel_size` :math:`(kT, kS)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, t, s) ={} & \max_{m=0, \ldots, kT-1} \max_{n=0, \ldots, kS-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times t + m,
                                                   \text{stride[1]} \times s + n)
        \end{aligned}

    The pooling over the :math:`T` dimension is continual (progressively cached) and the other is regular.
    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    .. note::

        When stride > 1, the forward_step will only produce non-None values every stride steps.

    .. note::
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        temporal_fill: How temporal states are initialized.

    Shape:
        - Input: :math:`(N, C, T_{in}, S_{in})`
        - Output: :math:`(N, C, T_{out}, S_{out})`, where

          .. math::
              T_{out} = \left\lfloor\frac{T_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              S_{out} = \left\lfloor\frac{S_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    Examples::

        m = MaxPool2d(3, stride=2)
        x = torch.randn(20, 16, 50, 32)
        assert torch.allclose(m.forward(x), m.forward_steps(x))

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
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
    r"""Applies a Continual 3D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, T, H, W)`,
    output :math:`(N, C, T_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kT, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \max_{k=0, \ldots, kT-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                              & \text{input}(N_i, C_j, \text{stride[0]} \times d + k,
                                                             \text{stride[1]} \times h + m, \text{stride[2]} \times w + n)
        \end{aligned}

    The pooling over the :math:`T` dimension is continual (progressively cached) and the others are regular.
    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    .. note::

        When stride > 1, the forward_step will only produce non-None values every stride steps.

    .. note::
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on all three sides
        dilation: a parameter that controls the stride of elements in the window
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        temporal_fill: How temporal states are initialized.

    Shape:
        - Input: :math:`(N, C, T_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, T_{out}, H_{out}, W_{out})`, where

          .. math::
              T_{out} = \left\lfloor\frac{T_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times
                (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times
                (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times
                (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        m = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2))
        x = torch.randn(20, 16, 50,44, 31)
        assert torch.allclose(m.forward(x), m.forward_steps(x))

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
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
    r"""Applies a Continual 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size T x W, for any input size.
    The pooling over the T dimension is continual (progressively cached) and the other is regular.
    During continual inference, the temporal pooling size is determined by the :attr:`kernel_size`.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form T x W.
                     Can be a tuple (T, W) or a single T for a square image T x T.
                     T and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
        kernel_size: Temporal kernel size to use for ``forward_step`` and ``forward_steps``.
        temporal_fill: How temporal states are initialized.

    Shape:
        - Input: :math:`(N, C, T_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1})`, where
          :math:`S=\text{output\_size}`.

    Examples::

        # target output size of 1x1
        m = co.AdaptiveAvgPool2d((1, 1), kernel_size=5)
        x = torch.randn(1, 64, 5, 16)
        assert torch.allclose(m.forward(x), m.forward_steps(x))

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
    r"""Applies a Continual 3D adaptive average pooling over an input signal composed of several input planes.

    The output is of size T x H x W, for any input size.
    The pooling over the T dimension is continual (progressively cached) and the other is regular.
    During continual inference, the temporal pooling size is determined by the :attr:`kernel_size`.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the form T x H x W.
                     Can be a tuple (T, H, W) or a single number T for a cube T x T x T.
                     T, H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
        kernel_size: Temporal kernel size to use for ``forward_step`` and ``forward_steps``.
        temporal_fill: How temporal states are initialized.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1}, S_{2})`,
          where :math:`S=\text{output\_size}`.

    Examples::

        # target output size of 1x1x1
        m = co.AdaptiveAvgPool3d((1, 1, 1), kernel_size=5)
        x = torch.randn(1, 64, 5, 16, 16)
        assert torch.allclose(m.forward(x), m.forward_steps(x))

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
    r"""Applies a Continual 2D adaptive max pooling over an input signal composed of several input planes.

    The output is of size T x W, for any input size.
    The pooling over the T dimension is continual (progressively cached) and the other is regular.
    During continual inference, the temporal pooling size is determined by the :attr:`kernel_size`.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form T x W.
                     Can be a tuple (T, W) or a single T for a square image T x T.
                     T and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
        kernel_size: Temporal kernel size to use for ``forward_step`` and ``forward_steps``.
        temporal_fill: How temporal states are initialized.

    Shape:
        - Input: :math:`(N, C, T_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1})`, where
          :math:`S=\text{output\_size}`.

    Examples::

        # target output size of 1x1
        m = co.AdaptiveMaxPool2d((1, 1), kernel_size=5)
        x = torch.randn(1, 64, 5, 16)
        assert torch.allclose(m.forward(x), m.forward_steps(x))

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
    r"""Applies a Continual 3D adaptive max pooling over an input signal composed of several input planes.

    The output is of size T x H x W, for any input size.
    The pooling over the T dimension is continual (progressively cached) and the other is regular.
    During continual inference, the temporal pooling size is determined by the :attr:`kernel_size`.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the form T x H x W.
                     Can be a tuple (T, H, W) or a single number T for a cube T x T x T.
                     T, H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
        kernel_size: Temporal kernel size to use for ``forward_step`` and ``forward_steps``.
        temporal_fill: How temporal states are initialized.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1}, S_{2})`,
          where :math:`S=\text{output\_size}`.

    Examples::

        # target output size of 1x1x1
        m = co.AdaptiveMaxPool3d((1, 1, 1), kernel_size=5)
        x = torch.randn(1, 64, 5, 16, 16)
        assert torch.allclose(m.forward(x), m.forward_steps(x))

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
