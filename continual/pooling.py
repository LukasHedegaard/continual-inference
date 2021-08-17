from functools import partial
from typing import Callable, Tuple, Type, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.utils import _ntuple

from .interface import CoModule, FillMode

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


class _DummyAvgPool(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)


class _DummyMaxPool(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)


def _co_window_pooled(  # noqa: C901
    InnerClass: Type[T],
    FromClass: Type[U],
    functional_fn: Callable[[Tensor, Tuple, Tuple, int, bool, bool], Tensor],
) -> Type[T]:
    """Wraps a pooling module to create a recursive version which pools across execusions

    Args:
        InnerClass (T): A pooling module to be used for pooling internally
        FromClass (T): The torch.nn.Module which this module can be `build_from`
        functional_fn (Callable): A torch.nn.function pooling function.
    """

    class CoPool(CoModule, InnerClass):
        def __init__(
            self,
            temporal_kernel_size: int = 1,
            temporal_stride: int = 1,
            temporal_dilation: int = 1,
            temporal_fill: FillMode = "zeros",
            *args,
            **kwargs,
        ):
            InnerClass.__init__(self, *args, **kwargs)

            class_name = InnerClass.__name__.lower()

            self.input_shape_desciption = ("batch_size", "channel", "time")
            if "1d" in class_name:
                self.input_shape_desciption += ("space",)
            elif "2d" in class_name:
                self.input_shape_desciption += ("height", "width")

            assert temporal_kernel_size > 0
            temporal_fill = FillMode(temporal_fill)
            self.temporal_kernel_size = temporal_kernel_size
            assert (
                temporal_stride == 1
            ), "Temporal stride > 1 is not supported currently."
            self.temporal_stride = temporal_stride
            self.temporal_dilation = temporal_dilation
            self.make_padding = {
                temporal_fill.ZEROS: torch.zeros_like,
                temporal_fill.REPLICATE: torch.clone,
            }[temporal_fill]

            self.temporal_pool = (
                nn.AdaptiveAvgPool1d if "avg" in class_name else nn.AdaptiveMaxPool1d
            )(1)

            if self.temporal_dilation > 1:
                self.frame_index_selection = torch.tensor(
                    range(0, self.temporal_kernel_size, self.temporal_dilation)
                )

            # Directly assign other kwargs to self. This is need to make AdaptiveAvgPool work
            for k, v in kwargs.items():
                setattr(self, k, v)

            # Select forward reshape mode depending on dimensionality
            def pooling_with_1d_reshape(frame_selection: Tensor) -> Tensor:
                _, B, C = frame_selection.shape
                x = self.temporal_pool(
                    frame_selection.permute(1, 2, 0)  # T, B, C -> B, C, T
                ).reshape(B, C)
                return x

            def pooling_with_2d_reshape(frame_selection: Tensor) -> Tensor:
                T, B, C, S = frame_selection.shape
                x = frame_selection.permute(1, 3, 2, 0)  # B, S, C, T
                x = x.reshape(B * S, C, T)
                x = self.temporal_pool(x)
                x = x.reshape(B, S, C)
                x = x.permute(0, 2, 1)  # B, C, S
                return x

            def pooling_with_3d_reshape(frame_selection: Tensor) -> Tensor:
                T, B, C, H, W = frame_selection.shape
                x = frame_selection.permute(1, 3, 4, 2, 0)  # B, H, W, C, T
                x = x.reshape(B * H * W, C, T)
                x = self.temporal_pool(x)
                x = x.reshape(B, H, W, C)
                x = x.permute(0, 3, 1, 2)  # B, C, H, W
                return x

            self._apply_pooling = {
                3: pooling_with_1d_reshape,
                4: pooling_with_2d_reshape,
                5: pooling_with_3d_reshape,
            }[len(self.input_shape_desciption)]

            # Select functional call depending on module type
            if InnerClass in {_DummyAvgPool, _DummyMaxPool}:
                self._functional_call = partial(
                    functional_fn,
                    kernel_size=(self.temporal_kernel_size,),
                    stride=(self.temporal_dilation,),
                )
            elif "Adaptive" in FromClass.__name__:
                self._functional_call = partial(
                    functional_fn,
                    output_size=(
                        getattr(self, "_temporal_adaptive_output_size", 1),
                        *self.output_size,
                    ),
                )
            else:
                # E.g. isolate the "2" in MaxPool2d and ensure the tuple matches
                _tuple = _ntuple(int(InnerClass.__name__[-2]))
                self._functional_call = partial(
                    functional_fn,
                    kernel_size=(self.temporal_kernel_size, *_tuple(self.kernel_size)),
                    stride=(self.temporal_stride, *_tuple(self.stride)),
                    padding=(0, *_tuple(self.padding)),
                )
                if "Max" in FromClass.__name__:
                    self._functional_call = partial(
                        self._functional_call,
                        dilation=(self.temporal_dilation, *_tuple(self.dilation)),
                    )

        def init_state(
            self,
            first_output: Tensor,
        ) -> State:
            padding = self.make_padding(first_output)
            state_buffer = torch.stack(
                [padding for _ in range(self.temporal_kernel_size)], dim=0
            )
            state_index = 0
            if not hasattr(self, "state_buffer"):
                self.register_buffer("state_buffer", state_buffer, persistent=False)
            return state_buffer, state_index

        def clean_state(self):
            self.state_buffer = None
            self.state_index = None

        def get_state(self):
            if (
                hasattr(self, "state_buffer")
                and self.state_buffer is not None
                and hasattr(self, "state_index")
                and self.state_buffer is not None
            ):
                return (self.state_buffer, self.state_index)
            else:
                return None

        def _forward_step(
            self,
            input: Tensor,
            prev_state: State,
        ) -> Tuple[Tensor, State]:
            assert (
                len(input.shape) == len(self.input_shape_desciption) - 1
            ), f"A tensor of size {(*self.input_shape_desciption[:2], *self.input_shape_desciption[3:])} should be passed as input."

            pooled_frame = (
                InnerClass.forward(self, input)  # 2D and 3D pooling
                if len(self.input_shape_desciption) > 3
                else input  # 1D pooling
            )

            if prev_state is None:
                buffer, index = self.init_state(pooled_frame)
            else:
                buffer, index = prev_state

            buffer[index] = pooled_frame

            if self.temporal_dilation == 1:
                frame_selection = buffer
            else:
                frame_selection = buffer.index_select(
                    dim=0, index=self.frame_index_selection
                )

            # Pool along temporal dimension
            pooled_window = self._apply_pooling(frame_selection)

            new_index = (index + 1) % self.temporal_kernel_size
            new_buffer = buffer  # .clone() if self.training else buffer.detach()

            return pooled_window, (new_buffer, new_index)

        def forward_step(self, input: Tensor) -> Tensor:
            output, (self.state_buffer, self.state_index) = self._forward_step(
                input, self.get_state()
            )
            return output

        def forward_steps(self, input: Tensor):
            """Performs a full forward computation in a frame-wise manner, updating layer states along the way.

            If input.shape[2] == self.temporal_kernel_size, a global pooling along temporal dimension is performed
            Otherwise, the pooling is performed per frame

            Args:
                input (Tensor): Layer input

            Returns:
                Tensor: Layer output
            """
            assert len(input.shape) == len(
                self.input_shape_desciption
            ), f"A tensor of size {self.input_shape_desciption} should be passed as input."

            outs = []
            for t in range(input.shape[2]):
                o = self.forward_step(input[:, :, t])
                if self.temporal_kernel_size - 1 <= t:
                    outs.append(o)

            if len(outs) == 0:
                return torch.tensor([])

            if input.shape[2] == self.temporal_kernel_size:
                # In order to be compatible with downstream forward_steps, select only last frame
                # This corrsponds to the regular global pool
                return outs[-1].unsqueeze(2)
            else:
                return torch.stack(outs, dim=2)

        def forward(self, input: Tensor) -> Tensor:
            """Performs a full forward computation exactly as the regular layer would.

            Args:
                input (Tensor): Layer input

            Returns:
                Tensor: Layer output
            """
            assert len(input.shape) == len(
                self.input_shape_desciption
            ), f"A tensor of size {self.input_shape_desciption} should be passed as input."

            return self._functional_call(input)

        @property
        def delay(self):
            return self.temporal_dilation * (self.temporal_kernel_size - 1)

        @staticmethod
        def build_from(
            module: FromClass, temporal_kernel_size: int = 1, *args, **kwargs
        ) -> "CoPool":
            FromType = type(module)
            assert FromType == FromClass, f"Can only build from {FromClass.__name__}"

            def unpack(tuple_or_int, index_or_slice):
                if type(tuple_or_int) == tuple:
                    return tuple_or_int[index_or_slice]
                else:
                    assert type(tuple_or_int) == int
                    return tuple_or_int

            if FromType in {nn.AvgPool1d, nn.MaxPool1d}:
                kwargs = dict(temporal_kernel_size=unpack(module.kernel_size, 0))
            elif "Adaptive" in FromType.__name__:
                kwargs = dict(
                    temporal_kernel_size=temporal_kernel_size,
                    output_size=unpack(module.output_size, slice(1, None)),
                    _temporal_adaptive_output_size=unpack(module.output_size, 0),
                )
            else:
                kwargs = dict(
                    temporal_kernel_size=unpack(module.kernel_size, 0),
                    temporal_dilation=unpack(module.stride, 0),
                    kernel_size=unpack(module.kernel_size, slice(1, None)),
                    stride=unpack(module.stride, slice(1, None)),
                    padding=unpack(module.padding, slice(1, None)),
                    ceil_mode=module.ceil_mode,
                )

            return CoPool(**kwargs)

    return CoPool


class AvgPool1d(_co_window_pooled(_DummyAvgPool, nn.AvgPool1d, F.avg_pool1d)):
    """
    Continual Average Pool in 1D

    This is the continual version of the regular :class:`torch.nn.AvgPool1d`
    """

    ...


class AvgPool2d(_co_window_pooled(nn.AvgPool1d, nn.AvgPool2d, F.avg_pool2d)):
    """
    Continual Average Pool in 2D

    This is the continual version of the regular :class:`torch.nn.AvgPool2d`
    """

    ...


class AvgPool3d(_co_window_pooled(nn.AvgPool2d, nn.AvgPool3d, F.avg_pool3d)):
    """
    Continual Average Pool in 3D

    This is the continual version of the regular :class:`torch.nn.AvgPool3d`
    """

    ...


class MaxPool1d(_co_window_pooled(_DummyMaxPool, nn.MaxPool1d, F.max_pool1d)):
    """
    Continual Max Pool in 1D

    This is the continual version of the regular :class:`torch.nn.MaxPool1d`
    """

    ...


class MaxPool2d(_co_window_pooled(nn.MaxPool1d, nn.MaxPool2d, F.max_pool2d)):
    """
    Continual Max Pool in 2D

    This is the continual version of the regular :class:`torch.nn.MaxPool2d`
    """

    ...


class MaxPool3d(_co_window_pooled(nn.MaxPool2d, nn.MaxPool3d, F.max_pool3d)):
    """
    Continual Max Pool in 3D

    This is the continual version of the regular :class:`torch.nn.MaxPool2d`
    """

    ...


class AdaptiveAvgPool2d(
    _co_window_pooled(nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d)
):
    """
    Continual Adaptive Average Pool in 2D

    This is the continual version of the regular :class:`torch.nn.AdaptiveAvgPool2d`
    """

    ...


class AdaptiveAvgPool3d(
    _co_window_pooled(nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, F.adaptive_avg_pool3d)
):
    """
    Continual Adaptive Average Pool in 3D

    This is the continual version of the regular :class:`torch.nn.AdaptiveAvgPool3d`
    """

    ...


class AdaptiveMaxPool2d(
    _co_window_pooled(nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, F.adaptive_max_pool2d)
):
    """
    Continual Adaptive Max Pool in 2D

    This is the continual version of the regular :class:`torch.nn.AdaptiveMaxPool2d`
    """

    ...


class AdaptiveMaxPool3d(
    _co_window_pooled(nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d, F.adaptive_max_pool3d)
):
    """
    Continual Adaptive Max Pool in 3D

    This is the continual version of the regular :class:`torch.nn.AdaptiveMaxPool3d`
    """

    ...
