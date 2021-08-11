from typing import Callable, Tuple, Type, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.pooling import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    AvgPool1d,
    AvgPool2d,
    MaxPool1d,
    MaxPool2d,
)

from .interface import _CoModule
from .utils import FillMode

State = Tuple[Tensor, int]

T = TypeVar("T")

__all__ = [
    "AvgPoolCo1d",
    "MaxPoolCo1d",
    "AvgPoolCo2d",
    "MaxPoolCo2d",
    "AdaptiveAvgPoolCo2d",
    "AdaptiveMaxPoolCo2d",
    "AvgPoolCo3d",
    "MaxPoolCo3d",
    "AdaptiveAvgPoolCo3d",
    "AdaptiveMaxPoolCo3d",
]

State = Tuple[Tensor, int]


class _DummyAvgPool(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)


class _DummyMaxPool(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)


def _co_window_pooled(  # noqa: C901
    cls: T,
    functional_fn: Callable[[Tensor, Tuple, Tuple, int, bool, bool], Tensor] = None,
) -> Type[T]:
    """Wraps a pooling module to create a recursive version which pools across execusions

    Args:
        cls (T): A pooling module
        functional_fn (Callable): A torch.nn.function pooling function.
    """
    assert cls in {
        _DummyAvgPool,
        _DummyMaxPool,
        AdaptiveAvgPool1d,
        MaxPool1d,
        AvgPool1d,
        AdaptiveMaxPool1d,
        AvgPool2d,
        MaxPool2d,
        AdaptiveAvgPool2d,
        AdaptiveMaxPool2d,
    }

    class CoPool1d(_CoModule, cls):
        def __init__(
            self,
            window_size: int,
            temporal_fill: FillMode = "replicate",
            temporal_dilation: int = 1,
            *args,
            **kwargs,
        ):
            cls.__init__(self, *args, **kwargs)

            class_name = cls.__name__.lower()

            self.input_shape_desciption = ("batch_size", "channel", "time")
            if "1d" in class_name:
                self.input_shape_desciption += ("space",)
            elif "2d" in class_name:
                self.input_shape_desciption += ("height", "width")

            assert window_size > 0
            assert temporal_fill in {"zeros", "replicate"}
            self.window_size = window_size
            self.temporal_dilation = temporal_dilation
            self.make_padding = {"zeros": torch.zeros_like, "replicate": torch.clone}[
                temporal_fill
            ]

            self.temporal_pool = (
                AdaptiveAvgPool1d if "avg" in class_name else AdaptiveMaxPool1d
            )(1)

            if self.temporal_dilation > 1:
                self.frame_index_selection = torch.tensor(
                    range(0, self.window_size, self.temporal_dilation)
                )

            # Directly assign other kwargs to self. This is need to make AdaptiveAvgPool work
            for k, v in kwargs.items():
                setattr(self, k, v)

            # Select reshape mode depending on dimensionality
            def apply_1d_pooling(frame_selection: Tensor) -> Tensor:
                _, B, C = frame_selection.shape
                x = self.temporal_pool(
                    frame_selection.permute(1, 2, 0)  # T, B, C -> B, C, T
                ).reshape(B, C)
                return x

            def apply_2d_pooling(frame_selection: Tensor) -> Tensor:
                T, B, C, S = frame_selection.shape
                x = frame_selection.permute(1, 3, 2, 0)  # B, S, C, T
                x = x.reshape(B * S, C, T)
                x = self.temporal_pool(x)
                x = x.reshape(B, S, C)
                x = x.permute(0, 2, 1)  # B, C, S
                return x

            def apply_3d_pooling(frame_selection: Tensor) -> Tensor:
                T, B, C, H, W = frame_selection.shape
                x = frame_selection.permute(1, 3, 4, 2, 0)  # B, H, W, C, T
                x = x.reshape(B * H * W, C, T)
                x = self.temporal_pool(x)
                x = x.reshape(B, H, W, C)
                x = x.permute(0, 3, 1, 2)  # B, C, H, W
                return x

            self.apply_pooling = {
                3: apply_1d_pooling,
                4: apply_2d_pooling,
                5: apply_3d_pooling,
            }[len(self.input_shape_desciption)]

        def init_state(
            self,
            first_output: Tensor,
        ) -> State:
            padding = self.make_padding(first_output)
            state_buffer = torch.stack(
                [padding for _ in range(self.window_size)], dim=0
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

        def _forward(
            self,
            input: Tensor,
            prev_state: State,
        ) -> Tuple[Tensor, State]:
            assert (
                len(input.shape) == len(self.input_shape_desciption) - 1
            ), f"A tensor of size {(*self.input_shape_desciption[:2], *self.input_shape_desciption[3:])} should be passed as input."

            pooled_frame = (
                cls.forward(self, input)  # 2D and 3D pooling
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
            pooled_window = self.apply_pooling(frame_selection)

            new_index = (index + 1) % self.window_size
            new_buffer = buffer  # .clone() if self.training else buffer.detach()

            return pooled_window, (new_buffer, new_index)

        def forward(self, input: Tensor) -> Tensor:
            output, (self.state_buffer, self.state_index) = self._forward(
                input, self.get_state()
            )
            return output

        def forward_regular(self, input: Tensor):
            """Performs a full forward computation in a frame-wise manner, updating layer states along the way.

            If input.shape[2] == self.window_size, a global pooling along temporal dimension is performed
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
                o = self.forward(input[:, :, t])
                if self.window_size - 1 <= t:
                    outs.append(o)

            if len(outs) == 0:
                return torch.tensor([])

            if input.shape[2] == self.window_size:
                # In order to be compatible with downstream forward_regular, select only last frame
                # This corrsponds to the regular global pool
                return outs[-1].unsqueeze(2)

            else:
                return torch.stack(outs, dim=2)

        def forward_regular_unrolled(self, input: Tensor) -> Tensor:
            """Performs a full forward computation exactly as the regular layer would.

            Args:
                input (Tensor): Layer input

            Returns:
                Tensor: Layer output
            """
            # For now, use implementation in forward_regular
            # TODO: Update impl
            assert len(input.shape) == len(
                self.input_shape_desciption
            ), f"A tensor of size {self.input_shape_desciption} should be passed as input."
            if functional_fn:
                return functional_fn(
                    input,
                    kernel_size=self.window_size,
                    stride=self.temporal_dilation,
                    padding=0,
                    ceil_mode=False,
                    count_include_pad=True,
                )
            else:
                return self.forward_regular(input)

        @property
        def delay(self):
            return self.window_size - 1

    return CoPool1d


class AvgPoolCo1d(_co_window_pooled(_DummyAvgPool, F.avg_pool1d)):
    """
    Continual Average Pool in 1D

    This is the continual version of the regular :class:`torch.nn.AvgPool1d`
    """

    ...


class MaxPoolCo1d(_co_window_pooled(_DummyMaxPool, F.max_pool1d)):
    """
    Continual Max Pool in 1D

    This is the continual version of the regular :class:`torch.nn.MaxPool1d`
    """

    ...


class AvgPoolCo2d(_co_window_pooled(AvgPool1d, F.avg_pool2d)):
    """
    Continual Average Pool in 2D

    This is the continual version of the regular :class:`torch.nn.AvgPool2d`
    """

    ...


class MaxPoolCo2d(_co_window_pooled(MaxPool1d, F.max_pool2d)):
    """
    Continual Max Pool in 2D

    This is the continual version of the regular :class:`torch.nn.MaxPool2d`
    """

    ...


class AdaptiveAvgPoolCo2d(_co_window_pooled(AdaptiveAvgPool1d)):
    """
    Continual Adaptive Average Pool in 2D

    This is the continual version of the regular :class:`torch.nn.AdaptiveAvgPool2d`
    """

    ...


class AdaptiveMaxPoolCo2d(_co_window_pooled(AdaptiveMaxPool1d)):
    """
    Continual Adaptive Max Pool in 2D

    This is the continual version of the regular :class:`torch.nn.AdaptiveMaxPool2d`
    """

    ...


class AvgPoolCo3d(_co_window_pooled(AvgPool2d, F.avg_pool3d)):
    """
    Continual Average Pool in 3D

    This is the continual version of the regular :class:`torch.nn.AvgPool3d`
    """

    ...


class MaxPoolCo3d(_co_window_pooled(MaxPool2d, F.max_pool3d)):
    """
    Continual Max Pool in 3D

    This is the continual version of the regular :class:`torch.nn.MaxPool2d`
    """

    ...


class AdaptiveAvgPoolCo3d(_co_window_pooled(AdaptiveAvgPool2d)):
    """
    Continual Adaptive Average Pool in 3D

    This is the continual version of the regular :class:`torch.nn.AdaptiveAvgPool3d`
    """

    ...


class AdaptiveMaxPoolCo3d(_co_window_pooled(AdaptiveMaxPool2d)):
    """
    Continual Adaptive Max Pool in 3D

    This is the continual version of the regular :class:`torch.nn.AdaptiveMaxPool3d`
    """

    ...
