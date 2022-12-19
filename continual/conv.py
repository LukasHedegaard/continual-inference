from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.conv import (
    _ConvNd,
    _pair,
    _reverse_repeat_tuple,
    _single,
    _size_1_t,
    _size_2_t,
    _size_3_t,
    _triple,
)

from .logging import getLogger
from .module import CoModule, PaddingMode

logger = getLogger(__name__)

State = Tuple[Tensor, Tensor, Tensor]

# _forward_step_impl = None
# from pathlib import Path
# try:
#     from torch.utils.cpp_extension import load as load_cpp

#     _forward_step_impl = load_cpp(
#         name="cpp_impl",
#         sources=[str(Path(__file__).parent / "conv.cpp")],
#         verbose=False,
#     ).forward_step
# except Exception as e:  # pragma: no cover
#     logger.warning(
#         "Unable to compile CoConv C++ implementation. Falling back to Python version."
#     )
#     logger.warning(e)


__all__ = ["Conv1d", "Conv2d", "Conv3d"]


class _ConvCoNd(CoModule, _ConvNd):
    def __init__(
        self,
        ConvClass: torch.nn.Module,
        conv_func: Callable,
        input_shape_desciption: Tuple[str],
        size_fn: Callable,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding,
        dilation,
        groups: int = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        device=None,
        dtype=None,
        temporal_fill: PaddingMode = "zeros",
    ):
        assert issubclass(
            ConvClass, _ConvNd
        ), "The ConvClass should be a subclass of `_ConvNd`"

        kernel_size = size_fn(kernel_size)
        padding = size_fn(padding)
        stride = size_fn(stride)

        self._ConvClass = ConvClass
        self._conv_func = conv_func
        self.input_shape_desciption = input_shape_desciption
        self._input_len = len(self.input_shape_desciption)

        if stride[0] > 1:
            logger.warning(
                f"Temporal stride of {stride[0]} will result in skipped outputs every {stride[0]-1} / {stride[0]} steps"
            )

        dilation = size_fn(dilation)
        assert dilation[0] == 1, "Temporal dilation > 1 is not supported currently."

        self.padding_mode = PaddingMode(padding_mode).value
        self.t_padding_mode = PaddingMode(temporal_fill).value

        _ConvNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed=False,
            output_padding=size_fn(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self._make_padding = {
            PaddingMode.ZEROS.value: torch.zeros_like,
            PaddingMode.REPLICATE.value: torch.clone,
        }[self.t_padding_mode]

        # Padding used in for `forward`
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        # Padding used in for `forward_step`
        self._step_space_rprt = _reverse_repeat_tuple((0, *self.padding[1:]), 2)
        self._step_time_pad = (
            self.kernel_size[0] - 1,
            *[0 for _ in self.padding[1:]],
        )
        self._step_padding = (self.kernel_size[0] - 1, *self.padding[1:])
        self._step_stride = (1, *self.stride[1:])

        self.register_buffer("state_buffer", torch.tensor([]), persistent=False)
        self.register_buffer("state_index", torch.tensor(0), persistent=False)
        self.register_buffer("stride_index", torch.tensor(0), persistent=False)

    @property
    def _stateless(self) -> bool:
        return self.kernel_size[0] == 1 and self.padding[0] == 0 and self.stride[0] == 1

    @property
    def _state_shape(self) -> int:
        if self._stateless:
            return 0
        else:
            return 3

    @property
    def _dynamic_state_inds(self) -> List[bool]:
        if self._stateless:
            return []
        else:
            return [True, False, False]

    def init_state(
        self,
        first_output: Tensor,
    ) -> State:
        padding = self._make_padding(first_output)
        repeat_shape = [self.kernel_size[0] - 1]
        repeat_shape.extend((1,) * len(self.input_shape_desciption))
        state_buffer = padding.repeat(repeat_shape)
        state_index = torch.tensor(0)
        stride_index = torch.tensor(
            self.stride[0] - len(state_buffer) - 1 + self.padding[0]
        )
        return (state_buffer, state_index, stride_index)

    def clean_state(self):
        self.state_buffer = torch.tensor([])
        self.state_index = torch.tensor(0)
        self.stride_index = torch.tensor(0)

    def get_state(self) -> Optional[State]:
        if len(self.state_buffer) > 0:
            return (self.state_buffer, self.state_index, self.stride_index)
        return None

    def set_state(self, state: State):
        self.state_buffer, self.state_index, self.stride_index = state

    @torch.jit.export
    def _forward_step(
        self, input: Tensor, prev_state: Optional[State]
    ) -> Tuple[Optional[Tensor], Optional[State]]:
        # assert (
        #     len(input.shape) == self._input_len - 1
        # ), f"A tensor of shape {(*self.input_shape_desciption[:2], *self.input_shape_desciption[3:])} should be passed as input but got {input.shape}"

        # if (
        #     _forward_step_impl is not None
        #     and not self.training
        #     and self.padding_mode == "zeros"
        # ):
        #     # Call C++ impl
        #     output, next_state = _forward_step_impl(
        #         input,
        #         self.weight,
        #         self.bias,
        #         self.stride,
        #         self.padding,
        #         self._step_padding,
        #         self.dilation,
        #         self.groups,
        #         *(prev_state or (None, None, None)),
        #     )
        #     if output is None:
        #         output = None
        #     return output, next_state
        return self._forward_step_py(input, prev_state)

    def _forward_step_py(
        self, input: Tensor, prev_state: Optional[State]
    ) -> Tuple[Optional[Tensor], Optional[State]]:
        # e.g. B, C -> B, C, 1
        x = input.unsqueeze(2).to(device=self.weight.device)

        if self._stateless:
            return self.forward(x).squeeze(2), prev_state

        if self.padding_mode == "zeros":
            x = self._conv_func(
                input=x,
                weight=self.weight,
                bias=None,
                stride=self._step_stride,
                padding=self._step_padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            x = self._conv_func(
                input=F.pad(x, self._step_space_rprt, mode=self.padding_mode),
                weight=self.weight,
                bias=None,
                stride=self._step_stride,
                padding=self._step_time_pad,
                dilation=self.dilation,
                groups=self.groups,
            )

        x_out, x_rest = x[:, :, 0], x[:, :, 1:]

        # Prepare previous state
        if prev_state is None:
            buffer, index, stride_index = self.init_state(x_rest)
        else:
            buffer, index, stride_index = prev_state

        assert index is not None
        assert stride_index is not None

        tot = self.kernel_size[0] - 1
        output_is_valid = stride_index == self.stride[0] - 1

        if output_is_valid:
            x_out = x_out + (
                torch.sum(
                    buffer[
                        torch.remainder(torch.arange(tot) + index, tot),
                        :,
                        :,
                        torch.arange(tot - 1, -1, -1),
                    ],
                    dim=0,
                )
            )

            if self.bias is not None:
                bias = self.bias.unsqueeze(0)
                for _ in range(self._input_len - 3):
                    bias = bias.unsqueeze(-1)
                x_out += bias

        # Update next state
        if self.kernel_size[0] > 1:
            next_buffer = buffer.clone() if self.training else buffer
            next_buffer[index] = x_rest
            next_index = (index + 1) % tot
        else:
            next_buffer = buffer
            next_index = index

        next_stride_index = stride_index + 1
        if next_stride_index > 0:
            next_stride_index = next_stride_index % self.stride[0]

        if output_is_valid:
            return x_out, (next_buffer, next_index, next_stride_index)
        return None, (next_buffer, next_index, next_stride_index)

    def forward_steps(
        self, input: Tensor, pad_end: bool = False, update_state: bool = True
    ) -> Optional[Tensor]:
        # assert (
        #     len(input.shape) == self._input_len
        # ), f"A tensor of shape {self.input_shape_desciption} should be passed as input but got {input.shape}."
        return self._forward_steps_impl(input, pad_end, update_state)

    def forward(self, input: Tensor) -> Tensor:
        """Performs a full forward computation exactly as the regular layer would.
        This method is handy for efficient training on clip-based data.

        Args:
            input (Tensor): Layer input

        Returns:
            Tensor: Layer output
        """
        assert (
            len(input.shape) == self._input_len
        ), f"A tensor of shape {self.input_shape_desciption} should be passed as input but got {input.shape}."
        # output = self._ConvClass._conv_forward(self, input, self.weight, self.bias)
        if self.padding_mode == "zeros":
            output = self._conv_func(
                input=input,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            output = self._conv_func(
                input=F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=(0,) * len(self.padding),
                dilation=self.dilation,
                groups=self.groups,
            )

        return output

    @property
    def receptive_field(self) -> int:
        return self.kernel_size[0] + (self.kernel_size[0] - 1) * (self.dilation[0] - 1)


class Conv1d(_ConvCoNd):
    r"""Continual 1D convolution over a temporal input signal.

    Continual Convolutions were proposed by
    Hedegaard et al.: "Continual 3D Convolutional Neural Networks for Real-time Processing of Videos", in ECCV (2022),
    https://arxiv.org/pdf/2106.00050.pdf (paper) https://www.youtube.com/watch?v=Jm2A7dVEaF4 (video).

    Assuming an input of shape `(B, C, T)`, it computes the convolution over one temporal instant `t` at a time
    where `t` ∈ `range(T)`, and keeps an internal state.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. NB: stride > 1 over the first channel is not supported. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. NB: padding over the first channel is not supported. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. NB: dilation > 1 over the first channel is not supported. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        temporal_fill (string, optional): ``'zeros'`` or ``'replicate'`` (= "boring video"). `temporal_fill` determines how state is initialised and which padding is applied during `forward_steps` along the temporal dimension. Default: ``'replicate'``

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                        :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                        :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                        The values of these weights are sampled from
                        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                        :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                        then the values of these weights are
                        sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                        :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        state (List[Tensor]):  a running buffer of partial computations from previous frames which are used for
                        the calculation of subsequent outputs.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        device=None,
        dtype=None,
        temporal_fill: PaddingMode = "zeros",
    ):
        _ConvCoNd.__init__(
            self,
            nn.Conv1d,
            F.conv1d,
            ("batch_size", "channel", "time"),
            _single,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
            temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.Conv1d, temporal_fill: PaddingMode = None, **kwargs
    ) -> "Conv1d":
        comodule = Conv1d(
            **{
                **dict(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    padding_mode=module.padding_mode,
                    temporal_fill=temporal_fill or module.padding_mode,
                ),
                **kwargs,
            }
        )
        with torch.no_grad():
            comodule.weight.copy_(module.weight)
            if module.bias is not None:
                comodule.bias.copy_(module.bias)
        return comodule


class Conv2d(_ConvCoNd):
    r"""Continual 2D convolution over a spatio-temporal input signal.

    Continual Convolutions were proposed by
    Hedegaard et al.: "Continual 3D Convolutional Neural Networks for Real-time Processing of Videos", in ECCV (2022),
    https://arxiv.org/pdf/2106.00050.pdf (paper) https://www.youtube.com/watch?v=Jm2A7dVEaF4 (video).

    Assuming an input of shape `(B, C, T, S)`, it computes the convolution over one temporal instant `t` at a time
    where `t` ∈ `range(T)`, and keeps an internal state.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. NB: stride > 1 over the first channel is not supported. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. NB: padding over the first channel is not supported. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. NB: dilation > 1 over the first channel is not supported. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        temporal_fill (string, optional): ``'zeros'`` or ``'replicate'`` (= "boring video"). `temporal_fill` determines how state is initialised and which padding is applied during `forward_steps` along the temporal dimension. Default: ``'replicate'``

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                        :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                        :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                        The values of these weights are sampled from
                        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                        :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                        then the values of these weights are
                        sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                        :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        state (List[Tensor]):  a running buffer of partial computations from previous frames which are used for
                        the calculation of subsequent outputs.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        device=None,
        dtype=None,
        temporal_fill: PaddingMode = "zeros",
    ):
        _ConvCoNd.__init__(
            self,
            nn.Conv2d,
            F.conv2d,
            ("batch_size", "channel", "time", "space"),
            _pair,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
            temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.Conv2d, temporal_fill: PaddingMode = None, **kwargs
    ) -> "Conv2d":
        comodule = Conv2d(
            **{
                **dict(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    padding_mode=module.padding_mode,
                    temporal_fill=temporal_fill or module.padding_mode,
                ),
                **kwargs,
            }
        )

        with torch.no_grad():
            comodule.load_state_dict(module.state_dict())

        return comodule


class Conv3d(_ConvCoNd):
    r"""Continual 3D convolution over a spatio-temporal input signal.

    Continual Convolutions were proposed by
    Hedegaard et al.: "Continual 3D Convolutional Neural Networks for Real-time Processing of Videos", in ECCV (2022),
    https://arxiv.org/pdf/2106.00050.pdf (paper) https://www.youtube.com/watch?v=Jm2A7dVEaF4 (video).

    Assuming an input of shape `(B, C, T, H, W)`, it computes the convolution over one temporal instant `t` at a time
    where `t` ∈ `range(T)`, and keeps an internal state. Two forward modes are supported here.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. NB: stride > 1 over the first channel is not supported. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. NB: padding over the first channel is not supported. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. NB: dilation > 1 over the first channel is not supported. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        temporal_fill (string, optional): ``'zeros'`` or ``'replicate'`` (= "boring video"). `temporal_fill` determines how state is initialised and which padding is applied during `forward_steps` along the temporal dimension. Default: ``'replicate'``

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                        :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                        :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                        The values of these weights are sampled from
                        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                        :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                        then the values of these weights are
                        sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                        :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        state (List[Tensor]):  a running buffer of partial computations from previous frames which are used for
                        the calculation of subsequent outputs.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        device=None,
        dtype=None,
        temporal_fill: PaddingMode = "zeros",
    ):
        _ConvCoNd.__init__(
            self,
            nn.Conv3d,
            F.conv3d,
            ("batch_size", "channel", "time", "height", "width"),
            _triple,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
            temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.Conv3d, temporal_fill: PaddingMode = None, **kwargs
    ) -> "Conv3d":
        comodule = Conv3d(
            **{
                **dict(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    padding_mode=module.padding_mode,
                    temporal_fill=temporal_fill or module.padding_mode,
                ),
                **kwargs,
            }
        )
        with torch.no_grad():
            comodule.weight.copy_(module.weight)
            if module.bias is not None:
                comodule.bias.copy_(module.bias)
        return comodule
