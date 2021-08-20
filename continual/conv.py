from typing import Callable, Tuple

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

from .interface import CoModule, Padded, PaddingMode, TensorPlaceholder
from .logging import getLogger

logger = getLogger(__name__)

State = Tuple[Tensor, int, int]


__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
]


class _ConvCoNd(_ConvNd, Padded, CoModule):
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
        temporal_fill: PaddingMode = "zeros",
    ):
        assert issubclass(
            ConvClass, _ConvNd
        ), "The ConvClass should be a subclass of `_ConvNd`"
        self._ConvClass = ConvClass
        self._conv_func = conv_func
        self.input_shape_desciption = input_shape_desciption
        self._input_len = len(self.input_shape_desciption)

        kernel_size = size_fn(kernel_size)

        padding = size_fn(padding)
        if padding[0] != 0:
            logger.debug(
                "Padding along the temporal dimension only affects the computation in `forward_steps`. In `forward` it is omitted."
            )

        stride = size_fn(stride)
        if stride[0] > 1:
            logger.warning(
                f"Temporal stride of {stride[0]} will result in skipped outputs every {stride[0]-1} / {stride[0]} steps"
            )

        dilation = size_fn(dilation)
        assert dilation[0] == 1, "Temporal dilation > 1 is not supported currently."

        temporal_fill = PaddingMode(temporal_fill)
        padding_mode = PaddingMode(padding_mode)

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
            padding_mode=padding_mode.value,
        )
        self.make_padding = {
            PaddingMode.ZEROS: torch.zeros_like,
            PaddingMode.REPLICATE: torch.clone,
        }[temporal_fill]

        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
            (self.kernel_size[0] - 1, *self.padding[1:]), 2
        )

        # init_state is called in `_forward_step`

    def init_state(
        self,
        first_output: Tensor,
    ) -> State:
        padding = self.make_padding(first_output)
        state_buffer = padding.repeat(
            self.kernel_size[0] - 1, *[1 for _ in self.input_shape_desciption]
        )
        state_index = 0
        stride_index = self.stride[0] - len(state_buffer) - 1 + self.padding[0]
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

    def _forward_step(self, input: Tensor, prev_state: State) -> Tuple[Tensor, State]:
        assert (
            len(input.shape) == self._input_len - 1
        ), f"A tensor of shape {(*self.input_shape_desciption[:2], *self.input_shape_desciption[3:])} should be passed as input."

        # e.g. B, C -> B, C, 1
        x = input.unsqueeze(2)

        if self.padding_mode == "zeros":
            x = self._conv_func(
                input=x,
                weight=self.weight,
                bias=None,
                stride=(1, *self.stride[1:]),
                padding=(self.kernel_size[0] - 1, *self.padding[1:]),
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            x = self._conv_func(
                input=F.pad(
                    x, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight=self.weight,
                bias=None,
                stride=(1, *self.stride[1:]),
                padding=(self.kernel_size[0] - 1, 0),
                dilation=self.dilation,
                groups=self.groups,
            )

        x_out, x_rest = x[:, :, 0], x[:, :, 1:]

        # Prepare previous state
        buffer, index, stride_index = prev_state or self.init_state(x_rest)

        tot = len(buffer)
        # if stride_index == 0:
        if stride_index == self.stride[0] - 1:
            x_out = x_out.clone()
            for i in range(tot):
                x_out += buffer[(i + index) % tot, :, :, tot - i - 1]

            if self.bias is not None:
                x_out += self.bias[
                    (None, slice(None), *(None for _ in range(self._input_len - 3)))
                ]
        else:
            x_out = TensorPlaceholder(x_out.shape)

        # Update next state
        if self.kernel_size[0] > 1:
            next_buffer = buffer.clone() if self.training else buffer.detach()
            next_buffer[index] = x_rest
            next_index = (index + 1) % tot
        else:
            next_buffer = buffer
            next_index = index

        next_stride_index = stride_index + 1
        if next_stride_index > 0:
            next_stride_index = next_stride_index % self.stride[0]

        return x_out, (next_buffer, next_index, next_stride_index)

    def forward_step(self, input: Tensor, update_state=True) -> Tensor:
        output, (
            new_buffer,
            new_state_index,
            new_stride_index,
        ) = self._forward_step(input, self.get_state())
        if update_state:
            self.state_buffer = new_buffer
            self.state_index = new_state_index
            self.stride_index = new_stride_index
        return output

    def forward_steps(self, input: Tensor, pad_end=True):
        assert (
            len(input.shape) == self._input_len
        ), f"A tensor of shape {self.input_shape_desciption} should be passed as input."

        outs = []
        for t in range(input.shape[2]):
            o = self.forward_step(input[:, :, t])
            if isinstance(o, Tensor):
                outs.append(o)

        if pad_end:
            # Don't save state for the end-padding
            (tmp_buffer, tmp_index, tmp_stride_index) = self.get_state()
            for t, i in enumerate(
                [self.make_padding(input[:, :, -1]) for _ in range(self.padding[0])]
            ):
                o, (tmp_buffer, tmp_index, tmp_stride_index) = self._forward_step(
                    i, (tmp_buffer, tmp_index, tmp_stride_index)
                )
                if isinstance(o, Tensor):
                    outs.append(o)

        if len(outs) > 0:
            outs = torch.stack(outs, dim=2)
        else:
            outs = torch.tensor([])
        return outs

    def forward(self, input: Tensor):
        """Performs a full forward computation exactly as the regular layer would.
        This method is handy for effient training on clip-based data.

        Args:
            input (Tensor): Layer input

        Returns:
            Tensor: Layer output
        """
        assert (
            len(input.shape) == self._input_len
        ), f"A tensor of shape {self.input_shape_desciption} should be passed as input."
        output = self._ConvClass._conv_forward(self, input, self.weight, self.bias)
        return output

    @property
    def delay(self):
        return self.kernel_size[0] - 1 - self.padding[0]


class Conv1d(_ConvCoNd):
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
        temporal_fill: PaddingMode = "zeros",
    ):
        r"""Applies a continual 1D convolution over an input signal composed of several input
        planes.

        Assuming an input of shape `(B, C, T)`, it computes the convolution over one temporal instant `t` at a time
        where `t` ∈ `range(T)`, and keeps an internal state. Two forward modes are supported here.

        `forward`   takes an input of shape `(B, C)`, and computes a single-frame output (B, C') based on its internal state.
                    On the first execution, the state is initialised with either ``'zeros'`` (corresponding to a zero padding of kernel_size[0]-1)
                    or with a `'replicate'`` of the first frame depending on the choice of `temporal_fill`.
                    `forward` also supports a functional-style exercution, by passing a `prev_state` explicitely as parameters, and by
                    optionally returning the updated `next_state` via the `return_next_state` parameter.
                    NB: The output when recurrently applying forward will be delayed by the `kernel_size[0] - 1`.

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
            temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.Conv1d, temporal_fill: PaddingMode = "zeros", **kwargs
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
                    temporal_fill=temporal_fill,
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
        temporal_fill: PaddingMode = "zeros",
    ):
        r"""Applies a continual 2D convolution over an input signal composed of several input
        planes.

        Assuming an input of shape `(B, C, T, S)`, it computes the convolution over one temporal instant `t` at a time
        where `t` ∈ `range(T)`, and keeps an internal state. Two forward modes are supported here.

        `forward`   takes an input of shape `(B, C, S)`, and computes a single-frame output (B, C', S') based on its internal state.
                    On the first execution, the state is initialised with either ``'zeros'`` (corresponding to a zero padding of kernel_size[0]-1)
                    or with a `'replicate'`` of the first frame depending on the choice of `temporal_fill`.
                    `forward` also supports a functional-style exercution, by passing a `prev_state` explicitely as parameters, and by
                    optionally returning the updated `next_state` via the `return_next_state` parameter.
                    NB: The output when recurrently applying forward will be delayed by the `kernel_size[0] - 1`.

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
            temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.Conv2d, temporal_fill: PaddingMode = "zeros", **kwargs
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
                    temporal_fill=temporal_fill,
                ),
                **kwargs,
            }
        )

        with torch.no_grad():
            comodule.load_state_dict(module.state_dict())

        return comodule


class Conv3d(_ConvCoNd):
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
        temporal_fill: PaddingMode = "zeros",
    ):
        r"""Applies a continual 3D convolution over an input signal composed of several input
        planes.

        Assuming an input of shape `(B, C, T, H, W)`, it computes the convolution over one temporal instant `t` at a time
        where `t` ∈ `range(T)`, and keeps an internal state. Two forward modes are supported here.

        `forward_steps` operates identically to `nn.Conv3d.forward`

        `forward`   takes an input of shape `(B, C, H, W)`, and computes a single-frame output (B, C', H', W') based on its internal state.
                    On the first execution, the state is initialised with either ``'zeros'`` (corresponding to a zero padding of kernel_size[0]-1)
                    or with a `'replicate'`` of the first frame depending on the choice of `temporal_fill`.
                    `forward` also supports a functional-style exercution, by passing a `prev_state` explicitely as parameters, and by
                    optionally returning the updated `next_state` via the `return_next_state` parameter.
                    NB: The output when recurrently applying forward will be delayed by the `kernel_size[0] - 1`.

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
            temporal_fill,
        )

    @staticmethod
    def build_from(
        module: nn.Conv3d, temporal_fill: PaddingMode = "zeros", **kwargs
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
                    temporal_fill=temporal_fill,
                ),
                **kwargs,
            }
        )
        with torch.no_grad():
            comodule.weight.copy_(module.weight)
            if module.bias is not None:
                comodule.bias.copy_(module.bias)
        return comodule
