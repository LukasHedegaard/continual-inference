import torch
from torch import Tensor, nn
from torch.nn.functional import linear

from .module import CoModule

__all__ = ["Linear"]


class Linear(CoModule, nn.Linear):
    r"""Applies a linear transformation to a dimension of the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        channel_dim: Channel dimension index over which to perform linear projection. Default: -1.

    Shape:
        - Input: :math:`(B, C_{in}, T, *)` where :math:`*` means any number of
          additional dimensions and :math:`C_{in} = \text{in\_features}` if `channel_dim = 2`.
          If channel_dim = -1, the order of input dimensions is :math:`(*, C_{in})`.
        - Output: :math:`(B, C_{out}, T, *)` where all but the last dimension are the
          same shape as the input and :math:`C_{out} = \text{out\_features}` if `channel_dim = 2`.
          If channel_dim = -1, the order of input dimensions is :math:`(*, C_{in})`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        # Use like torch.nn.Linear
        m = co.Linear(20, 30)
        input = torch.randn(128, 20)
        output = m(input)
        assert output.size() == torch.Size([128, 30])

        # Or in conjunction with other continual modules
        #                   B  C  T   H    W
        input = torch.randn(1, 3, 16, 128, 128)
        net = co.Sequential(
            co.Conv3d(3, 32, 3),
            co.AdaptiveAvgPool3d((1, 1, 1), 32),
            co.Linear(32, 10, channel_dim=1),
        )
        output = net(input)
        assert output.size() == torch.Size([1, 10, 1, 1, 1])

    """

    _state_shape = 0
    _dynamic_state_inds = []

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        channel_dim=-1,
    ) -> None:
        nn.Linear.__init__(self, in_features, out_features, bias, device, dtype)
        self.channel_dim = channel_dim

    def extra_repr(self):
        return nn.Linear.extra_repr(self) + f", channel_dim={self.channel_dim}"

    def forward(self, input: Tensor) -> Tensor:
        if self.channel_dim != -1:
            input = input.swapaxes(self.channel_dim, -1)

        output = linear(input, self.weight, self.bias)  # Assumes channel-last

        if self.channel_dim != -1:
            output = output.swapaxes(self.channel_dim, -1)

        return output

    def forward_step(self, input: Tensor, update_state=True) -> Tensor:
        return self.forward(input)

    def _forward_step(self, input: Tensor, prev_state=None) -> Tensor:
        return self.forward(input), prev_state

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True) -> Tensor:
        return self.forward(input)

    @staticmethod
    def build_from(
        module: nn.Linear,
        channel_dim=-1,
        **kwargs,
    ) -> "Linear":
        comodule = Linear(
            **{
                **dict(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                    channel_dim=channel_dim,
                ),
                **kwargs,
            }
        )
        with torch.no_grad():
            comodule.weight.copy_(module.weight)
            if module.bias is not None:
                comodule.bias.copy_(module.bias)
        return comodule
