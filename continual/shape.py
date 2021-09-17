from typing import Sequence, overload

from torch import Tensor, nn

from .module import CoModule


class Reshape(CoModule, nn.Module):
    """Reshape of non-temporal dimensions"""

    @overload
    def __init__(self, shape: Sequence[int]):
        ...  # pragma: no cover

    @overload
    def __init__(self, *shape: int):
        ...  # pragma: no cover

    def __init__(self, *shape):
        nn.Module.__init__(self)

        assert len(shape) > 0
        if isinstance(shape[0], int):
            self.shape = shape
        else:
            assert isinstance(shape[0], Sequence)
            assert isinstance(shape[0][0], int)
            self.shape = shape[0]

    def extra_repr(self):
        return ", ".join(str(s) for s in self.shape)

    def forward(self, input: Tensor) -> Tensor:
        T = input.shape[2]
        x = input.moveaxis(2, 0).reshape(T, *self.shape).moveaxis(0, 2)
        return x

    def forward_steps(self, input: Tensor, pad_end=False, update_state=True) -> Tensor:
        return self.forward(input)

    def forward_step(self, input: Tensor, update_state=True) -> Tensor:
        return input.reshape(self.shape)
