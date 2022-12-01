from typing import Dict, List

import torch

from continual import CoModule
from continual.utils import flatten


def _shape_list(lst, shape, idx=0):
    if isinstance(shape, int):
        return lst[idx : idx + shape], idx + shape

    assert hasattr(shape, "__len__")
    ret = []
    for s in shape:
        o, idx = _shape_list(lst, s, idx)
        ret.append(o)
    return ret, idx


class OnnxWrapper(torch.nn.Module):
    """Collapses input args and flattens output args.
    This is necessary as the ``dynamic_axes`` arg for
    :py:meth:`torch.onnx.export` doesn't accept nested Tuples.
    Args:
        model: A co.CoModule
    """

    def __init__(self, model: CoModule):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, *states: torch.Tensor):
        shaped_state, _ = _shape_list(states, self.model._state_shape)

        out, next_states = self.model._forward_step(x, shaped_state)
        return out, *flatten(next_states, remove_none=False)

    @staticmethod
    def _i2o_name(i_name: str) -> str:
        return f"n{i_name}"

    @property
    def state_input_names(self) -> List[str]:
        return [f"s{i}" for i in range(sum(flatten(self.model._state_shape)))]

    @property
    def state_output_names(self) -> List[str]:
        return [self._i2o_name(s) for s in self.state_input_names]

    @property
    def state_dynamic_axes(self) -> Dict[str, List[int]]:
        isdyn = flatten(self.model._dynamic_state_inds)
        ins = {sn: [0] for sn, i in zip(self.state_input_names, isdyn) if i}
        outs = {self._i2o_name(k): v for k, v in ins.items()}
        return {**ins, **outs}
