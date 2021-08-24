from collections import OrderedDict
from contextlib import contextmanager
from functools import reduce

from torch import Tensor, nn


@contextmanager
def temporary_parameter(obj, attr: str, val):
    do_del = False
    try:
        prev_val = rgetattr(obj, attr)
    except AttributeError:
        do_del = True
        assert (
            attr.count(".") == 0
        ), "Nonexisting attributes can only be set one level deep."

    rsetattr(obj, attr, val)

    yield obj

    if do_del:
        delattr(obj, attr)
    else:
        rsetattr(obj, attr, prev_val)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split("."))


class _FlatStateDict(object):
    """Context-manager state holder."""

    def __init__(self):
        self.flatten = False

    def __enter__(self):
        self.flatten = True

    def __exit__(self, *args, **kwargs):
        self.flatten = False


flat_state_dict = _FlatStateDict()
"""Context-manager that flattens the state dict of containers.

If a container module was not explicitely named by means of an OrderedDict,
it will attempt to flatten the keys during both the `state_dict` and `load_state_dict` operations.

Example:
    >>> with co.flat_state_dict:
    >>>     sd = module.state_dict()  # All unnamed nested keys are flattened, e.g. "0.weight" -> "weight"
    >>>     module.load_state_dict(sd)  # Automatically unflattened during loading "weight" -> "0.weight"
"""


def state_dict(
    module: nn.Module, destination=None, prefix="", keep_vars=False, flatten=False
) -> "OrderedDict[str, Tensor]":
    """Returns a dictionary containing a whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.

    Args:
        destination (OrderedDict, optional): a dict to which the state is saved.
        prefix (str, optional): A string prefix for state keys.
        keep_vars (bool, optional): Whether parameters should not be detached.
        flatten (bool, optional): whether the state dict keys are flattened (no numeric keys).

    Returns:
        dict:
            a dictionary containing a whole state of the module

    Example::

        >>> module.state_dict().keys()
        ['bias', 'weight']

    """
    d = nn.Module.state_dict(module, destination, prefix, keep_vars)

    if flatten or flat_state_dict.flatten:
        flat_keys = [
            ".".join(part for part in name.split(".") if not part.isdigit())
            for name in list(d.keys())
        ]
        if len(set(flat_keys)) == len(d.keys()):
            d = OrderedDict(list(zip(flat_keys, d.values())))

    return d


def load_state_dict(
    module: nn.Module,
    state_dict: "OrderedDict[str, Tensor]",
    strict: bool = True,
    flatten=False,
):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        flatten (bool, optional): whether the loaded state dict is flattened (no numeric keys) during loading.


    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys
    """

    if flatten or flat_state_dict.flatten:
        long_keys = nn.Module.state_dict(module, keep_vars=True).keys()
        short2long = {
            ".".join(part for part in key.split(".") if not part.isdigit()): key
            for key in list(long_keys)
        }
        state_dict = OrderedDict(
            [(short2long[key], val) for key, val in state_dict.items()]
        )

    return nn.Module.load_state_dict(module, state_dict, strict)
