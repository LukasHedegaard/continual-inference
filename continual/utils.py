from contextlib import contextmanager
from functools import reduce


@contextmanager
def temporary_parameter(obj, attr, val):
    prev_val = rgetattr(obj, attr)
    rsetattr(obj, attr, val)
    yield obj
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
