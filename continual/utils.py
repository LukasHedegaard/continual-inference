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


_FLATTEN_STATE_DICT: bool = False


class flat_state_dict(object):
    r"""Context-manager that flattens the state dict of containers.

    If a container module was not explicitely named by means of an OrderedDict,
    it will attempt to flatten the keys during both the `state_dict` and `load_state_dict` operations.
    """

    def __enter__(self):
        global _FLATTEN_STATE_DICT
        _FLATTEN_STATE_DICT = True

    def __exit__(self, *args, **kwargs):
        global _FLATTEN_STATE_DICT
        _FLATTEN_STATE_DICT = False
