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
