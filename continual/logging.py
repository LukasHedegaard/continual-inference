""" PyTorch Lightning compatible logging """

import logging
from functools import wraps
from typing import Callable

__all__ = ["getLogger"]


def _process_rank():
    try:  # pragma: no cover
        import pytorch_lightning as pl

        try:
            import horovod.torch as hvd

            hvd.init()
            return hvd.rank()

        except ModuleNotFoundError:
            return pl.utilities.rank_zero_only.rank

    except ModuleNotFoundError:
        return 0


process_rank = _process_rank()


def once(fn: Callable):
    mem = set()

    @wraps(fn)
    def wrapped(*args, **kwargs):
        h = hash((args, str(kwargs)))
        if h in mem:
            return
        mem.add(h)
        return fn(*args, **kwargs)

    return wrapped


def if_rank_zero(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        global process_rank
        if process_rank == 0:
            fn(*args, **kwargs)

    return wrapped


def getLogger(name, log_once=False):
    logger = logging.getLogger(name)
    if log_once:
        logger._log = once(logger._log)
    logger._log = if_rank_zero(logger._log)
    return logger
