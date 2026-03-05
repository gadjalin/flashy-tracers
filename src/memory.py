from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np


@dataclass
class ShmMeta(object):
    name: str
    shape: Tuple[int, ...]
    dtype: object


def make_shared(arr: np.ndarray):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    buf[:] = arr[:]

    return shm, ShmMeta(name=shm.name, shape=arr.shape, dtype=arr.dtype)


