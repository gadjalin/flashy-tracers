from __future__ import annotations
from typing import Tuple, List, Union, Optional
from abc import ABC, abstractmethod

import numpy as np


SNAP_FIELDS = ['density', 'temperature', 'electron fraction', 'entropy',
               'velocity-x', 'velocity-y', 'velocity-z',
               'energy', 'gravitational potential']
SNAP_FIELDS_NU = ['lum nue', 'lum anue', 'lum nux', 'lum anux',
                  'ener nue', 'ener anue', 'ener nux', 'ener anux']


class SnapshotProxy(ABC):
    _current_time: float
    _dim: int

    _xmin: float
    _xmax: float
    _ymin: float
    _ymax: float
    _zmin: float
    _zmax: float

    def __init__(self):
        pass

    @abstractmethod
    def get_quantity(
        self,
        fields: Union[List[str], str],
        x: float,
        y: Optional[float],
        z: Optional[float]
    ) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def get_field(self, fields: Union[List[str], Tuple[str], str]) -> np.ndarray:
        pass

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def dimensionality(self) -> int:
        return self._dim

    @property
    def xmin(self) -> float:
        return self._xmin

    @property
    def xmax(self) -> float:
        return self._xmax

    @property
    def ymin(self) -> float:
        return self._ymin

    @property
    def ymax(self) -> float:
        return self._ymax

    @property
    def zmin(self) -> float:
        return self._zmin

    @property
    def zmax(self) -> float:
        return self._zmax


class Snapshot(SnapshotProxy):
    def __init__(self):
        pass

    @abstractmethod
    def get_proxy_descriptor(self):
        pass

