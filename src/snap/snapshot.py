from __future__ import annotations
from typing import Tuple, List, Union, Optional
from abc import ABC, abstractmethod
from enum import IntEnum

import numpy as np


SNAP_FIELDS = ['density', 'temperature', 'electron fraction', 'entropy',
               'velocity-x', 'velocity-y', 'velocity-z',
               'energy', 'gravitational potential']
SNAP_FIELDS_NU = ['lum nue', 'lum anue', 'lum nux', 'lum anux',
                  'ener nue', 'ener anue', 'ener nux', 'ener anux']

#class FIELD(IntEnum):
#    DENSITY                 = 0
#    TEMPERATURE             = 1
#    ELECTRON_FRACTION       = 2
#    ENTROPY                 = 3
#    VELOCITY_X              = 4
#    VELOCITY_Y              = 5
#    VELOCITY_Z              = 6
#    ENERGY                  = 7
#    GRAVITATIONAL_POTENTIAL = 8
#    LUM_NUE                 = 9
#    LUM_ANUE                = 10
#    LUM_NUX                 = 11
#    LUM_ANUX                = 12
#    ENER_NUE                = 13
#    ENER_ANUE               = 14
#    ENER_NUX                = 15
#    ENER_ANUX               = 16

SNAP_FIELD_MAP = {k: v for v,k in enumerate(SNAP_FIELDS + SNAP_FIELDS_NU)}


class SnapshotProxy(ABC):
    _field_list: List[str]
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

    def __contains__(self, key: str) -> bool:
        return key in self._field_list

    @property
    def field_list(self) -> List[str]:
        return self._field_list

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

