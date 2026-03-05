from __future__ import annotations
from typing import List, Tuple, Dict, Union
from abc import ABC, abstractmethod
from multiprocessing import shared_memory

import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

from .memory import ShmMeta, make_shared

KELVIN_TO_MEV = 1.0/1.1604447522806e10


class EosProxy(ABC):
    # Zero-point shift
    _energy_shift: float
    _vars: List[str]

    # Local interpolator
    _table_interpolator: RegularGridInterpolator

    # Table bounds
    _min_rho: float
    _max_rho: float
    _min_temp: float
    _max_temp: float
    _min_ye: float
    _max_ye: float

    _shm_handles: List[shared_memory.SharedMemory]
    _shm_data: Dict[str, np.ndarray]

    def __init__(self, desc: Dict[str, Any]):
        self._energy_shift = desc['energy_shift']
        self._vars = desc['vars']

        self._min_rho = desc['min_rho']
        self._max_rho = desc['max_rho']
        self._min_temp = desc['min_temp']
        self._max_temp = desc['max_temp']
        self._min_ye = desc['min_ye']
        self._max_ye = desc['max_ye']

        self._shm_handles = []
        self._shm_data = {}

        for k,v in desc['data'].items():
            handle = shared_memory.SharedMemory(name=v.name)
            self._shm_data[k] = np.ndarray(v.shape, dtype=v.dtype, buffer=handle.buf)
            self._shm_handles.append(handle)

        # initalize interpolator
        self._table_interpolator = RegularGridInterpolator(
            (self._shm_data['ye'], self._shm_data['log_temp'], self._shm_data['log_rho']), self._shm_data['table'],
            method='linear', bounds_error=False, fill_value=None
        )


    def nuc_eos_zone(
        self,
        xrho: Union[float, np.ndarray],
        xtemp: Union[float, np.ndarray],
        xye: Union[float, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        logrho = np.log10(xrho)
        logtemp = np.log10(np.asarray(xtemp) * KELVIN_TO_MEV)
        ye = np.asarray(xye)

        zone_points = np.asarray([ye, logtemp, logrho]).T
        data = self._table_interpolator(zone_points)

        zones = {}
        zones['logrho'] = logrho[()]
        zones['logtemp'] = logtemp[()]
        zones['ye'] = ye[()]
        for i in range(len(self._vars)):
            zones[self._vars[i]] = data[:,i]

        return zones

    @property
    def field_list(self) -> list[str]:
        return self._vars

    @property
    def energy_shift(self) -> float:
        return self._energy_shift

    @property
    def minimum_density(self) -> float:
        return self._min_rho

    @property
    def minimum_temperature(self) -> float:
        return self._min_temp

    @property
    def minimum_ye(self) -> float:
        return self._min_ye

    @property
    def maximum_density(self) -> float:
        return self._max_rho

    @property
    def maximum_temperature(self) -> float:
        return self._max_temp

    @property
    def maximum_ye(self) -> float:
        return self._max_ye

    def close(self) -> None:
        for handle in self._shm_handles:
            handle.close()
        self._shm_handles.clear()

    def __del__(self):
        self.close()


# Also taken from https://github.com/gadjalin/flashy.git
class EosTable(EosProxy):
    # Grid info
    _log_rho: np.ndarray
    _log_temp: np.ndarray
    _ye: np.ndarray

    # Table fields list and indexing
    _table_data: np.ndarray

    # Shared memory
    _shm_handles: List[shared_memory.SharedMemory]
    _shm_data: Dict[str, np.ndarray]

    # Internal proxy
    _desc: Dict[str, Any]
    _proxy: EosProxy

    def __init__(self, filename: str):
        self._init_table(filename)
        self._setup_shm()

    def _init_table(self, filename: str) -> None:
        with h5py.File(filename, 'r') as f:
            # Get the table grid
            self._log_rho  = f['/logrho'][()]
            self._log_temp = f['/logtemp'][()]
            self._ye       = f['/ye'][()]

            # Table bounds
            self._min_rho  = 10**np.min(self._log_rho)
            self._max_rho  = 10**np.max(self._log_rho)
            self._min_temp = 10**np.min(self._log_temp)/KELVIN_TO_MEV
            self._max_temp = 10**np.max(self._log_temp)/KELVIN_TO_MEV
            self._min_ye   = np.min(self._ye)
            self._max_ye   = np.max(self._ye)

            # Get the actual tabulated quantities available in the table
            dataShape = (len(self._ye), len(self._log_temp), len(self._log_rho))
            self._vars = [k for k in list(f.keys()) if f[k].shape == dataShape]
            self._vars.sort()

            # Get EoS energy shift
            self._energy_shift = f['/energy_shift'][()][0]

            # load EoS data
            varData = np.array([f['/' + var][()] for var in self._vars])
            self._table_data = np.moveaxis(varData, 0, -1)

    def nuc_eos_zone(
        self,
        xrho: Union[float, np.ndarray],
        xtemp: Union[float, np.ndarray],
        xye: Union[float, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        return self._proxy.nuc_eos_zone(xrho, xtemp, xye)

    def get_proxy_descriptor(self):
        return self._desc

    def _setup_shm(self) -> None:
        self._shm_handles = []

        self._shm_data = {
            'log_rho' : self._to_shared(self._log_rho),
            'log_temp': self._to_shared(self._log_temp),
            'ye'      : self._to_shared(self._ye),
            'table'   : self._to_shared(self._table_data)
        }

        self._desc = {
            'energy_shift': self._energy_shift,
            'vars': self._vars,
            'min_rho': self._min_rho,
            'max_rho': self._max_rho,
            'min_temp': self._min_temp,
            'max_temp': self._max_temp,
            'min_ye': self._min_ye,
            'max_ye': self._max_ye,
            'data': self._shm_data
        }
        self._proxy = EosProxy(self._desc)

    def _to_shared(self, arr: np.ndarray) -> ShmMeta:
        shm, meta = make_shared(arr)
        self._shm_handles.append(shm)
        return meta

    def close(self) -> None:
        self._proxy.close()
        for handle in self._shm_handles:
            handle.close()
            handle.unlink()
        self._shm_handles.clear()

    def __del__(self):
        self.close()

