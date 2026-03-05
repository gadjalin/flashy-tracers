from __future__ import annotations
from typing import List, Dict

import numpy as np

from .progenitor import Progenitor
from ..network import find_isotope, Nucleus

class FLASHProgenitor(Progenitor):
    _data: np.ndarray
    _network: List[Nucleus]

    def __init__(self, filename: str):
        self._read_model1d(filename)
        self._detect_network()

    def get_seed(self, r: float) -> Dict[Nucleus, float]:
        res = {}
        for nuc in self._network:
            res[nuc] = np.interp(r, self._data['r'], self._data[nuc.name])
        return res

#    def get_seed_m(self, m: float) -> Dict[Nucleus, float]:
#        r = np.interp(m, self._field_list['mass'], self._field_list['r'])
#        return self.get_seed_r(r)

    def _read_model1d(self, filename: str) -> None:
        data_start = 0
        with open(filename, 'r') as f:
            line = f.readline().strip()
            # Skip comment on first line if any
            if line.startswith('#'):
                data_start += 1
                line = f.readline().strip()
            # Read "number of variables" line
            num_vars = int(line.split()[-1])
            data_start += num_vars + 1

            # Read variables names
            var_names = ['r']
            var_names += [f.readline().split()[0].strip() for i in range(num_vars)]

        # Read columns
        self._data = np.genfromtxt(
            fname=filename,
            skip_header=data_start,
            names=var_names,
            dtype=None,
            encoding='ascii'
        )
        
        self._field_list = self._data.dtype.names

    def _detect_network(self) -> None:
        self._network = []
        for field in self._field_list:
            try:
                species = find_isotope(field)
                self._network.append(species)
            except ValueError:
                continue

