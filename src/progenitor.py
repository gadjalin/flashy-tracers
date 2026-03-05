from __future__ import annotations
from typing import Tuple, List, Dict, Union, Optional
from abc import ABC, abstractmethod

import numpy as np

from .network import Nucleus

class Progenitor(ABC):
    _field_list: List[str]
    _network: List[Nucleus]

    def __init__(self):
        self._field_list = []
        self._network = []

    @abstractmethod
    def get_seed(self, r: float) -> Dict[Nucleus, float]:
        pass

    @property
    def field_list(self) -> List[str]:
        return self._field_list

    @property
    def network(self) -> List[Nucleus]:
        return self._network

