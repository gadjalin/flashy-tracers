from typing import Dict, List, Tuple, Type
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

TQDM_FORMAT = '{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'
MAX_WORKERS = 4

# Available output variables
_UNITS = {
    'time': 's',
    'x': 'km',
    'y': 'km',
    'z': 'km',
    'r': 'km',
    'density': 'g/cm3',
    'temperature': 'GK',
    'electron fraction': None,
    'entropy': 'kb/baryon',
    'lum nue': 'erg/s',
    'lum anue': 'erg/s',
    'lum nux': 'erg/s',
    'lum anux': 'erg/s',
    'ener nue': 'MeV',
    'ener anue': 'MeV',
    'ener nux': 'MeV',
    'ener anux': 'MeV',
    'ejected': None,
}


#def init(n_workers: int) -> None:
#    global N_WORKERS
#    N_WORKERS = min(n_workers, 8)


class StateBuffer(object):
    _states: List[np.ndarray]
    _vars: List[str]
    _dtype: List[Tuple[str, Type]]
    _len: int
    _output_dir: str

    def __init__(self, length: int, state_vars: List[str], output_dir: str):
        self._vars = state_vars
        self._len = length
        self._output_dir = output_dir

        # Overwrite tracer files if exist
        n_digits = int(np.log10(max(self._len - 1, 1))) + 1
        tracer_files = [os.path.join(self._output_dir, 'tracer' + f'{i}'.zfill(n_digits) + '.dat') for i in range(self._len)]
        header = [f'{var} [{_UNITS[var]}]' if _UNITS[var] is not None else f'{var}' for var in self._vars]
        header = '# ' + '        '.join(header) + '\n'

        def init_tracer_file(file: str, header: str) -> None:
            with open(file, 'w') as f:
                f.write(header)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(init_tracer_file, file, header) for file in tracer_files]
            with tqdm(total=len(futures), bar_format=TQDM_FORMAT, file=sys.stdout) as pbar:
                for _ in as_completed(futures):
                    pbar.update()

        self._dtype = [(var, float) for var in state_vars]
        self._states = [np.empty(0, dtype=self._dtype) for _ in range(self._len)]

    def append(self, index: int, new_state: np.ndarray) -> None:
        self._states[index] = np.append(self._states[index], new_state)

    # Assuming all entries have the same size at this point
    def sizeof(self) -> int:
        return self._len*self._states[0].nbytes

    def flush(self) -> None:
        n_digits = int(np.log10(max(self._len - 1, 1))) + 1
        tracer_files = [os.path.join(self._output_dir, 'tracer' + f'{i}'.zfill(n_digits) + '.dat') for i in range(len(self._states))]

        def append_tracer_file(file: str, data: np.ndarray) -> None:
            if len(data) > 0:
                with open(file, 'a') as f:
                    np.savetxt(f, data)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(append_tracer_file, file, state) for file,state in zip(tracer_files, self._states)]
            with tqdm(total=len(futures), bar_format=TQDM_FORMAT, file=sys.stdout) as pbar:
                for _ in as_completed(futures):
                    pbar.update()

        for i in range(len(self._states)):
            self._states[i] = np.empty(0, dtype=self._dtype)

    def delete_failed_output(self, index: List[int]) -> None:
        n_digits = int(np.log10(max(self._len - 1, 1))) + 1
        tracer_files = [os.path.join(self._output_dir, 'tracer' + f'{i}'.zfill(n_digits) + '.dat') for i in index]

        def remove_tracer_file(filename: str) -> None:
            if os.path.exists(filename):
                os.remove(filename)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            pool.map(remove_tracer_file, tracer_files)

    def reverse_output(self) -> None:
        n_digits = int(np.log10(max(self._len - 1, 1))) + 1
        tracer_files = [os.path.join(self._output_dir, 'tracer' + f'{i}'.zfill(n_digits) + '.dat') for i in range(self._len)]
        header = [f'{var} [{_UNITS[var]}]' if _UNITS[var] is not None else f'{var}' for var in self._vars]
        header = '        '.join(header)

        def reverse_tracer_file(file: str, header: str) -> None:
            history = np.genfromtxt(file)
            np.savetxt(file, history[::-1], header=header)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(reverse_tracer_file, file, header) for file in tracer_files]
            with tqdm(total=len(futures), bar_format=TQDM_FORMAT, file=sys.stdout) as pbar:
                for _ in as_completed(futures):
                    pbar.update()

