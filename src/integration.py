from typing import Dict, List, Tuple, Type, Any, Optional
import sys
import queue

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

from .snapshot import SnapshotProxy, Snapshot
from .eos import EosProxy


TQDM_FORMAT = '{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'


def loader_thread(filenames: List[str], load_nu: bool, queue: queue.Queue, SnapshotCls: Snapshot) -> None:
    with tqdm(total=len(filenames), bar_format=TQDM_FORMAT, file=sys.stdout) as pbar:
        for file in filenames:
            snap = SnapshotCls(file, use_nu=load_nu)
            queue.put(snap)
            pbar.update()
    print(f'File loader: reached EOL')
    queue.put(None) # Sentinel


def worker_process(
    start_desc: Dict[str, Any],
    end_desc: Dict[str, Any],
    identifier: int,
    start_pos: np.ndarray,
    exports: Dict[str, Any]
) -> Tuple[int, np.ndarray, np.ndarray]:
    start_proxy = exports['ProxyCls'](start_desc)
    end_proxy = exports['ProxyCls'](end_desc)

    end_pos = None
    state = None

    try:
        end_pos, state = integrate_tracer(start_proxy, end_proxy, start_pos, exports)
    except Exception as e:
        print('An error occured: ', e)
        sys.stdout.flush()
    finally:
        start_proxy.close()
        end_proxy.close()

    # If reached MAX_TEMP do not integrate this tracer further
    if exports['max_temp'] is not None and state is not None:
        if state['temperature']*1e9 > exports['max_temp']:
            end_pos = None

    return (identifier, end_pos, state)


def initial_state_process(
    desc: Dict[str, Any],
    identifier: int,
    pos: np.ndarray,
    exports: Dict[str, Any]
) -> Tuple[int, np.ndarray]:
    proxy = exports['ProxyCls'](desc)

    state = None
    try:
        state = save_state(pos, proxy, exports['output_vars'], exports['eos_desc'])
    finally:
        proxy.close()

    return (identifier, state)


def integrate_tracer(
    start_snap: SnapshotProxy,
    end_snap: SnapshotProxy,
    start_pos: np.ndarray,
    exports: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    t0 = start_snap.current_time
    t1 = end_snap.current_time

    def velocity(t: float, pos: List[float]) -> List[float]:
        v0 = start_snap.get_quantity(('velocity-x', 'velocity-y', 'velocity-z'), *pos)
        v1 = end_snap.get_quantity(('velocity-x', 'velocity-y', 'velocity-z'), *pos)
        # Linearly interpolate velocity
        return v0 + (v1 - v0) * (t - t0)/(t1 - t0)

    def out_of_bounds(t: float, pos: List[float]) -> float:
        if end_snap.dimensionality == 1:
            return min(
                pos[0] - end_snap.xmin, end_snap.xmax - pos[0],
            )
        elif end_snap.dimensionality == 2:
            return min(
                pos[0] - end_snap.xmin, end_snap.xmax - pos[0],
                pos[1] - end_snap.ymin, end_snap.ymax - pos[1],
            )
        else:
            return min(
                pos[0] - end_snap.xmin, end_snap.xmax - pos[0],
                pos[1] - end_snap.ymin, end_snap.ymax - pos[1],
                pos[2] - end_snap.zmin, end_snap.zmax - pos[2],
            )

    out_of_bounds.terminal = True
    out_of_bounds.direction = -1

    solution = solve_ivp(
        velocity,
        t_span=[t0, t1],
        y0=list(start_pos),
        rtol=exports['rtol'],
        atol=exports['atol'],
        max_step=exports['max_step'],
        method='RK45',
        dense_output=True,
        first_step=exports['init_step'],
        events=out_of_bounds
    )

    # Tracer reached domain's boundaries
    if solution.t_events[0].size > 0:
        raise RuntimeError(f'Tracer at {start_pos[0]}, {start_pos[1]}, {start_pos[2]} went out of bound')

    end_pos = solution.sol(t1)
    return end_pos, save_state(end_pos, end_snap, exports['output_vars'], exports['eos_desc'])


def save_state(
    pos: np.ndarray,
    snap: SnapshotProxy,
    output_vars: List[str],
    eos_desc: Dict[str, Any]
) -> np.ndarray:
    dtype = [(var, float) for var in output_vars]
    state = np.zeros((), dtype=dtype)
    snap_vars = []

    eos = None
    if 'ejected' in output_vars:
        eos = EosProxy(eos_desc)

    # Treat special variables
    for var in output_vars:
        if var == 'time':
            state[var] = snap.current_time
        elif var == 'x':
            state[var] = pos[0]/1e5 # cm to km
        elif var == 'y':
            state[var] = pos[1]/1e5 # cm to km
        elif var == 'z':
            state[var] = pos[2]/1e5 # cm to km
        elif var == 'r':
            state[var] = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)/1e5 # cm to km
        elif var == 'ejected':
            state[var] = 1 if is_unbound(pos, snap, eos) else 0
        else:
            snap_vars.append(var)

    # Remaining variables can be obtained directly by interpolation
    rem = snap.get_quantity(snap_vars, *pos)
    for i,var in enumerate(snap_vars):
        if var == 'temperature':
            state[var] = rem[i]/1e9 # K to GK
        else:
            state[var] = rem[i]

    return state


def is_unbound(pos: np.ndarray, snap: SnapshotProxy, eos: EosProxy) -> bool:
    var = [
        'volume', 'density', 'electron fraction',
        'velocity-x', 'velocity-y', 'velocity-z',
        'energy', 'gravitational potential'
    ]
    cell = snap.get_quantity(var, *pos)
    xrho = cell[var.index('density')]
    xtemp = eos.minimum_temperature
    xye = cell[var.index('electron fraction')]

    coldenergydensity = eos.nuc_eos_zone(xrho, xtemp, xye)['logenergy']

    r = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    vrad = (pos[0]*cell[var.index('velocity-x')] + pos[1]*cell[var.index('velocity-y')] + pos[2]*cell[var.index('velocity-z')]) / r
    xener = cell[var.index('energy')]
    xgpot = cell[var.index('gravitational potential')]
    vol = cell[var.index('volume')]

    coldenergy = (10**(coldenergydensity) - eos.energy_shift)*xrho*vol
    dener = (xener - eos.energy_shift)*xrho*vol
    dgrav = xgpot*xrho*vol
    detot = dener + dgrav - coldenergy
    return (detot > 0.0) & (vrad > 0.0)

