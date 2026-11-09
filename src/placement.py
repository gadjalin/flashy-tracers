from typing import Tuple, List, Dict, Union, Optional

import numpy as np

from .snap.snapshot import Snapshot
from .eos import EosTable


_COLUMNS = ['id', 'x', 'y', 'z', 'mass']
_FMTS = [' %-22d', ' % .15e', ' % .15e', ' % .15e', ' % .15e']
_DTYPE = [(c, np.float64) for c in _COLUMNS]


def from_file(filename: str) -> np.ndarray:
    tracers = np.atleast_1d(np.genfromtxt(filename, names=_COLUMNS))
    if len(tracers) == 0:
        raise RuntimeError(f'No tracers found in file {filename}')

    return tracers


def sample_uniform_space(
    n: int,
    snap: Snapshot,
    max_dens: Optional[float] = None,
    max_rad: Optional[float] = None
) -> np.ndarray:
    # Retrieve simulation domain
    grid, fields = snap.get_field(('density', 'temperature'))

    mask = np.ones_like(fields, dtype=bool)
    if max_dens is not None:
        mask &= fields['density'] < max_dens
    if max_rad is not None:
        r = np.sqrt(grid['x']**2 + grid['y']**2 + grid['z']**2)
        mask &= r < max_rad

    grid = grid[mask]
    fields = fields[mask]

    print(f'Sampling {n} tracers uniformly in space')

    cell_mass = fields['density']*grid['volume']
    cell_size = grid['dx'].copy()
    cell_size *= grid['dy'] if snap.dimensionality >= 2 else 1.0
    cell_size *= grid['dz'] if snap.dimensionality == 3 else 1.0

    print('Total mass: ', np.sum(cell_mass))

    occupation, sample = _sample_saturated(n, grid, cell_size)

    # Mass of the sampled region
    sample_mass = fields[sample]['density']*grid[sample]['volume']
    sample_total_mass = np.sum(sample_mass)

    #x_jitter, y_jitter, z_jitter = _calculate_jitter(grid, sample)

    tracers = np.empty(len(sample), dtype=_DTYPE)
    tracers['id'] = list(range(len(sample)))
    tracers['x'] = grid[sample]['x']# + x_jitter
    tracers['y'] = grid[sample]['y']# + y_jitter
    tracers['z'] = grid[sample]['z']# + z_jitter
    tracers['mass'] = sample_mass/occupation[sample]

    print('Total sampled mass: ', sample_total_mass)
    print('Total tracers mass: ', np.sum(tracers['mass']))
    print('Min/Max tracer mass: ', np.min(tracers['mass']), ' / ', np.max(tracers['mass']))

    return tracers


def sample_uniform_mass(
    n: int,
    snap: Snapshot,
    max_dens: Optional[float] = None,
    max_rad: Optional[float] = None
) -> np.ndarray:
    # Retrieve simulation domain
    grid, fields = snap.get_field(('density', 'temperature'))

    mask = np.ones_like(fields, dtype=bool)
    if max_dens is not None:
        mask &= fields['density'] < max_dens
    if max_rad is not None:
        r = np.sqrt(grid['x']**2 + grid['y']**2 + grid['z']**2)
        mask &= r < max_rad

    grid = grid[mask]
    fields = fields[mask]
    cell_mass = fields['density']*grid['volume']
    print(f'Sampling {n} tracers of uniform mass in {np.sum(cell_mass):.4e} [g]')

    prob = cell_mass/np.sum(cell_mass)
    # Allow placing multiple tracers in the same cell
    # Otherwise we would need to limit the tracer mass (and number) to the
    # heaviest cell
    sample = np.random.choice(len(grid), size=n, replace=True, p=prob)

    # Jitter particles away from cell centre, in particular those placed in the
    # same cell
    x_jitter, y_jitter, z_jitter = _calculate_jitter(grid, sample)

    sample_total_mass = np.sum(fields[sample]['density']*grid[sample]['volume'])

    tracers = np.empty(len(sample), dtype=_DTYPE)
    tracers['id'] = list(range(len(sample)))
    tracers['x'] = grid[sample]['x'] + x_jitter
    tracers['y'] = grid[sample]['y'] + y_jitter
    tracers['z'] = grid[sample]['z'] + z_jitter
    tracers['mass'] = np.sum(cell_mass)/n

    print('Total sampled mass: ', sample_total_mass)
    print('Sample tracer mass: ', tracers['mass'][0])
    print('Total tracers mass: ', np.sum(tracers['mass']))

    return tracers


def sample_unbound(
    n: int,
    snap: Snapshot,
    eos: EosTable,
    max_dens: Optional[float] = None,
    max_temp: Optional[float] = None
) -> np.ndarray:
    # Retrieve simulation domain
    grid, fields = snap.get_field((
        'density', 'temperature', 'electron fraction',
        'energy', 'gravitational potential',
        'velocity-x', 'velocity-y', 'velocity-z'
    ))

    mask = np.ones_like(fields, dtype=bool)
    if max_dens is not None:
        mask &= fields['density'] < max_dens
    if max_temp is not None:
        mask &= fields['temperature'] < max_temp
    mask &= _unbound_mask(grid, fields, eos)

    grid = grid[mask]
    fields = fields[mask]

    print(f'Sampling {n} tracers in the unbound region ({len(grid)} cells)')

    cell_mass = fields['density']*grid['volume']
    cell_size = grid['dx'].copy()
    cell_size *= grid['dy'] if snap.dimensionality >= 2 else 1.0
    cell_size *= grid['dz'] if snap.dimensionality == 3 else 1.0

    print('Total unbound mass: ', np.sum(cell_mass))

    occupation, sample = _sample_saturated(n, grid, cell_size)

    # Mass of the sampled region
    sample_mass = fields[sample]['density']*grid[sample]['volume']
    sample_total_mass = np.sum(sample_mass)

    # Jitter particles away from cell centre
    #x_jitter, y_jitter, z_jitter = _calculate_jitter(grid, sample)

    # Calculate each tracer mass and coordinates
    tracers = np.empty(len(sample), dtype=_DTYPE)
    tracers['id'] = list(range(len(sample)))
    tracers['x'] = grid[sample]['x']# + x_jitter
    tracers['y'] = grid[sample]['y']# + y_jitter
    tracers['z'] = grid[sample]['z']# + z_jitter
    tracers['mass'] = sample_mass/occupation[sample]

    print('Total sampled mass: ', sample_total_mass)
    print('Total tracers mass: ', np.sum(tracers['mass']))
    print('Min/Max tracer mass: ', np.min(tracers['mass']), ' / ', np.max(tracers['mass']))

    return tracers


def sample_user(
    n: int,
    snap: Snapshot,
    eos: EosTable,
    max_dens: Optional[float],
    max_temp: Optional[float],
    max_rad: Optional[float]
) -> np.ndarray:
    raise NotImplementedError('User placement method not implemented')


def save_tracers(file: str, tracers: np.ndarray) -> None:
    header = ''.join([col.ljust(24) for col in _COLUMNS]).strip()
    np.savetxt(file, tracers, fmt=_FMTS, header=header)


def _unbound_mask(grid: np.ndarray, fields: np.ndarray, eos: EosTable) -> np.ndarray:
    xrho = fields['density']
    xtemp = np.full_like(xrho, eos.minimum_temperature)
    xye = fields['electron fraction']

    coldenergydensity = eos.nuc_eos_zone(xrho, xtemp, xye)['logenergy']

    r = np.sqrt(grid['x']**2 + grid['y']**2 + grid['z']**2)
    vrad = (grid['x']*fields['velocity-x'] + grid['y']*fields['velocity-y'] + grid['z']*fields['velocity-z']) / r
    xener = fields['energy']
    xgpot = fields['gravitational potential']
    vol = grid['volume']

    coldenergy = (10**(coldenergydensity) - eos.energy_shift)*xrho*vol
    dener = (xener - eos.energy_shift)*xrho*vol
    dgrav = xgpot*xrho*vol
    detot = dener + dgrav - coldenergy
    return (detot > 0.0) & (vrad > 0.0)


def _sample_saturated(
    n: int,
    grid: np.ndarray,
    cell_size: np.ndarray,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng() if rng is None else rng

    # Ensure no more than one tracer per cell
    if n == len(grid):
        occupation = np.ones(len(grid))
        sample = np.arange(len(grid))
    elif n < len(grid):
        # Weigh by cell size/mass to ensure uniform spatial distribution
        prob = cell_size/np.sum(cell_size)

        # Compute occupation probability for each cell
        occupation = n*prob
        filled = occupation >= 1.
        num_filled = np.sum(filled)
        if num_filled > 0:
            while True: # Adjust distribution
                prob = cell_size[~filled]/np.sum(cell_size[~filled])
                occupation[filled] = 1.
                occupation[~filled] = (n-num_filled)*prob
                new_filled = occupation >= 1.
                if np.any(~filled & new_filled):
                    filled |= new_filled
                    num_filled = np.sum(filled)
                else:
                    break

        order = rng.permutation(len(grid))
        cumulative = np.cumsum(occupation[order])
        cumulative[-1] = n # absorb round-off so that the last point always lands
        sample = order[np.searchsorted(cumulative, rng.random() + np.arange(n), side='right')]

        if len(np.unique(sample)) != n:
            raise RuntimeError('Systematic sampling selected a cell twice')
    else:
        raise ValueError(f'Requesting more tracers than available cells: {n}/{len(grid)}')

    return occupation, sample


def _calculate_jitter(grid, sample):
    rng = np.random.default_rng(seed=42)
    x = rng.uniform(-grid[sample]['dx']/2., grid[sample]['dx']/2., size=len(sample))
    y = rng.uniform(-grid[sample]['dy']/2., grid[sample]['dy']/2., size=len(sample))
    z = rng.uniform(-grid[sample]['dz']/2., grid[sample]['dz']/2., size=len(sample))
    return x, y, z

