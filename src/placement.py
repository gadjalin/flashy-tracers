from typing import Tuple, List, Dict, Union, Optional

import numpy as np

from .snap.snapshot import Snapshot
from .eos import EosTable


def from_file(filename: str) -> np.ndarray:
    names = ['x', 'y', 'z', 'mass']
    tracers = np.genfromtxt(filanem, names=names)
    if len(tracers) == 0:
        raise RuntimeError(f'No tracers found in file {filename}')

    return tracers


def sample_uniform_space(
    n: int,
    snap: Snapshot,
    max_dens: Optional[float] = None
) -> np.ndarray:
    # Retrieve simulation domain
    grid, fields = snap.get_field(('density', 'temperature'))

    mask = np.ones_like(fields, dtype=bool)
    if max_dens is not None:
        mask &= fields['density'] < max_dens

    # Weigh by cell size to cover uniform area in 2D independent of refinement
    grid = grid[mask]
    fields = fields[mask]
    cell_size = grid['dx']
    if snap.dimensionality >= 2:
        cell_size *= grid['dy']
    if snap.dimensionality == 3:
        cell_size *= grid['dz']

    print(f'Sampling {n} tracers uniformly in space')

    # Ensure no more than one tracer per cell
    if n == len(grid):
        sample = np.full_like(grid, True)
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

            sample = np.random.choice(np.where(~filled)[0], size=(n-num_filled), replace=False, p=prob)
            sample = np.concatenate((np.where(filled)[0], sample))
        else:
            sample = np.random.choice(len(grid), size=n, replace=False, p=prob)
    else:
        raise ValueError(f'Requesting more tracers than available cells: {n}/{len(grid)}')

    # Mass of the sampled region
    sample_mass = fields[sample]['density']*grid[sample]['volume']
    sample_total_mass = np.sum(sample_mass)

    dtypes = [('x', float), ('y', float), ('z', float), ('mass', float)]
    tracers = np.zeros(len(sample), dtype=dtypes)
    tracers['x'] = grid[sample]['x']
    tracers['y'] = grid[sample]['y']
    tracers['z'] = grid[sample]['z']
    tracers['mass'] = sample_mass/occupation[sample]

    print('Total sampled mass: ', sample_total_mass)
    print('Total tracers mass: ', np.sum(tracers['mass']))
    print('Min tracer mass: ', np.min(tracers['mass']))
    print('Max tracer mass: ', np.max(tracers['mass']))

    return tracers


def sample_uniform_mass(
    n: int,
    snap: Snapshot,
    max_dens: Optional[float] = None
) -> np.ndarray:
    # Retrieve simulation domain
    grid, fields = snap.get_field(('density', 'temperature'))

    mask = np.ones_like(fields, dtype=bool)
    if max_dens is not None:
        mask &= fields['density'] < max_dens

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
    rng = np.random.default_rng(seed=42)
    x_jitter = rng.uniform(-grid[sample]['dx']/2., grid[sample]['dx']/2., size=len(sample))
    y_jitter = rng.uniform(-grid[sample]['dy']/2., grid[sample]['dy']/2., size=len(sample))
    z_jitter = rng.uniform(-grid[sample]['dz']/2., grid[sample]['dz']/2., size=len(sample))

    sample_total_mass = np.sum(fields[sample]['density']*grid[sample]['volume'])

    dtypes = [('x', float), ('y', float), ('z', float), ('mass', float)]
    tracers = np.zeros(len(sample), dtype=dtypes)
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

    # Weigh by cell size to cover uniform area in 2D independent of refinement
    grid = grid[mask]
    fields = fields[mask]

    print(f'Sampling {n} tracers in the unbound region ({len(grid)} cells)')

    cell_mass = fields['density']*grid['volume']
    cell_size = grid['dx'].copy()
    cell_size *= grid['dy'] if snap.dimensionality >= 2 else 1.0
    cell_size *= grid['dz'] if snap.dimensionality == 3 else 1.0

    print('Total unbound mass: ', np.sum(cell_mass))

    # Ensure no more than one tracer per cell
    if n == len(grid):
        sample = np.full_like(grid, True)
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

            sample = np.random.choice(np.where(~filled)[0], size=(n-num_filled), replace=False, p=prob)
            sample = np.concatenate((np.where(filled)[0], sample))
        else:
            sample = np.random.choice(len(grid), size=n, replace=False, p=prob)
    else:
        raise ValueError(f'Requesting more tracers than available cells: {n}/{len(grid)}')

    # Mass of the sampled region
    sample_mass = fields[sample]['density']*grid[sample]['volume']
    sample_total_mass = np.sum(sample_mass)

    # Jitter particles away from cell centre, in particular those placed in the
    # same cell
    rng = np.random.default_rng(seed=42)
    x_jitter = rng.uniform(-grid[sample]['dx']/2., grid[sample]['dx']/2., size=len(sample))
    y_jitter = rng.uniform(-grid[sample]['dy']/2., grid[sample]['dy']/2., size=len(sample))
    z_jitter = rng.uniform(-grid[sample]['dz']/2., grid[sample]['dz']/2., size=len(sample))

    # Calculate each tracer mass and coordinates
    dtypes = [('x', float), ('y', float), ('z', float), ('mass', float)]
    tracers = np.zeros(len(sample), dtype=dtypes)
    tracers['x'] = grid[sample]['x'] + x_jitter
    tracers['y'] = grid[sample]['y'] + y_jitter
    tracers['z'] = grid[sample]['z'] + z_jitter
    tracers['mass'] = sample_mass/occupation[sample]

    print('Total sampled mass: ', sample_total_mass)
    print('Total tracers mass: ', np.sum(tracers['mass']))
    print('Min tracer mass: ', np.min(tracers['mass']))
    print('Max tracer mass: ', np.max(tracers['mass']))

    return tracers


def sample_user(
    n: int,
    snap: Snapshot,
    eos: EosTable,
    max_dens: Optional[float],
    max_temp: Optional[float]
) -> np.ndarray:
    raise NotImplementedError('User placement method not implemented')


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

