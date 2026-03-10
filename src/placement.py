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
    domain = snap.get_field(('density', 'temperature'))

    mask = np.ones_like(domain, dtype=bool)
    if max_dens is not None:
        mask &= domain['density'] < max_dens

    # Weigh by cell size to cover uniform area in 2D independent of refinement
    domain = domain[mask]
    cell_size = domain['dx']
    if snap.dimensionality >= 2:
        cell_size *= domain['dy']
    if snap.dimensionality == 3:
        cell_size *= domain['dz']

    print(f'Sampling {n} tracers uniformly in space')

    # Ensure no more than one tracer per cell
    if n == len(domain):
        sample = np.full_like(domain, True)
    elif n < len(domain):
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
            sample = np.random.choice(len(domain), size=n, replace=False, p=prob)
    else:
        raise ValueError(f'Requesting more tracers than available cells: {n}/{len(domain)}')

    # Mass of the sampled region
    sample_mass = domain[sample]['density']*domain[sample]['volume']
    sample_total_mass = np.sum(sample_mass)

    dtypes = [('x', float), ('y', float), ('z', float), ('mass', float)]
    tracers = np.zeros(len(sample), dtype=dtypes)
    tracers['x'] = domain[sample]['x']
    tracers['y'] = domain[sample]['y']
    tracers['z'] = domain[sample]['z']
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
    domain = snap.get_field(('density', 'temperature'))

    mask = np.ones_like(domain, dtype=bool)
    if max_dens is not None:
        mask &= domain['density'] < max_dens

    domain = domain[mask]
    cell_mass = domain['density']*domain['volume']
    print(f'Sampling {n} tracers of uniform mass in {np.sum(cell_mass):.4e} [g]')

    prob = cell_mass/np.sum(cell_mass)
    # Allow placing multiple tracers in the same cell
    # Otherwise we would need to limit the tracer mass (and number) to the
    # heaviest cell
    sample = np.random.choice(len(domain), size=n, replace=True, p=prob)

    # Jitter particles away from cell centre, in particular those placed in the
    # same cell
    rng = np.random.default_rng(seed=42)
    x_jitter = rng.uniform(-domain[sample]['dx']/2., domain[sample]['dx']/2., size=len(sample))
    y_jitter = rng.uniform(-domain[sample]['dy']/2., domain[sample]['dy']/2., size=len(sample))
    z_jitter = rng.uniform(-domain[sample]['dz']/2., domain[sample]['dz']/2., size=len(sample))

    sample_total_mass = np.sum(domain[sample]['density']*domain[sample]['volume'])

    dtypes = [('x', float), ('y', float), ('z', float), ('mass', float)]
    tracers = np.zeros(len(sample), dtype=dtypes)
    tracers['x'] = domain[sample]['x'] + x_jitter
    tracers['y'] = domain[sample]['y'] + y_jitter
    tracers['z'] = domain[sample]['z'] + z_jitter
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
    domain = snap.get_field((
        'density', 'temperature', 'electron fraction',
        'energy', 'gravitational potential',
        'velocity-x', 'velocity-y', 'velocity-z'
    ))

    mask = np.ones_like(domain, dtype=bool)
    if max_dens is not None:
        mask &= domain['density'] < max_dens
    if max_temp is not None:
        mask &= domain['temperature'] < max_temp
    mask &= _unbound_mask(domain, eos)

    # Weigh by cell size to cover uniform area in 2D independent of refinement
    domain = domain[mask]

    print(f'Sampling {n} tracers in the unbound region ({len(domain)} cells)')

    cell_mass = domain['density']*domain['volume']
    cell_size = domain['dx']
    if snap.dimensionality >= 2:
        cell_size *= domain['dy']
    if snap.dimensionality == 3:
        cell_size *= domain['dz']

    print('Total unbound mass: ', np.sum(cell_mass))

    # Ensure no more than one tracer per cell
    if n == len(domain):
        sample = np.full_like(domain, True)
    elif n < len(domain):
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
            sample = np.random.choice(len(domain), size=n, replace=False, p=prob)
    else:
        raise ValueError(f'Requesting more tracers than available cells: {n}/{len(domain)}')

    # Mass of the sampled region
    sample_mass = domain[sample]['density']*domain[sample]['volume']
    sample_total_mass = np.sum(sample_mass)

    # Calculate each tracer mass and coordinates
    dtypes = [('x', float), ('y', float), ('z', float), ('mass', float)]
    tracers = np.zeros(len(sample), dtype=dtypes)
    tracers['x'] = domain[sample]['x']
    tracers['y'] = domain[sample]['y']
    tracers['z'] = domain[sample]['z']
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


def _unbound_mask(cells: np.ndarray, eos: EosTable) -> np.ndarray:
    xrho = cells['density']
    xtemp = np.full_like(xrho, eos.minimum_temperature)
    xye = cells['electron fraction']

    coldenergydensity = eos.nuc_eos_zone(xrho, xtemp, xye)['logenergy']

    r = np.sqrt(cells['x']**2 + cells['y']**2 + cells['z']**2)
    vrad = (cells['x']*cells['velocity-x'] + cells['y']*cells['velocity-y'] + cells['z']*cells['velocity-z']) / r
    xener = cells['energy']
    xgpot = cells['gravitational potential']
    vol = cells['volume']

    coldenergy = (10**(coldenergydensity) - eos.energy_shift)*xrho*vol
    dener = (xener - eos.energy_shift)*xrho*vol
    dgrav = xgpot*xrho*vol
    detot = dener + dgrav - coldenergy
    return (detot > 0.0) & (vrad > 0.0)

