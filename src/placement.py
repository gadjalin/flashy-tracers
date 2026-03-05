from typing import Tuple, List, Dict, Union, Optional

import numpy as np

from .snap.snapshot import Snapshot
from .eos import EosTable

_MAX_DENS = 1e11
_MAX_TEMP = None


def init(max_dens: float, max_temp: Optional[float]) -> None:
    global _MAX_DENS
    global _MAX_TEMP

    _MAX_DENS = max_dens
    _MAX_TEMP = max_temp


def from_file(filename: str) -> np.ndarray:
    names = ['x', 'y', 'z', 'mass']
    tracers = np.genfromtxt(filanem, names=names)
    if len(tracers) == 0:
        raise RuntimeError(f'No tracers found in file {filename}')

    return tracers


def sample_uniform(n: int, snap: Snapshot) -> np.ndarray:
    # Retrieve simulation domain
    if _MAX_TEMP is not None:
        domain = snap.get_field(('density', 'temperature'))
        mask = (domain['density'] < _MAX_DENS) & (domain['temperature'] < _MAX_TEMP)
    else:
        domain = snap.get_field(('density'))
        mask = domain['density'] < _MAX_DENS

    # Weigh by cell size to cover uniform area in 2D independent of refinement
    domain = domain[mask]
    cell_size = domain['size']
    #prob = cell_size/np.sum(cell_size)
    #sample = np.random.choice(len(domain), size=n, replace=False, p=prob)
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


def sample_unbound(n: int, snap: Snapshot, eos: EosTable) -> np.ndarray:
    # Retrieve simulation domain
    domain = snap.get_field((
        'density', 'temperature', 'electron fraction',
        'energy', 'gravitational potential',
        'velocity-x', 'velocity-y', 'velocity-z'
    ))

    mask = domain['density'] < _MAX_DENS
    if _MAX_TEMP is not None:
        mask &= domain['temperature'] < _MAX_TEMP
    mask &= _unbound_mask(domain, eos)

    # Weigh by cell size to cover uniform area in 2D independent of refinement
    domain = domain[mask]

    print(f'Sampling {n} tracers in the unbound region ({len(domain)} cells)')

    cell_mass = domain['density']*domain['volume']
    cell_size = domain['size']

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

