from __future__ import annotations
from typing import Tuple, Sequence, List, Dict, Union, Optional, Any
from dataclasses import dataclass
from multiprocessing import shared_memory
from math import floor
import itertools

import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

from .snapshot import Snapshot, SnapshotProxy, SNAP_FIELDS, SNAP_FIELDS_NU, SNAP_FIELD_MAP
from ..memory import ShmMeta, make_shared
from ..interp import interp1d, interp2d, interp3d


FLASH_FIELD_MAP = {
    'dens': 'density',
    'temp': 'temperature',
    'ye'  : 'electron fraction',
    'entr': 'entropy',
    'velx': 'velocity-x',
    'vely': 'velocity-y',
    'velz': 'velocity-z',
    'ener': 'energy',
    'gpot': 'gravitational potential',
}
_FLASH_FIELDS = ['dens', 'temp', 'ye', 'entr', 'velx', 'vely', 'velz', 'ener', 'gpot']
_FLASH_NU_FIELDS = ['enue', 'enua', 'enux', 'fnue', 'fnua', 'fnux']


class FLASHSnapshotProxy(SnapshotProxy):
    # Grid geometry
    _lrefine_max: int
    _nblockx: int
    _nblocky: int
    _nblockz: int
    _nxb: int
    _nyb: int
    _nzb: int
    _ngx: int
    _nyx: int
    _nzx: int

    # Grid data
    _x: np.ndarray
    _y: np.ndarray
    _z: np.ndarray
    _dx: np.ndarray
    _dy: np.ndarray
    _dz: np.ndarray
    _vol: np.ndarray
    _bbox: np.ndarray

    # Physical quantities data
    _unk: np.ndarray

    # Block structure
    _dxb: np.ndarray
    _dyb: np.ndarray
    _dzb: np.ndarray
    _hash_map: Dict[Tuple[int, int, int, int], int]

    # Shared memory
    _shm_handles: List[shared_memory.SharedMemory]

    def __init__(self, desc: Dict[str, Any]):
        def push_shared_memory(meta: ShmMeta) -> np.ndarray:
            handle = shared_memory.SharedMemory(name=meta.name)
            self._shm_handles.append(handle)
            return np.ndarray(meta.shape, dtype=meta.dtype, buffer=handle.buf)

        self._field_list = desc['field_list']
        self._current_time = desc['current_time']
        self._dim = desc['dim']
        self._geometry = desc['geometry']

        self._xmin = desc['xmin']
        self._xmax = desc['xmax']
        self._ymin = desc['ymin']
        self._ymax = desc['ymax']
        self._zmin = desc['zmin']
        self._zmax = desc['zmax']

        self._lrefine_max = desc['lrefine_max']
        self._nblockx = desc['nblockx']
        self._nblocky = desc['nblocky']
        self._nblockz = desc['nblockz']
        self._nxb = desc['nxb']
        self._nyb = desc['nyb']
        self._nzb = desc['nzb']
        self._ngx = desc['ngx']
        self._ngy = desc['ngy']
        self._ngz = desc['ngz']

        self._shm_handles = []
        self._hash_map = desc['hash_map']

        self._dxb = desc['dxb']
        self._dyb = desc['dyb']
        self._dzb = desc['dzb']

        self._x      = push_shared_memory(desc['grid']['x'])
        self._y      = push_shared_memory(desc['grid']['y'])
        self._z      = push_shared_memory(desc['grid']['z'])
        self._dx     = push_shared_memory(desc['grid']['dx'])
        self._dy     = push_shared_memory(desc['grid']['dy'])
        self._dz     = push_shared_memory(desc['grid']['dz'])
        self._vol    = push_shared_memory(desc['grid']['volume'])
        self._bbox   = push_shared_memory(desc['grid']['bbox'])

        self._unk = push_shared_memory(desc['fields'])

    def get_quantity(
        self,
        fields: Sequence[str],
        x: float,
        y: Optional[float] = None,
        z: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        if isinstance(fields, str):
            fields = [fields]
        elif isinstance(fields, (Sequence, np.ndarray)) and all(isinstance(field, str) for field in fields):
            fields = list(fields)
        else:
            raise TypeError('Field must be a string or list of string')

        ifields = [SNAP_FIELD_MAP[field] for field in fields]
        return self._interp_block(ifields, x, y, z)

    def get_field(
        self,
        fields: Sequence[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(fields, str):
            fields = [fields]

        nblk = self._bbox.shape[0]
        ncells = self._nzb*self._nyb*self._nxb

        ifields = [SNAP_FIELD_MAP[field] for field in fields]
        ixstart, ixend = self._ngx, self._nxb + self._ngx
        iystart, iyend = self._ngy, self._nyb + self._ngy
        izstart, izend = self._ngz, self._nzb + self._ngz

        grid_dtype = [('x', float), ('y', float), ('z', float), ('dx', float), ('dy', float), ('dz', float), ('volume', float)]
        field_dtype = [(field, float) for field in fields]
        grid = np.empty(nblk*ncells, dtype=grid_dtype)
        unk = np.empty(nblk*ncells, dtype=field_dtype)
        grid['x'][:] = np.broadcast_to(self._x[:,None,None,ixstart:ixend], (nblk, self._nzb, self._nyb, self._nxb)).reshape(-1)
        grid['y'][:] = np.broadcast_to(self._y[:,None,iystart:iyend,None], (nblk, self._nzb, self._nyb, self._nxb)).reshape(-1)
        grid['z'][:] = np.broadcast_to(self._z[:,izstart:izend,None,None], (nblk, self._nzb, self._nyb, self._nxb)).reshape(-1)
        grid['dx'][:] = np.repeat(self._dx[:], ncells)
        grid['dy'][:] = np.repeat(self._dy[:], ncells)
        grid['dz'][:] = np.repeat(self._dz[:], ncells)
        grid['volume'][:] = self._vol[:,izstart:izend,iystart:iyend,ixstart:ixend].ravel()
        for ifield,field in zip(ifields, fields):
            unk[field][:] = self._unk[ifield,:,izstart:izend,iystart:iyend,ixstart:ixend].ravel()

        return grid, unk

    def find_block(
        self,
        x: float,
        y: Optional[float] = None,
        z: Optional[float] = None
    ) -> int:
        if (self._dim >= 2 and y is None) or (self._dim == 3 and z is None):
            raise ValueError(f'Invalid coordinates for {self._dim}D block lookup: x = {x}, y = {y}, z = {z}')

        for level in range(self._lrefine_max, 0, -1):
            ix = int((x - self._xmin)//self._dxb[level-1])
            iy = int((y - self._ymin)//self._dyb[level-1]) if self._dim >= 2 else 0
            iz = int((z - self._zmin)//self._dzb[level-1]) if self._dim == 3 else 0

            key = (level, ix, iy, iz)
            if key in self._hash_map:
                return self._hash_map[key]
        else:
            raise RuntimeError(f'No block found at ({x:.5e}, {y if y is not None else np.nan:.5e}, {z if z is not None else np.nan:.5e})')

    def _interp_block(
        self,
        fields: List[int],
        x: float,
        y: Optional[float] = None,
        z: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        blockID = self.find_block(x, y, z)
        xbmin = self._bbox[blockID,0,0]
        dx = self._dx[blockID]

        ix = self._ngx + int((x - (xbmin + dx*0.5)) // dx)
        x0 = self._x[blockID,ix]
        x1 = x0 + dx

        q = np.empty(len(fields))
        if self._dim == 1:
            for i,field in enumerate(fields):
                f = self._stencil_1d(field, blockID, ix, ix+1)
                q[i] = interp1d(x, x0, x1, f)
        elif self._dim == 2:
            if y is None:
                raise ValueError('Y coordinate must be specified in 2D')

            ybmin = self._bbox[blockID,1,0]
            dy = self._dy[blockID]

            iy = self._ngy + int((y - (ybmin + dy*0.5)) // dy)
            y0 = self._y[blockID,iy]
            y1 = y0 + dy

            for i,field in enumerate(fields):
                f = self._stencil_2d(field, blockID, ix, ix+1, iy, iy+1)
                q[i] = interp2d(x, y, x0, x1, y0, y1, f)
        elif self._dim == 3:
            if y is None or z is None:
                raise ValueError('Y and Z coordinates must be specified in 3D')

            ybmin = self._bbox[blockID,1,0]
            zbmin = self._bbox[blockID,2,0]
            dy = self._dy[blockID]
            dz = self._dz[blockID]

            iy = self._ngy + int((y - (ybmin + dy*0.5)) // dy)
            iz = self._ngz + int((z - (zbmin + dz*0.5)) // dz)
            y0 = self._y[blockID, iy]
            y1 = y0 + dy
            z0 = self._z[blockID, iz]
            z1 = z0 + dz

            for i,field in enumerate(fields):
                f = self._stencil_3d(field, blockID, ix, ix+1, iy, iy+1, iz, iz+1)
                q[i] = interp3d(x, y, z, x0, x1, y0, y1, z0, z1, f)

        if len(q) == 1:
            return q[0]
        else:
            return q

    def _stencil_1d(
        self,
        field: int,
        blockID: int,
        ix0: int, ix1: int
    ) -> List[float]:
        return [
            self._unk[field, blockID, 0, 0, ix0],
            self._unk[field, blockID, 0, 0, ix1]
        ]

    def _stencil_2d(
        self,
        field: int,
        blockID: int,
        ix0: int, ix1: int,
        iy0: int, iy1: int
    ) -> List[List[float]]:
        return [[
            self._unk[field, blockID, 0, iy0, ix0],
            self._unk[field, blockID, 0, iy1, ix0]
        ], [
            self._unk[field, blockID, 0, iy0, ix1],
            self._unk[field, blockID, 0, iy1, ix1]
        ]]

    def _stencil_3d(
        self,
        field: int,
        blockID: int,
        ix0: int, ix1: int,
        iy0: int, iy1: int,
        iz0: int, iz1: int
    ) -> List[List[List[float]]]:
        return [[[
            self._unk[field, blockID, iz0, iy0, ix0],
            self._unk[field, blockID, iz1, iy0, ix0]
        ], [
            self._unk[field, blockID, iz0, iy1, ix0],
            self._unk[field, blockID, iz1, iy1, ix0]
        ]], [[
            self._unk[field, blockID, iz0, iy0, ix1],
            self._unk[field, blockID, iz1, iy0, ix1]
        ], [
            self._unk[field, blockID, iz0, iy1, ix1],
            self._unk[field, blockID, iz1, iy1, ix1]
        ]]]

    def close(self) -> None:
        for handle in self._shm_handles:
            handle.close()
        self._shm_handles.clear()

    def __del__(self):
        self.close()


class FLASHSnapshot(Snapshot):
    _lrefine_max: int
    _nblockx: int
    _nblocky: int
    _nblockz: int
    _nxb: int
    _nyb: int
    _nzb: int
    _ngx: int
    _nyx: int
    _nzx: int
    _dxb: np.ndarray
    _dyb: np.ndarray
    _dzb: np.ndarray

    _x: np.ndarray
    _y: np.ndarray
    _z: np.ndarray
    _dx: np.ndarray
    _dy: np.ndarray
    _dz: np.ndarray
    _vol: np.ndarray
    _bbox: np.ndarray
    _levels: np.ndarray
    _hash: List[Tuple[int, int, int, int]]

    _hash_map: Dict[Tuple[int, int, int, int], int]
    _neighbours: List[Dict[Tuple[int, int, int], Tuple[str, List[int]]]]
    _unk: np.ndarray

    _desc: Dict[str, Any]
    _proxy: FLASHSnapshotProxy
    _shm_handles: List[shared_memory.SharedMemory]
    _shm_grid: Dict[str, ShmMeta]
    _shm_fields: ShmMeta

    def __init__(self, filename: str, use_nu: bool = False, nguard: int = 1):
        self._field_list = (SNAP_FIELDS + SNAP_FIELDS_NU) if use_nu else SNAP_FIELDS

        self._read_data(filename, use_nu, nguard)
        self._find_neighbours()
        self._fill_gcs()

        self._setup_shm()

    # Utility method to sort plot files without loading everything
    @staticmethod
    def get_time(filename: str) -> float:
        with h5py.File(filename, 'r') as f:
            real_scalars = {k.decode('ascii').strip(): v for k,v in f['real scalars'][()]}
        return real_scalars['time']

    def get_quantity(
        self,
        fields: Sequence[str],
        x: float,
        y: Optional[float] = None,
        z: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        return self._proxy.get_quantity(fields, x, y, z)

    def get_field(self, fields: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        return self._proxy.get_field(fields)

    def _read_data(self, filename: str, use_nu: bool, nguard: int) -> None:
        with h5py.File(filename, 'r') as f:
            # Decode datasets
            integer_scalars = {k.decode('ascii').strip(): v for k,v in f['integer scalars'][()]}
            real_scalars = {k.decode('ascii').strip(): v for k,v in f['real scalars'][()]}
            real_runtime_parameters = {k.decode('ascii').strip(): v for k,v in f['real runtime parameters'][()]}
            integer_runtime_parameters = {k.decode('ascii').strip(): v for k,v in f['integer runtime parameters'][()]}
            string_runtime_parameters = {
                k.decode('ascii').strip(): v.decode('ascii').strip()
                for k,v in f['string runtime parameters'][()]
            } if 'string runtime parameters' in f else {}

            self._geometry = string_runtime_parameters['geometry'].lower()
            self._bc = [
                [string_runtime_parameters[f'{ax}{side}_boundary_type'].lower() for side in ('l', 'r')]
                for ax in ('x', 'y', 'z')
            ]

            # Read data
            self._dim = integer_scalars['dimensionality']
            self._current_time = real_scalars['time']

            # Number of cells per block in each direction
            self._nxb = integer_scalars['nxb']
            self._nyb = integer_scalars['nyb']
            self._nzb = integer_scalars['nzb']
            self._ngx = nguard
            self._ngy = nguard if self._dim >= 2 else 0
            self._ngz = nguard if self._dim == 3 else 0

            # Domain boundaries
            self._xmin = real_runtime_parameters['xmin']
            self._xmax = real_runtime_parameters['xmax']
            self._ymin = real_runtime_parameters['ymin'] if self._dim >= 2 else 0
            self._ymax = real_runtime_parameters['ymax'] if self._dim >= 2 else np.pi
            self._zmin = real_runtime_parameters['zmin'] if self._dim == 3 else 0
            self._zmax = real_runtime_parameters['zmax'] if self._dim == 3 else 2.*np.pi

            self._lrefine_max = integer_runtime_parameters['lrefine_max']
            self._nblockx = integer_runtime_parameters['nblockx']
            self._nblocky = integer_runtime_parameters['nblocky']
            self._nblockz = integer_runtime_parameters['nblockz']

            self._dxb = np.array([(self._xmax - self._xmin)/(self._nblockx * 2**l) for l in range(0, self._lrefine_max)])

            if self._dim >= 2:
                self._dyb = np.array([(self._ymax - self._ymin)/(self._nblocky * 2**l) for l in range(0, self._lrefine_max)])
            else:
                self._dyb = np.ones(self._lrefine_max)

            if self._dim == 3:
                self._dzb = np.array([(self._zmax - self._zmin)/(self._nblockz * 2**l) for l in range(0, self._lrefine_max)])
            else:
                self._dzb = np.ones(self._lrefine_max)

            # Keep quantities only from leaf blocks
            node_type = f['node type'][()]
            leaf_mask = (node_type == 1)

            self._bbox = f['bounding box'][()][leaf_mask]
            self._levels = f['refine level'][()][leaf_mask]

            # Calculate cell-centred coordinates
            nblk = self._bbox.shape[0]
            nxGC = self._nxb + 2*self._ngx
            nyGC = self._nyb + 2*self._ngy
            nzGC = self._nzb + 2*self._ngz
            ixstart = self._ngx
            ixend   = self._nxb + self._ngx
            iystart = self._ngy
            iyend   = self._nyb + self._ngy
            izstart = self._ngz
            izend   = self._nzb + self._ngz

            xbmin = self._bbox[:, 0, 0]
            xbmax = self._bbox[:, 0, 1]
            self._dx = np.abs(xbmax - xbmin) / self._nxb
            self._x = xbmin[:, None] + (np.arange(nxGC) + 0.5 - self._ngx)*self._dx[:, None]

            if self._dim >= 2:
                ybmin = self._bbox[:, 1, 0]
                ybmax = self._bbox[:, 1, 1]
                self._dy = np.abs(ybmax - ybmin) / self._nyb
                self._y = ybmin[:, None] + (np.arange(nyGC) + 0.5 - self._ngy)*self._dy[:, None]
            else:
                self._dy = np.zeros(nblk)
                self._y = np.zeros((nblk, nyGC))

            if self._dim == 3:
                zbmin = self._bbox[:, 2, 0]
                zbmax = self._bbox[:, 2, 1]
                self._dz = np.abs(zbmax - zbmin) / self._nzb
                self._z = zbmin[:, None] + (np.arange(nzGC) + 0.5 - self._ngz)*self._dz[:, None]
            else:
                self._dz = np.zeros(nblk)
                self._z = np.zeros((nblk, nzGC))

            # Fill cell volumes
            if self._dim == 1:
                # Spherical shell volume assuming cell-centred radius
                self._vol = (4./3.)*np.pi*((self._x + 0.5*self._dx[:, None])**3 - (self._x - 0.5*self._dx[:, None])**3)
                self._vol = self._vol.reshape(nblk, 1, 1, nxGC)
            elif self._dim == 2:
                # annular volume 2*pi*r_c*dr*dz, where r_c is cell-centred radius
                self._vol = 2.*np.pi*self._x*self._dx[:, None]*self._dy[:, None]
                self._vol = np.broadcast_to(self._vol[:, None, None,:], (nblk, 1, nyGC, nxGC))
            elif self._dim == 3:
                # Every cell in the block has a volume of dx*dy*dz
                self._vol = self._dx*self._dy*self._dz
                self._vol = np.broadcast_to(self._vol[:, None, None, None], (nblk, nzGC, nyGC, nxGC))

            # Setup grid quantities
            nfields = len(self._field_list)
            self._unk = np.full((nfields, nblk, nzGC, nyGC, nxGC), np.nan)
            for field in _FLASH_FIELDS:
                ifield = SNAP_FIELD_MAP[FLASH_FIELD_MAP[field]]
                self._unk[ifield,:,izstart:izend,iystart:iyend,ixstart:ixend] = f[field][()][leaf_mask]

            # Fill neutrino quantities
            if use_nu:
                X = np.broadcast_to(self._x[:, None, None, ixstart:ixend], (nblk, self._nzb, self._nyb, self._nxb))
                Y = np.broadcast_to(self._y[:, None, iystart:iyend, None], (nblk, self._nzb, self._nyb, self._nxb))
                Z = np.broadcast_to(self._z[:, izstart:izend, None, None], (nblk, self._nzb, self._nyb, self._nxb))
                r = np.sqrt(X**2 + Y**2 + Z**2)

                self._unk[SNAP_FIELD_MAP['lum nue'  ],:,izstart:izend,iystart:iyend,ixstart:ixend] = 4.*np.pi*r**2*f['fnue'][()][leaf_mask]*1e51
                self._unk[SNAP_FIELD_MAP['lum anue' ],:,izstart:izend,iystart:iyend,ixstart:ixend] = 4.*np.pi*r**2*f['fnua'][()][leaf_mask]*1e51
                self._unk[SNAP_FIELD_MAP['lum nux'  ],:,izstart:izend,iystart:iyend,ixstart:ixend] = 4.*np.pi*r**2*f['fnux'][()][leaf_mask]*1e51*0.5
                self._unk[SNAP_FIELD_MAP['lum anux' ],:,izstart:izend,iystart:iyend,ixstart:ixend] = 4.*np.pi*r**2*f['fnux'][()][leaf_mask]*1e51*0.5
                self._unk[SNAP_FIELD_MAP['ener nue' ],:,izstart:izend,iystart:iyend,ixstart:ixend] = f['enue'][()][leaf_mask]
                self._unk[SNAP_FIELD_MAP['ener anue'],:,izstart:izend,iystart:iyend,ixstart:ixend] = f['enua'][()][leaf_mask]
                self._unk[SNAP_FIELD_MAP['ener nux' ],:,izstart:izend,iystart:iyend,ixstart:ixend] = f['enux'][()][leaf_mask]
                self._unk[SNAP_FIELD_MAP['ener anux'],:,izstart:izend,iystart:iyend,ixstart:ixend] = f['enux'][()][leaf_mask]

            # Fill block hash table
            xb = lambda n: (self._bbox[n,0,0] + self._bbox[n,0,1])*0.5
            yb = lambda n: (self._bbox[n,1,0] + self._bbox[n,1,1])*0.5
            zb = lambda n: (self._bbox[n,2,0] + self._bbox[n,2,1])*0.5

            self._hash_map = {}
            self._hash = np.empty((nblk, 4), dtype=np.int64)
            for n,l in enumerate(self._levels):
                l = int(l)
                block_hash = (
                    l,
                    int((xb(n) - self._xmin)/self._dxb[l-1]),
                    int((yb(n) - self._ymin)/self._dyb[l-1]) if self._dim >= 2 else 0,
                    int((zb(n) - self._zmin)/self._dzb[l-1]) if self._dim == 3 else 0
                )
                self._hash_map[block_hash] = n
                self._hash[n] = block_hash

    def _find_neighbours(self) -> None:
        active_dims = [True, self._dim > 1, self._dim == 3]
        nroot = [self._nblockx, self._nblocky, self._nblockz]
        offsets = [
            offset for offset in
            itertools.product(*[(-1, 0, 1) if active else (0,) for active in active_dims])
            if any(offset)
        ]
        self._neighbours = []
        for n, row in enumerate(self._hash):
            l, ix, iy, iz = (int(v) for v in row)
            nmax = [nb << (l - 1) for nb in nroot]
            neighbours = {}
            for offset in offsets:
                direct = [i + o for i,o in zip([ix, iy, iz], offset)]
                out = [active and not (0 <= t < m) for active,t,m in zip(active_dims, direct, nmax)]
                if any(out):
                    if all(o_out or o == 0 for o_out,o in zip(out, offset)):
                        kind = 'boundary'
                    else:
                        kind = 'partial'
                    match = [axis for axis in range(3) if out[axis]]
                elif (l, *direct) in self._hash_map:
                    kind = 'same'
                    match = [self._hash_map[(l, *direct)]]
                elif (l - 1, *[i >> 1 for i in direct]) in self._hash_map:
                    kind = 'coarse'
                    match = [self._hash_map[(l - 1, *[i >> 1 for i in direct])]]
                else:
                    side = [
                        ((1,) if o < 0 else (0,) if o > 0 else (0, 1)) if active else (0,)
                        for active,o in zip(active_dims, offset)
                    ]
                    keys = [
                        (l + 1,
                         2*direct[0] + ox,
                         2*direct[1] + oy if active_dims[1] else 0,
                         2*direct[2] + oz if active_dims[2] else 0)
                        for ox in side[0] for oy in side[1] for oz in side[2]
                    ]
                    kind = 'fine'
                    match = [self._hash_map.get(key) for key in keys]
                    if None in match:
                        raise RuntimeError(f'Invalid mesh refinement at block {n}, offset {offset}')
                neighbours[offset] = (kind, match)
            self._neighbours.append(neighbours)

    def _fill_gcs(self) -> None:
        nblk = self._bbox.shape[0]

        # Reflect boundary, copy, and restrict
        for blockID in range(nblk):
            for offset,(kind, match) in self._neighbours[blockID].items():
                if kind == 'boundary':
                    self._amr_reflect_block(blockID, offset, mirror_axes=match)
                elif kind == 'same':
                    self._amr_copy_block(blockID, offset, match[0])
                elif kind == 'fine':
                    self._amr_restrict_block(blockID, offset, match)

        # Prolongation and partial reflection
        for blockID in np.argsort(self._levels, kind='stable'):
            items = self._neighbours[blockID].items()
            for offset,(kind, match) in items:
                if kind == 'coarse':
                    self._amr_prolong_block(blockID, offset, match[0])
            for offset,(kind, match) in items:
                if kind == 'partial':
                    self._amr_reflect_block(blockID, offset, mirror_axes=match)

        unfilled = np.where(~np.isfinite(self._unk).all(axis=(0, 2, 3, 4)))[0]
        if unfilled.size > 0:
            raise RuntimeError(f'Missed guard cells in {unfilled.size} blocks: {unfilled[:10]}')

    def _amr_dest_slice(
        self,
        offset: int,
        n: int,
        ng: int
    ) -> Tuple[int, int]:
        if offset == 0:
            return (ng, n + ng)
        else:
            istart = 0 if offset < 0 else (n + ng)
            return (istart, istart + ng)

    def _amr_restrict_slice(
        self, 
        offset: int,
        n: int,
        ng: int,
        ratio: int
    ) -> Tuple[int, int]:
        if offset == 0:
            return (ng, n + ng)
        else:
            istart = n - (ratio - 1)*ng if offset < 0 else ng
            return (istart, istart + ratio*ng)

    # Boundary types for which the normal velocity component changes sign
    _ODD_BC = ('reflect', 'reflecting', 'axisymmetric', 'eqtsymmetric')

    def _amr_reflect_block(
        self,
        blockID: int,
        offset: Tuple[int, int, int],
        mirror_axes: List[int]
    ) -> None:
        n  = (self._nxb, self._nyb, self._nzb)
        ng = (self._ngx, self._ngy, self._ngz)
        dst, src = [], []
        for axis in range(3):
            i0, i1 = self._amr_dest_slice(offset[axis], n[axis], ng[axis])
            dst.append(slice(i0, i1))
            if axis in mirror_axes:
                j0, j1 = i0 - offset[axis]*ng[axis], i1 - offset[axis]*ng[axis]
                src.append(slice(j1 - 1, j0 - 1 if j0 > 0 else None, -1))
            else:
                src.append(slice(i0, i1))

        # Mirror velocity components
        sign = {}
        vel = ('velocity-x', 'velocity-y', 'velocity-z')
        for axis in mirror_axes:
            bc = self._bc[axis][0 if offset[axis] < 0 else 1]
            if bc in self._ODD_BC:
                sign[vel[axis]] = -sign.get(vel[axis], 1.)
                if bc == 'axisymmetric' and axis == 0 and self._geometry == 'cylindrical':
                    sign['velocity-z'] = -sign.get('velocity-z', 1.)

        for ifield,field in enumerate(self._field_list):
            self._unk[ifield, blockID, dst[2], dst[1], dst[0]] = \
                sign.get(field, 1.) * self._unk[ifield, blockID, src[2], src[1], src[0]]

    def _amr_copy_block(
        self,
        blockID: int,
        offset: Tuple[int, int, int],
        neighbourID: int
    ) -> None:
        ixstart, ixend = self._amr_dest_slice(offset[0], self._nxb, self._ngx)
        iystart, iyend = self._amr_dest_slice(offset[1], self._nyb, self._ngy)
        izstart, izend = self._amr_dest_slice(offset[2], self._nzb, self._ngz)
        jxstart, jxend = ixstart - (offset[0]*self._nxb), ixend - (offset[0]*self._nxb)
        jystart, jyend = iystart - (offset[1]*self._nyb), iyend - (offset[1]*self._nyb)
        jzstart, jzend = izstart - (offset[2]*self._nzb), izend - (offset[2]*self._nzb)

        for ifield in range(len(self._field_list)):
            self._unk[ifield, blockID, izstart:izend, iystart:iyend, ixstart:ixend] = \
                self._unk[ifield, neighbourID, jzstart:jzend, jystart:jyend, jxstart:jxend]

    def _amr_restrict_block(
        self,
        blockID: int,
        offset: Tuple[int, int, int],
        neighbourIDs: List[int]
    ) -> None:
        dl = self._levels[neighbourIDs[0]] - self._levels[blockID]
        ratio = (1 << dl)

        ixstart, ixend = self._amr_dest_slice(offset[0], self._nxb, self._ngx)
        iystart, iyend = self._amr_dest_slice(offset[1], self._nyb, self._ngy)
        izstart, izend = self._amr_dest_slice(offset[2], self._nzb, self._ngz)
        jxstart, jxend = self._amr_restrict_slice(offset[0], self._nxb, self._ngx, ratio)
        jystart, jyend = self._amr_restrict_slice(offset[1], self._nyb, self._ngy, ratio)
        jzstart, jzend = self._amr_restrict_slice(offset[2], self._nzb, self._ngz, ratio)
        nxp = max(1, self._nxb // ratio) if offset[0] == 0 else self._ngx
        nyp = max(1, self._nyb // ratio) if offset[1] == 0 else self._ngy
        nzp = max(1, self._nzb // ratio) if offset[2] == 0 else self._ngz
        npx = ratio if offset[0] == 0 else 1
        npy = ratio if offset[1] == 0 and self._dim >= 2 else 1
        npz = ratio if offset[2] == 0 and self._dim == 3 else 1

        if self._dim == 1:
            mean_shape = (1, 1, self._ngx, ratio)
            mean_axes = 3
        elif self._dim == 2:
            mean_shape = (1,
                self._nyb//ratio if offset[1] == 0 else self._ngy, ratio,
                self._nxb//ratio if offset[0] == 0 else self._ngx, ratio
            )
            mean_axes = (2,4)
        elif self._dim == 3:
            mean_shape = (
                self._nzb//ratio if offset[2] == 0 else self._ngz, ratio,
                self._nyb//ratio if offset[1] == 0 else self._ngy, ratio,
                self._nxb//ratio if offset[0] == 0 else self._ngx, ratio
            )
            mean_axes = (1,3,5)

        for i,neighbourID in enumerate(neighbourIDs):
            ixp = i // (npy*npz)
            iyp = (i // npz) % npy
            izp = i % npz

            for ifield in range(len(self._field_list)):
                self._unk[
                    ifield,
                    blockID,
                    izstart+izp*nzp:izstart+(izp+1)*nzp,
                    iystart+iyp*nyp:iystart+(iyp+1)*nyp,
                    ixstart+ixp*nxp:ixstart+(ixp+1)*nxp
                ] = \
                self._unk[
                    ifield,
                    neighbourID,
                    jzstart:jzend,
                    jystart:jyend,
                    jxstart:jxend
                ].reshape(mean_shape).mean(axis=mean_axes)

    def _amr_prolong_block(
        self,
        blockID: int,
        offset: Tuple[int, int, int],
        neighbourID: int
    ) -> None:
        n  = (self._nxb, self._nyb, self._nzb)
        ng = (self._ngx, self._ngy, self._ngz)
        dst = [slice(*self._amr_dest_slice(offset[axis], n[axis], ng[axis])) for axis in range(3)]
        centres = (self._x[blockID, dst[0]], self._y[blockID, dst[1]], self._z[blockID, dst[2]])
        spacing = (self._dx[neighbourID], self._dy[neighbourID], self._dz[neighbourID])
        snap = 2 << int(self._levels[blockID] - self._levels[neighbourID])

        idx, wgt = [], []
        for axis in range(3):
            if axis < self._dim:
                s = (centres[axis] - (self._bbox[neighbourID, axis, 0] + 0.5*spacing[axis])) / spacing[axis]
                s = np.round(s*snap) / snap
                i0 = np.floor(s).astype(int)
                idx.append((ng[axis] + i0, ng[axis] + i0 + 1))
                wgt.append((1. - (s - i0), s - i0))
            else:
                zero = np.zeros(len(centres[axis]), dtype=int)
                idx.append((zero,))
                wgt.append((np.ones(len(centres[axis])),))

        coarse = self._unk[:, neighbourID]
        result = 0.
        for cz in range(len(idx[2])):
            for cy in range(len(idx[1])):
                for cx in range(len(idx[0])):
                    w = wgt[2][cz][:, None, None] * wgt[1][cy][None, :, None] * wgt[0][cx][None, None, :]
                    result = result + w * coarse[:, idx[2][cz][:, None, None], idx[1][cy][None, :, None], idx[0][cx][None, None, :]]

        self._unk[:, blockID, dst[2], dst[1], dst[0]] = result

    def get_proxy_descriptor(self) -> Dict[str, Any]:
        return self._desc

    def _setup_shm(self) -> None:
        self._shm_handles = []
        self._shm_grid = {}

        self._shm_grid = {
            'x'     : self._to_shared(self._x),
            'y'     : self._to_shared(self._y),
            'z'     : self._to_shared(self._z),
            'dx'    : self._to_shared(self._dx),
            'dy'    : self._to_shared(self._dy),
            'dz'    : self._to_shared(self._dz),
            'volume': self._to_shared(self._vol),
            'bbox'  : self._to_shared(self._bbox),
        }

        self._shm_fields = self._to_shared(self._unk)

        self._desc = {
            'field_list'  : self._field_list,
            'current_time': self._current_time,
            'dim'         : self._dim,
            'geometry'    : self._geometry,
            'xmin'        : self._xmin,
            'xmax'        : self._xmax,
            'ymin'        : self._ymin,
            'ymax'        : self._ymax,
            'zmin'        : self._zmin,
            'zmax'        : self._zmax,
            'lrefine_max' : self._lrefine_max,
            'nblockx'     : self._nblockx,
            'nblocky'     : self._nblocky,
            'nblockz'     : self._nblockz,
            'nxb'         : self._nxb,
            'nyb'         : self._nyb,
            'nzb'         : self._nzb,
            'ngx'         : self._ngx,
            'ngy'         : self._ngy,
            'ngz'         : self._ngz,
            'dxb'         : self._dxb,
            'dyb'         : self._dyb,
            'dzb'         : self._dzb,
            'hash_map'    : self._hash_map,
            'grid'        : self._shm_grid,
            'fields'      : self._shm_fields
        }

        self._proxy = FLASHSnapshotProxy(self._desc)

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

