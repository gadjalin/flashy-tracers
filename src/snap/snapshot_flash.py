from __future__ import annotations
from typing import Tuple, Sequence, List, Dict, Union, Optional, Any
from dataclasses import dataclass
from multiprocessing import shared_memory
from math import floor

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

    # Grid data
    _x: np.ndarray
    _y: np.ndarray
    _z: np.ndarray
    _dx: np.ndarray
    _dy: np.ndarray
    _dz: np.ndarray
    _vol: np.ndarray
    _bbox: np.ndarray
    _levels: np.ndarray

    # Physical quantities data
    _unk: np.ndarray

    # Block structure
    _dxb: np.ndarray
    _dyb: np.ndarray
    _dzb: np.ndarray
    _hash_map: Dict[Tuple[int, int, int, int], int]
    _neighbours: List[Dict[Tuple[int, int, int], List[int]]]

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

        self._shm_handles = []
        self._hash_map = desc['hash_map']
        self._neighbours = desc['neighbours']

        self._dxb = np.array([(self._xmax - self._xmin)/(self._nblockx * 2**l) for l in range(0, self._lrefine_max)])

        if self._dim >= 2:
            self._dyb = np.array([(self._ymax - self._ymin)/(self._nblocky * 2**l) for l in range(0, self._lrefine_max)])
        else:
            self._dyb = np.ones(self._lrefine_max)

        if self._dim == 3:
            self._dzb = np.array([(self._zmax - self._zmin)/(self._nblockz * 2**l) for l in range(0, self._lrefine_max)])
        else:
            self._dzb = np.ones(self._lrefine_max)

        self._x      = push_shared_memory(desc['grid']['x'])
        self._y      = push_shared_memory(desc['grid']['y'])
        self._z      = push_shared_memory(desc['grid']['z'])
        self._dx     = push_shared_memory(desc['grid']['dx'])
        self._dy     = push_shared_memory(desc['grid']['dy'])
        self._dz     = push_shared_memory(desc['grid']['dz'])
        self._vol    = push_shared_memory(desc['grid']['volume'])
        self._bbox   = push_shared_memory(desc['grid']['bbox'])
        self._levels = push_shared_memory(desc['grid']['level'])

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

        if len(fields) == 0:
            raise ValueError('Field list must contain at least one field')

        ifields = [SNAP_FIELD_MAP[field] for field in fields]

        nblk = len(self._bbox)
        ncells = self._nzb*self._nyb*self._nxb

        grid_dtype = [('x', float), ('y', float), ('z', float), ('dx', float), ('dy', float), ('dz', float), ('volume', float)]
        field_dtype = [(field, float) for field in fields]
        grid = np.empty(nblk*ncells, dtype=grid_dtype)
        unk = np.empty(nblk*ncells, dtype=field_dtype)
        grid['x'][:] = np.broadcast_to(self._x[:,None,None,:], (nblk, self._nzb, self._nyb, self._nxb)).reshape(-1)
        grid['y'][:] = np.broadcast_to(self._y[:,None,:,None], (nblk, self._nzb, self._nyb, self._nxb)).reshape(-1)
        grid['z'][:] = np.broadcast_to(self._z[:,:,None,None], (nblk, self._nzb, self._nyb, self._nxb)).reshape(-1)
        grid['dx'][:] = np.repeat(self._dx[:], ncells)
        grid['dy'][:] = np.repeat(self._dy[:], ncells)
        grid['dz'][:] = np.repeat(self._dz[:], ncells)
        grid['volume'][:] = self._vol[:,:,:,:].ravel()
        for ifield,field in zip(ifields, fields):
            unk[field][:] = self._unk[ifield,:,:,:,:].ravel()

        return grid, unk

    def find_block(
        self,
        x: float,
        y: Optional[float],
        z: Optional[float]
    ) -> int:
        for level in range(self._lrefine_max, 0, -1):
            ix = int((x - self._xmin)//self._dxb[level-1])
            iy = int((y - self._ymin)//self._dyb[level-1]) if y is not None else 0
            iz = int((z - self._zmin)//self._dzb[level-1]) if z is not None else 0

            key = (level, ix, iy, iz)
            if key in self._hash_map:
                return self._hash_map[key]
        else:
            raise RuntimeError(f'No block found at ({x:.5f}, {y if y is not None else np.nan:.5f}, {z if z is not None else np.nan:.5f})')

    def _interp_block(
        self,
        fields: List[int],
        x: float,
        y: Optional[float],
        z: Optional[float]
    ) -> Union[float, np.ndarray]:
        blockID = self.find_block(x, y, z)
        dx = self._dx[blockID]
        xbmin = self._bbox[blockID,0,0]

        ix = int((x - (xbmin + dx*0.5)) // dx)
        x0 = (xbmin + dx*0.5) + ix*dx
        x1 = x0 + dx

        q = np.empty(len(fields))
        if self._dim == 1:
            for i,field in enumerate(fields):
                f = self._stencil_1d(field, blockID, ix, ix+1)
                q[i] = interp1d(x, x0, x1, f)
        elif self._dim == 2:
            if y is None:
                raise ValueError('Y coordinate must be specified in 2D')

            dy = self._dy[blockID]
            ybmin = self._bbox[blockID,1,0]

            iy = int((y - (ybmin + dy*0.5)) // dy)
            y0 = (ybmin + dy*0.5) + iy*dy
            y1 = y0 + dy

            for i,field in enumerate(fields):
                f = self._stencil_2d(field, blockID, ix, ix+1, iy, iy+1)
                q[i] = interp2d(x, y, x0, x1, y0, y1, f)
        elif self._dim == 3:
            if y is None or z is None:
                raise ValueError('Y and Z coordinates must be specified in 3D')

            dy = self._dy[blockID]
            dz = self._dz[blockID]
            ybmin = self._bbox[blockID,1,0]
            zbmin = self._bbox[blockID,2,0]

            iy = int((y - (ybmin + dy*0.5)) // dy)
            iz = int((z - (zbmin + dz*0.5)) // dz)
            y0 = (ybmin + dy*0.5) + iy*dy
            y1 = y0 + dy
            z0 = (zbmin + dz*0.5) + iz*dz
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
        # Fast array access
        if ix0 >= 0 and ix1 < self._nxb:
            return (
                self._unk[field, blockID, 0, 0, ix0],
                self._unk[field, blockID, 0, 0, ix1]
            )
        else: # Slow AMR query
            return (
                self._get_amr_data(field, blockID, ix0),
                self._get_amr_data(field, blockID, ix1)
            )

    def _stencil_2d(
        self,
        field: int,
        blockID: int,
        ix0: int, ix1: int,
        iy0: int, iy1: int
    ) -> List[List[float]]:
        # Fast array access
        if (ix0 >= 0 and ix1 < self._nxb) and \
           (iy0 >= 0 and iy1 < self._nyb):
            return ((   
                self._unk[field, blockID, 0, iy0, ix0],
                self._unk[field, blockID, 0, iy1, ix0]
            ), (
                self._unk[field, blockID, 0, iy0, ix1],
                self._unk[field, blockID, 0, iy1, ix1]
            ))
        else: # Slow AMR query
            return ((
                self._get_amr_data(field, blockID, ix0, iy0),
                self._get_amr_data(field, blockID, ix0, iy1)
            ), ( 
                self._get_amr_data(field, blockID, ix1, iy0),
                self._get_amr_data(field, blockID, ix1, iy1)
            ))

    def _stencil_3d(
        self,
        field: int,
        blockID: int,
        ix0: int, ix1: int,
        iy0: int, iy1: int,
        iz0: int, iz1: int
    ) -> List[List[List[float]]]:
        # Fast array access
        if (ix0 >= 0 and ix1 < self._nxb) and \
           (iy0 >= 0 and iy1 < self._nyb) and \
           (iz0 >= 0 and iz1 < self._nzb):
            return (((
                self._unk[field, blockID, iz0, iy0, ix0],
                self._unk[field, blockID, iz1, iy0, ix0]
            ), (
                self._unk[field, blockID, iz0, iy1, ix0],
                self._unk[field, blockID, iz1, iy1, ix0]
            )), ((
                self._unk[field, blockID, iz0, iy0, ix1],
                self._unk[field, blockID, iz1, iy0, ix1]
            ), (
                self._unk[field, blockID, iz0, iy1, ix1],
                self._unk[field, blockID, iz1, iy1, ix1]
            )))
        else: # Slow AMR query
            return (((
                self._get_amr_data(field, blockID, ix0, iy0, iz0),
                self._get_amr_data(field, blockID, ix0, iy0, iz1)
            ), (
                self._get_amr_data(field, blockID, ix0, iy1, iz0),
                self._get_amr_data(field, blockID, ix0, iy1, iz1)
            )), ((
                self._get_amr_data(field, blockID, ix1, iy0, iz0),
                self._get_amr_data(field, blockID, ix1, iy0, iz1)
            ), (
                self._get_amr_data(field, blockID, ix1, iy1, iz0),
                self._get_amr_data(field, blockID, ix1, iy1, iz1)
            )))

    # TODO vectorise field variable
    def _get_amr_data(
        self,
        field: int,
        blockID: int,
        ix: int,
        iy: int = 0,
        iz: int = 0
    ) -> float:
        # Cell is within block
        if (0 <= ix and ix < self._nxb) and \
           (0 <= iy and iy < self._nyb) and \
           (0 <= iz and iz < self._nzb):
            return self._unk[field, blockID, iz, iy, ix]

        # Compute neighbour offset
        x_offset = ix // self._nxb
        y_offset = iy // self._nyb
        z_offset = iz // self._nzb
        offset = (x_offset, y_offset, z_offset)
        offset_count = self._count_offset(offset)

        # Cell requires neighbour data
        # FIXME diagonal neighbours (in corners) may be 2 refinement level difference
        neighbours = self._neighbours[blockID][offset]
        if len(neighbours) >= 1:
            if all(self._levels[i] == self._levels[blockID] for i in neighbours): # Single neighbour, same level of refinement
                return self._unk[field, neighbours[0], iz % self._nzb, iy % self._nyb, ix % self._nxb]
            elif all(self._levels[i] > self._levels[blockID] for i in neighbours): # 1 (1D), 1/2 (2D), 1/2/8 (3D) neighbours, finer
                return self._amr_restrict(field, offset, blockID, ix, iy, iz)
            elif all(self._levels[i] < self._levels[blockID] for i in neighbours): # 1 neighbour, coarser
                return self._amr_prolong(field, offset, blockID, ix, iy, iz)
        else:
            return self._amr_reflect(field, offset, blockID, ix, iy, iz) # Domain boundary, no neighbours -> reflect

    def _amr_reflect(
        self,
        field: int,
        offset: Tuple[int, int, int],
        blockID: int,
        ix: int,
        iy: int,
        iz: int
    ) -> float:
        # Calculate corresponding reflected index
        def reflect_index(i, n):
            i = i % (2*n)
            return i if i < n else 2*n - 1 - i

        # Get basis offsets for this edge (2D/3D), to find common neighbours
        # i.e. offsets correponding to face neighbours adjacent to this edge neighbour
        def edgetofaces(edge):
            if edge[0] == 0:
                return ((0, edge[1], 0), (0, 0, edge[2]))
            elif edge[1] == 0:
                return ((edge[0], 0, 0), (0, 0, edge[2]))
            elif edge[2] == 0:
                return ((edge[0], 0, 0), (0, edge[1], 0))

        # Get basis offsets for this corner (3D), to find common neighbours
        # i.e. offset corresponding to edge neighbours adjacent to this corner neighbour
        def cornertoedges(corner):
            return (
                (corner[0], corner[1], 0),
                (corner[0], 0, corner[2]),
                (0, corner[1], corner[2]),
            )

        offset_count = self._count_offset(offset)
        if offset_count == 1: # face offset -> reflect a face
            ix = reflect_index(ix, self._nxb) if offset[0] != 0 else ix
            iy = reflect_index(iy, self._nyb) if offset[1] != 0 else iy
            iz = reflect_index(iz, self._nzb) if offset[2] != 0 else iz
            return self._unk[field, blockID, iz, iy, ix]
        elif offset_count == 2: # edge offset -> reflect an edge
            axes = [axis for axis,face in enumerate(offset) if face != 0]
            faces = edgetofaces(offset)
            # Find face neighbour corresponding to this edge location
            # Or extrapolate current block if no neighbour (i.e. at an edge of the domain)
            if len(self._neighbours[blockID][faces[0]]) != 0:
                ix = reflect_index(ix, self._nxb) if axes[1] == 0 else ix
                iy = reflect_index(iy, self._nyb) if axes[1] == 1 else iy
                iz = reflect_index(iz, self._nzb) if axes[1] == 2 else iz
                return self._get_amr_data(field, blockID, ix, iy, iz)
            elif len(self._neighbours[blockID][faces[1]]) != 0:
                ix = reflect_index(ix, self._nxb) if axes[0] == 0 else ix
                iy = reflect_index(iy, self._nyb) if axes[0] == 1 else iy
                iz = reflect_index(iz, self._nzb) if axes[0] == 2 else iz
                return self._get_amr_data(field, blockID, ix, iy, iz)
            else:
                ix = reflect_index(ix, self._nxb) if offset[0] != 0 else ix
                iy = reflect_index(iy, self._nyb) if offset[1] != 0 else iy
                iz = reflect_index(iz, self._nzb) if offset[2] != 0 else iz
                return self._unk[field, blockID, iz, iy, ix]
        elif offset_count == 3: # reflect a corner
            edges = cornertoedges(offset)
            not_axes = [axis for edge in edges for axis,face in edge if face == 0]
            # Find edge neighbours adjacent to this corner
            # Or extrapolate if no neighbour (i.e. in corner o the domain)
            if len(self._neighbours[blockID][edges[0]]) != 0:
                ix = reflect_index(ix, self._nxb) if not_axes[0] == 0 else ix
                iy = reflect_index(iy, self._nyb) if not_axes[0] == 1 else iy
                iz = reflect_index(iz, self._nzb) if not_axes[0] == 2 else iz
                return self._get_amr_data(field, blockID, ix, iy, iz)
            elif len(self._neighbours[blockID][edges[1]]) != 0:
                ix = reflect_index(ix, self._nxb) if not_axes[1] == 0 else ix
                iy = reflect_index(iy, self._nyb) if not_axes[1] == 1 else iy
                iz = reflect_index(iz, self._nzb) if not_axes[1] == 2 else iz
                return self._get_amr_data(field, blockID, ix, iy, iz)
            elif len(self._neighbours[blockID][edges[2]]) != 0:
                ix = reflect_index(ix, self._nxb) if not_axes[2] == 0 else ix
                iy = reflect_index(iy, self._nyb) if not_axes[2] == 1 else iy
                iz = reflect_index(iz, self._nzb) if not_axes[2] == 2 else iz
                return self._get_amr_data(field, blockID, ix, iy, iz)
            else:
                ix = reflect_index(ix, self._nxb)
                iy = reflect_index(iy, self._nyb)
                iz = reflect_index(iz, self._nzb)
                return self._unk[field, blockID, iz, iy, ix]

    def _amr_restrict(
        self,
        field: int,
        offset: Tuple[int, int, int],
        blockID: int,
        ix: int,
        iy: int,
        iz: int
    ) -> float:
        neighbourIDs = self._neighbours[blockID][offset]
        ineighbour = 0
        stride = 1
        indices = (iz, iy, ix)
        ncells = (self._nzb, self._nyb, self._nxb)
        # Index correct neighbour in which restriction should happen
        for o,i,N in zip(offset[self._dim - 1::-1], indices[-self._dim:], ncells[-self._dim:]):
            if o == 0:
                if i >= max(1, N//2):
                    ineighbour += stride
                stride <<= 1
        neighbourID = neighbourIDs[ineighbour]

        # Generic but slightly slower
        #dl = self._levels[neighbourID] - self._levels[blockID]
        #ix0 = ((1 << dl)*ix) % self._nxb + (1 << (dl - 1)) - 1
        #iy0 = ((1 << dl)*iy) % self._nyb + (1 << (dl - 1)) - 1
        #iz0 = ((1 << dl)*iz) % self._nzb + (1 << (dl - 1)) - 1

        # Refinement can jump by 2 levels in corners
        if self._levels[neighbourID] - self._levels[blockID] == 2:
            ix0 = (4*ix) % self._nxb + 1
            iy0 = (4*iy) % self._nyb + 1
            iz0 = (4*iz) % self._nzb + 1
        else:
            ix0 = (2*ix) % self._nxb
            iy0 = (2*iy) % self._nyb
            iz0 = (2*iz) % self._nzb

        if self._dim == 1:
            return 0.5 * (
                self._unk[field, neighbourID, 0, 0, ix0  ] +
                self._unk[field, neighbourID, 0, 0, ix0+1]
            )
        elif self._dim == 2:
            return 0.25 * (
                self._unk[field, neighbourID, 0, iy0  , ix0  ] +
                self._unk[field, neighbourID, 0, iy0  , ix0+1] +
                self._unk[field, neighbourID, 0, iy0+1, ix0  ] +
                self._unk[field, neighbourID, 0, iy0+1, ix0+1]
            )
        else:
            return 0.125 * (
                self._unk[field, neighbourID, iz0  , iy0  , ix0  ] +
                self._unk[field, neighbourID, iz0  , iy0  , ix0+1] +
                self._unk[field, neighbourID, iz0  , iy0+1, ix0  ] +
                self._unk[field, neighbourID, iz0  , iy0+1, ix0+1] +
                self._unk[field, neighbourID, iz0+1, iy0  , ix0  ] +
                self._unk[field, neighbourID, iz0+1, iy0  , ix0+1] +
                self._unk[field, neighbourID, iz0+1, iy0+1, ix0  ] +
                self._unk[field, neighbourID, iz0+1, iy0+1, ix0+1]
            )

    # This is a little less accurate but works 100%
    # TODO improve by interpolating fine cells with coarse cells
    def _amr_prolong(
        self,
        field: int,
        offset: Tuple[int, int, int],
        blockID: int,
        ix: int,
        iy: int,
        iz: int
    ) -> float:
        neighbourID = self._neighbours[blockID][offset][0]
        xbmin = self._bbox[blockID,0,0]
        xnmin = self._bbox[neighbourID,0,0]
        dx = self._dx[blockID]
        dxn = self._dx[neighbourID]
        x = (xbmin + dx*0.5) + ix*dx
        ixn0 = int((x - (xnmin + dxn*0.5)) // dxn)

        if self._dim >= 2:
            ybmin = self._bbox[blockID,1,0]
            ynmin = self._bbox[neighbourID,1,0]
            dy = self._dy[blockID]
            dyn = self._dy[neighbourID]
            y = (ybmin + dy*0.5) + iy*dy
            iyn0 = int((y - (ynmin + dyn*0.5)) // dyn)

        if self._dim == 3:
            zbmin = self._bbox[blockID,2,0]
            znmin = self._bbox[neighbourID,2,0]
            dz = self._dz[blockID]
            dzn = self._dz[neighbourID]
            z = (zbmin + dz*0.5) + iz*dz
            izn0 = int((z - (znmin + dzn*0.5)) // dzn)

        if self._dim == 1:
            x0 = (xnmin + dxn*0.5) + ixn0*dxn
            f = self._stencil_1d(field, neighbourID, ixn0, ixn0+1)
            return interp1d(x, x0, x0+dxn, f)
        elif self._dim == 2:
            x0 = (xnmin + dxn*0.5) + ixn0*dxn
            y0 = (ynmin + dyn*0.5) + iyn0*dyn
            f = self._stencil_2d(field, neighbourID, ixn0, ixn0+1, iyn0, iyn0+1)
            return interp2d(x, y, x0, x0+dxn, y0, y0+dyn, f)
        elif self._dim == 3:
            x0 = (xnmin + dxn*0.5) + ixn0*dxn
            y0 = (ynmin + dyn*0.5) + iyn0*dyn
            z0 = (znmin + dzn*0.5) + izn0*dzn
            f = self._stencil_3d(field, neighbourID, ixn0, ixn0+1, iyn0, iyn0+1, izn0, izn0+1)
            return interp3d(x, y, z, x0, x0+dxn, y0, y0+dyn, z0, z0+dzn, f)

#    def _amr_prolong1d(
#        self,
#        field: int,
#        offset: Tuple[int, int, int],
#        blockID: int,
#        ix: int
#    ) -> float:
#        neighbourID = self._neighbours[blockID][offset][0]
#
#        block0 = blockID if (ix == self._nxb) else neighbourID
#        block1 = blockID if (ix == -1       ) else neighbourID
#
#        #prolong_index = lambda i, N: ((i % N) - 1) // 2
#        #prolong_index = lambda i, N: (N - 1 - (-i) // 2) if i < 0 else ((i - N + 1) // 2 - 1)
#        prolong_index = lambda i, N: ((i - 1) >> 1) - (8 if i > 0 else 0)
#
#        ix0 = prolong_index(ix)
#        ix1 = ix0 + 1
#
#        if block0 == blockID:
#            c0 = 1./3.
#        elif block1 == blockID:
#            c0 = 2./3.
#        else:
#            c0 = 0.25 + 0.5*(ix & 1)
#
#        f0 = self._unk[field, block0, 0, 0, ix0]
#        f1 = self._unk[field, block1, 0, 0, ix1]
#        return c0*f0 + (1 - c0)*f1

#    def _amr_prolong2d(
#        self,
#        field: int,
#        offset: Tuple[int, int, int],
#        blockID: int,
#        ix: int,
#        iy: int
#    ) -> float:
#        neighbourID = self._neighbours[blockID][offset][0]
#        xbmin = self._bbox[blockID,0,0]
#        ybmin = self._bbox[blockID,1,0]
#        xnmin = self._bbox[neighbourID,0,0]
#        ynmin = self._bbox[neighbourID,1,0]
#        dx = self._dx[blockID]
#        dy = self._dy[blockID]
#        dxn = self._dx[neighbourID]
#        dyn = self._dy[neighbourID]
#        x = (xbmin + dx*0.5) + ix*dx
#        y = (ybmin + dy*0.5) + iy*dy
#
#        prolong_index = lambda i, N: (N - 1 - (-i) // 2) if i < 0 else ((i - N + 1) // 2 - 1)
#
#        ix0 = int((x - (xnmin + dxn*0.5)) // dxn) if offset[0] == 0 else prolong_index(ix, self._nxb)
#        ix1 = ix0 + 1
#        iy0 = int((y - (ynmin + dyn*0.5)) // dyn) if offset[1] == 0 else prolong_index(iy, self._nyb)
#        iy1 = iy0 + 1
#
#        if offset[1] == 0: # Prolong on X face neighbour
#            block0 = blockID if (ix == self._nxb) else neighbourID
#            block1 = blockID if (ix == -1       ) else neighbourID
#
#            if block0 == neighbourID:
#                x0  = (xnmin + dxn*0.5) + ix0*dxn
#                yn0 = (ynmin + dyn*0.5) + iy0*dyn
#                yn1 = yn0 + dyn
#
#                fy0 = self._get_amr_data(field, neighbourID, ix0, iy0)
#                fy1 = self._get_amr_data(field, neighbourID, ix0, iy1)
#
#                f0 = interp1d(y, yn0, yn1, [fy0, fy1])
#            else:
#                x0 = self._x[blockID,-1]
#                f0 = self._unk[field, blockID, 0, iy, -1]
#
#            if block1 == neighbourID:
#                x1  = (xnmin + dxn*0.5) + ix1*dxn
#                yn0 = (ynmin + dyn*0.5) + iy0*dyn
#                yn1 = yn0 + dyn
#
#                fy0 = self._get_amr_data(field, neighbourID, ix1, iy0)
#                fy1 = self._get_amr_data(field, neighbourID, ix1, iy1)
#
#                f1 = interp1d(y, yn0, yn1, [fy0, fy1])
#            else:
#                x1 = self._x[blockID,0]
#                f1 = self._unk[field, blockID, 0, iy, 0]
#
#            return interp1d(x, x0, x1, [f0, f1])
#        elif offset[0] == 0: # Prolong on Y face neighbour
#            block0 = blockID if (iy == self._nyb) else neighbourID
#            block1 = blockID if (iy == -1       ) else neighbourID
#
#            if block0 == neighbourID:
#                xn0 = (xnmin + dxn*0.5) + ix0*dxn
#                xn1 = xn0 + dxn
#                y0  = (ynmin + dyn*0.5) + iy0*dyn
#
#                fx0 = self._get_amr_data(field, neighbourID, ix0, iy0)
#                fx1 = self._get_amr_data(field, neighbourID, ix1, iy0)
#
#                f0 = interp1d(x, xn0, xn1, [fx0, fx1])
#            else:
#                y0 = self._y[blockID,-1]
#                f0 = self._unk[field, blockID, 0, -1, ix]
#
#            if block1 == neighbourID:
#                xn0 = (xnmin + dxn*0.5) + ix0*dxn
#                xn1 = xn0 + dxn
#                y1  = (ynmin + dyn*0.5) + iy1*dyn
#
#                fx0 = self._get_amr_data(field, neighbourID, ix0, iy1)
#                fx1 = self._get_amr_data(field, neighbourID, ix1, iy1)
#
#                f1 = interp1d(x, xn0, xn1, [fx0, fx1])
#            else:
#                y1 = self._y[blockID,0]
#                f1 = self._unk[field, blockID, 0, 0, ix]
#
#            return interp1d(y, y0, y1, [f0, f1])
#        # FIXME handle corner case, when prolong in a corner neighbour
#        else: 
#            return 0.0

    def _count_offset(self, offset: Tuple[int, int, int]) -> int:
        return (offset[0] != 0) + (offset[1] != 0) + (offset[2] != 0)

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
    _neighbours: List[Dict[Tuple[int, int, int], List[int]]]
    _unk: np.ndarray

    _desc: Dict[str, Any]
    _proxy: FLASHSnapshotProxy
    _shm_handles: List[shared_memory.SharedMemory]
    _shm_grid: Dict[str, ShmMeta]
    _shm_fields: ShmMeta

    def __init__(self, filename: str, use_nu: Optional[bool] = False):
        self._field_list = (SNAP_FIELDS + SNAP_FIELDS_NU) if use_nu else SNAP_FIELDS

        self._read_data(filename, use_nu)
        self._find_neighbours()

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

    def get_field(self, fields: Sequence[str]) -> np.ndarray:
        return self._proxy.get_field(fields)

    def _read_data(self, filename: str, use_nu: bool) -> None:
        with h5py.File(filename, 'r') as f:
            # Decode datasets
            integer_scalars = {k.decode('ascii').strip(): v for k,v in f['integer scalars'][()]}
            real_scalars = {k.decode('ascii').strip(): v for k,v in f['real scalars'][()]}
            real_runtime_parameters = {k.decode('ascii').strip(): v for k,v in f['real runtime parameters'][()]}
            integer_runtime_parameters = {k.decode('ascii').strip(): v for k,v in f['integer runtime parameters'][()]}

            # Read data
            self._dim = integer_scalars['dimensionality']
            self._current_time = real_scalars['time']

            # Number of cells per block in each direction
            self._nxb = integer_scalars['nxb']
            self._nyb = integer_scalars['nyb']
            self._nzb = integer_scalars['nzb']

            # Domain boundaries
            if self._dim == 1: # 1D spherical symmetry
                self._xmin = real_runtime_parameters['xmin']
                self._xmax = real_runtime_parameters['xmax']
                self._ymin = 0
                self._ymax = np.pi
                self._zmin = 0
                self._zmax = 2.*np.pi
            if self._dim == 2: # 2D Axisymmetry (cylindrical)
                self._xmin = real_runtime_parameters['xmin']
                self._xmax = real_runtime_parameters['xmax']
                self._ymin = real_runtime_parameters['ymin']
                self._ymax = real_runtime_parameters['ymax']
                self._zmin = 0
                self._zmax = 2.*np.pi
            if self._dim == 3: # Full 3D / Octant
                self._xmin = real_runtime_parameters['xmin']
                self._xmax = real_runtime_parameters['xmax']
                self._ymin = real_runtime_parameters['ymin']
                self._ymax = real_runtime_parameters['ymax']
                self._zmin = real_runtime_parameters['zmin']
                self._zmax = real_runtime_parameters['zmax']

            self._lrefine_max = integer_runtime_parameters['lrefine_max']
            self._nblockx = integer_runtime_parameters['nblockx']
            self._nblocky = integer_runtime_parameters['nblocky']
            self._nblockz = integer_runtime_parameters['nblockz']

            # Keep quantities only from leaf blocks
            node_type = f['node type'][()]
            leaf_mask = (node_type == 1)

            self._bbox = f['bounding box'][()][leaf_mask]
            self._levels = f['refine level'][()][leaf_mask]

            # Calculate cell-centred coordinates
            nblk = self._bbox.shape[0]
            self._x = np.empty((nblk, self._nxb))
            self._y = np.empty((nblk, self._nyb))
            self._z = np.empty((nblk, self._nzb))
            self._dx = np.empty(nblk)
            self._dy = np.empty(nblk)
            self._dz = np.empty(nblk)
            self._vol = np.empty((nblk, self._nzb, self._nyb, self._nxb))
            for n in range(nblk):
                xbmin = self._bbox[n, 0, 0]
                xbmax = self._bbox[n, 0, 1]

                if self._dim >= 2:
                    ybmin = self._bbox[n, 1, 0]
                    ybmax = self._bbox[n, 1, 1]
                else:
                    ybmin = 0
                    ybmax = np.pi

                if self._dim == 3:
                    zbmin = self._bbox[n, 2, 0]
                    zbmax = self._bbox[n, 2, 1]
                else:
                    zbmin = 0
                    zbmax = 2.*np.pi

                dx = abs(xbmax - xbmin) / self._nxb
                dy = abs(ybmax - ybmin) / self._nyb
                dz = abs(zbmax - zbmin) / self._nzb

                self._x[n,:] = xbmin + (np.arange(self._nxb) + 0.5)*dx
                self._y[n,:] = ybmin + (np.arange(self._nyb) + 0.5)*dy if self._dim >= 2 else 0.0
                self._z[n,:] = zbmin + (np.arange(self._nzb) + 0.5)*dz if self._dim == 3 else 0.0
                self._dx[n] = dx
                self._dy[n] = dy if self._dim >= 2 else 0.0
                self._dz[n] = dz if self._dim == 3 else 0.0

            # Fill cell volumes
            for n in range(nblk):
                if self._dim == 1:
                    # Spherical shell volume assuming cell-centred radius
                    vol = (4./3.)*np.pi*((self._x[n,:] + self._dx[n]*0.5)**3 - (self._x[n,:] - self._dx[n]*0.5)**3)
                    vol = vol.reshape((1, 1, self._nxb))
                elif self._dim == 2:
                    # annular volume 2*pi*r_c*dr*dz, where r_c is cell-centred radius
                    vol = 2.*np.pi*self._x[n,:]*self._dx[n]*self._dy[n]
                    vol = np.tile(vol[None,:], (self._nyb,1)) # Broadcast to 2D array
                    vol = vol.reshape((1, self._nyb, self._nxb))
                elif self._dim == 3:
                    # Every cell in the block has a volume of dx*dy*dz
                    vol = np.full((self._nzb, self._nyb, self._nxb), self._dx[n]*self._dy[n]*self._dz[n])

                self._vol[n,:,:,:] = vol

            # Setup grid quantities
            nfields = len(self._field_list)
            self._unk = np.empty((nfields, nblk, self._nzb, self._nyb, self._nxb))
            for field in _FLASH_FIELDS:
                ifield = SNAP_FIELD_MAP[FLASH_FIELD_MAP[field]]
                self._unk[ifield,:,:,:,:] = f[field][()][leaf_mask]

            # Fill neutrino quantities
            if use_nu:
                unk = {field: f[field][()][leaf_mask] for field in _FLASH_NU_FIELDS}
                for n in range(nblk):
                    if self._dim == 1:
                        X = self._x[n,:].reshape(1, 1, self._nxb)
                        Y = np.zeros_like(X)
                        Z = np.zeros_like(X)
                    elif self._dim == 2:
                        Y, X = np.meshgrid(self._y[n,:], self._x[n,:], indexing='ij')
                        X = X.reshape(1, self._nyb, self._nxb)
                        Y = Y.reshape(1, self._nyb, self._nxb)
                        Z = np.zeros_like(X)
                    elif self._dim == 3:
                        Z, Y, X = np.meshgrid(self._z[n,:], self._y[n,:], self._x[n,:], indexing='ij')
                    r = np.sqrt(X**2 + Y**2 + Z**2)

                    self._unk[SNAP_FIELD_MAP['lum nue'  ] ,n,:,:,:] = 4.*np.pi*r**2*unk['fnue'][n]*1e51
                    self._unk[SNAP_FIELD_MAP['lum anue' ] ,n,:,:,:] = 4.*np.pi*r**2*unk['fnua'][n]*1e51
                    self._unk[SNAP_FIELD_MAP['lum nux'  ] ,n,:,:,:] = 4.*np.pi*r**2*unk['fnux'][n]*1e51*0.5
                    self._unk[SNAP_FIELD_MAP['lum anux' ] ,n,:,:,:] = 4.*np.pi*r**2*unk['fnux'][n]*1e51*0.5
                    self._unk[SNAP_FIELD_MAP['ener nue' ],n,:,:,:] = unk['enue'][n]
                    self._unk[SNAP_FIELD_MAP['ener anue'],n,:,:,:] = unk['enua'][n]
                    self._unk[SNAP_FIELD_MAP['ener nux' ],n,:,:,:] = unk['enux'][n]
                    self._unk[SNAP_FIELD_MAP['ener anux'],n,:,:,:] = unk['enux'][n]

            # Fill lookup hash table
            xb = lambda n: (self._bbox[n,0,0] + self._bbox[n,0,1])*0.5
            yb = lambda n: (self._bbox[n,1,0] + self._bbox[n,1,1])*0.5
            zb = lambda n: (self._bbox[n,2,0] + self._bbox[n,2,1])*0.5
            dxb = lambda l: (self._xmax - self._xmin)/(self._nblockx * 2**(l-1))
            dyb = lambda l: (self._ymax - self._ymin)/(self._nblocky * 2**(l-1))
            dzb = lambda l: (self._zmax - self._zmin)/(self._nblockz * 2**(l-1))

            self._hash_map = {}
            self._hash = np.empty((nblk, 4))
            for n,l in enumerate(self._levels):
                block_hash = (
                    l,
                    int((xb(n) - self._xmin)/dxb(l)),
                    int((yb(n) - self._ymin)/dyb(l)) if self._dim >= 2 else 0,
                    int((zb(n) - self._zmin)/dzb(l)) if self._dim == 3 else 0
                )
                self._hash_map[block_hash] = n
                self._hash[n] = block_hash

    def _find_neighbours(self) -> None:
        xbmin = self._bbox[:,0,0]
        xbmax = self._bbox[:,0,1]
        ybmin = self._bbox[:,1,0]
        ybmax = self._bbox[:,1,1]
        zbmin = self._bbox[:,2,0]
        zbmax = self._bbox[:,2,1]

        xi_lo = xbmin[:,None]
        yi_lo = ybmin[:,None]
        zi_lo = zbmin[:,None]
        xi_hi = xbmax[:,None]
        yi_hi = ybmax[:,None]
        zi_hi = zbmax[:,None]

        xj_lo = xbmin[None,:]
        yj_lo = ybmin[None,:]
        zj_lo = zbmin[None,:]
        xj_hi = xbmax[None,:]
        yj_hi = ybmax[None,:]
        zj_hi = zbmax[None,:]

        eps = 1e-10
        match_x = np.minimum(xi_hi, xj_hi) > np.maximum(xi_lo, xj_lo) + eps # blocks aligned on x-axis
        match_y = np.minimum(yi_hi, yj_hi) > np.maximum(yi_lo, yj_lo) + eps # blocks aligned on y-axis
        match_z = np.minimum(zi_hi, zj_hi) > np.maximum(zi_lo, zj_lo) + eps # blocks aligned on z-axis

        mask_xlo = (np.abs(xi_lo - xj_hi) < eps) & match_y & match_z # neighbours on lower x face
        mask_xhi = (np.abs(xi_hi - xj_lo) < eps) & match_y & match_z # neighbours on upper x face

        if self._dim >= 2:
            mask_ylo = (np.abs(yi_lo - yj_hi) < eps) & match_x & match_z # neighbours on lower y face
            mask_yhi = (np.abs(yi_hi - yj_lo) < eps) & match_x & match_z # neighbours on upper y face

            mask_xy00 = (np.abs(xi_lo - xj_hi) < eps) & (np.abs(yi_lo - yj_hi) < eps) & match_z # diagonal xlo/ylo
            mask_xy10 = (np.abs(xi_hi - xj_lo) < eps) & (np.abs(yi_lo - yj_hi) < eps) & match_z # diagonal xhi/ylo
            mask_xy01 = (np.abs(xi_lo - xj_hi) < eps) & (np.abs(yi_hi - yj_lo) < eps) & match_z # diagonal xlo/yhi
            mask_xy11 = (np.abs(xi_hi - xj_lo) < eps) & (np.abs(yi_hi - yj_lo) < eps) & match_z # diagonal xhi/yhi
        if self._dim == 3:
            mask_zlo = (np.abs(zi_lo - zj_hi) < eps) & match_x & match_y # neighbours on lower z face
            mask_zhi = (np.abs(zi_hi - zj_lo) < eps) & match_x & match_y # neighbours on upper z face

            mask_xz00 = (np.abs(xi_lo - xj_hi) < eps) & match_y & (np.abs(zi_lo - zj_hi) < eps) # diagonal xlo/zlo
            mask_xz10 = (np.abs(xi_hi - xj_lo) < eps) & match_y & (np.abs(zi_lo - zj_hi) < eps) # diagonal xhi/zlo
            mask_xz01 = (np.abs(xi_lo - xj_hi) < eps) & match_y & (np.abs(zi_hi - zj_lo) < eps) # diagonal xlo/zhi
            mask_xz11 = (np.abs(xi_hi - xj_lo) < eps) & match_y & (np.abs(zi_hi - zj_lo) < eps) # diagonal xhi/zhi

            mask_yz00 = match_x & (np.abs(yi_lo - yj_hi) < eps) & (np.abs(zi_lo - zj_hi) < eps) # diagonal ylo/zlo
            mask_yz10 = match_x & (np.abs(yi_hi - yj_lo) < eps) & (np.abs(zi_lo - zj_hi) < eps) # diagonal yhi/zlo
            mask_yz01 = match_x & (np.abs(yi_lo - yj_hi) < eps) & (np.abs(zi_hi - zj_lo) < eps) # diagonal ylo/zhi
            mask_yz11 = match_x & (np.abs(yi_hi - yj_lo) < eps) & (np.abs(zi_hi - zj_lo) < eps) # diagonal yhi/zhi

            mask_xyz000 = (np.abs(xi_lo - xj_hi) < eps) & (np.abs(yi_lo - yj_hi) < eps) & (np.abs(zi_lo - zj_hi) < eps) # corner xlo/ylo/zlo
            mask_xyz100 = (np.abs(xi_hi - xj_lo) < eps) & (np.abs(yi_lo - yj_hi) < eps) & (np.abs(zi_lo - zj_hi) < eps) # corner xhi/ylo/zlo
            mask_xyz010 = (np.abs(xi_lo - xj_hi) < eps) & (np.abs(yi_hi - yj_lo) < eps) & (np.abs(zi_lo - zj_hi) < eps) # corner xlo/yhi/zlo
            mask_xyz110 = (np.abs(xi_hi - xj_lo) < eps) & (np.abs(yi_hi - yj_lo) < eps) & (np.abs(zi_lo - zj_hi) < eps) # corner xhi/yhi/zlo
            mask_xyz001 = (np.abs(xi_lo - xj_hi) < eps) & (np.abs(yi_lo - yj_hi) < eps) & (np.abs(zi_hi - zj_lo) < eps) # corner xlo/ylo/zhi
            mask_xyz101 = (np.abs(xi_hi - xj_lo) < eps) & (np.abs(yi_lo - yj_hi) < eps) & (np.abs(zi_hi - zj_lo) < eps) # corner xhi/ylo/zhi
            mask_xyz011 = (np.abs(xi_lo - xj_hi) < eps) & (np.abs(yi_hi - yj_lo) < eps) & (np.abs(zi_hi - zj_lo) < eps) # corner xlo/yhi/zhi
            mask_xyz111 = (np.abs(xi_hi - xj_lo) < eps) & (np.abs(yi_hi - yj_lo) < eps) & (np.abs(zi_hi - zj_lo) < eps) # corner xhi/yhi/zhi

        nblk = self._bbox.shape[0]
        self._neighbours = [{} for n in range(nblk)]

        def push_neighbour(blockID, offset, neighbourIDs):
            # Sort neighbours by coordinates to easily locate finer neighbours
            # for restriction
            neighbourIDs = list(neighbourIDs[np.lexsort((
                self._hash[neighbourIDs,3],
                self._hash[neighbourIDs,2],
                self._hash[neighbourIDs,1],
            ))])
            self._neighbours[blockID][offset] = neighbourIDs

        for n in range(nblk):
            push_neighbour(n, (-1, 0, 0), np.where(mask_xlo[n])[0])
            push_neighbour(n, (+1, 0, 0), np.where(mask_xhi[n])[0])
            if self._dim >= 2:
                # Y Faces
                push_neighbour(n, (0, -1, 0), np.where(mask_ylo[n])[0])
                push_neighbour(n, (0, +1, 0), np.where(mask_yhi[n])[0])

                # X-Y Edges
                push_neighbour(n, (-1, -1, 0), np.where(mask_xy00[n])[0])
                push_neighbour(n, (+1, -1, 0), np.where(mask_xy10[n])[0])
                push_neighbour(n, (-1, +1, 0), np.where(mask_xy01[n])[0])
                push_neighbour(n, (+1, +1, 0), np.where(mask_xy11[n])[0])
            if self._dim == 3:
                # Z Faces
                push_neighbour(n, (0, 0, -1), np.where(mask_zlo[n])[0])
                push_neighbour(n, (0, 0, +1), np.where(mask_zhi[n])[0])

                # X-Z Edges
                push_neighbour(n, (-1, 0, -1), np.where(mask_xz00[n])[0])
                push_neighbour(n, (+1, 0, -1), np.where(mask_xz10[n])[0])
                push_neighbour(n, (-1, 0, +1), np.where(mask_xz01[n])[0])
                push_neighbour(n, (+1, 0, +1), np.where(mask_xz11[n])[0])

                # Y-Z Edges
                push_neighbour(n, (0, -1, -1), np.where(mask_yz00[n])[0])
                push_neighbour(n, (0, +1, -1), np.where(mask_yz10[n])[0])
                push_neighbour(n, (0, -1, +1), np.where(mask_yz01[n])[0])
                push_neighbour(n, (0, +1, +1), np.where(mask_yz11[n])[0])

                # Corners
                push_neighbour(n, (-1, -1, -1), np.where(mask_xyz000[n])[0])
                push_neighbour(n, (+1, -1, -1), np.where(mask_xyz100[n])[0])
                push_neighbour(n, (-1, +1, -1), np.where(mask_xyz010[n])[0])
                push_neighbour(n, (+1, +1, -1), np.where(mask_xyz110[n])[0])
                push_neighbour(n, (-1, -1, +1), np.where(mask_xyz001[n])[0])
                push_neighbour(n, (+1, -1, +1), np.where(mask_xyz101[n])[0])
                push_neighbour(n, (-1, +1, +1), np.where(mask_xyz011[n])[0])
                push_neighbour(n, (+1, +1, +1), np.where(mask_xyz111[n])[0])

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
            'level' : self._to_shared(self._levels)
        }

        self._shm_fields = self._to_shared(self._unk)

        self._desc = {
            'field_list'  : self._field_list,
            'current_time': self._current_time,
            'dim'         : self._dim,
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
            'hash_map'    : self._hash_map,
            'neighbours'  : self._neighbours,
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

