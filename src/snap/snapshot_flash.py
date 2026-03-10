from __future__ import annotations
from typing import Tuple, List, Dict, Union, Optional, Any
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

from .snapshot import Snapshot, SnapshotProxy, SNAP_FIELDS, SNAP_FIELDS_NU
from ..memory import ShmMeta, make_shared


_FIELD_MAP = {
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
    _nxb: int
    _nyb: int
    _nzb: int
    _ngx: int
    _ngy: int
    _ngz: int

    _shm_handles: List[shared_memory.SharedMemory]
    _grid: Dict[str, np.ndarray]
    _data: Dict[str, np.ndarray]

    def __init__(self, desc: Dict[str, Any]):
        self._field_list = desc['vars']
        self._current_time = desc['current_time']
        self._dim = desc['dim']

        self._nxb = desc['nxb']
        self._nyb = desc['nyb']
        self._nzb = desc['nzb']
        self._ngx = desc['ngx']
        self._ngy = desc['ngy']
        self._ngz = desc['ngz']

        self._xmin = desc['xmin']
        self._xmax = desc['xmax']
        self._ymin = desc['ymin']
        self._ymax = desc['ymax']
        self._zmin = desc['zmin']
        self._zmax = desc['zmax']

        self._shm_handles = []
        self._grid = {}
        self._data = {}
        for k,v in desc['grid'].items():
            handle = shared_memory.SharedMemory(name=v.name)
            self._grid[k] = np.ndarray(v.shape, dtype=v.dtype, buffer=handle.buf)
            self._shm_handles.append(handle)

        for k,v in desc['data'].items():
            handle = shared_memory.SharedMemory(name=v.name)
            self._data[k] = np.ndarray(v.shape, dtype=v.dtype, buffer=handle.buf)
            self._shm_handles.append(handle)

    def get_quantity(
        self,
        fields: Union[List[str], Tuple[str], str],
        x: float,
        y: Optional[float] = None,
        z: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        return self._interp_block(fields, x, y, z)

    def get_field(self, fields: Union[List[str], Tuple[str], str]) -> np.ndarray:
        if isinstance(fields, str):
            fields = [fields]
        elif isinstance(fields, (list, tuple, np.ndarray)) and all(isinstance(field, str) for field in fields):
            fields = list(fields)
        else:
            raise TypeError('Field must be a string or list of string')

        if len(fields) == 0:
            raise ValueError('Field list must contain at least one field')

        nblk = len(self._grid['bbox'])
        ncells = self._nxb*self._nyb*self._nzb

        ix_start = self._ngx
        ix_end = self._nxb + self._ngx if self._ngx > 0 else None
        iy_start = self._ngy
        iy_end = self._nyb + self._ngy if self._ngy > 0 else None
        iz_start = self._ngz
        iz_end = self._nzb + self._ngz if self._ngz > 0 else None

        dtypes = [('x', float), ('y', float), ('z', float), ('dx', float), ('dy', float), ('dz', float), ('volume', float)] + [(field, float) for field in fields]
        q = np.zeros(nblk*ncells, dtype=dtypes)
        for n in range(nblk):
            if self._dim == 1:
                X = self._grid['x'][n,ix_start:ix_end].reshape(1, 1, self._nxb)
                Y = np.zeros_like(X)
                Z = np.zeros_like(X)
            elif self._dim == 2:
                Y, X = np.meshgrid(self._grid['y'][n,iy_start:iy_end], self._grid['x'][n,ix_start:ix_end], indexing='ij')
                X = X.reshape(1, self._nyb, self._nxb)
                Y = Y.reshape(1, self._nyb, self._nxb)
                Z = np.zeros_like(X)
            elif self._dim == 3:
                Z, Y, X = np.meshgrid(self._grid['z'][n,iz_start:iz_end], self._grid['y'][n,iy_start:iy_end], self._grid['x'][n,ix_start:ix_end], indexing='ij')

            q['x'][n*ncells:(n+1)*ncells] = X.ravel()
            q['y'][n*ncells:(n+1)*ncells] = Y.ravel()
            q['z'][n*ncells:(n+1)*ncells] = Z.ravel()
            q['dx'][n*ncells:(n+1)*ncells] = self._grid['dx'][n]
            q['dy'][n*ncells:(n+1)*ncells] = self._grid['dy'][n] if self._dim >= 2 else 0.0
            q['dz'][n*ncells:(n+1)*ncells] = self._grid['dz'][n] if self._dim == 3 else 0.0
            for field in (fields + ['volume']):
                q[field][n*ncells:(n+1)*ncells] = \
                    self._data[field][n,iz_start:iz_end,iy_start:iy_end,ix_start:ix_end].ravel()

        return q

#    def cell_coords(self) -> Tuple[np.ndarray, ...]:
#        return self._x[blockID,:], self._y[blockID,:], self._z[blockID,:]

#    def cell_edges(self) -> Tuple[np.ndarray, ...]:
#        x_edges = np.concatenate(([self._x[blockID,0] - 0.5*self._dx[blockID]], self._x[blockID,:] + 0.5*self._dx[blockID]))
#        y_edges = np.concatenate(([self._y[blockID,0] - 0.5*self._dy[blockID]], self._y[blockID,:] + 0.5*self._dy[blockID]))
#        z_edges = np.concatenate(([self._z[blockID,0] - 0.5*self._dz[blockID]], self._z[blockID,:] + 0.5*self._dz[blockID]))
#        return x_edges, y_edges, z_edges

#    def cell_volumes(self) -> np.ndarray:
#        nblk = len(self._x)
#        ncells = self._nxb*self._nyb*self._nzb
#
#        ix_start = self._ngx
#        ix_end = self._nxb + self._ngx if self._ngx > 0 else None
#        iy_start = self._ngy
#        iy_end = self._nyb + self._ngy if self._ngy > 0 else None
#        iz_start = self._ngz
#        iz_end = self._nzb + self._ngz if self._ngz > 0 else None
#
#        dtypes = [('x', float), ('y', float), ('z', float), ('volume', float)]
#        res = np.zeros(nblk*ncells, dtype=dtypes)
#        for n in range(nblk):
#            if self._dim == 1:
#                X = self._x[n,ix_start:ix_end].reshape(1, 1, self._nxb)
#                Y = np.zeros_like(X)
#                Z = np.zeros_like(X)
#            elif self._dim == 2:
#                Y, X = np.meshgrid(self._y[n,iy_start:iy_end], self._x[n,ix_start:ix_end], indexing='ij')
#                X = X.reshape(1, self._nyb, self._nxb)
#                Y = Y.reshape(1, self._nyb, self._nxb)
#                Z = np.zeros_like(X)
#            elif self._dim == 3:
#                Z, Y, X = np.meshgrid(self._z[n,iz_start:iz_end], self._y[n,iy_start:iy_end], self._x[n,ix_start:ix_end], indexing='ij')
#
#            res['x'][n*ncells:(n+1)*ncells] = X.ravel()
#            res['y'][n*ncells:(n+1)*ncells] = Y.ravel()
#            res['z'][n*ncells:(n+1)*ncells] = Z.ravel()
#
#            if self._dim == 1:
#                # Spherical shell volume assuming cell-centred radius
#                vol = (4./3.)*np.pi*((self._x[n,ix_start:ix_end] + self._dx[n]/2.)**3 - (self._x[n,ix_start:ix_end] - self._dx[n]/2.)**3)
#            elif self._dim == 2:
#                # annular volume 2*pi*r_c*dr*dz, where r_c is cell-centred radius
#                vol = 2.*np.pi*self._x[n,ix_start:ix_end]*self._dx[n]*self._dy[n]
#                vol = np.tile(vol[None,:], (self._nyb,1)) # Broadcast to 2D array
#            elif self._dim == 3:
#                # Every cell in the block has a volume of dx*dy*dz
#                vol = np.full((self._nzb, self._nyb, self._nxb), self._dx[n]*self._dy[n]*self._dz[n])
#
#            res['volume'][n*ncells:(n+1)*ncells] = vol.ravel()
#
#        return res

    # TODO Try KD-Tree for O(logN)
    def find_block(self, x: float, y: Optional[float] = None, z: Optional[float] = None) -> int:
        if self._dim == 1:
            x_mask = (x >= self._grid['bbox'][:,0,0]) & (x < self._grid['bbox'][:,0,1])
            y = 0.0
            z = 0.0
            match_ids = np.where(x_mask)[0]
        elif self._dim == 2:
            if y is None:
                raise ValueError('Y coordinate must be specified in 2D')
            x_mask = (x >= self._grid['bbox'][:,0,0]) & (x < self._grid['bbox'][:,0,1])
            y_mask = (y >= self._grid['bbox'][:,1,0]) & (y < self._grid['bbox'][:,1,1])
            z = 0.0
            match_ids = np.where(x_mask & y_mask)[0]
        elif self._dim == 3:
            if y is None or z is None:
                raise ValueError('Y and Z coordinates must be specified in 3D')
            x_mask = (x >= self._grid['bbox'][:,0,0]) & (x < self._grid['bbox'][:,0,1])
            y_mask = (y >= self._grid['bbox'][:,1,0]) & (y < self._grid['bbox'][:,1,1])
            z_mask = (z >= self._grid['bbox'][:,2,0]) & (z < self._grid['bbox'][:,2,1])
            match_ids = np.where(x_mask & y_mask & z_mask)[0]

        if len(match_ids) == 1:
            return match_ids[0]
        elif len(match_ids) > 1:
            raise RuntimeError(f'PANIC: Multiple matching blocks at x={x:.5f}, y={y:.5f}, z={z:.5f}')
        else:
            raise RuntimeError(f'Coordinates x={x:.5f}, y={y:.5f}, z={z:.5f} are outside the domain')

    def _interp_block(
        self,
        fields: Union[List[str], Tuple[str], str],
        x: float,
        y: Optional[float],
        z: Optional[float]
    ) -> Union[float, np.ndarray]:
        if isinstance(fields, str):
            fields = [fields]
        elif isinstance(fields, (list, tuple, np.ndarray)) and all(isinstance(field, str) for field in fields):
            fields = list(fields)
        else:
            raise TypeError('Field must be a string or list of string')

        if len(fields) == 0:
            raise ValueError('Field list must contain at least one field')

        if self._dim == 1:
            return self._interp1d_block(fields, x)
        elif self._dim == 2:
            if y is None:
                raise ValueError('Y coordinate must be specified in 2D')
            return self._interp2d_block(fields, x, y)
        elif self._dim == 3:
            if y is None or z is None:
                raise ValueError('Y and Z coordinates must be specified in 3D')
            return self._interp3d_block(fields, x, y, z)

    def _interp1d_block(self, fields: List[str], x: float) -> Union[float, np.ndarray]:
        blockID = self.find_block(x)
        dx = self._grid['dx'][blockID]
        xmin = self._grid['bbox'][blockID,0,0]
        x = min(self._xmax - dx/2., max(self._xmin + dx/2., x))

        ix = 1 + int((x - (xmin - dx/2.)) / dx)
        x0 = self._grid['x'][blockID,ix-1]
        x1 = self._grid['x'][blockID,ix  ]

        q = np.zeros(len(fields))
        f = np.zeros(2)
        for i,field in enumerate(fields):
            f[0] = self._data[field][blockID,0,0,ix-1]
            f[1] = self._data[field][blockID,0,0,ix  ]
            q[i] = self._interp1d(x, x0, x1, f)

        if len(q) == 1:
            return q[0]
        else:
            return q

    def _interp2d_block(self, fields: List[str], x: float, y: float) -> Union[float, np.ndarray]:
        blockID = self.find_block(x, y)
        dx = self._grid['dx'][blockID]
        dy = self._grid['dy'][blockID]
        xmin = self._grid['bbox'][blockID,0,0]
        ymin = self._grid['bbox'][blockID,1,0]
        x = min(self._xmax - dx/2., max(self._xmin + dx/2., x))
        y = min(self._ymax - dy/2., max(self._ymin + dy/2., y))

        ix = 1 + int((x - (xmin - dx/2.)) / dx)
        iy = 1 + int((y - (ymin - dy/2.)) / dy)
        x0 = self._grid['x'][blockID,ix-1]
        x1 = self._grid['x'][blockID,ix  ]
        y0 = self._grid['y'][blockID,iy-1]
        y1 = self._grid['y'][blockID,iy  ]

        q = np.zeros(len(fields))
        f = np.zeros((2, 2))
        for i,field in enumerate(fields):
            f[0,0] = self._data[field][blockID,0, iy-1, ix-1]
            f[0,1] = self._data[field][blockID,0, iy  , ix-1]
            f[1,0] = self._data[field][blockID,0, iy-1, ix  ]
            f[1,1] = self._data[field][blockID,0, iy  , ix  ]
            q[i] = self._interp2d(x, y, x0, x1, y0, y1, f)

        if len(q) == 1:
            return q[0]
        else:
            return q

    def _interp3d_block(self, fields: List[str], x: float, y: float, z: float) -> Union[float, np.ndarray]:
        blockID = self.find_block(x, y, z)
        dx = self._grid['dx'][blockID]
        dy = self._grid['dy'][blockID]
        dz = self._grid['dz'][blockID]
        xmin = self._grid['bbox'][blockID,0,0]
        ymin = self._grid['bbox'][blockID,1,0]
        zmin = self._grid['bbox'][blockID,2,0]
        x = min(self._xmax - dx/2., max(self._xmin + dx/2., x))
        y = min(self._ymax - dy/2., max(self._ymin + dy/2., y))
        z = min(self._zmax - dz/2., max(self._zmin + dz/2., z))

        ix = 1 + int((x - (xmin - dx/2.)) / dx)
        iy = 1 + int((y - (ymin - dy/2.)) / dy)
        iz = 1 + int((z - (zmin - dz/2.)) / dz)
        x0 = self._grid['x'][blockID,ix-1]
        x1 = self._grid['x'][blockID,ix  ]
        y0 = self._grid['y'][blockID,iy-1]
        y1 = self._grid['y'][blockID,iy  ]
        z0 = self._grid['z'][blockID,iz-1]
        z1 = self._grid['z'][blockID,iz  ]

        q = np.zeros(len(fields))
        f = np.zeros((2, 2, 2))
        for i,field in enumerate(fields):
            f[0,0,0] = self._data[field][blockID,iz-1, iy-1, ix-1]
            f[0,0,1] = self._data[field][blockID,iz  , iy-1, ix-1]
            f[0,1,0] = self._data[field][blockID,iz-1, iy  , ix-1]
            f[0,1,1] = self._data[field][blockID,iz  , iy  , ix-1]
            f[1,0,0] = self._data[field][blockID,iz-1, iy-1, ix  ]
            f[1,0,1] = self._data[field][blockID,iz  , iy-1, ix  ]
            f[1,1,0] = self._data[field][blockID,iz-1, iy  , ix  ]
            f[1,1,1] = self._data[field][blockID,iz  , iy  , ix  ]
            q[i] = self._interp3d(x, y, z, x0, x1, y0, y1, z0, z1, f)

        if len(q) == 1:
            return q[0]
        else:
            return q

    @staticmethod
    def _interp1d(
        x: float,
        x0: float, x1: float,
        f: np.ndarray
    ) -> float:
        t = (x - x0)/(x1 - x0)
        return f[0] + t*(f[1] - f[0])

    @staticmethod
    def _interp2d(
        x: float, y: float,
        x0: float, x1: float, y0: float, y1: float,
        f: np.ndarray
    ) -> float:
        t = (x - x0)/(x1 - x0)
        f0 = f[0,0] + t*(f[1,0] - f[0,0])
        f1 = f[0,1] + t*(f[1,1] - f[0,1])
        return FLASHSnapshotProxy._interp1d(y, y0, y1, np.array([f0, f1]))

    @staticmethod
    def _interp3d(
        x: float, y: float, z: float,
        x0: float, x1: float, y0: float, y1: float, z0: float, z1: float,
        f: np.ndarray
    ) -> float:
        raise NotImplementedError('3D interpolation')

    def close(self) -> None:
        for handle in self._shm_handles:
            handle.close()
        self._shm_handles.clear()

    def __del__(self):
        self.close()


class FLASHSnapshot(Snapshot):
    _bbox: np.ndarray
    _data: Dict[str, np.ndarray]

    _levels: np.ndarray
    _neighbours: List[Dict[str, List[int]]]
    _faces_meta: Dict[str, Tuple[int, ...]]

    _nxb: int
    _nyb: int
    _nzb: int
    _ngx: int
    _ngy: int
    _ngz: int
    _x: np.ndarray
    _y: np.ndarray
    _z: np.ndarray
    _dx: np.ndarray
    _dy: np.ndarray
    _dz: np.ndarray

    _desc: Dict[str, Any]
    _proxy: FLASHSnapshotProxy
    _shm_handles: List[shared_memory.SharedMemory]
    _shm_grid: Dict[str, ShmMeta]
    _shm_data: Dict[str, ShmMeta]

    def __init__(self, filename: str, use_nu: Optional[bool] = False):
        self._field_list = (SNAP_FIELDS + SNAP_FIELDS_NU) if use_nu else SNAP_FIELDS

        self._read_data(filename, use_nu)

        self._faces_meta = {
            'x-': (2, -1, self._ngx),
            'x+': (2,  1, self._ngx),
            'y-': (1, -1, self._ngy),
            'y+': (1,  1, self._ngy),
            'z-': (0, -1, self._ngz),
            'z+': (0,  1, self._ngz),
        }

        self._find_neighbours()
        self._fill_gc()

        self._setup_shm()

    # Utility method to sort plot files without loading everything
    @staticmethod
    def get_time(filename: str) -> float:
        with h5py.File(filename, 'r') as f:
            real_scalars = {k.decode('ascii').strip(): v for k,v in f['real scalars'][()]}
        return real_scalars['time']

    def get_quantity(
        self,
        fields: Union[List[str], str],
        x: float,
        y: Optional[float] = None,
        z: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        return self._proxy.get_quantity(fields, x, y, z)

    def get_field(self, fields: Union[List[str], Tuple[str], str]) -> np.ndarray:
        return self._proxy.get_field(fields)

    def _read_data(self, filename: str, use_nu: bool) -> None:
        with h5py.File(filename, 'r') as f:
            # Decode datasets
            integer_scalars = {k.decode('ascii').strip(): v for k,v in f['integer scalars'][()]}
            real_scalars = {k.decode('ascii').strip(): v for k,v in f['real scalars'][()]}
            real_runtime_parameters = {k.decode('ascii').strip(): v for k,v in f['real runtime parameters'][()]}

            # Read simulation metadata
            self._dim = integer_scalars['dimensionality']
            if self._dim < 1 or self._dim > 3:
                raise NotImplementedError(f'Invalid dimensionality: {self._dim}')

            self._current_time = real_scalars['time']

            # Grid boundaries, block size, ghost cells
            self._nxb = integer_scalars['nxb']
            self._nyb = integer_scalars['nyb']
            self._nzb = integer_scalars['nzb']

            if self._dim == 1:
                self._xmin = real_runtime_parameters['xmin']
                self._xmax = real_runtime_parameters['xmax']
                self._ymin = 0
                self._ymax = np.pi
                self._zmin = 0
                self._zmax = 2.*np.pi
                self._ngx = 1
                self._ngy = 0
                self._ngz = 0
            if self._dim == 2:
                self._xmin = real_runtime_parameters['xmin']
                self._xmax = real_runtime_parameters['xmax']
                self._ymin = real_runtime_parameters['ymin']
                self._ymax = real_runtime_parameters['ymax']
                self._zmin = 0
                self._zmax = 2.*np.pi
                self._ngx = 1
                self._ngy = 1
                self._ngz = 0
            if self._dim == 3:
                self._xmin = real_runtime_parameters['xmin']
                self._xmax = real_runtime_parameters['xmax']
                self._ymin = real_runtime_parameters['ymin']
                self._ymax = real_runtime_parameters['ymax']
                self._zmin = real_runtime_parameters['zmin']
                self._zmax = real_runtime_parameters['zmax']
                self._ngx = 1
                self._ngy = 1
                self._ngz = 1

            # Keep quantities from leaf blocks
            node_type = f['node type'][()]
            leaf_mask = (node_type == 1)

            self._bbox = f['bounding box'][()][leaf_mask]
            self._levels = f['refine level'][()][leaf_mask]

            # Calculate cell-centred coordinates
            nblk = self._bbox.shape[0]
            self._x = np.zeros((nblk, self._nxb+self._ngx*2))
            self._y = np.zeros((nblk, self._nyb+self._ngy*2))
            self._z = np.zeros((nblk, self._nzb+self._ngz*2))
            self._dx = np.zeros(nblk)
            self._dy = np.zeros(nblk)
            self._dz = np.zeros(nblk)
            for n in range(nblk):
                xmin = self._bbox[n, 0, 0]
                xmax = self._bbox[n, 0, 1]

                if self._dim >= 2:
                    ymin = self._bbox[n, 1, 0]
                    ymax = self._bbox[n, 1, 1]
                else:
                    ymin = 0
                    ymax = np.pi

                if self._dim == 3:
                    zmin = self._bbox[n, 2, 0]
                    zmax = self._bbox[n, 2, 1]
                else:
                    zmin = 0
                    zmax = 2.*np.pi

                dx = abs(xmax - xmin) / self._nxb
                dy = abs(ymax - ymin) / self._nyb
                dz = abs(zmax - zmin) / self._nzb

                self._x[n,:] = xmin - (dx*self._ngx) + (np.arange(self._nxb+self._ngx*2) + 0.5)*dx
                if self._dim >= 2:
                    self._y[n,:] = ymin - (dy*self._ngy) + (np.arange(self._nyb+self._ngy*2) + 0.5)*dy
                if self._dim == 2:
                    self._z[n,:] = zmin - (dz*self._ngz) + (np.arange(self._nzb+self._ngz*2) + 0.5)*dz
                self._dx[n] = dx
                self._dy[n] = dy
                self._dz[n] = dz

            # Setup grid quantities with ghost cells
            self._data = {
                field: np.zeros((nblk, self._nzb+self._ngz*2, self._nyb+self._ngy*2, self._nxb+self._ngx*2))
                for field in (self._field_list + ['volume'])
            }

            # Exclude ghost cells
            ix_start = self._ngx
            ix_end = self._nxb + self._ngx if self._ngx > 0 else None
            iy_start = self._ngy
            iy_end = self._nyb + self._ngy if self._ngy > 0 else None
            iz_start = self._ngz
            iz_end = self._nzb + self._ngz if self._ngz > 0 else None

            # Fill grid quantities
            tmp = {field: f[field][()][leaf_mask] for field in _FLASH_FIELDS}
            for k,v in tmp.items():
                for n in range(nblk):
                    self._data[_FIELD_MAP[k]][n,iz_start:iz_end,iy_start:iy_end,ix_start:ix_end] = v[n]

            # Just in case
            if self._dim < 3:
                self._data['velocity-z'][:] = 0.0
            if self._dim < 2:
                self._data['velocity-y'][:] = 0.0

            # Fill cell volumes
            for n in range(nblk):
                if self._dim == 1:
                    # Spherical shell volume assuming cell-centred radius
                    vol = (4./3.)*np.pi*((self._x[n,ix_start:ix_end] + self._dx[n]/2.)**3 - (self._x[n,ix_start:ix_end] - self._dx[n]/2.)**3)
                    vol = vol.reshape((1, 1, self._nxb))
                elif self._dim == 2:
                    # annular volume 2*pi*r_c*dr*dz, where r_c is cell-centred radius
                    vol = 2.*np.pi*self._x[n,ix_start:ix_end]*self._dx[n]*self._dy[n]
                    vol = np.tile(vol[None,:], (self._nyb,1)) # Broadcast to 2D array
                    vol = vol.reshape((1, self._nyb, self._nxb))
                elif self._dim == 3:
                    # Every cell in the block has a volume of dx*dy*dz
                    vol = np.full((self._nzb, self._nyb, self._nxb), self._dx[n]*self._dy[n]*self._dz[n])

                self._data['volume'][n,iz_start:iz_end,iy_start:iy_end,ix_start:ix_end] = vol

            # Fill neutrino quantities
            if use_nu:
                tmp = {field: f[field][()][leaf_mask] for field in _FLASH_NU_FIELDS}
                for n in range(nblk):
                    if self._dim == 1:
                        X = self._x[n,ix_start:ix_end].reshape(1, 1, self._nxb)
                        Y = np.zeros_like(X)
                        Z = np.zeros_like(X)
                    elif self._dim == 2:
                        Y, X = np.meshgrid(self._y[n,iy_start:iy_end], self._x[n,ix_start:ix_end], indexing='ij')
                        X = X.reshape(1, self._nyb, self._nxb)
                        Y = Y.reshape(1, self._nyb, self._nxb)
                        Z = np.zeros_like(X)
                    elif self._dim == 3:
                        Z, Y, X = np.meshgrid(self._z[n,iz_start:iz_end], self._y[n,iy_start:iy_end], self._x[n,ix_start:ix_end], indexing='ij')
                    r = np.sqrt(X**2 + Y**2 + Z**2)

                    self._data['lum nue'  ][n,iz_start:iz_end,iy_start:iy_end,ix_start:ix_end] = 4.*np.pi*r**2*tmp['fnue'][n]*1e51
                    self._data['lum anue' ][n,iz_start:iz_end,iy_start:iy_end,ix_start:ix_end] = 4.*np.pi*r**2*tmp['fnua'][n]*1e51
                    self._data['lum nux'  ][n,iz_start:iz_end,iy_start:iy_end,ix_start:ix_end] = 4.*np.pi*r**2*tmp['fnux'][n]*1e51*0.5
                    self._data['lum anux' ][n,iz_start:iz_end,iy_start:iy_end,ix_start:ix_end] = 4.*np.pi*r**2*tmp['fnux'][n]*1e51*0.5
                    self._data['ener nue' ][n,iz_start:iz_end,iy_start:iy_end,ix_start:ix_end] = tmp['enue'][n]
                    self._data['ener anue'][n,iz_start:iz_end,iy_start:iy_end,ix_start:ix_end] = tmp['enua'][n]
                    self._data['ener nux' ][n,iz_start:iz_end,iy_start:iy_end,ix_start:ix_end] = tmp['enux'][n]
                    self._data['ener anux'][n,iz_start:iz_end,iy_start:iy_end,ix_start:ix_end] = tmp['enux'][n]

    def _find_neighbours(self) -> None:
        xmin = self._bbox[:,0,0]
        xmax = self._bbox[:,0,1]
        ymin = self._bbox[:,1,0]
        ymax = self._bbox[:,1,1]
        zmin = self._bbox[:,2,0]
        zmax = self._bbox[:,2,1]

        xi_lo = xmin[:,None]
        yi_lo = ymin[:,None]
        zi_lo = zmin[:,None]
        xi_hi = xmax[:,None]
        yi_hi = ymax[:,None]
        zi_hi = zmax[:,None]

        xj_lo = xmin[None,:]
        yj_lo = ymin[None,:]
        zj_lo = zmin[None,:]
        xj_hi = xmax[None,:]
        yj_hi = ymax[None,:]
        zj_hi = zmax[None,:]

        eps = 1e-10
        match_x = np.minimum(xi_hi, xj_hi) > np.maximum(xi_lo, xj_lo) + eps
        match_y = np.minimum(yi_hi, yj_hi) > np.maximum(yi_lo, yj_lo) + eps
        match_z = np.minimum(zi_hi, zj_hi) > np.maximum(zi_lo, zj_lo) + eps

        mask_xlo = (np.abs(xi_lo - xj_hi) < eps) & match_y & match_z
        mask_xhi = (np.abs(xi_hi - xj_lo) < eps) & match_y & match_z
        mask_ylo = (np.abs(yi_lo - yj_hi) < eps) & match_x & match_z
        mask_yhi = (np.abs(yi_hi - yj_lo) < eps) & match_x & match_z
        mask_zlo = (np.abs(zi_lo - zj_hi) < eps) & match_x & match_y
        mask_zhi = (np.abs(zi_hi - zj_lo) < eps) & match_x & match_y

        nblk = self._bbox.shape[0]
        self._neighbours = [{} for n in range(nblk)]
        for n in range(nblk):
            self._neighbours[n] = {
                'x-': list(np.where(mask_xlo[n])[0]),
                'x+': list(np.where(mask_xhi[n])[0]),
                'y-': list(np.where(mask_ylo[n])[0]),
                'y+': list(np.where(mask_yhi[n])[0]),
                'z-': list(np.where(mask_zlo[n])[0]),
                'z+': list(np.where(mask_zhi[n])[0])
            }

    def _fill_gc(self) -> None:
        nblk = self._bbox.shape[0]

        # Boundary, copy, coarse from fine
        for n in range(nblk):
            for face,neighbours in self._neighbours[n].items():
                _, _, ng = self._faces_meta[face]
                if ng == 0:
                    continue

                if len(neighbours) == 0:
                    self._amr_boundary(n, face)
                elif len(neighbours) == 1 and self._levels[n] == self._levels[neighbours[0]]:
                    self._amr_copy(n, neighbours[0], face)
                elif len(neighbours) > 1:
                    self._amr_restrict(n, neighbours, face)

        # fine from coarse
        for n in range(nblk):
            for face,neighbours in self._neighbours[n].items():
                _, _, ng = self._faces_meta[face]
                if ng == 0:
                    continue

                if len(neighbours) == 1 and self._levels[n] > self._levels[neighbours[0]]:
                    self._amr_prolong(n, neighbours[0], face)
            # FIXME: Guard cells in block corners are not set
            for field in self._field_list:
                if self._ngx > 0 and self._ngy > 0:
                    self._data[field][n,:, self._ngy-1, self._ngx-1] = self._data[field][n,:, self._ngy  , self._ngx  ]
                    self._data[field][n,:, self._ngy-1,-self._ngx  ] = self._data[field][n,:, self._ngy  ,-self._ngx-1]
                    self._data[field][n,:,-self._ngy  ,-self._ngx  ] = self._data[field][n,:,-self._ngy-1,-self._ngx-1]
                    self._data[field][n,:,-self._ngy  , self._ngx-1] = self._data[field][n,:,-self._ngy-1, self._ngx  ]
                if self._ngx > 0 and self._ngz > 0:
                    self._data[field][n, self._ngz-1,:, self._ngx-1] = self._data[field][n, self._ngz  ,:, self._ngx  ]
                    self._data[field][n, self._ngz-1,:,-self._ngx  ] = self._data[field][n, self._ngz  ,:,-self._ngx-1]
                    self._data[field][n,-self._ngz  ,:,-self._ngx  ] = self._data[field][n,-self._ngz-1,:,-self._ngx-1]
                    self._data[field][n,-self._ngz  ,:, self._ngx-1] = self._data[field][n,-self._ngz-1,:, self._ngx  ]
                if self._ngy > 0 and self._ngz > 0:
                    self._data[field][n, self._ngz-1, self._ngy-1,:] = self._data[field][n, self._ngz  , self._ngy  ,:]
                    self._data[field][n, self._ngz-1,-self._ngy  ,:] = self._data[field][n, self._ngz  ,-self._ngy-1,:]
                    self._data[field][n,-self._ngz  ,-self._ngy  ,:] = self._data[field][n,-self._ngz-1,-self._ngy-1,:]
                    self._data[field][n,-self._ngz  , self._ngy-1,:] = self._data[field][n,-self._ngz-1, self._ngy  ,:]
                if self._ngx > 0 and self._ngy > 0 and self._ngz > 0:
                    self._data[field][n, self._ngz-1, self._ngy-1, self._ngx-1] = self._data[field][n, self._ngz  , self._ngy  , self._ngx  ]
                    self._data[field][n, self._ngz-1, self._ngy-1,-self._ngx  ] = self._data[field][n, self._ngz  , self._ngy  ,-self._ngx-1]
                    self._data[field][n, self._ngz-1,-self._ngy  ,-self._ngx  ] = self._data[field][n, self._ngz  ,-self._ngy-1,-self._ngx-1]
                    self._data[field][n, self._ngz-1,-self._ngy  , self._ngx-1] = self._data[field][n, self._ngz  ,-self._ngy-1, self._ngx  ]
                    self._data[field][n,-self._ngz  , self._ngy-1, self._ngx-1] = self._data[field][n,-self._ngz-1, self._ngy  , self._ngx  ]
                    self._data[field][n,-self._ngz  , self._ngy-1,-self._ngx  ] = self._data[field][n,-self._ngz-1, self._ngy  ,-self._ngx-1]
                    self._data[field][n,-self._ngz  ,-self._ngy  ,-self._ngx  ] = self._data[field][n,-self._ngz-1,-self._ngy-1,-self._ngx-1]
                    self._data[field][n,-self._ngz  ,-self._ngy  , self._ngx-1] = self._data[field][n,-self._ngz-1,-self._ngy-1, self._ngx  ]

    def _amr_boundary(self, n, face) -> None:
        ixlo, ixhi, iylo, iyhi, izlo, izhi = self._blck_lim()

        sl_src = [slice(izlo,izhi), slice(iylo,iyhi), slice(ixlo,ixhi)]
        sl_dst = [slice(izlo,izhi), slice(iylo,iyhi), slice(ixlo,ixhi)]
        axis, sign, ng = self._faces_meta[face]

        if sign < 0:
            sl_dst[axis] = slice(None, ng)
            sl_src[axis] = slice(ng,   ng+1)
        else:
            sl_dst[axis] = slice(-ng,   None)
            sl_src[axis] = slice(-ng-1, -ng)

        sl_dst = tuple([n, *sl_dst])
        sl_src = tuple([n, *sl_src])
        for field in self._field_list:
            self._data[field][sl_dst] = self._data[field][sl_src]

    def _amr_copy(self, n_dst, n_src, face) -> None:
        ixlo, ixhi, iylo, iyhi, izlo, izhi = self._blck_lim()

        sl_src = [slice(izlo,izhi), slice(iylo,iyhi), slice(ixlo,ixhi)]
        sl_dst = [slice(izlo,izhi), slice(iylo,iyhi), slice(ixlo,ixhi)]
        axis, sign, ng = self._faces_meta[face]

        if sign < 0:
            sl_dst[axis] = slice(None, ng)
            sl_src[axis] = slice(-2*ng,  -ng)
        else:
            sl_dst[axis] = slice(-ng,  None)
            sl_src[axis] = slice(ng, 2*ng)

        sl_dst = tuple([n_dst, *sl_dst])
        sl_src = tuple([n_src, *sl_src])
        for field in self._field_list:
            self._data[field][sl_dst] = self._data[field][sl_src]

    def _amr_restrict(self, n_dst, n_srcs, face) -> None:
        ixlo, ixhi, iylo, iyhi, izlo, izhi = self._blck_lim()

        sl_src = [slice(izlo,izhi), slice(iylo,iyhi), slice(ixlo,ixhi)]
        sl_dst = [slice(izlo,izhi), slice(iylo,iyhi), slice(ixlo,ixhi)]
        sl_fine_dst = [[slice(None,None),slice(None,None),slice(None,None)] for _ in range(len(n_srcs))]
        axis, sign, ng = self._faces_meta[face]

        if sign < 0:
            sl_dst[axis] = slice(None, ng)
            sl_src[axis] = slice(-3*ng,  -ng)
        else:
            sl_dst[axis] = slice(-ng,  None)
            sl_src[axis] = slice(ng, 3*ng)

        if self._dim == 1:
            fine_face_shape = [1, 1, 2*self._nxb]
            coarse_face_shape = [1, 1, self._nxb]
        elif self._dim == 2:
            fine_face_shape = [1, 2*self._nyb, 2*self._nxb]
            coarse_face_shape = [1, self._nyb, self._nxb]
        elif self._dim == 3:
            fine_face_shape = [2*self._nzb, 2*self._nyb, 2*self._nxb]
            coarse_face_shape = [self._nzb, self._nyb, self._nxb]
        fine_face_shape[axis] = 2*ng
        coarse_face_shape[axis] = ng

        xilo, xihi = self._bbox[n_dst,0,0], self._bbox[n_dst,0,1]
        yilo, yihi = self._bbox[n_dst,1,0], self._bbox[n_dst,1,1]
        zilo, zihi = self._bbox[n_dst,2,0], self._bbox[n_dst,2,1]
        dx = self._dx[n_dst]/2.
        dy = self._dy[n_dst]/2.
        dz = self._dz[n_dst]/2.

        for it,j in enumerate(n_srcs):
            xjlo, xjhi = self._bbox[j,0,0], self._bbox[j,0,1]
            yjlo, yjhi = self._bbox[j,1,0], self._bbox[j,1,1]
            zjlo, zjhi = self._bbox[j,2,0], self._bbox[j,2,1]

            xmin = max(xilo, xjlo)
            xmax = min(xihi, xjhi)
            ymin = max(yilo, yjlo)
            ymax = min(yihi, yjhi)
            zmin = max(zilo, zjlo)
            zmax = min(zihi, zjhi)

            x0 = max(int((xmin - (xilo - dx/2.))/dx), 0)
            y0 = max(int((ymin - (yilo - dy/2.))/dy), 0)
            z0 = max(int((zmin - (zilo - dz/2.))/dz), 0)
            x1 =     int((xmax - (xilo - dx/2.))/dx)
            y1 =     int((ymax - (yilo - dy/2.))/dy)
            z1 =     int((zmax - (zilo - dz/2.))/dz)
            if x0 >= (self._nxb*2 - 1) or x0 <= 0: x0 = None
            if y0 >= (self._nyb*2 - 1) or y0 <= 0: y0 = None
            if z0 >= (self._nzb*2 - 1) or z0 <= 0: z0 = None
            if x1 >= (self._nxb*2 - 1) or x1 <= 0: x1 = None
            if y1 >= (self._nyb*2 - 1) or y1 <= 0: y1 = None
            if z1 >= (self._nzb*2 - 1) or z1 <= 0: z1 = None

            sl_fine_dst[it][2] = slice(x0,x1)
            sl_fine_dst[it][1] = slice(y0,y1)
            sl_fine_dst[it][0] = slice(z0,z1)

        sl_dst = tuple([n_dst, *sl_dst])
        for field in self._field_list:
            fine_grid = np.zeros(fine_face_shape)

            for it,j in enumerate(n_srcs):
                sl_fine_src = tuple([j, *sl_src])
                fine_grid[tuple(sl_fine_dst[it])] = self._data[field][sl_fine_src]

            if self._dim == 1:
                self._data[field][sl_dst] = \
                    fine_grid.reshape(1, 1, coarse_face_shape[2], 2).mean(axis=3)
            elif self._dim == 2:
                self._data[field][sl_dst] = \
                    fine_grid.reshape(1, coarse_face_shape[1], 2, coarse_face_shape[2], 2).mean(axis=(2, 4))
            elif self._dim == 3:
                self._data[field][sl_dst] = \
                    fine_grid.reshape(coarse_face_shape[0], 2, coarse_face_shape[1], 2, coarse_face_shape[2], 2).mean(axis=(1,3,5))

    def _amr_prolong(self, n_dst, n_src, face) -> None:
        ixlo, ixhi, iylo, iyhi, izlo, izhi = self._blck_lim()

        sl_src = [slice(izlo,izhi), slice(iylo,iyhi), slice(ixlo,ixhi)]
        sl_dst = [slice(izlo,izhi), slice(iylo,iyhi), slice(ixlo,ixhi)]
        sl_fine_src = [slice(None,None), slice(None,None), slice(None,None)]
        axis, sign, ng = self._faces_meta[face]

        xilo, xihi = self._bbox[n_dst,0,0], self._bbox[n_dst,0,1]
        yilo, yihi = self._bbox[n_dst,1,0], self._bbox[n_dst,1,1]
        zilo, zihi = self._bbox[n_dst,2,0], self._bbox[n_dst,2,1]
        xjlo, xjhi = self._bbox[n_src,0,0], self._bbox[n_src,0,1]
        yjlo, yjhi = self._bbox[n_src,1,0], self._bbox[n_src,1,1]
        zjlo, zjhi = self._bbox[n_src,2,0], self._bbox[n_src,2,1]
        dx = self._dx[n_src]
        dy = self._dy[n_src]
        dz = self._dz[n_src]

        xmin = max(xilo, xjlo)
        xmax = min(xihi, xjhi)
        ymin = max(yilo, yjlo)
        ymax = min(yihi, yjhi)
        zmin = max(zilo, zjlo)
        zmax = min(zihi, zjhi)

        x0 = max(int((xmin - (xjlo - dx/2.))/dx), 0)
        y0 = max(int((ymin - (yjlo - dy/2.))/dy), 0)
        z0 = max(int((zmin - (zjlo - dz/2.))/dz), 0)
        x1 =     int((xmax - (xjlo - dx/2.))/dx) + 2
        y1 =     int((ymax - (yjlo - dy/2.))/dy) + 2
        z1 =     int((zmax - (zjlo - dz/2.))/dz) + 2
        if x0 <= 0 or x0 >= (self._nxb + 1): x0 = None
        if x1 <= 0 or x1 >= (self._nxb + 1): x1 = None
        if y0 <= 0 or y0 >= (self._nyb + 1): y0 = None
        if y1 <= 0 or y1 >= (self._nyb + 1): y1 = None
        if z0 <= 0 or z0 >= (self._nzb + 1): z0 = None
        if z1 <= 0 or z1 >= (self._nzb + 1): z1 = None

        sl_src[2] = slice(x0,x1)
        sl_src[1] = slice(y0,y1)
        sl_src[0] = slice(z0,z1)

        if sign < 0:
            sl_dst[axis] = slice(None, ng)
            sl_src[axis] = slice(-ng-int(ng/2)-1, -ng+1 if ng > 1 else None)
        else:
            sl_dst[axis] = slice(-ng,  None)
            sl_src[axis] = slice(ng-1 if ng > 1 else None, ng+int(ng/2)+1)

        if self._dim == 1:
            fine_face_shape = [1, 1, self._nxb]
        elif self._dim == 2:
            fine_face_shape = [1, self._nyb, self._nxb]
        elif self._dim == 3:
            fine_face_shape = [self._nzb, self._nyb, self._nxb]
        fine_face_shape[axis] = ng

        sl_dst = tuple([n_dst, *sl_dst])
        sl_src = tuple([n_src, *sl_src])
        # FIXME: fine grid interpolation is offset in face direction by dxf/2
        for field in self._field_list:
            coarse_grid = self._data[field][sl_src]
            coarse_face_shape = coarse_grid.shape
            dxc = 1./coarse_face_shape[2]
            dyc = 1./coarse_face_shape[1]
            dzc = 1./coarse_face_shape[0]
            xc = dxc/2. + (np.arange(coarse_face_shape[2])*dxc)
            yc = dyc/2. + (np.arange(coarse_face_shape[1])*dyc)
            zc = dzc/2. + (np.arange(coarse_face_shape[0])*dzc)
            coarse_interp = RegularGridInterpolator((zc, yc, xc), coarse_grid, method='linear')

            dxf = (1.0 - 2.*dxc)/fine_face_shape[2]
            dyf = (1.0 - 2.*dyc)/fine_face_shape[1]
            dzf = (1.0 - 2.*dzc)/fine_face_shape[0]
            xf = dxc + dxf/2. + (np.arange(fine_face_shape[2])*dxf)
            yf = dyc + dyf/2. + (np.arange(fine_face_shape[1])*dyf)
            zf = dzc + dzf/2. + (np.arange(fine_face_shape[0])*dzf)
            Z, Y, X = np.meshgrid(zf, yf, xf, indexing='ij')

            self._data[field][sl_dst] = coarse_interp(np.array([Z.ravel(), Y.ravel(), X.ravel()]).T).reshape(fine_face_shape)

    def _blck_lim(self) -> Tuple[int, ...]:
        ixlo = self._ngx if self._ngx > 0 else None
        ixhi = self._ngx + self._nxb if self._ngx > 0 else None
        iylo = self._ngy if self._ngy > 0 else None
        iyhi = self._ngy + self._nyb if self._ngy > 0 else None
        izlo = self._ngz if self._ngz > 0 else None
        izhi = self._ngz + self._nzb if self._ngz > 0 else None
        return ixlo, ixhi, iylo, iyhi, izlo, izhi

    def get_proxy_descriptor(self):
        return self._desc

    def _setup_shm(self) -> None:
        self._shm_handles = []
        self._shm_grid = {}
        self._shm_data = {}

        self._shm_grid = {
            'bbox': self._to_shared(self._bbox),
            'x'   : self._to_shared(self._x),
            'y'   : self._to_shared(self._y),
            'z'   : self._to_shared(self._z),
            'dx'  : self._to_shared(self._dx),
            'dy'  : self._to_shared(self._dy),
            'dz'  : self._to_shared(self._dz),
        }

        self._shm_data = {
            k: self._to_shared(v)
            for k, v in self._data.items()
        }

        self._desc = {
            'vars': self._field_list,
            'current_time': self._current_time,
            'dim': self._dim,
            'nxb': self._nxb,
            'nyb': self._nyb,
            'nzb': self._nzb,
            'ngx': self._ngx,
            'ngy': self._ngy,
            'ngz': self._ngz,
            'xmin': self._xmin,
            'xmax': self._xmax,
            'ymin': self._ymin,
            'ymax': self._ymax,
            'zmin': self._zmin,
            'zmax': self._zmax,
            'grid': self._shm_grid,
            'data': self._shm_data
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

