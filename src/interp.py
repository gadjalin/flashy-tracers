from typing import Union, Sequence

import numpy as np


def interp1d(
    x: float,
    x0: float, x1: float,
    f: Union[Sequence, np.ndarray]
) -> float:
    t = (x - x0)/(x1 - x0)
    return f[0] + t*(f[1] - f[0])


def interp2d(
    x: float, y: float,
    x0: float, x1: float,
    y0: float, y1: float,
    f: Union[Sequence, np.ndarray]
) -> float:
    tx = (x - x0)/(x1 - x0)
    ty = (y - y0)/(y1 - y0)
    fy0 = f[0][0] + tx*(f[1][0] - f[0][0])
    fy1 = f[0][1] + tx*(f[1][1] - f[0][1])
    return fy0 + ty*(fy1 - fy0)


def interp3d(
    x: float, y: float, z: float,
    x0: float, x1: float,
    y0: float, y1: float,
    z0: float, z1: float,
    f: Union[Sequence, np.ndarray]
) -> float:
    tx = (x - x0)/(x1 - x0)
    ty = (y - y0)/(y1 - y0)
    tz = (z - z0)/(z1 - z0)
    fy00 = f[0][0][0] + tx*(f[1][0][0] - f[0][0][0])
    fy10 = f[0][1][0] + tx*(f[1][1][0] - f[0][1][0])
    fy01 = f[0][0][1] + tx*(f[1][0][1] - f[0][0][1])
    fy11 = f[0][1][1] + tx*(f[1][1][1] - f[0][1][1])
    fz0 = fy00 + ty*(fy10 - fy00)
    fz1 = fy01 + ty*(fy11 - fy01)
    return fz0 + tz*(fz1 - fz0)

