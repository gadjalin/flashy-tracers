import numpy as np

from src.model.progenitor_flash import FLASHProgenitor
from src.snap.snapshot_flash import FLASHSnapshot, FLASHSnapshotProxy

# Input
PATH_TO_SNAPSHOTS = 's20_hdf5_plt_cnt_*' # List of files or globbing pattern
PATH_TO_PROGENITOR = 's20.1d'
PATH_TO_EOS = 'SFHo.h5'

# Output
OUTPUT_DIR = 'output'
# Available variables are listed in src/buffer.py
OUTPUT_VARS = [
    'time', 'x', 'y', 'z', 'r', 'density', 'temperature', 'electron fraction', 'entropy',
    'lum nue', 'lum anue', 'lum nux', 'lum anux', 'ener nue', 'ener anue', 'ener nux', 'ener anux'
]

# Integration
NUM_TRACERS = 10000
INTEGRATE_BACKWARDS = True
TRACK_NU = True
MAX_TEMP = None
CALCULATE_SEEDS = True

# Tolerances
RTOL = 1e-2
ATOL = 1e4
MAXSTEP = 1e-4

# Tracer placement
PLACEMENT_METHOD = 'unbound' # 'file', 'uniform', 'unbound'
TRACER_FILE = 'tracers.dat'
#TRACER_MASS = 2e28 # Uniform tracer mass [g]
MAX_DENS = 1e11

# Protocols
MODEL_CLS = FLASHProgenitor
SNAP_CLS = FLASHSnapshot
PROXY_CLS = FLASHSnapshotProxy

# Misc
LOG_FILE = 'tracer_integration.log'
MAX_TRACER_BUF_SIZE = 1024*1024*1024 # Threshold tracer history buffer before saving [bytes]

