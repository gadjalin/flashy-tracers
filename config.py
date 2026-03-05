import numpy as np

from src.progenitor_flash import FLASHProgenitor
from src.snapshot_flash import FLASHSnapshot, FLASHSnapshotProxy

# Input
PATH_TO_SNAPSHOTS = '/home/gjalin/project/runs/RSG_s12.6_caseC/2D/RSG_hot/RSG_hot_hdf5_plt_cnt_*'
PATH_TO_PROGENITOR = '/home/gjalin/project/runs/RSG_s12.6_caseC/2D/RSG_s12.6.1d'
PATH_TO_EOS = '/home/gjalin/data/eos_tables/SFHo_low.h5'

# Output
OUTPUT_DIR = 'RSG_10000'
# Available variables in src/buffer.py
OUTPUT_VARS = [
    'time', 'x', 'y', 'z', 'r', 'density', 'temperature', 'electron fraction', 'entropy',
    'lum nue', 'lum anue', 'lum nux', 'lum anux', 'ener nue', 'ener anue', 'ener nux', 'ener anux'
]

# Integration
NUM_TRACERS = 10000
INTEGRATE_BACKWARDS = True
CALCULATE_SEEDS = True
TRACK_NU = True
MAX_TEMP = None
#MAX_TEMP = 5.8e9

# Tolerances
RTOL = 1e-2
ATOL = 1e4
MAXSTEP = 1e-4

# Tracer placement
PLACEMENT_METHOD = 'unbound' # 'file', 'uniform', 'unbound'
TRACER_FILE = 'tracers.dat'
TRACER_MASS = 2e28 # Uniform tracer mass [g]
MAX_DENS = 1e11

# Protocols
MODEL_CLS = FLASHProgenitor
SNAP_CLS = FLASHSnapshot
PROXY_CLS = FLASHSnapshotProxy

# Misc
LOG_FILE = 'tracer_integration.log'
MAX_TRACER_BUF_SIZE = 1024*1024*1024 # Threshold tracer history buffer before saving [bytes]

