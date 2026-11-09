from src.model.progenitor_flash import FLASHProgenitor
from src.snap.snapshot_flash import FLASHSnapshot, FLASHSnapshotProxy

# Input
PATH_TO_SNAPSHOTS =  '/home/gjalin/project/binary_network/runs/RSG_s12.6_caseC/2D/RSG_rn16e/RSG_rn16e_hdf5_plt_cnt_*' # List of files or globbing pattern
PATH_TO_PROGENITOR = '/home/gjalin/project/binary_network/runs/RSG_s12.6_caseC/2D/RSG_s12.6_iso16.1d'
PATH_TO_EOS = '/home/gjalin/data/eos_tables/SFHo_low.h5'

# Output
OUTPUT_DIR = '/home/gjalin/project/binary_network/tracers/RSG_2D_rn16e'
# Available variables are listed in src/buffer.py
OUTPUT_VARS = [
    'time', 'x', 'y', 'z', 'r', 'density', 'temperature', 'electron fraction', 'entropy',
    'lum nue', 'lum anue', 'lum nux', 'lum anux', 'ener nue', 'ener anue', 'ener nux', 'ener anux',
    'ejected'
]

# Integration
NUM_TRACERS = 10000
INTEGRATE_BACKWARDS = True
TRACK_NU = True
MAX_TEMP = 6e9
CALCULATE_SEEDS = True

# Tolerances
RTOL = 1e-3
ATOL = 1e4
MAXSTEP = 1e-4
TRACER_TIMEOUT = 20.0 # Failsafe timeout for tracer integration [s] (None to disable)

# Tracer placement
PLACEMENT_METHOD = 'unbound' # 'file', 'uniform space', 'uniform mass', 'unbound', 'user'
TRACER_FILE = 'tracers.dat'
MAX_DENS = 1e11
MAX_RAD = 1.2e9

# Protocols
MODEL_CLS = FLASHProgenitor
SNAP_CLS = FLASHSnapshot
PROXY_CLS = FLASHSnapshotProxy

# Misc
LOG_FILE = 'tracer_integration.log'
MAX_TRACER_BUF_SIZE = 2*1024*1024*1024 # Threshold tracer history buffer before saving [bytes]

