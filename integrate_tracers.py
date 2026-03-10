#!/usr/bin/env python
import os
import sys
import argparse
from glob import glob

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import threading
from functools import partial

import numpy as np
from tqdm import tqdm
import h5py

import config as cfg
import src.log as log
from src.eos import EosTable
import src.placement as plc
import src.integration as itg
from src.buffer import StateBuffer

TQDM_FORMAT = '{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'

MIN_TIME = 0
MAX_TIME = 0
OVERWRITE = False

TRACER_DIR = None
SEED_DIR = None

SNAPSHOT_FILES = []
TRACERS = None
NUC_EOS = None

N_WORKERS = 1


def parse_snapshots() -> None:
    global SNAPSHOT_FILES
    global MIN_TIME, MAX_TIME

    # Try globbing if there is only one path
    if isinstance(cfg.PATH_TO_SNAPSHOTS, str):
        cfg.PATH_TO_SNAPSHOTS = glob(cfg.PATH_TO_SNAPSHOTS)
    elif isinstance(cfg.PATH_TO_SNAPSHOTS, list) and len(cfg.PATH_TO_SNAPSHOTS) == 1:
        cfg.PATH_TO_SNAPSHOTS = glob(cfg.PATH_TO_SNAPSHOTS[0])

    assert len(cfg.PATH_TO_SNAPSHOTS) > 1, 'Invalid snapshot count'

    # Sort snapshots by simulation time
    snap_map = {}
    with tqdm(total=len(cfg.PATH_TO_SNAPSHOTS), bar_format=TQDM_FORMAT, file=sys.stdout) as pbar:
        for snap in cfg.PATH_TO_SNAPSHOTS:
            snap_map[snap] = cfg.SNAP_CLS.get_time(snap)
            pbar.update()

    if isinstance(cfg.PATH_TO_SNAPSHOTS, list) and len(cfg.PATH_TO_SNAPSHOTS) > 1:
        SNAPSHOT_FILES = sorted(set(cfg.PATH_TO_SNAPSHOTS), key=lambda snap: snap_map[snap])
    else:
        raise RuntimeError(f'Invalid input snapshots: {cfg.PATH_TO_SNAPSHOTS}')

    assert len(SNAPSHOT_FILES) > 1, 'Invalid snapshot count'

    MIN_TIME = cfg.SNAP_CLS.get_time(SNAPSHOT_FILES[0])
    MAX_TIME = cfg.SNAP_CLS.get_time(SNAPSHOT_FILES[-1])
    assert MAX_TIME > MIN_TIME, 'Invalid time range'

    if cfg.INTEGRATE_BACKWARDS:
        SNAPSHOT_FILES = list(reversed(SNAPSHOT_FILES))


def load_eos() -> None:
    global NUC_EOS

    if cfg.PATH_TO_EOS is None:
        NUC_EOS = None
    else:
        NUC_EOS = EosTable(cfg.PATH_TO_EOS)


def parse_command_line() -> None:
    global N_WORKERS
    global OVERWRITE

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--num-tracers', type=int, default=cfg.NUM_TRACERS, required=False,
                        help='Number of tracer particles')
    parser.add_argument('-j', '--jobs', type=int, default=None, required=False,
                        help='Number worker processes for integration')
    parser.add_argument('--snapshots', type=str, nargs='+', default=cfg.PATH_TO_SNAPSHOTS, required=False,
                        help='Paths to the snapshot files')
    parser.add_argument('--progenitor', type=str, default=cfg.PATH_TO_PROGENITOR, required=False,
                        help='Path to the progenitor file')
    parser.add_argument('--eos', type=str, default=cfg.PATH_TO_EOS, required=False,
                        help='Path to the EoS file')
    parser.add_argument('--output-dir', type=str, default=cfg.OUTPUT_DIR, required=False,
                        help='Output directory for the tracer calculations')
    parser.add_argument('--integrate-backwards', action='store_true', required=False,
                        help='Use backward integration instead of forward')
    parser.add_argument('--track-nu', action='store_true', required=False,
                        help='Track neutrino quantities in tracer history')
    parser.add_argument('--max-dens', type=float, default=cfg.MAX_DENS, required=False,
                        help='Maximum density where tracers can be placed')
    parser.add_argument('--max-temp', type=float, default=cfg.MAX_TEMP, required=False,
                        help='Maximum temperature at which to stop tracer backward integration')
    parser.add_argument('--calculate-seeds', action='store_true', required=False,
                        help='Calculate tracer seeds from progenitor composition')
    parser.add_argument('--overwrite', action='store_true', required=False,
                        help='Overwrite existing output directory contents without prompting')
    parser.add_argument('--log-file', type=str, default=cfg.LOG_FILE, required=False,
                        help='Path to the file where logs are to be written')

    args = parser.parse_args()

    OVERWRITE = args.overwrite

    # Setup configuration
    cfg.PATH_TO_SNAPSHOTS = args.snapshots
    cfg.PATH_TO_PROGENITOR = args.progenitor
    cfg.PATH_TO_EOS = args.eos

    cfg.OUTPUT_DIR = args.output_dir

    cfg.NUM_TRACERS = args.num_tracers
    cfg.INTEGRATE_BACKWARDS = True if args.integrate_backwards else cfg.INTEGRATE_BACKWARDS
    cfg.TRACK_NU = True if args.track_nu else cfg.TRACK_NU
    cfg.MAX_DENS = args.max_dens
    cfg.MAX_TEMP = args.max_temp if cfg.INTEGRATE_BACKWARDS else None
    cfg.CALCULATE_SEEDS = True if args.calculate_seeds else cfg.CALCULATE_SEEDS

    cfg.LOG_FILE = args.log_file

    assert cfg.PATH_TO_SNAPSHOTS, 'Must specify input snapshots'
    assert (cfg.PATH_TO_PROGENITOR and cfg.CALCULATE_SEEDS) or not cfg.CALCULATE_SEEDS, 'Must specify progenitor file for seed calculation'
    assert cfg.OUTPUT_DIR, 'Must specify output directory'

    if args.jobs is not None:
        N_WORKERS = args.jobs
    else:
        N_WORKERS = os.environ.get('SLURM_CPUS_PER_TASK')
        N_WORKERS = int(N_WORKERS) if N_WORKERS is not None else None

    if N_WORKERS is None or N_WORKERS < 1:
        if not sys.stdin.isatty():
            sys.exit('Invalid worker count.')

        answer = input('WARNING: Worker count could not be determined from neither SLURM nor the command line. How many workers? : ')
        try:
            N_WORKERS = int(answer)
            if N_WORKERS < 1:
                sys.exit('Invalid user input.')
        except:
            sys.exit('Invalid user input.')


def init() -> None:
    global TRACER_DIR
    global SEED_DIR

    print('Initialising...')
    log.init(cfg.LOG_FILE)

    print('\tParsing snapshots')
    parse_snapshots()

    print('\tReading equation of state')
    load_eos()

    print('\tPreparing output')
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    elif not OVERWRITE:
        if not sys.stdin.isatty():
            print('Output directory cannot be overwritten (use --overwrite to force)')
            sys.exit('Exiting.')
        answer = input('WARNING: output directory already exists. Contents will be overwritten. Confirm (y/N): ')
        if answer.lower() != 'y':
            sys.exit('Exiting.')
    TRACER_DIR = os.path.join(cfg.OUTPUT_DIR, 'tracers')
    if not os.path.exists(TRACER_DIR):
        os.makedirs(TRACER_DIR)
    SEED_DIR = os.path.join(cfg.OUTPUT_DIR, 'seeds')
    if cfg.CALCULATE_SEEDS and not os.path.exists(SEED_DIR):
        os.makedirs(SEED_DIR)

    print('\tPlacing tracers')
    place_tracers()

    log.write(f'\tNumber of tracers: {cfg.NUM_TRACERS}')
    log.write(f'\tNumber of snapshots: {len(SNAPSHOT_FILES)}')
    log.write(f'\tTime range: {MIN_TIME:.4f}-{MAX_TIME:.4f}')
    log.write(f'\tWriting output to: {cfg.OUTPUT_DIR}')
    log.write('')
    log.write(f'\tUse backward integration: {cfg.INTEGRATE_BACKWARDS}')
    log.write(f'\tTrack neutrinos: {cfg.TRACK_NU}')
    log.write(f'\tCalculate seeds: {cfg.CALCULATE_SEEDS}')
    log.write(f'\tProgenitor file: {cfg.PATH_TO_PROGENITOR}')
    log.write('')
    log.write(f'\tWorker process: {N_WORKERS}')
    print('Initialised.')


def place_tracers() -> None:
    global TRACERS

    place_snap = cfg.SNAP_CLS(SNAPSHOT_FILES[0], use_nu=cfg.TRACK_NU)
    match cfg.PLACEMENT_METHOD:
        case 'file':
            TRACERS = plc.from_file(cfg.TRACER_FILE)
        case 'uniform space':
            TRACERS = plc.sample_uniform_space(cfg.NUM_TRACERS, place_snap, cfg.MAX_DENS)
        case 'uniform mass':
            TRACERS = plc.sample_uniform_mass(cfg.NUM_TRACERS, place_snap, cfg.MAX_DENS)
        case 'unbound':
            TRACERS = plc.sample_unbound(cfg.NUM_TRACERS, place_snap, NUC_EOS, cfg.MAX_DENS, cfg.MAX_TEMP)
        case 'user':
            TRACERS = plc.sample_user(cfg.NUM_TRACERS, place_snap, NUC_EOS, cfg.MAX_DENS, cfg.MAX_TEMP)
        case _:
            raise RuntimeError('Invalid tracer placement method: {cfg.PLACEMENT_METHOD}')

    np.savetxt(os.path.join(cfg.OUTPUT_DIR, 'tracers.dat'), TRACERS, fmt='%1.18e', header='x [cm]\ty [cm]\tz [cm]\t mass [g]')


def integrate() -> None:
    log.write(f'Beginning integration of {cfg.NUM_TRACERS} tracers on {N_WORKERS} threads')

    print('Creating tracer buffer')
    state_buffer = StateBuffer(len(TRACERS), cfg.OUTPUT_VARS, TRACER_DIR)
    snap_queue = queue.Queue(maxsize=2)

    print('Starting integration')
    loader = threading.Thread(
        target=itg.loader_thread,
        args=(SNAPSHOT_FILES, cfg.TRACK_NU, snap_queue, cfg.SNAP_CLS),
        daemon=True
    )
    loader.start()

    snap0 = snap_queue.get()
    snap1 = snap_queue.get()
    tracer_pos = [np.array(list(pos)) for pos in TRACERS[['x', 'y', 'z']]]
    failed_tracers = []

    worker_exports = {
        'ProxyCls': cfg.PROXY_CLS,
        'max_temp': cfg.MAX_TEMP,
        'eos_desc': NUC_EOS.get_proxy_descriptor(),
        'output_vars': cfg.OUTPUT_VARS,
        'rtol': cfg.RTOL,
        'atol': cfg.ATOL,
        'max_step': cfg.MAXSTEP,
        'init_step': 1e-6,
    }

    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp.get_context('spawn')) as exe:
        # Get initial tracer state
        desc0 = snap0.get_proxy_descriptor()

        print('Computing initial tracer states')
        initial_state_process_p = partial(itg.initial_state_process, desc=desc0, exports=worker_exports)
        active_tracers = [(n, pos) for n,pos in enumerate(tracer_pos) if pos is not None]
        active_ids, active_pos = zip(*active_tracers)
        for index, state in exe.map(initial_state_process_p, active_ids, active_pos, chunksize=20):
            if state is None:
                tracer_pos[index] = None
                print(f'Tracer {index} failed')
                log.error(f'Tracer {index} failed')
                failed_tracers.append(index)
                continue
            state_buffer.append(index, state)

        # Integrate next states
        while True:
            desc0 = snap0.get_proxy_descriptor()
            desc1 = snap1.get_proxy_descriptor()

            log.info(f"Integrating {desc0['current_time']:.4f} - {desc1['current_time']:.4f}")

            # Prepare worker process and retrieve active tracers
            worker_process_p = partial(itg.worker_process, start_desc=desc0, end_desc=desc1, exports=worker_exports)
            active_tracers = [(n, pos) for n,pos in enumerate(tracer_pos) if pos is not None]
            active_ids, active_pos = zip(*active_tracers)
            # Start worker processes
            # Append tracer integration results and update positions
            for index,end_pos,state in exe.map(worker_process_p, active_ids, active_pos, chunksize=20):
                if state is None:
                    tracer_pos[index] = None
                    print(f'Tracer {index} failed')
                    log.error(f'Tracer {index} failed')
                    failed_tracers.append(index)
                    continue
                state_buffer.append(index, state)
                tracer_pos[index] = end_pos

            # Flush buffer if required
            if state_buffer.sizeof() > cfg.MAX_TRACER_BUF_SIZE:
                print('Flushing buffer')
                log.info('Flushing buffer')
                state_buffer.flush()

            # Preload next snapshot
            snap2 = snap_queue.get()

            # Check sentinel
            if snap2 is None:
                break

            # Delete previous snapshot
            snap0.close()

            # Slide window
            snap0 = snap1
            snap1 = snap2

            # Sometimes IO does not flush on the cluster
            sys.stdout.flush()

    print('Finishing up')
    # Clean up shared memory
    snap0.close()
    snap1.close()
    # Flush buffer
    state_buffer.flush()

    # Reverse order in output files
    if cfg.INTEGRATE_BACKWARDS:
        state_buffer.reverse_output()

    # Rid failed tracers files
    if len(failed_tracers) > 0:
        state_buffer.delete_failed_output(failed_tracers)

    if cfg.CALCULATE_SEEDS:
        print('Calculating seeds')
        log.info('Calculating seeds')
        calculate_seeds(tracer_pos)


def calculate_seeds(tracer_pos) -> None:
    model = cfg.MODEL_CLS(cfg.PATH_TO_PROGENITOR)

    header = '    '.join(['A', 'Z', 'X'])
    n_digits = int(np.log10(len(tracer_pos) - 1)) + 1
    with tqdm(total=len(tracer_pos), bar_format=TQDM_FORMAT, file=sys.stdout) as pbar:
        for i,pos in enumerate(tracer_pos):
            if pos is None:
                pbar.update()
                continue
            r = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            seed_file = os.path.join(SEED_DIR, 'seed' + f'{i}'.zfill(n_digits) + '.dat')
            seed = model.get_seed(r)
            A = np.array([nuc.A for nuc in seed.keys()])
            Z = np.array([nuc.Z for nuc in seed.keys()])
            X = np.array([X for X in seed.values()])
            np.savetxt(seed_file, np.column_stack((A, Z, X)), fmt='%d\t%d\t%.6e', header=header)
            pbar.update()


def terminate() -> None:
    print('Terminating')
    if NUC_EOS is not None:
        NUC_EOS.close()
    log.terminate()


if __name__ == '__main__':
    parse_command_line()
    init()
    integrate()
    terminate()
    sys.stdout.flush()

