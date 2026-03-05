from time import time
from datetime import datetime

_DEFAULT_LOG_FILE = 'out.log'
_STAMP_FORMAT = '[%d-%m-%Y - %H:%M:%S]'

_log_file = _DEFAULT_LOG_FILE
_t0 = 0


def init(log_file: str) -> None:
    global _t0
    global _log_file

    _t0 = time()
    _log_file = log_file if log_file else _DEFAULT_LOG_FILE

    # Truncates log file if it exists and prints run setup
    with open(_log_file, 'w') as f:
        f.write('Starting tracer calculation on ' + datetime.now().strftime('%d-%m-%Y at %H:%M:%S') + '\n')


def write(msg: str) -> None:
    with open(_log_file, 'a') as f:
        f.write(msg + '\n')


def stamp(msg: str) -> None:
    with open(_log_file, 'a') as f:
        timestamp = datetime.now().strftime(_STAMP_FORMAT)
        f.write(f'{timestamp} {msg}')


def debug(msg: str) -> None:
    stamp(f'<DBG> {msg}\n')


def info(msg: str) -> None:
    stamp(f'<INFO> {msg}\n')


def warn(msg: str) -> None:
    stamp(f'<WARN> {msg}\n')


def error(msg: str) -> None:
    stamp(f'<ERR> {msg}\n')


def terminate() -> None:
    with open(_log_file, 'a') as f:
        f.write(f'Tracer integration finished in {time() - _t0:.2f} [s]\n')

