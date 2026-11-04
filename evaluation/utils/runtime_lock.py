"""Runtime lock."""
from contextlib import contextmanager
from pathlib import Path
import fcntl

@contextmanager
def operation_lock(*args, **kwargs):
    root = Path(__file__).resolve().parents[2] / '.runtime'
    root.mkdir(exist_ok=True)
    with (root / 'model.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
