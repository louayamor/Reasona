import time
import contextlib
import signal
from functools import wraps
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/utils/retry_timeout.jsonl")
@contextlib.contextmanager
def timeout_context(seconds: int):
    def handler(signum, frame):
        raise TimeoutError(f"Operation exceeded {seconds}s timeout")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def retry_on_exception(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts > max_retries:
                        raise
                    logger.warning(f"Retry {attempts}/{max_retries} after error: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator
