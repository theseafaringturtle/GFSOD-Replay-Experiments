import functools
import time


def time_perf(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
            logger.info(f"{func.__name__} execution time: {elapsed_time:.4f} s")
            return result

        return wrapper

    return decorator
