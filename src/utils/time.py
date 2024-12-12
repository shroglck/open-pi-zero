import functools
import time


def log_execution_time(logger=None):
    """Decorator to log the execution time of a function"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if logger is None:
                print(f"{func.__name__} took {elapsed_time:.2f} seconds to execute.")
            else:
                logger.info(
                    f"{func.__name__} took {elapsed_time:.2f} seconds to execute."
                )
            return result

        return wrapper

    return decorator
