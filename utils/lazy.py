import time


def lazy(f, expire=60):

    def wrapper(*args, **kwargs):
        response = getattr(f, '__last_response__', None)
        last_call = getattr(f, '__last_call__', 0)

        if response is None or last_call + expire < time.time():
            f.__last_call__ = time.time()
            response = f.__last_response__ = f(*args, **kwargs)
            
        return response

    return wrapper
