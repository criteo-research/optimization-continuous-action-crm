import time


def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()

        if 'log_time' in kwargs:
            name = kwargs.get('log_name', func.__name__.upper())
            kwargs['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (func.__name__, (te - ts) * 1000))
        return result
    return timed


def accepts(*types):
    def check_accepts(func):
        assert len(types) == func.__code__.co_argcount - 1

        def wrapper(self, *args, **kwargs):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), \
                    "arg %r does not match %s" % (a, t)
            return func(self, *args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper
    return check_accepts


def accepts_(*types):
    def check_accepts(func):
        assert len(types) == func.__code__.co_argcount

        def wrapper(*args, **kwargs):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), \
                    "arg %r does not match %s" % (a, t)
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper
    return check_accepts


def fitted(func):
    def wrapper(self, *args, **kwargs):
        if self._fitted:
            return func(self, *args, **kwargs)
        else:
            raise RuntimeError("Model not fitted yet")
    return wrapper
