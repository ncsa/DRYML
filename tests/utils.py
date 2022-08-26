import importlib
import pytest
import functools


def import_check(*module_names):
    def _dec(f):
        @functools.wraps(f)
        def _f(*args, **kwargs):
            for module_name in module_names:
                try:
                    importlib.import_module(module_name)
                except ImportError:
                    pytest.skip(f"Missing required module {module_name}")
            return f(*args, **kwargs)

        return _f
    return _dec


def ray_wrap(f):
    ray = pytest.importorskip("ray")

    f_rem = ray.remote(f)

    orig_func_name = f.__name__

    def func():
        ray.get(f_rem.remote())

    func.__name__ = orig_func_name
    f.__name__ = f"{orig_func_name}_temp"

    return func
