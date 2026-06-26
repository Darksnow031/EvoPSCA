import importlib.util
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PC = os.path.join(_HERE, '_bin', '_tune.cpython-311.pyc')
if not os.path.isfile(_PC):
    raise ImportError(f'Missing protected module: {_PC}')
_spec = importlib.util.spec_from_file_location('_tune._impl', _PC)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
_defaults = getattr(_mod, '_defaults')
__all__ = ['_defaults']
