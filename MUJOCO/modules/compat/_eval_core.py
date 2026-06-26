import importlib.util
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PC = os.path.join(_HERE, '_bin', '_eval_core.cpython-311.pyc')
if not os.path.isfile(_PC):
    raise ImportError(f'Missing protected module: {_PC}')
_spec = importlib.util.spec_from_file_location('_eval_core._impl', _PC)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
_aggregate = getattr(_mod, '_aggregate')
__all__ = ['_aggregate']
