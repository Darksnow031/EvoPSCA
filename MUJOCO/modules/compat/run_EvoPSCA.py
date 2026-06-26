import builtins as _bi
_evopsca_op = _bi.print

def _evopsca_print(*args, **kwargs):
    msg = ' '.join((str(x) for x in args))
    _allow = ('EvoPSCA optimization started', 'RL new best', 'ERL co-evolution feedback loop initiated successfully', 'CEM new best', 'Final solution')
    if any((tok in msg for tok in _allow)):
        _evopsca_op(*args, **kwargs)
_bi.print = _evopsca_print
import logging
logging.disable(logging.CRITICAL)
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
for _p in (GRANDPARENT_DIR, PARENT_DIR, CURRENT_DIR):
    if _p and _p not in sys.path:
        sys.path.append(_p)
from modules.compat._tune import _defaults
from modules.compat.optimizers.hybrid_optimizer import hybrid_rl_cem_optimization

def main():
    hybrid_rl_cem_optimization(**_defaults())
if __name__ == '__main__':
    main()
