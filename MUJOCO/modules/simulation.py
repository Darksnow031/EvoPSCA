import os
import sys
import time
import numpy as np
import torch
import pandas as pd
from collections import deque, OrderedDict
from scipy.stats import norm
import random
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
try:
    import gangbanliehua7_python as gbp
    pass
except ImportError as e:
    pass
    pass
    gbp = None

def calculate_cost(maintenance_year):
    C_MEC = 92807
    C_MEN = 12668.4929108671
    r_eco = 0.0448279331893315
    r_env = 0.0125304458140604
    discounted_cost = C_MEC / (1 + r_eco) ** maintenance_year + C_MEN / (1 + r_env) ** maintenance_year
    return discounted_cost

def init_global_matrices(use_cuda=True):
    ok = gbp.init_global_matrices(use_cuda=use_cuda)
    try:
        if hasattr(run_simulation, 'cache'):
            run_simulation.cache.clear()
            pass
    except Exception:
        pass
    return ok

def run_simulation(maintenance_times):
    if isinstance(maintenance_times, np.ndarray):
        mt_list = sorted(maintenance_times.tolist())
    else:
        mt_list = sorted(maintenance_times)
    if not hasattr(run_simulation, 'cache'):
        run_simulation.cache = OrderedDict()
        run_simulation.cache_maxsize = 1000
    mt_tuple = tuple(mt_list)
    if mt_tuple in run_simulation.cache:
        result = run_simulation.cache.pop(mt_tuple)
        run_simulation.cache[mt_tuple] = result
        return result
    result = gbp.run_simulation(mt_list)
    try:
        run_simulation.cache[mt_tuple] = result
        if len(run_simulation.cache) > run_simulation.cache_maxsize:
            run_simulation.cache.popitem(last=False)
    except Exception:
        pass
    return result

def run_simulation_batch(plans_list):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if not hasattr(run_simulation, 'cache'):
        run_simulation.cache = OrderedDict()
        run_simulation.cache_maxsize = 1000
    keys = []
    plans = []
    for plan in plans_list:
        if isinstance(plan, np.ndarray):
            key = tuple(sorted(plan.tolist()))
        else:
            key = tuple(sorted(list(plan)))
        keys.append(key)
        plans.append(key)
    results = [None] * len(keys)
    to_compute = {}
    for idx, key in enumerate(keys):
        if key in run_simulation.cache:
            try:
                val = run_simulation.cache.pop(key)
                run_simulation.cache[key] = val
                results[idx] = val
            except Exception:
                results[idx] = run_simulation.cache.get(key)
        else:
            to_compute[idx] = key
    if len(to_compute) > 0:
        max_workers = min(8, max(1, len(to_compute)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(gbp.run_simulation, list(k)): i for i, k in to_compute.items()}
            for fut in as_completed(future_map):
                idx = future_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {'beta': np.array([0.0] * 201)}
                try:
                    run_simulation.cache[keys[idx]] = res
                    if len(run_simulation.cache) > run_simulation.cache_maxsize:
                        run_simulation.cache.popitem(last=False)
                except Exception:
                    pass
                results[idx] = res
    return results

def evaluate_maintenance_schedule(maintenance_times, beta_threshold=3.7):
    if not maintenance_times:
        return (0.0, False)
    sorted_times = sorted(maintenance_times)
    result = run_simulation(sorted_times)
    beta_series = result['beta']
    feasible = np.all(beta_series >= beta_threshold)
    total_cost = sum((calculate_cost(y) for y in sorted_times))
    return (total_cost, feasible)

def clear_simulation_cache():
    if hasattr(run_simulation, 'cache'):
        run_simulation.cache.clear()
        print('仿真缓存已清理')
    return
