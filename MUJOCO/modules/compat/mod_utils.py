from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import os
import json
import time
import math
import random
from collections import deque, defaultdict
import numpy as np
try:
    import torch
except Exception:
    torch = None

def years_to_onehot(years: Iterable[int], max_year: int) -> np.ndarray:
    v = np.zeros(max_year + 1, dtype=np.float32)
    for y in years:
        if 0 <= int(y) <= max_year:
            v[int(y)] = 1.0
    return v

def onehot_to_years(vec: np.ndarray, threshold: float=0.5) -> List[int]:
    vec = np.asarray(vec).reshape(-1)
    ys = [int(i) for i, v in enumerate(vec) if v >= float(threshold)]
    return sorted(list(dict.fromkeys(ys)))

def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    run = 0.0
    for t in range(len(x) - 1, -1, -1):
        run = run * gamma + float(x[t])
        y[t] = run
    return y

def gae_advantages(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float=0.99, lam: float=0.95) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values[:-1]
    return (adv, returns)

def linear_decay(step: int, total: int, start: float, end: float) -> float:
    if total <= 0:
        return end
    frac = max(0.0, min(1.0, 1.0 - step / float(total)))
    return end + (start - end) * frac

def cosine_decay(step: int, total: int, start: float, end: float) -> float:
    if total <= 0:
        return end
    t = max(0.0, min(1.0, step / float(total)))
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

def get_device(prefer: str='cuda'):
    if prefer == 'cuda' and torch is not None and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu') if torch is not None else 'cpu'

class ReplayBufferSimple:

    def __init__(self, capacity: int=100000):
        self.capacity = int(capacity)
        self.storage = deque(maxlen=self.capacity)

    def add(self, *transition):
        if len(transition) == 1:
            self.storage.append(transition[0])
        else:
            self.storage.append(tuple(transition))

    def sample(self, batch_size: int):
        k = min(int(batch_size), len(self.storage))
        return random.sample(self.storage, k)

    def __len__(self):
        return len(self.storage)

class OnPolicyRolloutBuffer:

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], act_shape: Tuple[int, ...], device: Any='cpu'):
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.full = False
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity + 1,), dtype=np.float32)
        self.logps = np.zeros((capacity,), dtype=np.float32)
        self.advs = np.zeros((capacity,), dtype=np.float32)
        self.returns = np.zeros((capacity,), dtype=np.float32)

    def reset(self):
        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward: float, done: bool, value: float, logp: float):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = 1.0 if done else 0.0
        self.values[i] = value
        self.logps[i] = logp
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True
            self.ptr = self.capacity

    def compute_gae(self, last_value: float, gamma: float=0.99, lam: float=0.95):
        n = self.capacity if self.full else self.ptr
        self.values[n] = float(last_value)
        adv, ret = gae_advantages(self.rewards[:n], self.values[:n + 1], self.dones[:n], gamma, lam)
        self.advs[:n] = adv
        self.returns[:n] = ret
        std = np.std(self.advs[:n]) + 1e-08
        self.advs[:n] = (self.advs[:n] - np.mean(self.advs[:n])) / std
        return n

    def iter(self, minibatch_size: int):
        n = self.capacity if self.full else self.ptr
        idx = np.random.permutation(n)
        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            mb = idx[start:end]
            yield (self.obs[mb], self.actions[mb], self.returns[mb], self.advs[mb], self.logps[mb])

class EMA:

    def __init__(self, alpha: float=0.1):
        self.alpha = float(alpha)
        self.v: Optional[float] = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v

class Tracker:

    def __init__(self, ema_alpha: float=0.0):
        self.raw: Dict[str, list] = defaultdict(list)
        self.ema: Dict[str, EMA] = {}
        self.last: Dict[str, float] = {}
        self.ema_alpha = float(ema_alpha)

    def record(self, name: str, value: float, step: Optional[int]=None, tag: Optional[str]=None):
        self.raw[name].append((step, float(value), tag))
        self.last[name] = float(value)
        if self.ema_alpha > 0:
            if name not in self.ema:
                self.ema[name] = EMA(self.ema_alpha)
            self.ema[name].update(float(value))

    def get(self, name: str, default: Optional[float]=None) -> Optional[float]:
        return self.last.get(name, default)

    def summary(self) -> Dict[str, float]:
        s = {k: v for k, v in self.last.items()}
        if self.ema_alpha > 0:
            for k, e in self.ema.items():
                if e.v is not None:
                    s[f'{k}_ema'] = float(e.v)
        return s

    def jsonl_append(self, path: str, extra: Optional[Dict[str, Any]]=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = self.summary()
        if extra:
            data.update(extra)
        with open(path, 'a') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

class timeit:

    def __init__(self, name: str, logger: Optional[Callable[[str], None]]=None):
        self.name = name
        self.logger = logger or (lambda msg: None)
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        t1 = time.time()
        self.logger(f'{self.name}: {t1 - self.t0:.3f}s')
        return False

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
__all__ = ['years_to_onehot', 'onehot_to_years', 'discounted_cumsum', 'gae_advantages', 'linear_decay', 'cosine_decay', 'set_global_seed', 'get_device', 'ReplayBufferSimple', 'OnPolicyRolloutBuffer', 'EMA', 'Tracker', 'timeit', 'NumpyEncoder']
