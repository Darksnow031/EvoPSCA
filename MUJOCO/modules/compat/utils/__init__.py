from __future__ import annotations
from typing import Iterable, List, Tuple, Optional, Any
import math
import random
import os
import json
import numpy as np
from ..mod_utils import discounted_cumsum, gae_advantages, linear_decay, cosine_decay, set_global_seed, get_device, ReplayBufferSimple, OnPolicyRolloutBuffer, Tracker, timeit, years_to_onehot, onehot_to_years

def discount(x, gamma: float):
    return discounted_cumsum(np.asarray(x, dtype=np.float32), float(gamma))

def combined_shape(length: int, shape: Optional[int | Tuple[int, ...]]=None):
    if shape is None:
        return (int(length),)
    return (int(length), int(shape)) if isinstance(shape, (int, np.integer)) else (int(length), *shape)

class Scaler:

    def __init__(self, obs_dim: int):
        self.vars = np.zeros(obs_dim, dtype=np.float32)
        self.means = np.zeros(obs_dim, dtype=np.float32)
        self.m = 0
        self.first_pass = True

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_var = np.var(x, axis=0)
            new_mean = np.mean(x, axis=0)
            new_mean_sq = np.square(new_mean)
            new_means = (self.means * self.m + new_mean * n) / (self.m + n)
            self.vars = (self.m * (self.vars + np.square(self.means)) + n * (new_var + new_mean_sq)) / (self.m + n) - np.square(new_means)
            self.vars = np.maximum(0.0, self.vars)
            self.means = new_means
            self.m += n

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        scale = 1.0 / (np.sqrt(self.vars) + 0.1) / 3.0
        offset = self.means
        return (scale.astype(np.float32), offset.astype(np.float32))

class ReplayBuffer:

    def __init__(self, max_size: int=int(1000000.0)):
        self.storage: list = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, data: tuple):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size: int):
        ind = np.random.randint(0, len(self.storage), size=int(batch_size))
        cols = None
        buckets = None
        for i in ind:
            tup = self.storage[i]
            if cols is None:
                cols = len(tup)
                buckets = [[] for _ in range(cols)]
            for j in range(cols):
                buckets[j].append(np.array(tup[j], copy=False))
        return tuple((np.array(b) for b in buckets))

    def get_current(self, k: int):
        k = int(k)
        if k <= 0:
            return tuple()
        if len(self.storage) == 0:
            return tuple()
        if k > len(self.storage):
            sample_list = list(range(len(self.storage)))
        elif len(self.storage) == self.max_size:
            if k <= self.ptr:
                sample_list = list(range(int(self.ptr - k), int(self.ptr)))
            else:
                sample_list = list(range(int(self.ptr))) + list(range(int(self.max_size - (k - self.ptr)), int(self.max_size)))
        else:
            sample_list = list(range(max(0, int(self.ptr - k)), int(self.ptr)))
        cols = len(self.storage[0])
        buckets = [[] for _ in range(cols)]
        for i in sample_list:
            tup = self.storage[i]
            for j in range(cols):
                buckets[j].append(np.array(tup[j], copy=False))
        return tuple((np.array(b) for b in buckets))

class ReplayBufferPPO:

    def __init__(self, obs_dim: int, act_dim: int, size: int, gamma: float=0.99, lam: float=0.95):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.size = int(size)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.buf = OnPolicyRolloutBuffer(capacity=self.size, obs_shape=(obs_dim,), act_shape=(act_dim,), device='cpu')

    def reset(self):
        self.buf.reset()

    def add(self, obs, act, rew, val, logp):
        assert self.buf.ptr < self.buf.capacity
        self.buf.add(obs, act, float(rew), False, float(val), float(logp))

    def finish_path(self, last_val: float=0.0):
        self.buf.compute_gae(last_value=float(last_val), gamma=self.gamma, lam=self.lam)

    def get(self):
        n = self.buf.capacity if self.buf.full else self.buf.ptr
        obs = self.buf.obs[:n].copy()
        act = self.buf.actions[:n].copy()
        adv = self.buf.advs[:n].copy()
        ret = self.buf.returns[:n].copy()
        logp = self.buf.logps[:n].copy()
        self.reset()
        return [obs, act, adv, ret, logp]

class ReplayBuffer_MC:

    def __init__(self, max_size: int=int(1000000.0)):
        self.storage: list = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, data: tuple):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size: int):
        ind = np.random.randint(0, len(self.storage), size=int(batch_size))
        x, u, r = ([], [], [])
        for i in ind:
            X, U, R = self.storage[i]
            x.append(np.array(X, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
        return (np.array(x), np.array(u), np.array(r).reshape(-1, 1))

class ReplayBuffer_VDFP:

    def __init__(self, max_size: int=int(100000.0)):
        self.storage: list = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, data: tuple):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size: int):
        ind = np.random.randint(0, len(self.storage), size=int(batch_size))
        s, a, u, x = ([], [], [], [])
        for i in ind:
            S, A, U, X = self.storage[i]
            s.append(np.array(S, copy=False))
            a.append(np.array(A, copy=False))
            u.append(np.array(U, copy=False))
            x.append(np.array(X, copy=False))
        return (np.array(s), np.array(a), np.array(u).reshape(-1, 1), np.array(x))

    def sample_traj(self, batch_size: int, offset: int=0):
        ind = np.random.randint(0, len(self.storage) - int(offset), size=int(batch_size))
        if len(self.storage) == self.max_size:
            ind = (self.ptr + self.max_size - ind) % self.max_size
        else:
            ind = len(self.storage) - ind - 1
        s, a, x = ([], [], [])
        for i in ind:
            S, A, _, X = self.storage[i]
            s.append(np.array(S, copy=False))
            a.append(np.array(A, copy=False))
            x.append(np.array(X, copy=False))
        return (np.array(s), np.array(a), np.array(x))

    def sample_traj_return(self, batch_size: int):
        ind = np.random.randint(0, len(self.storage), size=int(batch_size))
        u, x = ([], [])
        for i in ind:
            _, _, U, X = self.storage[i]
            u.append(np.array(U, copy=False))
            x.append(np.array(X, copy=False))
        return (np.array(u).reshape(-1, 1), np.array(x))

def store_experience(replay_buffer: Any, trajectory: tuple, s_dim: int, a_dim: int, sequence_length: int, min_sequence_length: int=0, is_padding: bool=False, gamma: float=0.99):
    s_traj, a_traj, r_traj = trajectory
    arr_s = np.array(s_traj)
    arr_a = np.array(a_traj)
    arr_r = np.array(r_traj)
    zero_pads = np.zeros(shape=(sequence_length, s_dim + a_dim), dtype=np.float32)
    for i in range(len(s_traj) - int(min_sequence_length)):
        tmp_s = arr_s[i]
        tmp_a = arr_a[i]
        tmp_soff = arr_s[i:i + sequence_length]
        tmp_aoff = arr_a[i:i + sequence_length]
        tmp_saoff = np.concatenate([tmp_soff, tmp_aoff], axis=1)
        tmp_saoff_padded = np.concatenate([tmp_saoff, zero_pads], axis=0)
        tmp_saoff_clip = tmp_saoff_padded[:sequence_length, :]
        tmp_roff = arr_r[i:i + sequence_length]
        tmp_u = np.matmul(tmp_roff, np.power(float(gamma), np.arange(len(tmp_roff))))
        replay_buffer.add((tmp_s, tmp_a, tmp_u, tmp_saoff_clip))

def elite_indices(scores: List[float], elite_frac: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    k = max(1, int(len(scores) * float(elite_frac)))
    return np.argsort(scores)[-k:]

def choose_elites(solutions: List[Any], scores: List[float], elite_frac: float) -> List[Any]:
    idx = elite_indices(scores, elite_frac)
    return [solutions[int(i)] for i in idx]

def update_probs_with_elites(probs: np.ndarray, elite_onehots: np.ndarray, alpha: float=0.5, eps: float=1e-06) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    elite = np.asarray(elite_onehots, dtype=np.float32)
    if elite.ndim == 1:
        elite = elite[None, :]
    target = elite.mean(axis=0)
    target = (target + eps) / (target.sum() + eps * len(target))
    newp = (1 - float(alpha)) * p + float(alpha) * target
    newp = np.clip(newp, eps, 1.0)
    newp /= newp.sum()
    return newp.astype(np.float32)

def sample_solutions_from_probs(probs: np.ndarray, count: int, max_select: int, min_year: int=0, rng: Optional[np.random.RandomState | np.random.Generator]=None) -> List[List[int]]:
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    years = np.arange(len(p), dtype=np.int64)
    sols: List[List[int]] = []
    rng = rng or np.random
    for _ in range(int(count)):
        k = min(int(max_select), len(years) - int(min_year))
        mask = years >= int(min_year)
        idxs = rng.choice(years[mask], size=k, replace=False, p=p[mask] / p[mask].sum())
        sols.append(sorted([int(i) for i in idxs]))
    return sols

def ensure_valid_years(years: Iterable[int], min_year: int, max_year: int) -> List[int]:
    ys = [int(y) for y in years if int(y) >= int(min_year) and int(y) <= int(max_year)]
    return sorted(list(dict.fromkeys(ys)))

def jsonl_append(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(dict(data), ensure_ascii=False) + '\n')
__all__ = ['discount', 'Scaler', 'ReplayBuffer', 'ReplayBufferPPO', 'ReplayBuffer_MC', 'ReplayBuffer_VDFP', 'ReplayBufferSimple', 'OnPolicyRolloutBuffer', 'store_experience', 'Tracker', 'timeit', 'discounted_cumsum', 'gae_advantages', 'linear_decay', 'cosine_decay', 'set_global_seed', 'get_device', 'combined_shape', 'years_to_onehot', 'onehot_to_years', 'elite_indices', 'choose_elites', 'update_probs_with_elites', 'sample_solutions_from_probs', 'ensure_valid_years', 'jsonl_append']
