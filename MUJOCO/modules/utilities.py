import os
import sys
import numpy as np
import torch
import random
import time
from collections import deque
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from modules.simulation import run_simulation, evaluate_maintenance_schedule, clear_simulation_cache

def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TrainingCallback:

    def __init__(self, total_steps, output_dir=None):
        self.total_steps = total_steps
        self.last_print = 0
        self.print_interval = 1000
        self.best_maintenance_plan = []
        self.best_reward = -float('inf')
        self.history = {'rewards': [], 'plans': [], 'steps': []}
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def __call__(self, locals_dict=None):
        env = locals_dict.get('self').env if locals_dict and 'self' in locals_dict else None
        model = locals_dict.get('self') if locals_dict and 'self' in locals_dict else None
        if not env or not model:
            return
        self.last_print += 1
        if self.last_print % self.print_interval == 0:
            env.render()
            print(f'训练进度: {self.last_print / self.total_steps * 100:.1f}%')
            test_env = env.__class__(beta_threshold=env.beta_threshold, max_maintenance=env.max_maintenance)
            obs = test_env.reset()
            actions = []
            action_years = []
            total_reward = 0
            for year in range(200):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = test_env.step(action)
                total_reward += reward
                if action == 1:
                    actions.append(year)
                    action_years.append(year)
                if done:
                    break
            self.history['rewards'].append(total_reward)
            self.history['plans'].append(actions.copy())
            self.history['steps'].append(self.last_print)
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.best_maintenance_plan = actions.copy()
                print(f'发现更好的维护计划! 奖励: {total_reward:.2f}')
                if self.output_dir:
                    self._plot_maintenance_plan(actions, env.beta_threshold, total_reward)
            maintenance_distribution = ''
            if len(actions) > 1:
                early = len([y for y in actions if y < 67])
                middle = len([y for y in actions if 67 <= y < 134])
                late = len([y for y in actions if y >= 134])
                maintenance_distribution = f'分布: 前期{early}次, 中期{middle}次, 后期{late}次'
            print(f'当前维护决策: {actions}')
            print(f'当前奖励: {total_reward:.2f}')
            print(f'最佳维护计划: {self.best_maintenance_plan}')
            print(f'最佳奖励: {self.best_reward:.2f}')
            if maintenance_distribution:
                print(maintenance_distribution)

    def _plot_maintenance_plan(self, actions, beta_threshold, reward):
        if not self.output_dir:
            return
        result = run_simulation(actions)
        beta_series = result['beta']
        time_years = result['time_years'] if 'time_years' in result else range(len(beta_series))
        plt.figure(figsize=(10, 4))
        plt.plot(time_years, beta_series)
        plt.axhline(y=beta_threshold, color='r', linestyle='--', label=f'阈值 {beta_threshold}')
        for mt in actions:
            plt.axvline(x=mt, color='g', linestyle='--', alpha=0.7)
        plt.xlabel('年份')
        plt.ylabel('可靠度指标 (Beta)')
        plt.title(f'维护计划的可靠度曲线 (奖励 = {reward:.2f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f'best_plan_reward_{reward:.2f}.png'))
        plt.close()

def evaluate_model_with_plan(model, beta_threshold=3.7, max_maintenance=5):
    from modules.environments import CorrosionMaintenanceEnvDiscrete
    env = CorrosionMaintenanceEnvDiscrete(beta_threshold=beta_threshold, max_maintenance=max_maintenance)
    obs = env.reset()
    done = False
    total_reward = 0
    maintenance_plan = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if action == 1:
            maintenance_plan.append(env.current_year - 1)
    return (total_reward, maintenance_plan)

def calculate_plan_uniformity(maintenance_plan):
    if len(maintenance_plan) <= 1:
        return 0.0
    maintenance_gaps = [maintenance_plan[i + 1] - maintenance_plan[i] for i in range(len(maintenance_plan) - 1)]
    mean_gap = np.mean(maintenance_gaps)
    gap_std = np.std(maintenance_gaps)
    if mean_gap > 0:
        uniformity = max(0.0, 1.0 - gap_std / mean_gap)
        return uniformity
    return 0.0

def analyze_maintenance_distribution(all_plans):
    early_count = 0
    middle_count = 0
    late_count = 0
    total_plans = len(all_plans)
    for plan in all_plans:
        for year in plan:
            if year < 67:
                early_count += 1
            elif year < 134:
                middle_count += 1
            else:
                late_count += 1
    return {'early': early_count / max(1, total_plans), 'middle': middle_count / max(1, total_plans), 'late': late_count / max(1, total_plans), 'total_plans': total_plans}

def get_runs_root():
    repo_root = os.path.dirname(parent_dir)
    env_path = os.environ.get('EVORAINBOW_RUNS_DIR')
    runs_root = env_path if env_path else os.path.join(repo_root, 'runs')
    os.makedirs(runs_root, exist_ok=True)
    return runs_root

def create_output_dir(prefix='output'):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    runs_root = get_runs_root()
    output_dir = os.path.join(runs_root, f'{prefix}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def postprocess_maintenance_plan(plan, windows=None, min_gap=15, min_year=50, max_year=200):
    try:
        yrs = []
        for x in plan or []:
            try:
                yrs.append(int(x))
            except Exception:
                continue
        yrs = sorted([y for y in yrs if min_year <= y <= max_year])
        if not yrs:
            return []
        window_map = {}
        if windows:
            for idx, (s, e) in enumerate(windows):
                s = max(min_year, int(s))
                e = min(max_year, int(e))
                if s <= e:
                    for y in range(s, e + 1):
                        window_map[y] = idx
        yrs = sorted(list(dict.fromkeys(yrs)))
        if len(yrs) <= 1:
            return yrs
        res = [yrs[0]]
        for i in range(1, len(yrs)):
            y = yrs[i]
            if y - res[-1] < min_gap:
                y = res[-1] + min_gap
                if windows:
                    w = window_map.get(yrs[i])
                    if w is not None:
                        s, e = windows[w]
                        y = min(max(y, s), e)
            res.append(int(y))
        for _ in range(5):
            if len(res) <= 2:
                break
            gaps = [res[i] - res[i - 1] for i in range(1, len(res))]
            ok = True
            for i in range(1, len(gaps)):
                if gaps[i] >= gaps[i - 1]:
                    y2 = res[i + 1]
                    target = res[i] + gaps[i - 1] - 1
                    y_new = max(res[i] + min_gap, min(y2, target))
                    if windows:
                        w = window_map.get(y2)
                        if w is not None:
                            s, e = windows[w]
                            y_new = min(max(y_new, s), e)
                    if y_new > res[i]:
                        res[i + 1] = int(y_new)
                        ok = False
                    else:
                        ok = True
                        break
            if ok:
                break
        res = [min(max_year, max(min_year, int(y))) for y in res]
        res = sorted(list(dict.fromkeys(res)))
        return res
    except Exception:
        safe = []
        for x in plan or []:
            try:
                v = int(x)
                if min_year <= v <= max_year:
                    safe.append(v)
            except Exception:
                continue
        return sorted(list(dict.fromkeys(safe)))
