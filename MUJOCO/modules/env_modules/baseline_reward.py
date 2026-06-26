import numpy as np
from ..simulation import run_simulation, calculate_cost, init_global_matrices

class BaselineRewardSystem:

    def __init__(self, beta_threshold=3.7, max_year=200):
        self.beta_threshold = beta_threshold
        self.max_year = max_year
        self.baseline_plan = [50, 100, 190]
        self.baseline_reward = None
        self.baseline_calculated = False
        self.debug_counter = 0

    def _calculate_baseline_reward(self):
        if not self.baseline_calculated:
            try:
                from ..simulation import init_global_matrices
                try:
                    init_global_matrices(use_cuda=False)
                    print('[baseline_reward] 已尝试初始化仿真全局矩阵（init_global_matrices）。')
                except Exception:
                    print('[baseline_reward] init_global_matrices 调用失败（可忽略）。')
            except Exception:
                pass
            self.baseline_plan = [50, 100, 190]
            self.baseline_reward = self._calculate_raw_reward(self.baseline_plan)
            self.baseline_calculated = True
            print(f'基线策略奖励: {self.baseline_reward:.2f}')
            print(f'=== 奖励尺度诊断 ===')
            print(f'基线奖励: {self.baseline_reward}')
        return self.baseline_reward

    def _calculate_raw_reward(self, maintenance_plan):
        from modules.compat.unified_reward import calculate_reward
        return calculate_reward(maintenance_plan, self.beta_threshold)

    def calculate_reward_with_baseline(self, maintenance_plan):
        current_reward = self._calculate_raw_reward(maintenance_plan)
        baseline_reward = self._calculate_baseline_reward()
        raw_advantage = current_reward - baseline_reward
        self.debug_counter += 1
        if self.debug_counter % 100 == 0:
            print(f'奖励诊断: 当前={current_reward:.3f}, 基线={baseline_reward:.3f}, 优势={raw_advantage:.3f}')
        return float(raw_advantage)

    def calculate_scaled_advantage(self, maintenance_plan, clip=False):
        current_reward = self._calculate_raw_reward(maintenance_plan)
        baseline_reward = self._calculate_baseline_reward()
        raw_advantage = current_reward - baseline_reward
        if clip:
            return float(np.clip(raw_advantage, -200.0, 200.0))
        return float(raw_advantage)

    def get_baseline_info(self):
        if not self.baseline_calculated:
            self._calculate_baseline_reward()
        return {'baseline_plan': self.baseline_plan, 'baseline_reward': self.baseline_reward}

class _LazyBaselineProxy:

    def __init__(self):
        self._instance = None

    def _create_instance(self):
        self._instance = BaselineRewardSystem()
        return self._instance

    def _get_instance(self):
        if self._instance is None:
            return self._create_instance()
        return self._instance

    def initialize(self, use_cuda=False):
        try:
            init_global_matrices(use_cuda=use_cuda)
        except Exception:
            pass
        inst = self._get_instance()
        try:
            inst._calculate_baseline_reward()
        except Exception:
            pass

    def __getattr__(self, name):
        inst = self._get_instance()
        return getattr(inst, name)
baseline_reward_system = _LazyBaselineProxy()
