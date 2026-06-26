import gym
import numpy as np
from .base_env import BaseMaintenanceEnv
from .baseline_reward import BaselineRewardSystem
from ..simulation import run_simulation, calculate_cost
from modules.compat._eval_core import _aggregate as _calc_unified

class DirectYearSelectionEnv(BaseMaintenanceEnv):

    def __init__(self, max_years=200, beta_threshold=3.7, max_maintenance=6, min_year=50, target_years=None, use_baseline_reward=True):
        super().__init__(max_years, beta_threshold, max_maintenance, False, target_years, use_baseline_reward)
        self.max_years = max_years
        self.min_year = min_year
        if self.use_baseline_reward:
            self.baseline_system = BaselineRewardSystem(beta_threshold=self.beta_threshold, max_year=self.max_years)
        else:
            self.baseline_system = None
        self.action_space = gym.spaces.Discrete(max_years + 1)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(max_years + 1,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.maintenance_plan = []
        self.available_years = list(range(self.min_year, self.max_years + 1))
        self.best_reward = float('-inf')
        self.best_plan = None
        return self._get_observation()

    def step(self, action):
        if action not in self.available_years:
            return (self._get_observation(), 0, False, {'invalid_action': True, 'action': action})
        self.maintenance_plan.append(action)
        self.available_years.remove(action)
        reward = 0.0
        done = len(self.maintenance_plan) >= self.max_maintenance
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_plan = self.maintenance_plan.copy()
        if done:
            try:
                if _calc_unified is not None:
                    final_reward = float(_calc_unified(self.maintenance_plan, self.beta_threshold))
                    reward = final_reward
                else:
                    final_reward = self._calculate_reward(self.maintenance_plan)
                    reward = final_reward
            except Exception:
                final_reward = self._calculate_reward(self.maintenance_plan)
                reward = final_reward
        return (self._get_observation(), reward, done, {})

    def _get_observation(self):
        obs = np.zeros(self.max_years + 1, dtype=np.float32)
        for year in self.maintenance_plan:
            obs[year] = 1.0
        return obs

    def _calculate_reward(self, maintenance_plan):
        if not maintenance_plan:
            return -200
        try:
            if _calc_unified is not None:
                return float(_calc_unified(maintenance_plan, self.beta_threshold))
        except Exception:
            pass
        return self._calculate_original_reward_direct(maintenance_plan)

    def _calculate_original_reward_direct(self, maintenance_plan):
        from modules.compat.unified_reward import calculate_reward
        return calculate_reward(maintenance_plan, self.beta_threshold)

    def render(self, mode='human'):
        if mode == 'human':
            plt.figure(figsize=(4, 3))
            plt.plot(range(self.max_years + 1), self._get_observation(), 'b-')
            plt.title('Maintenance Plan')
            plt.xlabel('Year')
            plt.ylabel('Maintenance')
            plt.grid(True)
            plt.show()
