import gym
import numpy as np
from .base_env import BaseMaintenanceEnv
from .baseline_reward import BaselineRewardSystem
from ..simulation import run_simulation, calculate_cost
from modules.compat._eval_core import _aggregate as _calc_unified

class CorrosionMaintenanceEnvDiscrete(BaseMaintenanceEnv):

    def __init__(self, max_year=200, beta_threshold=3.7, max_maintenance=6, normalize_rewards=False, target_years=None, use_baseline_reward=True):
        super().__init__(max_year, beta_threshold, max_maintenance, normalize_rewards, target_years, use_baseline_reward)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), high=np.array([float(self.max_year), 10.0, float(self.max_year), float(self.max_maintenance), float(self.max_maintenance), 1.0, 1.0]), shape=(7,), dtype=np.float32)
        if self.use_baseline_reward:
            self.baseline_system = BaselineRewardSystem(beta_threshold=self.beta_threshold, max_year=self.max_year)
        else:
            self.baseline_system = None
        self.current_year = 0
        self.maintenance_times = []
        self.last_maintenance_year = 0
        self.reliability_history = []
        self.current_reliability = 4.4172
        self.maintenance_cooldown = 10
        self.ideal_maintenance_gaps = self.max_year / (self.max_maintenance + 1)
        self.suggested_maintenance_times = [int(self.max_year * (i + 1) / (self.max_maintenance + 1)) for i in range(self.max_maintenance)]
        self.exploration_phase = True
        self.exploration_steps = 200
        self.step_count = 0
        self.prev_reliability = 4.4172
        self.distribution_score = 0.0

    def reset(self):
        self.current_year = 0
        self.maintenance_times = []
        self.last_maintenance_year = 0
        self.reliability_history = []
        self.current_reliability = 4.4172
        self.prev_reliability = 4.4172
        self.step_count = 0
        self.distribution_score = 0.0
        self.best_reward = float('-inf')
        self.best_plan = None
        result = run_simulation([])
        self.reliability_history = result['beta'].tolist()
        self.current_reliability = self.reliability_history[0]
        self.prev_reliability = self.current_reliability
        return self._get_state()

    def step(self, action):
        self.step_count += 1
        self.prev_reliability = self.current_reliability
        if action == 1 and self.current_year - self.last_maintenance_year < self.maintenance_cooldown:
            action = 0
        if action == 1:
            if len(self.maintenance_times) >= self.max_maintenance:
                action = 0
            else:
                self.maintenance_times.append(self.current_year)
                self.last_maintenance_year = self.current_year
        self.current_year += 1
        result = run_simulation(self.maintenance_times)
        self.reliability_history = result['beta'].tolist()
        self.current_reliability = self.reliability_history[self.current_year - 1]
        done = self.current_year >= self.max_year
        reward = 0.0
        if done:
            try:
                if _calc_unified is not None:
                    final_reward = float(_calc_unified(self.maintenance_times, self.beta_threshold))
                    reward = final_reward
                else:
                    final_reward = self._calculate_reward(self.maintenance_times, action, done)
                    reward = final_reward
            except Exception:
                final_reward = self._calculate_reward(self.maintenance_times, action, done)
                reward = final_reward
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_plan = self.maintenance_times.copy()
        info = {'maintenance_times': self.maintenance_times.copy(), 'current_reliability': self.current_reliability, 'best_plan': self.best_plan, 'best_reward': self.best_reward}
        return (self._get_state(), reward, done, info)

    def _calculate_reward(self, maintenance_years, action, done):
        reward = 0
        if action == 1:
            maintenance_cost = calculate_cost(self.current_year)
            reward -= maintenance_cost * 0.0005
        if done:
            if self.use_baseline_reward and self.baseline_system:
                reward = self.baseline_system.calculate_reward_with_baseline(maintenance_years)
            else:
                reward = self._calculate_original_reward(maintenance_years)
        return reward

    def _calculate_original_reward(self, maintenance_years):
        if not maintenance_years:
            return -200
        result = run_simulation(maintenance_years)
        beta_values = result['beta']
        reliability_reward = 0
        for beta in beta_values:
            if beta < self.beta_threshold:
                reliability_reward -= 20
            else:
                reliability_reward += 5
        maintenance_count = len(maintenance_years)
        maintenance_reward = 0
        if maintenance_count < 4:
            maintenance_reward = -50 * (4 - maintenance_count)
        elif maintenance_count == 4:
            maintenance_reward = 0
        elif maintenance_count == 5:
            maintenance_reward = 100
        elif maintenance_count == 6:
            maintenance_reward = 0
        else:
            maintenance_reward = -50 * (maintenance_count - 6)
        total_cost = sum((calculate_cost(year) for year in maintenance_years))
        cost_reward = -total_cost * 0.001
        interval_reward = 0
        if len(maintenance_years) > 1:
            gaps = [maintenance_years[i] - maintenance_years[i - 1] for i in range(1, len(maintenance_years))]
            for i in range(1, len(gaps)):
                if gaps[i] <= gaps[i - 1]:
                    interval_reward += 5
                else:
                    interval_reward -= 10
        reward = reliability_reward + maintenance_reward + cost_reward + interval_reward
        return reward

    def _get_state(self):
        years_since_maintenance = self.current_year - self.last_maintenance_year
        used_maintenance = len(self.maintenance_times)
        remaining_maintenance = self.max_maintenance - used_maintenance
        resource_usage_ratio = used_maintenance / self.max_maintenance if self.max_maintenance > 0 else 0
        time_progress_ratio = self.current_year / self.max_year if self.max_year > 0 else 0
        return np.array([self.current_year, self.current_reliability, years_since_maintenance, used_maintenance, remaining_maintenance, resource_usage_ratio, time_progress_ratio], dtype=np.float32)

    def render(self, mode='human'):
        plt.figure(figsize=(4, 3))
        years = range(len(self.reliability_history))
        plt.plot(years, self.reliability_history, 'b-', label='Reliability')
        for year in self.maintenance_times:
            plt.axvline(x=year, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=self.beta_threshold, color='g', linestyle='--', label='Threshold')
        plt.xlabel('Year')
        plt.ylabel('Reliability Index (β)')
        plt.title('Maintenance Plan Visualization')
        plt.legend()
        plt.grid(True)
        plt.show()
