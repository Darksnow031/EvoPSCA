import gym
import numpy as np
from abc import ABC, abstractmethod

class BaseMaintenanceEnv(gym.Env, ABC):

    def __init__(self, max_year=200, beta_threshold=3.7, max_maintenance=6, normalize_rewards=False, target_years=None, use_baseline_reward=True):
        super().__init__()
        self.max_year = max_year
        self.beta_threshold = beta_threshold
        self.max_maintenance = max_maintenance
        self.normalize_rewards = normalize_rewards
        self.target_years = target_years
        self.use_baseline_reward = use_baseline_reward
        self.current_step = 0
        self.maintenance_plan = []
        self.best_reward = float('-inf')
        self.best_plan = None
        self.reward_history = []

    def get_best_maintenance_plan(self):
        return sorted(self.best_plan) if self.best_plan else []

    def get_current_maintenance_plan(self):
        return sorted(self.maintenance_plan)

    @abstractmethod
    def _calculate_reward(self, *args, **kwargs):
        pass
