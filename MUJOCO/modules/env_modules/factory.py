from .discrete_env import CorrosionMaintenanceEnvDiscrete
from .direct_env import DirectYearSelectionEnv

class EnvironmentFactory:

    @staticmethod
    def create_environment(env_type, **kwargs):
        common_config = {'beta_threshold': 3.7, 'max_maintenance': 6, 'use_baseline_reward': True}
        if env_type == 'discrete':
            config = {'max_year': 200}
            config.update(common_config)
            config.update(kwargs)
            return CorrosionMaintenanceEnvDiscrete(**config)
        elif env_type == 'direct':
            config = {'max_years': 200, 'min_year': 50}
            config.update(common_config)
            config.update(kwargs)
            return DirectYearSelectionEnv(**config)
        else:
            raise ValueError(f'Unsupported environment type: {env_type}')
