from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Parameters:
    beta_threshold: float = 3.7
    max_maintenance: int = 6
    min_year: int = 50
    max_year: int = 200
    target_years: List[int] = field(default_factory=lambda: [50, 100, 190])
    verbose: bool = True
    rl_train_steps: int = 2000
    learning_rate: float = 0.0003
    hidden_size: int = 256
    rl_model: Optional[Any] = None
    gamma: float = 0.99
    entropy_coef: float = 0.02
    entropy_constant: bool = True
    max_grad_norm: float = 0.5
    cem_iterations: int = 60
    pop_size: int = 200
    elite_frac: float = 0.15
    sigma_init: float = 0.2
    elitism: bool = False
    antithetic: bool = False
    damp: float = 0.001
    damp_limit: float = 1e-05
    early_stopping: bool = True
    patience: int = 20
    feedback_rounds: int = 1
    seed: int = 42

    def update(self, **kwargs) -> 'Parameters':
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()
OptimizationConfig = Parameters
__all__ = ['Parameters', 'OptimizationConfig']
