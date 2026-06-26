from __future__ import annotations
import os
import json
import random
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import numpy as np
import torch
from modules.utilities import create_output_dir
from modules.simulation import init_global_matrices
from modules.compat.parameters import OptimizationConfig
from modules.compat.mod_utils import NumpyEncoder

class BaseOptimizer(ABC):

    def __init__(self, config: Optional[OptimizationConfig]=None):
        self.config = config or OptimizationConfig()
        self.output_dir: Optional[str] = None
        self.history: dict = {}
        self.best_plan: Any = None
        self.best_score: float = float('-inf')
        self._set_random_seed()
        init_global_matrices(use_cuda=torch.cuda.is_available())

    def _set_random_seed(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def _create_output_dir(self, prefix: str='optimization') -> str:
        self.output_dir = create_output_dir(prefix=prefix)
        if getattr(self.config, 'verbose', True):
            return self.output_dir

    def _save_results(self, results: dict, filename: str) -> None:
        if not self.output_dir:
            return
        try:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4, cls=NumpyEncoder)
            if getattr(self.config, 'verbose', True):
                print(f'结果已保存至: {filepath}')
        except Exception as e:
            if getattr(self.config, 'verbose', True):
                print(f'保存结果时出错: {e}')

    @abstractmethod
    def optimize(self):
        raise NotImplementedError

    def get_best_solution(self) -> Tuple[Any, float]:
        return (self.best_plan, self.best_score)

    def set_global_tracker(self, tracker: Any) -> None:
        try:
            self.global_tracker = tracker
        except Exception:
            self.global_tracker = None
