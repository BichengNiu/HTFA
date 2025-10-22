# -*- coding: utf-8 -*-
"""
训练协调层

训练流程管理和配置：
- trainer: DFMTrainer训练器（包含pipeline逻辑）
- config: TrainingConfig配置管理（扁平化结构）
"""

from dashboard.DFM.train_ref.training.trainer import (
    DFMTrainer,
    ModelEvaluator,
    EvaluationMetrics,
    DFMModelResult,
    TrainingResult
)
from dashboard.DFM.train_ref.training.config import TrainingConfig

__all__ = [
    'DFMTrainer',
    'ModelEvaluator',
    'EvaluationMetrics',
    'DFMModelResult',
    'TrainingResult',
    'TrainingConfig',
]
