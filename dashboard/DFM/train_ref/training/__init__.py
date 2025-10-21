# -*- coding: utf-8 -*-
"""
训练协调层

训练流程管理和配置：
- trainer: 训练协调器
- pipeline: 训练流水线
- config: 配置管理
"""

from dashboard.DFM.train_ref.training.trainer import Trainer
from dashboard.DFM.train_ref.training.pipeline import TrainingPipeline
from dashboard.DFM.train_ref.training.config import (
    TrainingConfig,
    ModelConfig,
    SelectionConfig,
    OptimizationConfig
)

__all__ = [
    'Trainer',
    'TrainingPipeline',
    'TrainingConfig',
    'ModelConfig',
    'SelectionConfig',
    'OptimizationConfig',
]
