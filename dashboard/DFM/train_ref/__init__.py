# -*- coding: utf-8 -*-
"""
DFM训练模块 - 重构版本
基于KISS, DRY, YAGNI, LOD原则的全新架构

主要组件：
- DFMTrainer: 统一训练接口
- TrainingConfig: 配置管理
- DFMResults: 结果对象
"""

from dashboard.DFM.train_ref.facade import DFMTrainer
from dashboard.DFM.train_ref.training.config import (
    TrainingConfig,
    ModelConfig,
    SelectionConfig,
    OptimizationConfig
)

__version__ = '2.0.0'

__all__ = [
    'DFMTrainer',
    'TrainingConfig',
    'ModelConfig',
    'SelectionConfig',
    'OptimizationConfig',
]
