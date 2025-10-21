# -*- coding: utf-8 -*-
"""
评估层

模型评估和验证组件：
- evaluator: 核心评估逻辑
- metrics: 指标计算
- validator: 数据验证
"""

from dashboard.DFM.train_ref.evaluation.evaluator import DFMEvaluator, evaluate_model
from dashboard.DFM.train_ref.evaluation.metrics import calculate_metrics, MetricsResult
from dashboard.DFM.train_ref.evaluation.validator import DataValidator, validate_data

__all__ = [
    'DFMEvaluator',
    'evaluate_model',
    'calculate_metrics',
    'MetricsResult',
    'DataValidator',
    'validate_data',
]
