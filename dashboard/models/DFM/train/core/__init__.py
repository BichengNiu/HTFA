# -*- coding: utf-8 -*-
"""
核心算法层

包含DFM模型的核心算法实现：
- models: 统一数据模型
- kalman: 卡尔曼滤波器
- factor_model: DFM主算法
- estimator: 参数估计
- prediction: 目标预测
"""

# 数据模型
from dashboard.models.DFM.train.core.models import (
    EvaluationMetrics,
    MetricsResult,
    DFMModelResult,
    KalmanFilterResult,
    KalmanSmootherResult,
    SelectionResult,
    TrainingResult
)

# 卡尔曼滤波
from dashboard.models.DFM.train.core.kalman import KalmanFilter, kalman_filter, kalman_smoother

# DFM模型
from dashboard.models.DFM.train.core.factor_model import DFMModel, fit_dfm

# 参数估计
from dashboard.models.DFM.train.core.estimator import estimate_loadings

# 目标预测
from dashboard.models.DFM.train.core.prediction import generate_target_forecast

# PCA工具
from dashboard.models.DFM.train.core.pca_utils import select_num_factors

__all__ = [
    # 数据模型（已合并DFMResults到DFMModelResult）
    'EvaluationMetrics',
    'MetricsResult',
    'DFMModelResult',
    'KalmanFilterResult',
    'KalmanSmootherResult',
    'SelectionResult',
    'TrainingResult',

    # 算法组件
    'KalmanFilter',
    'kalman_filter',
    'kalman_smoother',
    'DFMModel',
    'fit_dfm',
    'estimate_loadings',
    'generate_target_forecast',
    'select_num_factors',
]
