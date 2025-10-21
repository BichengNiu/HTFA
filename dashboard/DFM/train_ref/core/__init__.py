# -*- coding: utf-8 -*-
"""
核心算法层

包含DFM模型的核心算法实现：
- kalman: 卡尔曼滤波器
- factor_model: DFM主算法
- estimator: 参数估计
"""

from dashboard.DFM.train_ref.core.kalman import KalmanFilter, kalman_filter, kalman_smoother
from dashboard.DFM.train_ref.core.factor_model import DFMModel, fit_dfm
from dashboard.DFM.train_ref.core.estimator import estimate_loadings, estimate_parameters

__all__ = [
    'KalmanFilter',
    'kalman_filter',
    'kalman_smoother',
    'DFMModel',
    'fit_dfm',
    'estimate_loadings',
    'estimate_parameters',
]
