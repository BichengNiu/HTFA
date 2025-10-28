# -*- coding: utf-8 -*-
"""
工具函数模块

包含数据验证、辅助函数和异常处理等工具类。
"""

from .validators import (
    validate_model_data,
    validate_impact_parameters,
    validate_time_series
)
from .helpers import (
    ensure_numerical_stability,
    safe_matrix_division,
    align_time_series
)
from .exceptions import (
    DecompError,
    ModelLoadError,
    ValidationError,
    ComputationError
)

__all__ = [
    # 验证器
    'validate_model_data',
    'validate_impact_parameters',
    'validate_time_series',
    # 辅助函数
    'ensure_numerical_stability',
    'safe_matrix_division',
    'align_time_series',
    # 异常类
    'DecompError',
    'ModelLoadError',
    'ValidationError',
    'ComputationError'
]