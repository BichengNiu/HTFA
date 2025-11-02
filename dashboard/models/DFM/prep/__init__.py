# -*- coding: utf-8 -*-
"""
DFM数据准备模块包

这个包提供DFM数据准备的完整功能，包括：
- 数据加载和预处理
- 数据对齐和清理
- 平稳性处理
- 映射管理

使用 api.py 中的标准化API接口进行前后端分离开发。
"""

__version__ = "2.0.0"
__author__ = "DFM Data Preparation Team"

from dashboard.models.DFM.prep.api import (
    prepare_dfm_data,
    load_variable_mappings,
    apply_stationarity_transforms,
    validate_preparation_parameters
)

__all__ = [
    'prepare_dfm_data',
    'load_variable_mappings',
    'apply_stationarity_transforms',
    'validate_preparation_parameters'
]
