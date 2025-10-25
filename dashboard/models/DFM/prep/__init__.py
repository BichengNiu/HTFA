# -*- coding: utf-8 -*-
"""
DFM数据准备模块包

这个包提供DFM数据准备的完整功能，包括：
- 数据加载和预处理
- 数据对齐和清理
- 平稳性处理
- 映射管理
"""

__version__ = "1.0.0"
__author__ = "DFM Data Preparation Team"

# 导入主要接口函数
from dashboard.models.DFM.prep.data_preparation import prepare_data, load_mappings
from dashboard.models.DFM.prep.modules.stationarity_processor import apply_stationarity_transforms

__all__ = [
    'prepare_data',
    'load_mappings',
    'apply_stationarity_transforms'
]
