# -*- coding: utf-8 -*-
"""
DFM数据准备模块包

这个包提供DFM数据准备的完整功能，包括：
- 数据加载和预处理
- 数据对齐和清理
- 映射管理
- 时间范围统计（新增）
- 智能缺失值检测（根据频率关系）

使用 api.py 中的标准化API接口进行前后端分离开发。

重构说明（2025-11-13）：
- 简化为7步流程
- 消除Pipeline和Core层
- 删除工作表自动推断功能
- 新增时间范围统计功能
"""

__version__ = "3.0.0"
__author__ = "DFM Data Preparation Team"

from dashboard.models.DFM.prep.api import (
    prepare_dfm_data_simple,
    load_mappings_once,
    collect_time_ranges,
    validate_preparation_parameters,
    clear_mapping_cache
)

# 向后兼容别名
prepare_dfm_data = prepare_dfm_data_simple
load_variable_mappings = load_mappings_once

__all__ = [
    'prepare_dfm_data_simple',  # 新API
    'load_mappings_once',       # 新API
    'collect_time_ranges',      # 新API
    'validate_preparation_parameters',
    'clear_mapping_cache',      # 新API
    # 向后兼容
    'prepare_dfm_data',
    'load_variable_mappings'
]
