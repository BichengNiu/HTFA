# -*- coding: utf-8 -*-
"""
DFM模型分析组件模块

提供DFM模型分析相关的UI组件，包括：
- 模型加载组件
- 因子分析组件
- 预测分析组件
- 结果展示组件
"""

from dashboard.ui.components.dfm.model_analysis.model_load import ModelLoadComponent
from dashboard.ui.components.dfm.model_analysis.factor_analysis import FactorAnalysisComponent

__all__ = [
    'ModelLoadComponent',
    'FactorAnalysisComponent'
]
