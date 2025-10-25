# -*- coding: utf-8 -*-
"""
变量选择层

变量选择算法:
- backward_selector: 后向选择算法

注: 方案B精简架构 - 移除了SelectionEngine抽象层,直接使用BackwardSelector
"""

from dashboard.models.DFM.train.selection.backward_selector import BackwardSelector, SelectionResult

__all__ = [
    'BackwardSelector',
    'SelectionResult',
]
