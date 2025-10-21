# -*- coding: utf-8 -*-
"""
优化层

性能优化组件（可选）：
- cache: 统一缓存管理
- precompute: 预计算引擎
"""

from dashboard.DFM.train_ref.optimization.cache import CacheManager, get_cache
from dashboard.DFM.train_ref.optimization.precompute import PrecomputeEngine

__all__ = [
    'CacheManager',
    'get_cache',
    'PrecomputeEngine',
]
