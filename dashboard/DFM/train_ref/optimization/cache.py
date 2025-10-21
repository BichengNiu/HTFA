# -*- coding: utf-8 -*-
"""
缓存管理模块

提供基于LRU的评估结果缓存
参考: dashboard/DFM/train_model/evaluation_cache.py
"""

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import pandas as pd
from dashboard.DFM.train_ref.utils.logger import get_logger


logger = get_logger(__name__)


class CacheManager:
    """LRU缓存管理器

    特性：
    - 线程安全
    - LRU淘汰策略
    - 统计信息

    Args:
        max_size: 最大缓存条目数
        precision: 浮点数精度
    """

    def __init__(self, max_size: int = 1000, precision: int = 6):
        self.max_size = max_size
        self.precision = precision

        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()

        self._stats = {
            'hits': 0,
            'misses': 0,
            'puts': 0,
            'evictions': 0
        }

    def generate_key(
        self,
        variables: List[str],
        k_factors: int,
        target_variable: str,
        train_end: str,
        validation_end: str,
        max_lags: int = 1,
        **kwargs
    ) -> str:
        """生成缓存键

        Args:
            variables: 变量列表
            k_factors: 因子数量
            target_variable: 目标变量
            train_end: 训练结束日期
            validation_end: 验证结束日期
            max_lags: 最大滞后
            **kwargs: 其他参数

        Returns:
            str: 缓存键（MD5哈希）
        """
        key_dict = {
            'variables': sorted(variables),
            'k_factors': k_factors,
            'target_variable': target_variable,
            'train_end': train_end,
            'validation_end': validation_end,
            'max_lags': max_lags,
        }

        for k, v in kwargs.items():
            if isinstance(v, float):
                key_dict[k] = round(v, self.precision)
            else:
                key_dict[k] = v

        key_str = json.dumps(key_dict, sort_keys=True, ensure_ascii=False)
        key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()

        return key_hash

    def get(self, key: str) -> Optional[Tuple]:
        """获取缓存

        Args:
            key: 缓存键

        Returns:
            Optional[Tuple]: 缓存值，未命中返回None
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._stats['hits'] += 1
                logger.debug(f"缓存命中: {key[:8]}... (命中率: {self.get_hit_rate():.1%})")
                return self._cache[key]
            else:
                self._stats['misses'] += 1
                logger.debug(f"缓存未命中: {key[:8]}...")
                return None

    def put(self, key: str, value: Tuple) -> None:
        """存入缓存

        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._lock:
            if key not in self._cache and len(self._cache) >= self.max_size:
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]
                self._stats['evictions'] += 1
                logger.debug(f"缓存淘汰: {evicted_key[:8]}...")

            self._cache[key] = value
            self._cache.move_to_end(key)
            self._stats['puts'] += 1

            logger.debug(f"缓存存储: {key[:8]}... (大小: {len(self._cache)}/{self.max_size})")

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            logger.info("缓存已清空")

    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self._stats['hits'] + self._stats['misses']
        if total == 0:
            return 0.0
        return self._stats['hits'] / total

    def get_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': self.get_hit_rate(),
                **self._stats
            }

    def print_statistics(self) -> None:
        """打印统计信息"""
        stats = self.get_statistics()
        print(f"\n=== 缓存统计 ===")
        print(f"大小: {stats['size']}/{stats['max_size']}")
        print(f"命中率: {stats['hit_rate']:.1%}")
        print(f"命中: {stats['hits']}, 未命中: {stats['misses']}")
        print(f"存储: {stats['puts']}, 淘汰: {stats['evictions']}")
        print(f"=" * 16)


_global_cache: Optional[CacheManager] = None


def get_cache(max_size: int = 1000) -> CacheManager:
    """获取全局缓存实例（单例）

    Args:
        max_size: 最大缓存大小

    Returns:
        CacheManager: 全局缓存实例
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager(max_size=max_size)
    return _global_cache
