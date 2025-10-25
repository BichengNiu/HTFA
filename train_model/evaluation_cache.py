# -*- coding: utf-8 -*-
"""
DFM评估缓存模块
用于缓存DFM模型评估结果，减少重复计算，提升性能
"""

import hashlib
import pickle
import json
import time
import threading
import os
from collections import OrderedDict
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DFMEvaluationCache:
    """
    DFM评估结果缓存类
    
    特性：
    1. 基于LRU的内存缓存
    2. 可选的磁盘持久化
    3. 线程安全
    4. 缓存统计
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        enable_disk_cache: bool = False,
        cache_dir: Optional[str] = None,
        precision: int = 6
    ):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            enable_disk_cache: 是否启用磁盘缓存
            cache_dir: 磁盘缓存目录
            precision: 浮点数精度（用于键生成）
        """
        self.max_size = max_size
        self.enable_disk_cache = enable_disk_cache
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '.cache')
        self.precision = precision
        
        # 内存缓存（OrderedDict实现LRU）
        self._cache: OrderedDict = OrderedDict()
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 统计信息
        self._stats = {
            'hits': 0,
            'misses': 0,
            'puts': 0,
            'evictions': 0,
            'total_compute_time_saved': 0.0,
            'cache_creation_time': time.time()
        }
        
        # 如果启用磁盘缓存，确保目录存在
        if self.enable_disk_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        logger.info(f"DFM评估缓存初始化：max_size={max_size}, disk_cache={enable_disk_cache}")
    
    def generate_cache_key(
        self,
        variables: List[str],
        params: Dict,
        target_variable: str,
        validation_start: str,
        validation_end: str,
        train_end_date: str,
        target_freq: str,
        max_iter: int,
        target_mean_original: float,
        target_std_original: float,
        max_lags: int = 1
    ) -> str:
        """
        生成缓存键
        
        使用所有影响评估结果的参数生成唯一键
        变量列表会被排序以确保顺序无关性
        """
        # 创建键字典
        key_dict = {
            # 排序变量列表（确保顺序无关）
            'variables': sorted(variables),
            'target_variable': target_variable,
            # 参数
            'k_factors': params.get('k_factors'),
            # 日期参数
            'validation_start': validation_start,
            'validation_end': validation_end,
            'train_end_date': train_end_date,
            'target_freq': target_freq,
            # 其他参数
            'max_iter': max_iter,
            'max_lags': max_lags,
            # 处理浮点数精度
            'target_mean': round(target_mean_original, self.precision) if pd.notna(target_mean_original) else None,
            'target_std': round(target_std_original, self.precision) if pd.notna(target_std_original) else None,
        }
        
        # 转换为JSON字符串（确保稳定的序列化）
        key_str = json.dumps(key_dict, sort_keys=True, ensure_ascii=False)
        
        # 生成MD5哈希作为键
        key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()
        
        return key_hash
    
    def get(
        self,
        key: str,
        compute_time_estimate: float = 2.149
    ) -> Optional[Tuple]:
        """
        从缓存获取结果
        
        Args:
            key: 缓存键
            compute_time_estimate: 估计的计算时间（用于统计）
            
        Returns:
            缓存的评估结果元组，如果未命中则返回None
        """
        with self._lock:
            if key in self._cache:
                # 命中：移动到末尾（LRU）
                self._cache.move_to_end(key)
                self._stats['hits'] += 1
                self._stats['total_compute_time_saved'] += compute_time_estimate
                
                result = self._cache[key]
                logger.debug(f"缓存命中：key={key[:8]}... (命中率: {self.get_hit_rate():.1%})")
                return result
            else:
                # 未命中
                self._stats['misses'] += 1
                logger.debug(f"缓存未命中：key={key[:8]}...")
                
                # 尝试从磁盘加载（如果启用）
                if self.enable_disk_cache:
                    disk_result = self._load_from_disk_file(key)
                    if disk_result is not None:
                        # 加载到内存缓存
                        self._put_to_memory(key, disk_result)
                        self._stats['hits'] += 1
                        self._stats['misses'] -= 1  # 修正统计
                        self._stats['total_compute_time_saved'] += compute_time_estimate
                        logger.debug(f"从磁盘缓存恢复：key={key[:8]}...")
                        return disk_result
                
                return None
    
    def put(
        self,
        key: str,
        value: Tuple,
        save_to_disk: bool = None
    ) -> None:
        """
        将结果存入缓存
        
        Args:
            key: 缓存键
            value: 评估结果元组
            save_to_disk: 是否保存到磁盘（None则使用默认设置）
        """
        with self._lock:
            # 存入内存缓存
            self._put_to_memory(key, value)
            
            # 可选：保存到磁盘
            if save_to_disk or (save_to_disk is None and self.enable_disk_cache):
                self._save_to_disk_file(key, value)
            
            self._stats['puts'] += 1
            logger.debug(f"缓存存储：key={key[:8]}... (缓存大小: {len(self._cache)}/{self.max_size})")
    
    def _put_to_memory(self, key: str, value: Tuple) -> None:
        """内部方法：存入内存缓存"""
        # 检查是否需要淘汰
        if key not in self._cache and len(self._cache) >= self.max_size:
            # 淘汰最早的条目（LRU）
            evicted_key = next(iter(self._cache))
            del self._cache[evicted_key]
            self._stats['evictions'] += 1
            logger.debug(f"缓存淘汰：key={evicted_key[:8]}...")
        
        # 存入或更新
        self._cache[key] = value
        self._cache.move_to_end(key)
    
    def _get_disk_filepath(self, key: str) -> str:
        """获取磁盘缓存文件路径"""
        return os.path.join(self.cache_dir, f"dfm_cache_{key}.pkl")
    
    def _save_to_disk_file(self, key: str, value: Tuple) -> None:
        """保存单个条目到磁盘"""
        try:
            filepath = self._get_disk_filepath(key)
            with open(filepath, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"保存到磁盘：{filepath}")
        except Exception as e:
            logger.warning(f"保存到磁盘失败：{e}")
    
    def _load_from_disk_file(self, key: str) -> Optional[Tuple]:
        """从磁盘加载单个条目"""
        try:
            filepath = self._get_disk_filepath(key)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    value = pickle.load(f)
                return value
        except Exception as e:
            logger.warning(f"从磁盘加载失败：{e}")
        return None
    
    def save_to_disk(self, filepath: Optional[str] = None) -> bool:
        """
        保存整个缓存到磁盘
        
        Args:
            filepath: 保存路径，如果为None则使用默认路径
            
        Returns:
            是否成功
        """
        if filepath is None:
            filepath = os.path.join(self.cache_dir, 'dfm_cache_full.pkl')
        
        try:
            with self._lock:
                cache_data = {
                    'cache': dict(self._cache),
                    'stats': self._stats.copy(),
                    'config': {
                        'max_size': self.max_size,
                        'precision': self.precision
                    }
                }
                
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'wb') as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                logger.info(f"缓存已保存到磁盘：{filepath} (条目数: {len(self._cache)})")
                return True
                
        except Exception as e:
            logger.error(f"保存缓存到磁盘失败：{e}")
            return False
    
    def load_from_disk(self, filepath: Optional[str] = None) -> bool:
        """
        从磁盘加载缓存
        
        Args:
            filepath: 加载路径，如果为None则使用默认路径
            
        Returns:
            是否成功
        """
        if filepath is None:
            filepath = os.path.join(self.cache_dir, 'dfm_cache_full.pkl')
        
        if not os.path.exists(filepath):
            logger.warning(f"缓存文件不存在：{filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            with self._lock:
                self._cache = OrderedDict(cache_data['cache'])
                # 合并统计信息
                old_stats = cache_data['stats']
                self._stats['hits'] += old_stats.get('hits', 0)
                self._stats['misses'] += old_stats.get('misses', 0)
                self._stats['puts'] += old_stats.get('puts', 0)
                self._stats['evictions'] += old_stats.get('evictions', 0)
                self._stats['total_compute_time_saved'] += old_stats.get('total_compute_time_saved', 0)
                
                logger.info(f"缓存已从磁盘加载：{filepath} (条目数: {len(self._cache)})")
                return True
                
        except Exception as e:
            logger.error(f"从磁盘加载缓存失败：{e}")
            return False
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self._stats['hits'] + self._stats['misses']
        if total == 0:
            return 0.0
        return self._stats['hits'] / total
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            包含各种统计指标的字典
        """
        with self._lock:
            stats = self._stats.copy()
            stats['size'] = len(self._cache)
            stats['max_size'] = self.max_size
            stats['hit_rate'] = self.get_hit_rate()
            stats['memory_usage_mb'] = self._estimate_memory_usage() / (1024 * 1024)
            stats['uptime_seconds'] = time.time() - self._stats['cache_creation_time']
            return stats
    
    def _estimate_memory_usage(self) -> int:
        """估算内存使用（字节）"""
        # 简单估算：每个条目约1KB（包括键和值）
        return len(self._cache) * 1024
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            # 重置统计（保留创建时间）
            creation_time = self._stats['cache_creation_time']
            self._stats = {
                'hits': 0,
                'misses': 0,
                'puts': 0,
                'evictions': 0,
                'total_compute_time_saved': 0.0,
                'cache_creation_time': creation_time
            }
            logger.info("缓存已清空")
    
    def print_statistics(self) -> None:
        """打印缓存统计信息"""
        stats = self.get_statistics()
        print("\n=== DFM评估缓存统计 ===")
        print(f"缓存大小: {stats['size']}/{stats['max_size']}")
        print(f"命中率: {stats['hit_rate']:.1%}")
        print(f"命中次数: {stats['hits']}")
        print(f"未命中次数: {stats['misses']}")
        print(f"存储次数: {stats['puts']}")
        print(f"淘汰次数: {stats['evictions']}")
        print(f"节省计算时间: {stats['total_compute_time_saved']:.1f}秒")
        print(f"内存使用: {stats['memory_usage_mb']:.2f}MB")
        print(f"运行时间: {stats['uptime_seconds']:.1f}秒")
        print("=" * 25)


# 全局缓存实例（可选）
_global_cache: Optional[DFMEvaluationCache] = None


def get_global_cache(
    max_size: int = 1000,
    enable_disk_cache: bool = False,
    cache_dir: Optional[str] = None
) -> DFMEvaluationCache:
    """
    获取全局缓存实例（单例模式）
    
    Args:
        max_size: 最大缓存条目数
        enable_disk_cache: 是否启用磁盘缓存
        cache_dir: 磁盘缓存目录
        
    Returns:
        全局缓存实例
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = DFMEvaluationCache(
            max_size=max_size,
            enable_disk_cache=enable_disk_cache,
            cache_dir=cache_dir
        )
    return _global_cache


def clear_global_cache() -> None:
    """清空全局缓存"""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
        _global_cache = None