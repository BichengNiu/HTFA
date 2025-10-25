#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可重现性工具模块
提供统一的随机种子管理和确定性操作
"""

import os
import random
import numpy as np
import hashlib
from typing import Optional, Any, Dict, List
from functools import wraps
import logging

# 全局默认种子
DEFAULT_SEED = 42

# 日志配置
logger = logging.getLogger(__name__)

class ReproducibilityManager:
    """可重现性管理器"""
    
    def __init__(self, global_seed: int = DEFAULT_SEED):
        """
        初始化可重现性管理器
        
        Args:
            global_seed: 全局随机种子
        """
        self.global_seed = global_seed
        self.subprocess_seeds = {}
        self.operation_counter = 0
        
    def set_global_seed(self, seed: Optional[int] = None):
        """
        设置全局随机种子
        
        Args:
            seed: 随机种子，如果为None则使用默认种子
        """
        if seed is None:
            seed = self.global_seed
            
        # 设置Python随机种子
        random.seed(seed)
        
        # 设置NumPy随机种子
        np.random.seed(seed)
        
        # 设置环境变量（用于子进程）
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['REPRODUCIBILITY_SEED'] = str(seed)
        
        logger.info(f"全局随机种子已设置为: {seed}")
        
    def get_subprocess_seed(self, process_id: str) -> int:
        """
        为子进程生成确定性种子
        
        Args:
            process_id: 进程标识符
            
        Returns:
            确定性的子进程种子
        """
        if process_id not in self.subprocess_seeds:
            # 基于全局种子和进程ID生成确定性种子
            seed_str = f"{self.global_seed}_{process_id}"
            seed_hash = hashlib.md5(seed_str.encode()).hexdigest()
            subprocess_seed = int(seed_hash[:8], 16) % (2**31 - 1)
            self.subprocess_seeds[process_id] = subprocess_seed
            
        return self.subprocess_seeds[process_id]
    
    def get_operation_seed(self, operation_name: str) -> int:
        """
        为特定操作生成确定性种子
        
        Args:
            operation_name: 操作名称
            
        Returns:
            确定性的操作种子
        """
        self.operation_counter += 1
        seed_str = f"{self.global_seed}_{operation_name}_{self.operation_counter}"
        seed_hash = hashlib.md5(seed_str.encode()).hexdigest()
        operation_seed = int(seed_hash[:8], 16) % (2**31 - 1)
        
        return operation_seed
    
    def ensure_deterministic_dict_iteration(self, data: Dict[Any, Any]) -> List[tuple]:
        """
        确保字典迭代的确定性
        
        Args:
            data: 输入字典
            
        Returns:
            排序后的键值对列表
        """
        try:
            # 尝试按键排序
            return sorted(data.items())
        except TypeError:
            # 如果键不可比较，按字符串表示排序
            return sorted(data.items(), key=lambda x: str(x[0]))
    
    def ensure_deterministic_list_order(self, data: List[Any]) -> List[Any]:
        """
        确保列表顺序的确定性
        
        Args:
            data: 输入列表
            
        Returns:
            排序后的列表
        """
        try:
            return sorted(data)
        except TypeError:
            # 如果元素不可比较，按字符串表示排序
            return sorted(data, key=str)

# 全局可重现性管理器实例
_global_manager = ReproducibilityManager()

def set_global_seed(seed: Optional[int] = None):
    """设置全局随机种子的便捷函数"""
    _global_manager.set_global_seed(seed)

def get_subprocess_seed(process_id: str) -> int:
    """获取子进程种子的便捷函数"""
    return _global_manager.get_subprocess_seed(process_id)

def get_operation_seed(operation_name: str) -> int:
    """获取操作种子的便捷函数"""
    return _global_manager.get_operation_seed(operation_name)

def ensure_deterministic(func):
    """
    装饰器：确保函数执行的确定性
    
    Args:
        func: 要装饰的函数
        
    Returns:
        装饰后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 在函数执行前设置种子
        operation_seed = get_operation_seed(func.__name__)
        random.seed(operation_seed)
        np.random.seed(operation_seed)
        
        # 执行函数
        result = func(*args, **kwargs)
        
        return result
    
    return wrapper

def setup_subprocess_reproducibility():
    """
    为子进程设置可重现性环境
    应在子进程初始化时调用
    """
    # 从环境变量获取种子
    seed_str = os.environ.get('REPRODUCIBILITY_SEED', str(DEFAULT_SEED))
    try:
        seed = int(seed_str)
    except ValueError:
        seed = DEFAULT_SEED
    
    # 获取进程特定的种子
    process_id = f"subprocess_{os.getpid()}"
    subprocess_seed = _global_manager.get_subprocess_seed(process_id)
    
    # 设置随机种子
    random.seed(subprocess_seed)
    np.random.seed(subprocess_seed)
    
    logger.info(f"子进程 {process_id} 随机种子已设置为: {subprocess_seed}")

def verify_reproducibility(func, *args, num_runs: int = 3, **kwargs):
    """
    验证函数的可重现性
    
    Args:
        func: 要验证的函数
        *args: 函数参数
        num_runs: 运行次数
        **kwargs: 函数关键字参数
        
    Returns:
        bool: 是否可重现
    """
    results = []
    
    for i in range(num_runs):
        # 每次运行前重置种子
        set_global_seed()
        
        try:
            result = func(*args, **kwargs)
            results.append(result)
        except Exception as e:
            logger.error(f"验证运行 {i+1} 失败: {e}")
            return False
    
    # 检查所有结果是否相同
    if not results:
        return False
    
    first_result = results[0]
    for i, result in enumerate(results[1:], 1):
        if not _compare_results(first_result, result):
            logger.warning(f"运行 {i+1} 的结果与第一次运行不同")
            return False
    
    logger.info(f"函数 {func.__name__} 在 {num_runs} 次运行中保持一致")
    return True

def _compare_results(result1, result2):
    """比较两个结果是否相同"""
    try:
        if isinstance(result1, (list, tuple)) and isinstance(result2, (list, tuple)):
            if len(result1) != len(result2):
                return False
            return all(_compare_results(r1, r2) for r1, r2 in zip(result1, result2))
        
        elif isinstance(result1, dict) and isinstance(result2, dict):
            if set(result1.keys()) != set(result2.keys()):
                return False
            return all(_compare_results(result1[k], result2[k]) for k in result1.keys())
        
        elif isinstance(result1, np.ndarray) and isinstance(result2, np.ndarray):
            return np.array_equal(result1, result2)
        
        else:
            return result1 == result2
            
    except Exception:
        return False

# 在模块导入时自动设置全局种子
set_global_seed()
