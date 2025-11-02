# -*- coding: utf-8 -*-
"""
Core基础设施模块
提供统一的装饰器、基类和通用工具
"""

import threading
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


# ==================== 工具函数 ====================

def _get_logger_from_args(args: tuple):
    """
    从函数参数中获取日志记录器

    Args:
        args: 函数参数元组

    Returns:
        日志记录器实例
    """
    if args and hasattr(args[0], 'logger'):
        return args[0].logger
    return logger


# ==================== 装饰器 ====================

def safe_operation(default_return: Any = None, log_error: bool = True):
    """
    安全操作装饰器 - 统一异常处理

    Args:
        default_return: 发生异常时的默认返回值
        log_error: 是否记录错误日志

    Usage:
        @safe_operation(default_return=False)
        def some_method(self, key, value):
            return True
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    target_logger = _get_logger_from_args(args)
                    target_logger.error(f"Operation '{func.__name__}' failed: {e}")
                return default_return
        return wrapper
    return decorator


def timed_operation(log_level: str = 'DEBUG'):
    """
    计时装饰器 - 记录操作耗时

    Args:
        log_level: 日志级别 ('DEBUG', 'INFO', 'WARNING')

    Usage:
        @timed_operation(log_level='INFO')
        def some_method(self):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed_time = (time.time() - start_time) * 1000

                target_logger = _get_logger_from_args(args)
                log_message = f"Operation '{func.__name__}' completed in {elapsed_time:.2f}ms"

                if log_level == 'INFO':
                    target_logger.info(log_message)
                elif log_level == 'WARNING':
                    target_logger.warning(log_message)
                else:
                    target_logger.debug(log_message)

                return result
            except Exception as e:
                elapsed_time = (time.time() - start_time) * 1000
                target_logger = _get_logger_from_args(args)
                target_logger.error(
                    f"Operation '{func.__name__}' failed after {elapsed_time:.2f}ms: {e}"
                )
                raise
        return wrapper
    return decorator


def thread_safe(func: Callable) -> Callable:
    """
    线程安全装饰器 - 自动使用对象的锁

    要求被装饰对象必须有 _lock 属性

    Usage:
        @thread_safe
        def some_method(self, key):
            pass
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_lock'):
            raise AttributeError(
                f"Object {self.__class__.__name__} must have '_lock' attribute for @thread_safe decorator"
            )
        with self._lock:
            return func(self, *args, **kwargs)
    return wrapper


# ==================== 单例基类 ====================

class ThreadSafeSingleton:
    """
    线程安全单例基类

    使用继承方式实现单例模式，包含重置功能

    Usage:
        class MyManager(ThreadSafeSingleton):
            def __init__(self):
                if hasattr(self, '_initialized'):
                    return
                super().__init__()
                # 初始化代码
                self._initialized = True
    """
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__new__(cls)
                    cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def reset_instance(cls):
        """重置单例实例（主要用于测试）"""
        with cls._lock:
            if cls in cls._instances:
                del cls._instances[cls]

    @classmethod
    def get_instance(cls, *args, **kwargs):
        """获取单例实例"""
        return cls(*args, **kwargs)


# ==================== 工具函数 ====================

def validate_required_attributes(obj: Any, *attrs: str):
    """
    验证对象是否具有必需的属性

    Args:
        obj: 待验证对象
        *attrs: 必需的属性名列表

    Raises:
        AttributeError: 如果缺少必需属性
    """
    missing_attrs = [attr for attr in attrs if not hasattr(obj, attr)]
    if missing_attrs:
        raise AttributeError(
            f"Object {obj.__class__.__name__} missing required attributes: {', '.join(missing_attrs)}"
        )
