# -*- coding: utf-8 -*-
"""
统一状态管理辅助函数模块
用于替换重复的get_tools_manager等函数定义
"""

import streamlit as st
from typing import Any, Optional
from functools import lru_cache
import logging
from dashboard.core import get_unified_manager

# 设置日志
logger = logging.getLogger(__name__)

def get_tools_manager():
    """获取统一状态管理器实例"""
    return get_unified_manager()

# 全局实例
_tools_manager_instance = None

def get_tools_manager_instance():
    """获取统一状态管理器实例（单例模式）"""
    global _tools_manager_instance
    if _tools_manager_instance is None:
        _tools_manager_instance = get_tools_manager()
    return _tools_manager_instance

# === Dashboard专用状态管理函数 ===

def get_state_manager():
    """获取统一状态管理器实例"""
    return get_unified_manager()


def get_dashboard_state(key: str, default: Any = None) -> Any:
    """
    获取dashboard状态值

    Args:
        key: 状态键名
        default: 默认值

    Returns:
        Any: 状态值
    """
    state_manager = get_state_manager()
    full_key = f"dashboard.{key}"
    return state_manager.get_state(full_key, default)


def set_dashboard_state(key: str, value: Any) -> bool:
    """
    设置dashboard状态值

    Args:
        key: 状态键名
        value: 状态值

    Returns:
        bool: 设置是否成功
    """
    state_manager = get_state_manager()
    full_key = f"dashboard.{key}"
    return state_manager.set_state(full_key, value)


def get_staged_data() -> dict:
    """
    获取暂存数据

    Returns:
        dict: 暂存数据字典
    """
    return get_dashboard_state('staged_data', {})


def get_staged_data_options() -> list:
    """
    获取暂存数据选项列表

    Returns:
        list: 数据集名称列表
    """
    staged_data = get_staged_data()
    return [''] + list(staged_data.keys())


def clear_analysis_states(analysis_type: str) -> bool:
    """
    清理特定分析类型的状态

    Args:
        analysis_type: 分析类型 (stationarity, correlation, lead_lag)

    Returns:
        bool: 清理是否成功
    """
    state_manager = get_state_manager()
    return state_manager.clear_analysis_states(analysis_type)


def set_analysis_data(analysis_type: str, dataset_name: str, dataset_df: Any) -> bool:
    """
    设置分析数据

    Args:
        analysis_type: 分析类型
        dataset_name: 数据集名称
        dataset_df: 数据集DataFrame

    Returns:
        bool: 设置是否成功
    """
    state_manager = get_state_manager()
    return state_manager.set_analysis_data(analysis_type, dataset_name, dataset_df)


def clear_analysis_data(analysis_type: str) -> bool:
    """
    清理分析数据

    Args:
        analysis_type: 分析类型

    Returns:
        bool: 清理是否成功
    """
    state_manager = get_state_manager()
    return state_manager.clear_analysis_data(analysis_type)


# === 通用状态管理函数 ===

def get_tools_state(module_name: str, key: str, default: Any = None) -> Any:
    """获取工具模块状态值（通用版本）"""
    state_manager = get_tools_manager_instance()
    full_key = f"tools.{module_name}.{key}"
    return state_manager.get_state(full_key, default)

def set_tools_state(module_name: str, key: str, value: Any) -> bool:
    """设置工具模块状态值（通用版本）"""
    state_manager = get_tools_manager_instance()
    full_key = f"tools.{module_name}.{key}"
    return state_manager.set_state(full_key, value)

# === 特定模块的状态管理函数 ===

def get_property_state(key: str, default: Any = None) -> Any:
    """获取explore模块状态值"""
    return get_tools_state('explore', key, default)

def set_property_state(key: str, value: Any) -> bool:
    """设置explore模块状态值"""
    return set_tools_state('explore', key, value)

def get_exploration_state(module_name: str, key: str, default: Any = None) -> Any:
    """获取数据探索模块状态"""
    state_key = f'exploration.{module_name}.{key}'
    return get_tools_state('property', state_key, default)

def set_exploration_state(module_name: str, key: str, value: Any) -> bool:
    """设置数据探索模块状态"""
    state_key = f'exploration.{module_name}.{key}'
    return set_tools_state('property', state_key, value)

# === 特定功能的状态管理函数 ===

def get_dtw_state(key: str, default: Any = None) -> Any:
    """获取DTW分析状态"""
    return get_tools_state('property', f'dtw.{key}', default)

def set_dtw_state(key: str, value: Any) -> bool:
    """设置DTW分析状态"""
    return set_tools_state('property', f'dtw.{key}', value)

# === 数据输入组件状态管理函数 ===

def get_data_input_state(component_name: str, key: str, default: Any = None) -> Any:
    """获取数据输入组件状态"""
    return get_tools_state('data_input', f'{component_name}.{key}', default)

def set_data_input_state(component_name: str, key: str, value: Any) -> bool:
    """设置数据输入组件状态"""
    return set_tools_state('data_input', f'{component_name}.{key}', value)

def get_upload_state(key: str, default: Any = None) -> Any:
    """获取数据上传状态"""
    return get_data_input_state('upload', key, default)

def set_upload_state(key: str, value: Any) -> bool:
    """设置数据上传状态"""
    return set_data_input_state('upload', key, value)

def get_validation_state(key: str, default: Any = None) -> Any:
    """获取数据验证状态"""
    return get_data_input_state('validation', key, default)

def set_validation_state(key: str, value: Any) -> bool:
    """设置数据验证状态"""
    return set_data_input_state('validation', key, value)

def get_staging_state(key: str, default: Any = None) -> Any:
    """获取数据暂存状态"""
    return get_data_input_state('staging', key, default)

def set_staging_state(key: str, value: Any) -> bool:
    """设置数据暂存状态"""
    return set_data_input_state('staging', key, value)

def get_preview_state(key: str, default: Any = None) -> Any:
    """获取数据预览状态"""
    return get_data_input_state('preview', key, default)

def set_preview_state(key: str, value: Any) -> bool:
    """设置数据预览状态"""
    return set_data_input_state('preview', key, value)

# === 导航状态缓存管理 ===

def get_cached_navigation_state():
    """获取导航状态"""
    state_manager = get_unified_manager()
    return {
        'main_module': state_manager.get_state('navigation.main_module'),
        'sub_module': state_manager.get_state('navigation.sub_module')
    }

def is_in_data_exploration():
    """检查是否在数据探索模块中（缓存版本）"""
    nav_state = get_cached_navigation_state()
    return nav_state['main_module'] == "数据探索"

def get_current_navigation_info():
    """获取当前导航信息（缓存版本）"""
    nav_state = get_cached_navigation_state()
    return nav_state['main_module'], nav_state['sub_module']

# === 缓存管理 ===

def clear_state_cache():
    """清除状态管理缓存"""
    global _tools_manager_instance
    _tools_manager_instance = None
    get_tools_manager.clear()
    get_cached_navigation_state.clear()  # 清除导航状态缓存
    logger.info("状态管理缓存已清除")

# === 健康检查 ===

def check_state_manager_health() -> bool:
    """检查状态管理器健康状态"""
    try:
        tools_manager = get_tools_manager_instance()
        if tools_manager is None:
            return False
        
        # 尝试简单的状态操作
        test_key = "_health_check_test"
        test_value = "test"
        
        # 设置测试值
        if not set_tools_state('test', test_key, test_value):
            return False
        
        # 读取测试值
        retrieved_value = get_tools_state('test', test_key)
        if retrieved_value != test_value:
            return False
        
        logger.info("状态管理器健康检查通过")
        return True
        
    except Exception as e:
        logger.error(f"状态管理器健康检查失败: {e}")
        return False

# === 向后兼容性支持 ===

# 为了向后兼容，保留一些旧的函数名
get_global_tools_manager = get_tools_manager_instance

def detect_current_module():
    """检测当前活跃的分析模块"""
    state_manager = get_unified_manager()
    return state_manager.get_state('navigation.current_module', None)

# === 模块信息 ===

__version__ = "1.0.0"
__author__ = "UI优化团队"
__description__ = "统一状态管理辅助函数模块"

# 导出的公共接口
__all__ = [
    'get_tools_manager',
    'get_tools_manager_instance',
    'get_tools_state',
    'set_tools_state',
    'get_compute_state',
    'set_compute_state',
    'get_clean_state',
    'set_clean_state',
    'get_property_state',
    'set_property_state',
    'get_exploration_state',
    'set_exploration_state',
    'get_missing_ui_state',
    'set_missing_ui_state',
    'get_dtw_state',
    'set_dtw_state',
    'get_cached_navigation_state',
    'is_in_data_exploration',
    'get_current_navigation_info',
    'clear_state_cache',
    'check_state_manager_health',
    'detect_current_module',
    # 数据输入组件状态管理函数
    'get_data_input_state',
    'set_data_input_state',
    'get_upload_state',
    'set_upload_state',
    'get_validation_state',
    'set_validation_state',
    'get_staging_state',
    'set_staging_state',
    'get_preview_state',
    'set_preview_state'
]
