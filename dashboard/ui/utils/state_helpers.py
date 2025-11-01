# -*- coding: utf-8 -*-
"""
状态管理辅助函数模块
提供命名空间封装，直接使用st.session_state进行状态管理
"""

import streamlit as st
from typing import Any, Optional
from functools import lru_cache
import logging

# 设置日志
logger = logging.getLogger(__name__)

# === Dashboard专用状态管理函数 ===

def get_dashboard_state(key: str, default: Any = None) -> Any:
    """
    获取dashboard状态值

    Args:
        key: 状态键名
        default: 默认值

    Returns:
        Any: 状态值
    """
    full_key = f"dashboard.{key}"
    return st.session_state.get(full_key, default)


def set_dashboard_state(key: str, value: Any) -> bool:
    """
    设置dashboard状态值

    Args:
        key: 状态键名
        value: 状态值

    Returns:
        bool: 设置是否成功
    """
    try:
        full_key = f"dashboard.{key}"
        st.session_state[full_key] = value
        return True
    except Exception as e:
        logger.error(f"设置dashboard状态失败: {key}, 错误: {e}")
        return False


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
    try:
        prefix = f"analysis.{analysis_type}."
        keys_to_delete = [k for k in st.session_state.keys() if str(k).startswith(prefix)]
        for k in keys_to_delete:
            del st.session_state[k]
        return True
    except Exception as e:
        logger.error(f"清理分析状态失败: {analysis_type}, 错误: {e}")
        return False


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
    try:
        key = f"analysis.{analysis_type}.{dataset_name}.data"
        st.session_state[key] = dataset_df
        return True
    except Exception as e:
        logger.error(f"设置分析数据失败: {analysis_type}.{dataset_name}, 错误: {e}")
        return False


def clear_analysis_data(analysis_type: str) -> bool:
    """
    清理分析数据

    Args:
        analysis_type: 分析类型

    Returns:
        bool: 清理是否成功
    """
    return clear_analysis_states(analysis_type)


# === 通用状态管理函数 ===

def get_tools_state(module_name: str, key: str, default: Any = None) -> Any:
    """获取工具模块状态值（通用版本）"""
    full_key = f"tools.{module_name}.{key}"
    return st.session_state.get(full_key, default)

def set_tools_state(module_name: str, key: str, value: Any) -> bool:
    """设置工具模块状态值（通用版本）"""
    try:
        full_key = f"tools.{module_name}.{key}"
        st.session_state[full_key] = value
        return True
    except Exception as e:
        logger.error(f"设置工具状态失败: {module_name}.{key}, 错误: {e}")
        return False

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

def get_all_preview_data(cache_key: Optional[str] = None) -> dict:
    """
    批量获取所有预览数据

    Args:
        cache_key: 缓存键（通常是文件名，用于缓存失效判断）

    Returns:
        dict: 包含所有预览数据的字典
            - weekly_df, monthly_df, daily_df, ten_day_df, yearly_df: 各频率的DataFrame
            - weekly_industries, monthly_industries等: 各频率的行业列表
            - clean_industry_map: 行业映射
            - source_map: 来源映射
            - indicator_industry_map: 指标到行业的映射
            - indicator_type_map: 指标到类型的映射
            - indicator_unit_map: 指标到单位的映射
            - data_loaded_files: 已加载的文件名
    """
    return {
        'weekly_df': get_preview_state('weekly_df'),
        'monthly_df': get_preview_state('monthly_df'),
        'daily_df': get_preview_state('daily_df'),
        'ten_day_df': get_preview_state('ten_day_df'),
        'yearly_df': get_preview_state('yearly_df'),
        'weekly_industries': get_preview_state('weekly_industries', []),
        'monthly_industries': get_preview_state('monthly_industries', []),
        'daily_industries': get_preview_state('daily_industries', []),
        'ten_day_industries': get_preview_state('ten_day_industries', []),
        'yearly_industries': get_preview_state('yearly_industries', []),
        'clean_industry_map': get_preview_state('clean_industry_map', {}),
        'source_map': get_preview_state('source_map', {}),
        'indicator_industry_map': get_preview_state('indicator_industry_map', {}),
        'indicator_type_map': get_preview_state('indicator_type_map', {}),
        'indicator_unit_map': get_preview_state('indicator_unit_map', {}),
        'data_loaded_files': get_preview_state('data_loaded_files'),
    }

def clear_preview_data() -> bool:
    """
    清空所有预览数据状态

    Returns:
        bool: 清理是否成功
    """
    try:
        prefix = 'tools.data_input.preview.'
        keys_to_delete = [k for k in st.session_state.keys() if str(k).startswith(prefix)]
        for k in keys_to_delete:
            del st.session_state[k]
        logger.info(f"清空预览数据: 删除了{len(keys_to_delete)}个状态键")
        return True
    except Exception as e:
        logger.error(f"清空预览数据失败: {e}")
        return False

# === 导航状态缓存管理 ===

def get_cached_navigation_state():
    """获取导航状态"""
    return {
        'main_module': st.session_state.get("navigation.main_module"),
        'sub_module': st.session_state.get("navigation.sub_module")
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
# 注：直接使用st.session_state，无需缓存清理

# === 健康检查 ===

def check_state_manager_health() -> bool:
    """检查状态管理器健康状态"""
    try:
        # 尝试简单的状态操作
        test_key = "tools.test._health_check_test"
        test_value = "test"

        # 设置测试值
        st.session_state[test_key] = test_value

        # 读取测试值
        retrieved_value = st.session_state.get(test_key)
        if retrieved_value != test_value:
            return False

        # 清理测试值
        del st.session_state[test_key]

        logger.info("状态管理器健康检查通过")
        return True

    except Exception as e:
        logger.error(f"状态管理器健康检查失败: {e}")
        return False

def detect_current_module():
    """检测当前活跃的分析模块"""
    return st.session_state.get("navigation.current_module", None)

# === 模块信息 ===

__version__ = "1.0.0"
__author__ = "UI优化团队"
__description__ = "状态管理辅助函数模块"

# 导出的公共接口
__all__ = [
    'get_tools_state',
    'set_tools_state',
    'get_property_state',
    'set_property_state',
    'get_exploration_state',
    'set_exploration_state',
    'get_dtw_state',
    'set_dtw_state',
    'get_cached_navigation_state',
    'is_in_data_exploration',
    'get_current_navigation_info',
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
    'set_preview_state',
    'get_all_preview_data',
    'clear_preview_data'
]
