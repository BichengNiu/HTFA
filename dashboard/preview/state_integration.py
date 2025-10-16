# -*- coding: utf-8 -*-
"""
Preview模块状态管理集成
提供preview模块与统一状态管理系统的简化接口
"""

import pandas as pd
import streamlit as st
from typing import Any, Optional, Dict
import logging

# 导入统一状态管理系统
from dashboard.core import get_unified_manager

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False, ttl=1800)
def get_all_preview_data(cache_key: str = None) -> Dict[str, Any]:
    """一次性获取所有预览数据，减少重复查询

    使用基于文件名的缓存机制，文件变化时自动失效

    Args:
        cache_key: 缓存键（参与哈希计算，文件变化时缓存自动失效）

    Returns:
        包含所有常用预览数据的字典
    """
    logger.debug(f"[STATE] 开始获取所有预览数据（缓存键: {cache_key}）")

    # 直接从统一状态管理器读取
    manager = get_unified_manager()

    data = {
        # 数据框
        'weekly_df': manager.get_namespaced('preview', 'weekly_df', pd.DataFrame()),
        'monthly_df': manager.get_namespaced('preview', 'monthly_df', pd.DataFrame()),
        'daily_df': manager.get_namespaced('preview', 'daily_df', pd.DataFrame()),
        'ten_day_df': manager.get_namespaced('preview', 'ten_day_df', pd.DataFrame()),
        'yearly_df': manager.get_namespaced('preview', 'yearly_df', pd.DataFrame()),

        # 行业列表
        'weekly_industries': manager.get_namespaced('preview', 'weekly_industries', []),
        'monthly_industries': manager.get_namespaced('preview', 'monthly_industries', []),
        'daily_industries': manager.get_namespaced('preview', 'daily_industries', []),
        'ten_day_industries': manager.get_namespaced('preview', 'ten_day_industries', []),
        'yearly_industries': manager.get_namespaced('preview', 'yearly_industries', []),

        # 映射关系
        'source_map': manager.get_namespaced('preview', 'source_map', {}),
        'indicator_industry_map': manager.get_namespaced('preview', 'indicator_industry_map', {}),
        'indicator_unit_map': manager.get_namespaced('preview', 'indicator_unit_map', {}),
        'indicator_type_map': manager.get_namespaced('preview', 'indicator_type_map', {}),
        'clean_industry_map': manager.get_namespaced('preview', 'clean_industry_map', {}),

        # 元数据
        'data_loaded_files': manager.get_namespaced('preview', 'data_loaded_files'),
        'file_hash': manager.get_namespaced('preview', 'file_hash'),
        'data_processing_time': manager.get_namespaced('preview', 'data_processing_time')
    }

    logger.debug(f"[STATE] 获取数据完成 - 已加载文件: {data.get('data_loaded_files')}")

    return data


# 便捷函数：直接使用unified_manager，无需中间层
def get_preview_state(key: str, default: Any = None) -> Any:
    """获取预览状态值

    Args:
        key: 状态键名
        default: 默认值

    Returns:
        状态值或默认值
    """
    manager = get_unified_manager()
    return manager.get_namespaced('preview', key, default)


def set_preview_state(key: str, value: Any) -> bool:
    """设置预览状态值

    Args:
        key: 状态键名
        value: 状态值

    Returns:
        设置是否成功
    """
    manager = get_unified_manager()
    return manager.set_namespaced('preview', key, value)


def clear_preview_data() -> bool:
    """清理预览数据

    Returns:
        清理是否成功
    """
    manager = get_unified_manager()
    return manager.clear_namespace('preview')
