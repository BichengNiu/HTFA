"""
Industrial Analysis Unified Module
工业分析统一模块 - 提供统一的文件上传和数据共享功能

This module provides:
- Unified file upload functionality
- Data caching and sharing between macro and enterprise modules
- State management integration
- Tab-based interface for macro operations and enterprise operations
"""

import streamlit as st
import pandas as pd
from typing import Optional, Tuple
import logging

# 设置日志
logger = logging.getLogger(__name__)

# 导入统一的数据加载函数
from dashboard.analysis.industrial.utils import load_macro_data, load_weights_data

# 导入新的UI组件
from dashboard.analysis.industrial.ui import (
    IndustrialFileUploadComponent
)
# 导入统一状态管理
from dashboard.analysis.industrial.utils import IndustrialStateManager
from dashboard.analysis.industrial.constants import (
    STATE_NAMESPACE_INDUSTRIAL,
    STATE_KEY_UPLOADED_FILE,
    STATE_KEY_MACRO_TIME_RANGE_CHART1,
    STATE_KEY_MACRO_TIME_RANGE_CHART2,
    STATE_KEY_MACRO_TIME_RANGE_CHART3,
    STATE_KEY_ENTERPRISE_TIME_RANGE_CHART1,
    STATE_KEY_ENTERPRISE_TIME_RANGE_CHART2,
    STATE_KEY_ENTERPRISE_TIME_RANGE_CHART3,
    DEFAULT_TIME_RANGE
)


def initialize_industrial_states():
    """
    初始化工业分析模块的时间范围状态
    """
    # 初始化宏观分析图表的时间范围状态
    for key in [STATE_KEY_MACRO_TIME_RANGE_CHART1,
                STATE_KEY_MACRO_TIME_RANGE_CHART2,
                STATE_KEY_MACRO_TIME_RANGE_CHART3]:
        if IndustrialStateManager.get(key) is None:
            IndustrialStateManager.set(key, DEFAULT_TIME_RANGE)

    # 初始化企业分析图表的时间范围状态
    for key in [STATE_KEY_ENTERPRISE_TIME_RANGE_CHART1,
                STATE_KEY_ENTERPRISE_TIME_RANGE_CHART2,
                STATE_KEY_ENTERPRISE_TIME_RANGE_CHART3]:
        if IndustrialStateManager.get(key) is None:
            IndustrialStateManager.set(key, DEFAULT_TIME_RANGE)


def render_unified_file_upload(st_obj) -> Optional[object]:
    """
    渲染统一的文件上传功能 - 使用新的UI组件

    Returns:
        uploaded_file: 上传的文件对象，如果没有上传则返回None
    """
    # 使用新的UI组件
    file_upload_component = IndustrialFileUploadComponent()
    return file_upload_component.render(st_obj)


def load_and_cache_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    加载数据（缓存由data_loader模块的@st.cache_data装饰器自动处理）

    Returns:
        (df_macro, df_weights): 宏观数据和权重数据的元组
    """
    if uploaded_file is None:
        return None, None

    # 直接加载数据，缓存由@st.cache_data装饰器自动处理
    # 不需要手动缓存，避免双重缓存
    with st.spinner("正在处理数据..."):
        df_macro = load_macro_data(uploaded_file)
        df_weights = load_weights_data()  # 从内部CSV文件读取

    # 错误提示
    if df_macro is None:
        st.error("分行业工业增加值同比增速数据加载失败，请检查文件是否包含'分行业工业增加值同比增速'工作表")
    if df_weights is None:
        st.error("内部权重数据加载失败，请联系管理员检查配置文件")

    return df_macro, df_weights


def render_macro_operations_with_data(st_obj, df_macro: Optional[pd.DataFrame], df_weights: Optional[pd.DataFrame], uploaded_file=None):
    """
    使用共享数据渲染分行业工业增加值同比增速分析
    """
    if df_macro is not None and df_weights is not None:
        # Store uploaded file using unified state management
        if uploaded_file is not None:
            st.session_state[f'{STATE_NAMESPACE_INDUSTRIAL}.{STATE_KEY_UPLOADED_FILE}'] = uploaded_file

        # 调用宏观运行分析，传入数据
        from dashboard.analysis.industrial.macro_analysis import render_macro_operations_analysis_with_data
        render_macro_operations_analysis_with_data(st_obj, df_macro, df_weights)
    else:
        st_obj.info("请先上传Excel数据文件以开始分行业工业增加值同比增速分析")


def render_enterprise_operations_with_data(st_obj, df_macro: Optional[pd.DataFrame], df_weights: Optional[pd.DataFrame], uploaded_file=None):
    """
    使用共享数据渲染企业经营分析
    """
    # 调用企业经营分析，传入数据和上传的文件
    from dashboard.analysis.industrial.enterprise_analysis import render_enterprise_operations_analysis_with_data
    render_enterprise_operations_analysis_with_data(st_obj, df_macro, df_weights, uploaded_file)


def render_enterprise_profit_with_data(st_obj, df_macro: Optional[pd.DataFrame], df_weights: Optional[pd.DataFrame], uploaded_file=None):
    """
    使用共享数据渲染工业企业利润分析
    """
    from dashboard.analysis.industrial.enterprise_analysis import render_enterprise_profit_analysis_with_data
    render_enterprise_profit_analysis_with_data(st_obj, df_macro, df_weights, uploaded_file)


def render_enterprise_efficiency_with_data(st_obj, df_macro: Optional[pd.DataFrame], df_weights: Optional[pd.DataFrame], uploaded_file=None):
    """
    使用共享数据渲染工业企业经营效率分析
    """
    from dashboard.analysis.industrial.enterprise_analysis import render_enterprise_efficiency_analysis_with_data
    render_enterprise_efficiency_analysis_with_data(st_obj, df_macro, df_weights, uploaded_file)


def render_industrial_analysis(st_obj):
    """
    渲染统一的工业分析模块

    Args:
        st_obj: Streamlit对象
    """
    # 1. 统一文件上传（在侧边栏显示）
    uploaded_file = render_unified_file_upload(st_obj)

    # 2. 加载和缓存数据
    df_macro, df_weights = load_and_cache_data(uploaded_file)

    # 3. 始终创建标签页（右侧主区域）
    tab_macro, tab_profit, tab_efficiency = st_obj.tabs([
        "工业增加值分析",
        "工业企业利润分析",
        "工业企业经营效率分析"
    ])

    with tab_macro:
        render_macro_operations_with_data(st_obj, df_macro, df_weights, uploaded_file)

    with tab_profit:
        render_enterprise_profit_with_data(st_obj, df_macro, df_weights, uploaded_file)

    with tab_efficiency:
        render_enterprise_efficiency_with_data(st_obj, df_macro, df_weights, uploaded_file)
