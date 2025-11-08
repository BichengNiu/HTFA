# -*- coding: utf-8 -*-
"""
单变量分析页面
包含平稳性分析功能
"""

import streamlit as st
import pandas as pd
import logging
from typing import Optional

from dashboard.explore.ui.stationarity import StationarityAnalysisComponent
from dashboard.core.ui.components.data_input import UnifiedDataUploadComponent

logger = logging.getLogger(__name__)


def render_univariate_analysis_page():
    """渲染单变量分析页面"""
    # 创建平稳性分析标签页
    tab = st.tabs(["平稳性分析"])[0]

    with tab:
        # Tab内数据上传
        upload_component = UnifiedDataUploadComponent(
            accepted_types=['csv', 'xlsx', 'xls'],
            help_text="上传CSV或Excel文件进行平稳性分析",
            show_data_source_selector=False,
            show_staging_data_option=False,
            component_id="univariate_stationarity_upload"
        )

        data = upload_component.render_file_upload_section(
            st,
            upload_key="stationarity_tab_upload",
            show_overview=False,
            show_preview=False
        )

        if data is not None:
            file_name = upload_component.get_state('file_name')
            st.session_state['exploration.stationarity.upload_data'] = data
            st.session_state['exploration.stationarity.file_name'] = file_name
            st.info(f"数据已加载: {file_name} ({data.shape[0]} 行 x {data.shape[1]} 列)")

        # 渲染分析组件
        stationarity_component = StationarityAnalysisComponent()
        stationarity_component.render(st, tab_index=0)
