# -*- coding: utf-8 -*-
"""
数据探索侧边栏组件
"""

import streamlit as st
import pandas as pd
import logging
from typing import List, Optional
from dashboard.core.ui.components.sidebar.base import SidebarComponent

logger = logging.getLogger(__name__)


class DataExplorationSidebar(SidebarComponent):
    """
    数据探索专用侧边栏组件

    完全基于UnifiedDataUploadComponent实现。
    """

    def __init__(self,
                 title: str = "数据探索 - 数据上传",
                 accepted_types: List[str] = None,
                 help_text: str = None):
        super().__init__()
        self.title = title
        self.accepted_types = accepted_types or ['csv', 'xlsx', 'xls']
        self.help_text = help_text or "上传CSV或Excel文件进行时间序列数据探索分析"

        # 使用UnifiedDataUploadComponent
        from dashboard.core.ui.components.data_input import UnifiedDataUploadComponent
        self.upload_component = UnifiedDataUploadComponent(
            accepted_types=self.accepted_types,
            help_text=self.help_text,
            show_data_source_selector=False,
            show_staging_data_option=False,
            component_id="data_exploration_upload"
        )

    def render(self, st_obj, **kwargs) -> Optional[pd.DataFrame]:
        """渲染数据探索专用侧边栏"""
        with st_obj.sidebar:
            st_obj.markdown(f"### {self.title}")

            # 使用UnifiedDataUploadComponent渲染
            data = self.upload_component.render_file_upload_section(
                st_obj,
                upload_key="data_exploration_sidebar_upload",
                show_overview=False,
                show_preview=False
            )

            if data is not None:
                # 获取文件名
                file_name = self.upload_component.get_state('file_name')

                # 手动调用数据存储逻辑
                if file_name:
                    self.store_exploration_data(data, file_name)

                st_obj.info(f"数据形状: {data.shape[0]} 行 x {data.shape[1]} 列")

            return data

    def store_exploration_data(self, data: pd.DataFrame, file_name: str):
        """存储数据到session_state - 确保所有分析模块都能访问"""
        current_data_hash = hash(str(data.shape) + file_name)
        last_stored_hash = getattr(self, '_last_stored_data_hash', None)

        if current_data_hash != last_stored_hash:
            from dashboard.core.ui.constants import UIConstants
            for module in UIConstants.EXPLORATION_ANALYSIS_MODULES:
                st.session_state[f"exploration.{module}.data"] = data
                st.session_state[f"exploration.{module}.file_name"] = file_name
                st.session_state[f"exploration.{module}.data_source"] = 'upload'

            logger.info(f"数据已存储到所有分析模块: {data.shape}")

            self._last_stored_data_hash = current_data_hash

    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return ['data_exploration_sidebar_upload']
