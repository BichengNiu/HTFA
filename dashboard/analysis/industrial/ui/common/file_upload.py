# -*- coding: utf-8 -*-
"""
工业分析文件上传组件
"""

import streamlit as st
from typing import Optional
from dashboard.core.ui.components.base import UIComponent
from dashboard.core.ui.components.data_input import UnifiedDataUploadComponent


class IndustrialFileUploadComponent(UIComponent):
    """
    工业分析文件上传组件

    完全基于UnifiedDataUploadComponent实现。
    """

    def __init__(self):
        super().__init__()
        self.upload_component = UnifiedDataUploadComponent(
            accepted_types=['xlsx', 'xls'],
            help_text="请上传包含'宏观运行'和相关工作表的Excel文件",
            show_data_source_selector=False,
            show_staging_data_option=False,
            component_id="industrial_unified_upload",
            return_file_object=True
        )

    def render(self, st_obj, **kwargs) -> Optional[object]:
        """
        渲染文件上传功能

        Returns:
            uploaded_file: 上传的文件对象，如果没有上传则返回None
        """
        with st_obj.sidebar:
            uploaded_file = self.upload_component.render_file_upload_section(
                st_obj,
                upload_key="industrial_unified_file_uploader",
                show_overview=False,
                show_preview=False
            )

            if uploaded_file is None:
                st_obj.info("请上传Excel数据文件以开始工业分析")

        return uploaded_file

    def get_state_keys(self) -> list:
        """获取组件相关的状态键"""
        return self.upload_component.get_state_keys()
