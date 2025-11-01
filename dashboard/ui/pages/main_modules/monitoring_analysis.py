# -*- coding: utf-8 -*-
"""
监测分析欢迎页面组件
"""

import streamlit as st
from typing import List

from dashboard.ui.components.base import UIComponent
from dashboard.ui.constants import UIConstants


class MonitoringAnalysisWelcomePage(UIComponent):
    """监测分析欢迎页面"""

    def __init__(self):
        self.constants = UIConstants
        self.module_config = self.constants.MAIN_MODULES["监测分析"]

    def render(self, st_obj, **kwargs) -> None:
        """渲染监测分析欢迎页面"""
        self._render_header(st_obj)

    def _render_header(self, st_obj) -> None:
        """渲染页面头部"""
        st_obj.markdown(
            f"""
        <div style="text-align: center; padding: 3rem 0 2rem 0;">
            <div style="font-size: 4em; margin-bottom: 1rem;">{self.module_config['icon']}</div>
            <h1 style="color: #333; margin-bottom: 1rem; font-weight: 700;">监测分析</h1>
            <p style="font-size: 1.3em; color: #666; max-width: 600px; margin: 0 auto; line-height: 1.6;">
                {self.module_config['description']}
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def _render_sub_modules(self, st_obj) -> None:
        """渲染子模块选择卡片"""
        st_obj.markdown("### 选择分析领域")

        col1, col2 = st_obj.columns(2)

        with col1:
            if st_obj.button(
                "工业",
                key="nav_monitoring_analysis_industrial",
                use_container_width=True,
                help="进行工业增加值和工业企业利润拆解分析",
            ):
                import time
                current_time = time.time()
                st.session_state["dashboard.last_navigation_time"] = current_time
                st.session_state["navigation.navigate_to_sub_module"] = "工业"
                st_obj.rerun()

        st_obj.markdown(
            """
        <div style="margin-top: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
            <h4 style="color: #495057; margin-bottom: 0.5rem;">使用提示</h4>
            <ul style="color: #6c757d; margin-bottom: 0;">
                <li><strong>工业</strong>：提供工业增加值分析和工业企业利润拆解功能</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def _navigate_to_sub_module(self, st_obj, sub_module: str) -> None:
        """导航到子模块"""
        try:
            import streamlit as st
            from dashboard.core.navigation_manager import get_navigation_manager

            # 使用st.session_state获取导航管理器
            nav_manager = get_navigation_manager(st.session_state)
            nav_manager.set_current_sub_module(sub_module)
            st_obj.rerun()
        except Exception as e:
            st_obj.error(f"导航失败: {e}")

    def _handle_navigation(self, st_obj) -> None:
        """处理导航事件"""
        try:
            import streamlit as st

            sub_module = st.session_state.get("navigation.navigate_to_sub_module")
            if sub_module:
                if "navigation.navigate_to_sub_module" in st.session_state:
                    del st.session_state["navigation.navigate_to_sub_module"]
                self._navigate_to_sub_module(st_obj, sub_module)
        except Exception as e:
            st_obj.error(f"处理导航事件失败: {e}")

    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return ["navigate_to_sub_module"]
