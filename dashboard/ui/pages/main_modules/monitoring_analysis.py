# -*- coding: utf-8 -*-
"""
监测分析欢迎页面组件
"""

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
                from dashboard.core import get_unified_manager
                import time

                state_manager = get_unified_manager()
                if state_manager is None:
                    raise RuntimeError("统一状态管理器不可用，无法导航到工业子模块")

                current_time = time.time()
                state_manager.set_state("dashboard.last_navigation_time", current_time)
                state_manager.set_state("navigation.navigate_to_sub_module", "工业")
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
        from dashboard.core import get_unified_manager
        from dashboard.core.navigation_manager import get_navigation_manager

        unified_manager = get_unified_manager()
        if unified_manager is None:
            raise RuntimeError("统一状态管理器不可用，无法执行导航操作")

        nav_manager = get_navigation_manager(unified_manager)
        nav_manager.set_current_sub_module(sub_module)
        st_obj.rerun()

    def _handle_navigation(self, st_obj) -> None:
        """处理导航事件"""
        from dashboard.core import get_unified_manager

        state_manager = get_unified_manager()
        if state_manager is None:
            raise RuntimeError("统一状态管理器不可用，无法处理导航事件")

        sub_module = state_manager.get_state("navigation.navigate_to_sub_module")
        if sub_module:
            state_manager.clear_state("navigation.navigate_to_sub_module")
            self._navigate_to_sub_module(st_obj, sub_module)

    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return ["navigate_to_sub_module"]
