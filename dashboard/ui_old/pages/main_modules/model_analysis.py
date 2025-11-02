# -*- coding: utf-8 -*-
"""
模型分析欢迎页面组件
"""

import streamlit as st
from typing import List
from dashboard.ui.components.base import UIComponent
from dashboard.ui.constants import UIConstants


class ModelAnalysisWelcomePage(UIComponent):
    """模型分析欢迎页面"""
    
    def __init__(self):
        self.constants = UIConstants
        self.module_config = self.constants.MAIN_MODULES["模型分析"]
    
    def render(self, st_obj, **kwargs) -> None:
        """渲染模型分析欢迎页面"""
        # 显示标题和介绍
        self._render_header(st_obj)

        # 渲染子模块选择卡片
        self._render_sub_modules(st_obj)

        # 处理导航事件
        self._handle_navigation(st_obj)
    
    def _render_header(self, st_obj):
        """渲染页面头部"""
        st_obj.markdown(f"""
        <div style="text-align: center; padding: 3rem 0 2rem 0;">
            <div style="font-size: 4em; margin-bottom: 1rem;">{self.module_config['icon']}</div>
            <h1 style="color: #333; margin-bottom: 1rem; font-weight: 700;">模型分析</h1>
            <p style="font-size: 1.3em; color: #666; max-width: 600px; margin: 0 auto; line-height: 1.6;">
                {self.module_config['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    def _render_sub_modules(self, st_obj):
        """渲染子模块选择卡片"""
        st_obj.markdown("### 选择模型类型")

        # 创建单列布局（目前只有DFM模型）
        col1, col2, col3 = st_obj.columns([1, 2, 1])

        with col2:
            # DFM模型卡片
            if st_obj.button(
                "DFM 模型",
                key="nav_model_analysis_dfm",
                use_container_width=True,
                help="使用动态因子模型进行经济预测和分析"
            ):
                st.session_state["navigation.navigate_to_sub_module"] = 'DFM 模型'
                st_obj.rerun()

        # 添加一些说明文字
        st_obj.markdown("""
        <div style="margin-top: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
            <h4 style="color: #495057; margin-bottom: 0.5rem;">[INFO] 使用提示</h4>
            <ul style="color: #6c757d; margin-bottom: 0;">
                <li><strong>DFM 模型</strong>：动态因子模型，包含数据准备、模型训练、模型分析和新闻分析功能</li>
                <li><strong>数据准备</strong>：处理和准备用于模型训练的时间序列数据</li>
                <li><strong>模型训练</strong>：配置参数并训练DFM模型</li>
                <li><strong>模型分析</strong>：分析已训练模型的性能和结果</li>
                <li><strong>新闻分析</strong>：结合新闻数据进行综合分析</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    def _navigate_to_sub_module(self, st_obj, sub_module: str):
        """导航到子模块"""
        try:
            import streamlit as st
            from core.navigation_manager import get_navigation_manager

            # 使用st.session_state获取导航管理器
            nav_manager = get_navigation_manager(st.session_state)

            # 设置子模块
            nav_manager.set_current_sub_module(sub_module)

            # 强制重新渲染
            st_obj.rerun()

        except Exception as e:
            st_obj.error(f"导航失败: {e}")

    def _handle_navigation(self, st_obj):
        """处理导航事件"""
        sub_module = st.session_state.get("navigation.navigate_to_sub_module")
        if sub_module:
            if "navigation.navigate_to_sub_module" in st.session_state:
                del st.session_state["navigation.navigate_to_sub_module"]
            self._navigate_to_sub_module(st_obj, sub_module)
    
    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return ['navigate_to_sub_module']
