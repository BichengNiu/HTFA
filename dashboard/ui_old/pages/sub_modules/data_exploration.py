# -*- coding: utf-8 -*-
"""
数据探索子模块欢迎页面
简化版本：移除复杂的状态管理，仅保留基本的页面渲染功能
"""

import streamlit as st
from typing import List
from dashboard.ui.components.base import UIComponent
from dashboard.ui.constants import UIConstants

class DataExplorationWelcomePage(UIComponent):
    """数据探索欢迎页面 - 简化版本"""
    
    def __init__(self):
        super().__init__("DataExplorationWelcomePage")
        self.constants = UIConstants
        self.sub_module_config = self.constants.SUB_MODULES["数据探索"]
    
    def render(self, st_obj, **kwargs) -> None:
        """渲染数据探索页面 - 简化版本：仅显示欢迎信息"""
        # 显示简化的欢迎页面
        self._render_header(st_obj)
        
        # 显示提示信息，引导用户使用简化的标签页界面
        st_obj.info("数据探索功能已简化！请通过侧边栏导航直接访问三个分析模块的标签页界面。")

    def _render_header(self, st_obj):
        """渲染页面头部"""
        st_obj.markdown(f"""
        <div style="text-align: center; padding: 2rem 0;">
            <div style="font-size: 4em; margin-bottom: 1rem;">{self.sub_module_config['icon']}</div>
            <h1 style="color: #333; margin-bottom: 1rem; font-weight: 700;">数据探索</h1>
            <p style="font-size: 1.2em; color: #666; max-width: 700px; margin: 0 auto; line-height: 1.6;">
                {self.sub_module_config['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # 添加分隔线
        st_obj.markdown("---")
    
    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return []
