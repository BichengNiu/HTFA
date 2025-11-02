# -*- coding: utf-8 -*-
"""
数据预览欢迎页面组件
"""

from typing import List
from dashboard.ui.components.base import UIComponent
from dashboard.ui.constants import UIConstants


class DataPreviewWelcomePage(UIComponent):
    """数据预览欢迎页面"""
    
    def __init__(self):
        self.constants = UIConstants
        self.module_config = self.constants.MAIN_MODULES["数据预览"]
    
    def render(self, st_obj, **kwargs) -> None:
        """渲染数据预览页面 - 直接显示工业数据预览"""
        # 注意：实际的数据预览渲染由 content_router.py 的 render_data_preview_content() 处理
        # 这里不再重复调用 display_industrial_tabs，避免重复显示问题
        pass

    def _render_industrial_data_preview(self, st_obj):
        """直接渲染工业数据预览功能（已废弃，由 content_router 统一处理）"""
        # 这个方法已废弃，实际渲染由 content_router.py 处理
        pass

    def _render_welcome_message(self, st_obj):
        """渲染简洁的欢迎信息 (已废弃，保留为备用)"""
        st_obj.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <h1 style="font-size: 3rem; font-weight: bold; margin-bottom: 1rem;">欢迎使用数据预览模块</h1>
            <hr style="width: 50%; margin: 2rem auto; border: 2px solid #ddd;">
            <p style="font-size: 1.2rem; margin-top: 2rem;">数据预览功能已直接加载</p>
        </div>
        """, unsafe_allow_html=True)

    
    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return []


class UniversalWelcomePage(UIComponent):
    """通用欢迎页面，适用于所有主模块"""
    
    def __init__(self, module_name: str):
        self.constants = UIConstants
        self.module_name = module_name
    
    def render(self, st_obj, **kwargs) -> None:
        """渲染通用欢迎页面"""
        self._render_welcome_message(st_obj)
    
    def _render_welcome_message(self, st_obj):
        """渲染简洁的欢迎信息"""
        st_obj.markdown(f"""
        <div style="text-align: center; padding: 3rem 0;">
            <h1 style="font-size: 3rem; font-weight: bold; margin-bottom: 1rem;">欢迎使用{self.module_name}</h1>
            <hr style="width: 50%; margin: 2rem auto; border: 2px solid #ddd;">
            <p style="font-size: 1.2rem; margin-top: 2rem;">请在侧边栏选择子模块</p>
        </div>
        """, unsafe_allow_html=True)
    
    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return []
