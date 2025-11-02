# -*- coding: utf-8 -*-
"""
样式管理器
统一管理UI组件的样式
"""

import streamlit as st
from typing import Dict, Any, Optional

class StyleManager:
    """样式管理器"""
    
    def __init__(self):
        self._custom_css = ""
        self._theme = "light"
    
    def set_theme(self, theme: str):
        """设置主题"""
        self._theme = theme
    
    def get_theme(self) -> str:
        """获取当前主题"""
        return self._theme
    
    def add_custom_css(self, css: str):
        """添加自定义CSS"""
        self._custom_css += css + "\n"
    
    def apply_styles(self, st_obj):
        """应用样式到Streamlit"""
        if self._custom_css:
            st_obj.markdown(f"<style>{self._custom_css}</style>", unsafe_allow_html=True)
    
    def get_card_style(self, card_type: str = "default") -> str:
        """获取卡片样式"""
        base_style = """
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        """
        
        if card_type == "primary":
            return base_style + """
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
            """
        elif card_type == "success":
            return base_style + """
                background: #e8f5e8;
                border: 1px solid #4caf50;
                color: #2e7d32;
            """
        elif card_type == "warning":
            return base_style + """
                background: #fff3e0;
                border: 1px solid #ff9800;
                color: #ef6c00;
            """
        elif card_type == "error":
            return base_style + """
                background: #ffebee;
                border: 1px solid #f44336;
                color: #c62828;
            """
        else:
            return base_style + """
                background: white;
                border: 1px solid #e0e0e0;
                color: #333;
            """
    
    def get_button_style(self, button_type: str = "default") -> str:
        """获取按钮样式"""
        base_style = """
            border-radius: 6px;
            padding: 0.5rem 1rem;
            border: none;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s ease;
        """
        
        if button_type == "primary":
            return base_style + """
                background: #2196f3;
                color: white;
            """
        elif button_type == "secondary":
            return base_style + """
                background: #f5f5f5;
                color: #333;
                border: 1px solid #ddd;
            """
        elif button_type == "success":
            return base_style + """
                background: #4caf50;
                color: white;
            """
        elif button_type == "danger":
            return base_style + """
                background: #f44336;
                color: white;
            """
        else:
            return base_style + """
                background: #e0e0e0;
                color: #333;
            """
    
    def get_container_style(self, container_type: str = "default") -> str:
        """获取容器样式"""
        if container_type == "welcome":
            return """
                text-align: center;
                padding: 3rem 2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
                margin: 2rem 0;
                color: white;
            """
        elif container_type == "section":
            return """
                padding: 2rem;
                margin: 1rem 0;
                background: #fafafa;
                border-radius: 8px;
                border-left: 4px solid #2196f3;
            """
        elif container_type == "highlight":
            return """
                padding: 1.5rem;
                margin: 1rem 0;
                background: #fff3e0;
                border-radius: 6px;
                border: 1px solid #ff9800;
            """
        else:
            return """
                padding: 1rem;
                margin: 0.5rem 0;
            """
    
    def create_gradient_background(self, color1: str, color2: str) -> str:
        """创建渐变背景"""
        return f"background: linear-gradient(135deg, {color1} 0%, {color2} 100%);"
    
    def create_hover_effect(self, transform: str = "translateY(-2px)", 
                          shadow: str = "0 4px 8px rgba(0,0,0,0.15)") -> str:
        """创建悬停效果"""
        return f"""
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        """ + f"""
            :hover {{
                transform: {transform};
                box-shadow: {shadow};
            }}
        """

# 统一按钮样式定义
def render_action_button(st_obj, label: str, key: str, button_type: str = "primary", 
                        disabled: bool = False, help: str = None, **kwargs):
    """
    渲染统一样式的操作按钮（如：开始分析、下载数据等）
    
    Args:
        st_obj: Streamlit对象
        label: 按钮文本
        key: 按钮唯一标识
        button_type: 按钮类型 ("primary", "secondary", None)
        disabled: 是否禁用
        help: 帮助提示文本
        **kwargs: 其他参数
    
    Returns:
        bool: 是否被点击
    """
    return st_obj.button(
        label=label,
        key=key,
        type=button_type,
        disabled=disabled,
        help=help,
        **kwargs
    )

def render_download_button(st_obj, label: str, data, file_name: str, 
                          mime: str = "text/csv", key: str = None, **kwargs):
    """
    渲染统一样式的下载按钮（与操作按钮样式一致）
    
    Args:
        st_obj: Streamlit对象
        label: 按钮文本
        data: 下载数据
        file_name: 文件名
        mime: MIME类型
        key: 按钮唯一标识
        **kwargs: 其他参数
    
    Returns:
        bool: 是否被点击
    """
    return st_obj.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        key=key,
        type="primary",  # 添加primary类型，与操作按钮保持一致
        **kwargs
    )

# 全局样式管理器实例
_style_manager = None

def get_style_manager() -> StyleManager:
    """获取全局样式管理器实例"""
    global _style_manager
    if _style_manager is None:
        _style_manager = StyleManager()
    return _style_manager

def apply_global_styles(st_obj):
    """应用全局样式"""
    style_manager = get_style_manager()
    
    # 基础样式
    global_css = """
    /* 全局样式 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: 600;
    }
    
    /* 卡片悬停效果 */
    .hover-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* 按钮样式 */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        padding-top: 1rem;
    }
    """
    
    style_manager.add_custom_css(global_css)
    style_manager.apply_styles(st_obj)
