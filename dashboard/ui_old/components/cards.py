# -*- coding: utf-8 -*-
"""
卡片组件
提供各种卡片样式的UI组件
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Callable
from dashboard.ui.components.base import UIComponent

class ModuleCard(UIComponent):
    """模块卡片组件 - 使用统一状态管理"""

    def __init__(self, title: str, icon: str, description: str,
                 action_text: str = "进入", on_click: Optional[Callable] = None):
        # 调用父类初始化，集成统一状态管理
        super().__init__()

        self.title = title
        self.icon = icon
        self.description = description
        self.action_text = action_text
        self.on_click = on_click
    
    def render(self, st_obj, **kwargs) -> None:
        """渲染模块卡片"""
        container_height = kwargs.get('height', '200px')
        
        # 使用模板管理器渲染模块卡片
        from dashboard.ui.constants import TemplateManager
        card_html = TemplateManager.render_template(
            'module_card',
            height=container_height,
            icon=self.icon,
            title=self.title,
            description=self.description
        )
        
        st_obj.markdown(card_html, unsafe_allow_html=True)
        
        # 添加操作按钮
        button_key = f"card_btn_{self.title}_{id(self)}"
        if st_obj.button(self.action_text, key=button_key, use_container_width=True):
            try:
                if self.on_click:
                    self.on_click()
                else:
                    # 使用统一状态管理设置导航状态
                    self.set_state(f'navigate_to_{self.title}', True)
                    st_obj.rerun()
            except Exception as e:
                self.handle_error(st_obj, e, "处理卡片点击事件")
    
    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return [f'navigate_to_{self.title}']

class FeatureCard(UIComponent):
    """功能卡片组件 - 使用统一状态管理"""

    def __init__(self, name: str, icon: str, description: str,
                 status: str = "available", badge: Optional[str] = None):
        # 调用父类初始化，集成统一状态管理
        super().__init__()

        self.name = name
        self.icon = icon
        self.description = description
        self.status = status  # available, coming_soon, beta
        self.badge = badge
    
    def render(self, st_obj, **kwargs) -> None:
        """渲染功能卡片"""
        container_height = kwargs.get('height', '180px')
        clickable = kwargs.get('clickable', True)
        
        # 根据状态设置样式
        if self.status == "available":
            border_color = "#f0f0f0"
            bg_color = "#fafafa"
            text_color = "#333"
            opacity = "1"
        elif self.status == "coming_soon":
            border_color = "#e0e0e0"
            bg_color = "#f5f5f5"
            text_color = "#999"
            opacity = "0.7"
        elif self.status == "beta":
            border_color = "#ffd700"
            bg_color = "#fffbf0"
            text_color = "#333"
            opacity = "1"
        else:
            border_color = "#f0f0f0"
            bg_color = "#fafafa"
            text_color = "#333"
            opacity = "1"
        
        # 徽章HTML
        badge_html = ""
        if self.badge:
            badge_html = f"""
            <div style="
                position: absolute;
                top: 10px;
                right: 10px;
                background: #ff4b4b;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 0.7em;
                font-weight: bold;
            ">{self.badge}</div>
            """
        
        # 卡片HTML
        cursor_style = "cursor: pointer;" if clickable and self.status == "available" else "cursor: default;"
        hover_effect = ""
        if clickable and self.status == "available":
            hover_effect = """
            onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.15)';"
            onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)';"
            """
        
        card_html = f"""
        <div style="
            position: relative;
            border: 1px solid {border_color};
            border-radius: 6px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            background: {bg_color};
            text-align: center;
            height: {container_height};
            display: flex;
            flex-direction: column;
            justify-content: center;
            opacity: {opacity};
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            {cursor_style}
        " {hover_effect}>
            {badge_html}
            <div style="font-size: 2.5em; margin-bottom: 0.5rem;">{self.icon}</div>
            <h4 style="margin: 0.5rem 0; color: {text_color}; font-weight: 600;">{self.name}</h4>
            <p style="color: {text_color}; font-size: 0.85em; margin: 0; line-height: 1.4; opacity: 0.8;">{self.description}</p>
        </div>
        """
        
        st_obj.markdown(card_html, unsafe_allow_html=True)
        
        # 状态提示
        if self.status == "coming_soon":
            st_obj.caption("即将推出")
        elif self.status == "beta":
            st_obj.caption("测试版本")
    
    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return []

class StatCard(UIComponent):
    """统计卡片组件 - 使用统一状态管理"""

    def __init__(self, title: str, value: str, icon: str,
                 trend: Optional[str] = None, color: str = "blue"):
        # 调用父类初始化，集成统一状态管理
        super().__init__()

        self.title = title
        self.value = value
        self.icon = icon
        self.trend = trend
        self.color = color
    
    def render(self, st_obj, **kwargs) -> None:
        """渲染统计卡片"""
        # 颜色映射
        color_map = {
            "blue": {"bg": "#e3f2fd", "text": "#1976d2"},
            "green": {"bg": "#e8f5e8", "text": "#388e3c"},
            "red": {"bg": "#ffebee", "text": "#d32f2f"},
            "orange": {"bg": "#fff3e0", "text": "#f57c00"},
            "purple": {"bg": "#f3e5f5", "text": "#7b1fa2"}
        }
        
        colors = color_map.get(self.color, color_map["blue"])
        
        # 趋势指示器
        trend_html = ""
        if self.trend:
            trend_color = "#4caf50" if "↑" in self.trend else "#f44336" if "↓" in self.trend else "#666"
            trend_html = f"""
            <div style="
                font-size: 0.8em;
                color: {trend_color};
                margin-top: 0.5rem;
                font-weight: 500;
            ">{self.trend}</div>
            """
        
        card_html = f"""
        <div style="
            background: {colors['bg']};
            border-radius: 8px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            text-align: center;
            border-left: 4px solid {colors['text']};
        ">
            <div style="font-size: 2em; margin-bottom: 0.5rem;">{self.icon}</div>
            <h3 style="margin: 0; color: {colors['text']}; font-size: 2em; font-weight: bold;">{self.value}</h3>
            <p style="margin: 0.5rem 0 0 0; color: {colors['text']}; font-weight: 500;">{self.title}</p>
            {trend_html}
        </div>
        """
        
        st_obj.markdown(card_html, unsafe_allow_html=True)
    
    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return []

class InfoCard(UIComponent):
    """信息卡片组件 - 使用统一状态管理"""

    def __init__(self, title: str, content: str, card_type: str = "info"):
        # 调用父类初始化，集成统一状态管理
        super().__init__()

        self.title = title
        self.content = content
        self.card_type = card_type  # info, warning, success, error
    
    def render(self, st_obj, **kwargs) -> None:
        """渲染信息卡片"""
        # 类型样式映射
        type_styles = {
            "info": {"bg": "#e3f2fd", "border": "#2196f3", "icon": "ℹ️"},
            "warning": {"bg": "#fff3e0", "border": "#ff9800", "icon": "⚠️"},
            "success": {"bg": "#e8f5e8", "border": "#4caf50", "icon": "✅"},
            "error": {"bg": "#ffebee", "border": "#f44336", "icon": "❌"}
        }
        
        style = type_styles.get(self.card_type, type_styles["info"])
        
        card_html = f"""
        <div style="
            background: {style['bg']};
            border: 1px solid {style['border']};
            border-radius: 6px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid {style['border']};
        ">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.2em; margin-right: 0.5rem;">{style['icon']}</span>
                <h4 style="margin: 0; color: {style['border']};">{self.title}</h4>
            </div>
            <p style="margin: 0; color: #333; line-height: 1.5;">{self.content}</p>
        </div>
        """
        
        st_obj.markdown(card_html, unsafe_allow_html=True)
    
    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return []
