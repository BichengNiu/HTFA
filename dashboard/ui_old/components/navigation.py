# -*- coding: utf-8 -*-
"""
导航组件
提供导航相关的UI组件
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from dashboard.ui.components.base import UIComponent

class NavigationComponent(UIComponent):
    """导航组件基类"""
    
    def __init__(self):
        pass
    
    def render(self, st_obj, **kwargs) -> None:
        """渲染导航组件"""
        pass
    
    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return []

class Breadcrumb(NavigationComponent):
    """面包屑导航组件"""
    
    def __init__(self, items: List[Dict[str, str]]):
        super().__init__()
        self.items = items
    
    def render(self, st_obj, **kwargs) -> None:
        """渲染面包屑导航"""
        if not self.items:
            return
        
        breadcrumb_html = "<div style='margin-bottom: 1rem; color: #666;'>"
        
        for i, item in enumerate(self.items):
            if i > 0:
                breadcrumb_html += " > "
            
            if item.get('active', False):
                breadcrumb_html += f"<strong>{item['name']}</strong>"
            else:
                breadcrumb_html += item['name']
        
        breadcrumb_html += "</div>"
        
        st_obj.markdown(breadcrumb_html, unsafe_allow_html=True)

class StepIndicator(NavigationComponent):
    """步骤指示器组件"""
    
    def __init__(self, steps: List[str], current_step: int = 0):
        super().__init__()
        self.steps = steps
        self.current_step = current_step
    
    def render(self, st_obj, **kwargs) -> None:
        """渲染步骤指示器"""
        if not self.steps:
            return
        
        cols = st_obj.columns(len(self.steps))
        
        for i, (col, step) in enumerate(zip(cols, self.steps)):
            with col:
                if i < self.current_step:
                    # 已完成的步骤
                    st_obj.markdown(f"""
                    <div style="text-align: center;">
                        <div style="
                            width: 30px; height: 30px; 
                            background: #4caf50; 
                            border-radius: 50%; 
                            color: white; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            margin: 0 auto 0.5rem auto;
                            font-weight: bold;
                        ">[DONE]</div>
                        <small style="color: #4caf50;">{step}</small>
                    </div>
                    """, unsafe_allow_html=True)
                elif i == self.current_step:
                    # 当前步骤
                    st_obj.markdown(f"""
                    <div style="text-align: center;">
                        <div style="
                            width: 30px; height: 30px; 
                            background: #2196f3; 
                            border-radius: 50%; 
                            color: white; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            margin: 0 auto 0.5rem auto;
                            font-weight: bold;
                        ">{i+1}</div>
                        <small style="color: #2196f3; font-weight: bold;">{step}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # 未完成的步骤
                    st_obj.markdown(f"""
                    <div style="text-align: center;">
                        <div style="
                            width: 30px; height: 30px; 
                            background: #e0e0e0; 
                            border-radius: 50%; 
                            color: #999; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            margin: 0 auto 0.5rem auto;
                        ">{i+1}</div>
                        <small style="color: #999;">{step}</small>
                    </div>
                    """, unsafe_allow_html=True)
