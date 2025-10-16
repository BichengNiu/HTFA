# -*- coding: utf-8 -*-
"""
时间序列分析组件基类
提供时间序列分析组件的基础接口和通用功能
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import abstractmethod
import logging
import time

from dashboard.ui.components.base import UIComponent
from dashboard.ui.utils.state_helpers import (
    get_exploration_state,
    set_exploration_state
)
from dashboard.core import get_global_tools_manager, get_unified_manager

logger = logging.getLogger(__name__)


class TimeSeriesAnalysisComponent(UIComponent):
    """时间序列分析组件基类"""
    
    def __init__(self, analysis_type: str, title: str = None):
        # 先设置属性
        self.analysis_type = analysis_type
        self.title = title or analysis_type
        self.logger = logging.getLogger(f"{__name__}.{analysis_type}")

        # 调用父类初始化方法
        super().__init__(component_name=f"timeseries_{analysis_type}")
    
    def get_component_id(self) -> str:
        """获取组件ID"""
        return f"timeseries_{self.analysis_type}"
    
    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return [
            f'{self.analysis_type}_data',
            f'{self.analysis_type}_results',
            f'{self.analysis_type}_parameters',
            f'{self.analysis_type}_last_analysis_time',
            f'{self.analysis_type}_active'
        ]
    
    def get_state(self, key: str, default=None):
        """获取分析状态"""
        tools_manager = get_global_tools_manager()
        if tools_manager:
            return tools_manager.get_tools_state('analysis', f'{self.analysis_type}.{key}', default)
        else:
            return get_exploration_state(self.analysis_type, key, default)

    def set_state(self, key: str, value):
        """设置分析状态"""
        tools_manager = get_global_tools_manager()
        if tools_manager:
            return tools_manager.set_tools_state('analysis', f'{self.analysis_type}.{key}', value)
        else:
            return set_exploration_state(self.analysis_type, key, value)
    
    def set_state_value(self, key: str, value):
        """设置状态值"""
        tools_manager = get_global_tools_manager()
        if tools_manager:
            success = tools_manager.set_tools_state('ui_state', key, value)
            if not success:
                self.logger.debug(f"Failed to set UI state: {key}")
        else:
            self.logger.debug(f"ToolsModuleManager not available for state: {key}")

    def get_session_state_value(self, key: str, default=None):
        """获取session state值"""
        tools_manager = get_global_tools_manager()
        if tools_manager:
            value = tools_manager.get_tools_state('ui_state', f'session.{key}', None)
            return value if value is not None else default
        else:
            return default
    
    def detect_tab_activation(self, st_obj, tab_index: int) -> bool:
        """
        检测当前标签页是否激活

        Args:
            st_obj: Streamlit对象
            tab_index: 标签页索引

        Returns:
            bool: 是否激活
        """
        is_really_active = False

        tools_manager = get_global_tools_manager()

        if tools_manager:
            # 检查管理的标签页状态
            current_active_tab = tools_manager.get_tools_state('ui_state', 'data_exploration_active_tab')
            is_really_active = (current_active_tab == self.analysis_type)

        # 如果状态管理器未激活此标签，检查统一状态管理器
        if not is_really_active and self.state_manager:
            # 限制检查的键数量以提升性能
            tab_keys = [k for k in self.state_manager.get_all_keys() if 'TabState' in str(k) or 'tab' in str(k).lower()][:10]
            for key in tab_keys:
                try:
                    tab_state = self.state_manager.get_state(key)
                    if ((hasattr(tab_state, 'active_tab') and tab_state.active_tab == tab_index) or
                        (hasattr(tab_state, 'value') and tab_state.value == tab_index)):
                        is_really_active = True
                        break
                except Exception:
                    continue
        
        # 只在状态真正改变时才更新
        if is_really_active:
            previous_active_tab = self.get_session_state_value('data_exploration_active_tab', None)

            if previous_active_tab != self.analysis_type:
                self.logger.debug(f"Tab {self.analysis_type} activated")

                # 批量设置状态以减少调用次数
                self.set_state_value(f'currently_in_{self.analysis_type}_tab', True)
                self.set_state_value('data_exploration_active_tab', self.analysis_type)

                # 只在标签页切换时更新时间戳，避免每次渲染都更新
                self.set_state('last_activity_time', time.time())

        return is_really_active
    
    def get_module_data(self) -> Tuple[Optional[pd.DataFrame], str, str]:
        """
        获取当前模块的数据

        Returns:
            Tuple[Optional[pd.DataFrame], str, str]: (数据, 数据源描述, 数据名称)
        """
        self.logger.debug(f"Getting module data for analysis type: {self.analysis_type}")

        # 从统一状态管理器获取数据
        state_manager = get_unified_manager()

        if not state_manager:
            self.logger.error("Unified state manager not available")
            return None, "状态管理器不可用", ""

        # 尝试获取探索模块的数据
        data_key = f"exploration.{self.analysis_type}.data"
        file_name_key = f"exploration.{self.analysis_type}.file_name"
        data_source_key = f"exploration.{self.analysis_type}.data_source"

        selected_data = state_manager.get_state(data_key)

        if selected_data is not None:
            file_name = state_manager.get_state(file_name_key, '')
            data_source_type = state_manager.get_state(data_source_key, 'unknown')

            self.logger.debug(f"Found data with shape: {selected_data.shape}, file: {file_name}")

            if data_source_type == 'upload':
                data_source = f"上传文件: {file_name}" if file_name else "上传文件"
            else:
                data_source = f"数据源: {data_source_type}"

            selected_df_name = file_name or "data"
            return selected_data, data_source, selected_df_name
        else:
            self.logger.debug(f"No data found for module: {self.analysis_type}")
            return None, "未选择数据", ""
    
    def render_data_status(self, st_obj):
        """渲染数据状态信息"""
        selected_data, data_source, selected_df_name = self.get_module_data()
        
        if selected_data is None:
            st_obj.info("请在左侧侧边栏上传数据文件以进行分析")
            st_obj.markdown("""
            **使用说明：**
            1. **数据上传**：在左侧侧边栏上传数据文件
            2. **数据格式**：第一列为时间戳，其余列为变量数据
            3. **支持格式**：CSV、Excel (.xlsx, .xls)
            4. **数据共享**：上传的数据在所有分析模块间自动共享
            """)
            return None, "", ""
        else:
            st_obj.success(f"当前数据源: {data_source}")
            st_obj.info(f"数据形状: {selected_data.shape[0]} 行 × {selected_data.shape[1]} 列")
            return selected_data, data_source, selected_df_name
    
    @abstractmethod
    def render_analysis_interface(self, st_obj, data: pd.DataFrame, data_name: str) -> Any:
        """
        渲染分析界面
        
        Args:
            st_obj: Streamlit对象
            data: 分析数据
            data_name: 数据名称
            
        Returns:
            Any: 分析结果
        """
        pass
    
    def render(self, st_obj, **kwargs) -> Any:
        """
        渲染完整的时间序列分析组件
        
        Args:
            st_obj: Streamlit对象
            **kwargs: 其他参数
            
        Returns:
            Any: 分析结果
        """
        try:
            # 检测标签页激活状态
            tab_index = kwargs.get('tab_index', 0)
            self.detect_tab_activation(st_obj, tab_index)

            # 移除重复的标题渲染，因为标签页已经显示了标题
            # st_obj.markdown(f"### {self.title}")

            # 渲染数据状态和获取数据
            data, data_source, data_name = self.render_data_status(st_obj)
            
            if data is None:
                return None
            
            # 渲染分析界面
            return self.render_analysis_interface(st_obj, data, data_name)
            
        except Exception as e:
            self.handle_error(st_obj, e, f"渲染{self.title}组件")
            return None


__all__ = ['TimeSeriesAnalysisComponent']
