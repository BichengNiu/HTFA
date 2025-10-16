# -*- coding: utf-8 -*-
"""
数据输入组件基类
提供数据输入相关组件的基础接口和通用功能
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from abc import abstractmethod
import logging

from dashboard.ui.components.base import UIComponent
from dashboard.ui.utils.state_helpers import (
    get_tools_manager_instance,
    get_exploration_state,
    set_exploration_state
)
from dashboard.core import get_global_tools_manager

logger = logging.getLogger(__name__)


class DataInputComponent(UIComponent):
    """数据输入组件基类"""
    
    def __init__(self, component_name: str, title: str = None):
        self.component_name = component_name
        self.title = title or component_name
        self.logger = logging.getLogger(f"{__name__}.{component_name}")
    
    def get_component_id(self) -> str:
        """获取组件ID"""
        return f"data_input_{self.component_name}"
    
    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return [
            f'{self.component_name}_data',
            f'{self.component_name}_file_name',
            f'{self.component_name}_data_source',
            f'{self.component_name}_validation_status',
            f'{self.component_name}_last_update'
        ]
    
    def get_state(self, key: str, default=None):
        """获取组件状态 - 使用ToolsModuleManager"""
        tools_manager = get_global_tools_manager()
        if tools_manager:
            # 使用data_input作为工具类型，component_name作为子模块
            full_key = f'{self.component_name}.{key}'
            return tools_manager.get_tools_state('data_input', full_key, default)
        else:
            self.logger.warning(f"ToolsModuleManager不可用，使用fallback获取状态: {key}")
            full_key = f'{self.component_name}.{key}'
            return get_exploration_state('data_input', full_key, default)

    def set_state(self, key: str, value):
        """设置组件状态 - 使用ToolsModuleManager"""
        tools_manager = get_global_tools_manager()
        print(f"[DataInput] set_state - 组件: {self.component_name}, 键: {key}, 工具管理器: {tools_manager is not None}")

        if tools_manager:
            # 使用data_input作为工具类型，component_name作为子模块
            full_key = f'{self.component_name}.{key}'
            success = tools_manager.set_tools_state('data_input', full_key, value)
            print(f"[DataInput] set_state - 完整键: {full_key}, 保存结果: {success}")

            if success:
                self.logger.debug(f"设置数据输入状态成功: {self.component_name}.{key}")
                return True
            else:
                self.logger.warning(f"设置数据输入状态失败: {self.component_name}.{key}")
                print(f"[DataInput] WARNING - 状态保存失败: {full_key}")
                return False
        else:
            self.logger.warning(f"ToolsModuleManager不可用，使用fallback设置状态: {key}")
            print(f"[DataInput] WARNING - 工具管理器不可用，使用fallback")
            full_key = f'{self.component_name}.{key}'
            return set_exploration_state('data_input', full_key, value)
    
    def validate_data_format(self, df: pd.DataFrame) -> Tuple[bool, str, Optional[str]]:
        """
        验证数据格式是否符合标准要求
        
        Args:
            df: 待验证的DataFrame
            
        Returns:
            Tuple[bool, str, Optional[str]]: (是否有效, 消息, 时间列名)
        """
        if df is None or df.empty:
            return False, "数据为空", None
        
        # 检查是否至少有2列（时间列 + 至少1个数据列）
        if df.shape[1] < 2:
            return False, "数据至少需要包含时间列和一个数据列", None
        
        # 尝试识别时间列（通常是第一列）
        time_col = df.columns[0]
        
        # 检查第一列是否可以转换为时间格式
        try:
            pd.to_datetime(df[time_col], errors='raise')
            time_col_valid = True
        except:
            time_col_valid = False
        
        # 检查是否有数值列
        numeric_cols = [col for col in df.columns[1:] if pd.api.types.is_numeric_dtype(df[col])]
        
        if not time_col_valid:
            return False, f"第一列 '{time_col}' 无法识别为时间格式", None
        
        if not numeric_cols:
            return False, "除时间列外，没有找到有效的数值列", time_col
        
        return True, f"数据格式有效：时间列 '{time_col}'，{len(numeric_cols)} 个数值列", time_col
    
    def render_validation_status(self, st_obj, df: pd.DataFrame):
        """渲染数据验证状态"""
        if df is not None:
            is_valid, message, time_col = self.validate_data_format(df)
            
            if is_valid:
                st_obj.success(f"[SUCCESS] {message}")
                st_obj.info(f"[DATA] 数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
            else:
                st_obj.error(f"[ERROR] {message}")
        else:
            st_obj.info("[INFO] 请选择或上传数据")
    
    @abstractmethod
    def render_input_section(self, st_obj, **kwargs) -> Optional[pd.DataFrame]:
        """
        渲染数据输入部分
        
        Args:
            st_obj: Streamlit对象
            **kwargs: 其他参数
            
        Returns:
            Optional[pd.DataFrame]: 输入的数据
        """
        pass
    
    def render(self, st_obj, **kwargs) -> Optional[pd.DataFrame]:
        """
        渲染完整的数据输入组件
        
        Args:
            st_obj: Streamlit对象
            **kwargs: 其他参数
            
        Returns:
            Optional[pd.DataFrame]: 输入的数据
        """
        try:
            # 渲染标题
            if self.title:
                st_obj.markdown(f"### {self.title}")
            
            # 渲染输入部分
            data = self.render_input_section(st_obj, **kwargs)
            
            # 渲染验证状态
            self.render_validation_status(st_obj, data)
            
            return data
            
        except Exception as e:
            self.handle_error(st_obj, e, "渲染数据输入组件")
            return None


__all__ = ['DataInputComponent']
