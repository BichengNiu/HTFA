# -*- coding: utf-8 -*-
"""
分析组件基类
提供分析相关UI组件的基础功能
"""

import streamlit as st
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from dashboard.ui.components.base import UIComponent


class AnalysisComponent(UIComponent):
    """分析组件基类"""

    def __init__(self, component_name: str, display_name: str):
        self.component_name = component_name
        self.display_name = display_name
        self._state = {}
    
    @abstractmethod
    def render_analysis_interface(self, st_obj, data: pd.DataFrame) -> Dict[str, Any]:
        """
        渲染分析界面
        
        Args:
            st_obj: Streamlit对象
            data: 要分析的数据
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        pass
    
    def render_analysis_summary(self, st_obj, results: Dict[str, Any]) -> None:
        """
        渲染分析结果摘要
        
        Args:
            st_obj: Streamlit对象
            results: 分析结果
        """
        if not results:
            st_obj.info("暂无分析结果")
            return
        
        st_obj.markdown("#### 分析结果摘要")
        
        # 显示基本统计信息
        if 'data_shape' in results:
            st_obj.metric("数据形状", f"{results['data_shape'][0]} × {results['data_shape'][1]}")
        
        if 'analysis_type' in results:
            st_obj.info(f"分析类型: {results['analysis_type']}")
        
        if 'timestamp' in results:
            st_obj.caption(f"分析时间: {results['timestamp']}")
    
    def validate_data_for_analysis(self, data: pd.DataFrame) -> tuple[bool, str]:
        """
        验证数据是否适合分析
        
        Args:
            data: 要验证的数据
            
        Returns:
            tuple[bool, str]: (是否有效, 错误信息)
        """
        if data is None:
            return False, "数据为空"
        
        if data.empty:
            return False, "数据集为空"
        
        if len(data.columns) == 0:
            return False, "数据集没有列"
        
        return True, ""
    
    def get_numeric_columns(self, data: pd.DataFrame) -> List[str]:
        """
        获取数值列
        
        Args:
            data: 数据框
            
        Returns:
            List[str]: 数值列名列表
        """
        return data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    def get_categorical_columns(self, data: pd.DataFrame) -> List[str]:
        """
        获取分类列
        
        Args:
            data: 数据框
            
        Returns:
            List[str]: 分类列名列表
        """
        return data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def get_datetime_columns(self, data: pd.DataFrame) -> List[str]:
        """
        获取时间列

        Args:
            data: 数据框

        Returns:
            List[str]: 时间列名列表
        """
        datetime_cols = []
        for col in data.columns:
            try:
                pd.to_datetime(data[col], errors='raise')
                datetime_cols.append(col)
            except (ValueError, TypeError):
                continue
        return datetime_cols
    
    def render_data_info(self, st_obj, data: pd.DataFrame) -> None:
        """
        渲染数据信息
        
        Args:
            st_obj: Streamlit对象
            data: 数据框
        """
        st_obj.markdown("#### 数据信息")
        
        col1, col2, col3, col4 = st_obj.columns(4)
        
        with col1:
            st_obj.metric("总行数", data.shape[0])
        
        with col2:
            st_obj.metric("总列数", data.shape[1])
        
        with col3:
            numeric_cols = self.get_numeric_columns(data)
            st_obj.metric("数值列", len(numeric_cols))
        
        with col4:
            categorical_cols = self.get_categorical_columns(data)
            st_obj.metric("分类列", len(categorical_cols))
        
        # 显示列信息
        with st_obj.expander("查看列详情"):
            col_info = []
            for col in data.columns:
                col_info.append({
                    '列名': col,
                    '数据类型': str(data[col].dtype),
                    '非空值': data[col].notna().sum(),
                    '缺失值': data[col].isnull().sum(),
                    '唯一值': data[col].nunique()
                })
            
            col_df = pd.DataFrame(col_info)
            st_obj.dataframe(col_df, use_container_width=True)

    def get_state(self, key: str, default=None):
        """获取状态值"""
        return self._state.get(key, default)

    def set_state(self, key: str, value):
        """设置状态值"""
        self._state[key] = value

    def clear_state(self):
        """清空状态"""
        self._state.clear()


__all__ = ['AnalysisComponent']
