# -*- coding: utf-8 -*-
"""
分析相关图表UI组件
提供各种分析图表的UI组件
"""

import streamlit as st
import pandas as pd
from typing import Optional, List, Any
from dashboard.ui.components.base import UIComponent


class TimeSeriesChartComponent(UIComponent):
    """时间序列图表组件"""
    
    def __init__(self):
        super().__init__()
    
    def render(self, st_obj, data: pd.DataFrame, variables: List[str], 
               title: str = "", time_range: str = "全部", 
               custom_start: Optional[str] = None, 
               custom_end: Optional[str] = None, **kwargs):
        """
        渲染时间序列图表
        
        Args:
            st_obj: Streamlit对象
            data: 数据DataFrame
            variables: 要绘制的变量列表
            title: 图表标题
            time_range: 时间范围
            custom_start: 自定义开始时间
            custom_end: 自定义结束时间
        """
        try:
            # 这里应该调用实际的图表创建函数
            # 为了保持UI组件的纯净性，实际的图表创建逻辑应该在业务层
            if hasattr(kwargs, 'chart_creator'):
                fig = kwargs['chart_creator'](data, variables, title, time_range, custom_start, custom_end)
                if fig is not None:
                    st_obj.plotly_chart(fig, use_container_width=True)
            else:
                st_obj.info("图表创建器未提供")
                
        except Exception as e:
            st_obj.error(f"图表渲染失败: {str(e)}")
    
    def get_state_keys(self) -> list:
        """获取组件相关的状态键"""
        return []


class EnterpriseIndicatorsChartComponent(UIComponent):
    """企业指标图表组件"""
    
    def __init__(self):
        super().__init__()
    
    def render(self, st_obj, data: pd.DataFrame, chart_type: str = "profit_breakdown",
               time_range: str = "3年", custom_start: Optional[str] = None,
               custom_end: Optional[str] = None, **kwargs):
        """
        渲染企业指标图表
        
        Args:
            st_obj: Streamlit对象
            data: 企业数据DataFrame
            chart_type: 图表类型
            time_range: 时间范围
            custom_start: 自定义开始时间
            custom_end: 自定义结束时间
        """
        try:
            # 添加图表标题
            if chart_type == "profit_breakdown":
                st_obj.markdown("#### 利润总额拆解")
            
            # 这里应该调用实际的图表创建函数
            if hasattr(kwargs, 'chart_creator'):
                fig = kwargs['chart_creator'](data, time_range, custom_start, custom_end)
                if fig is not None:
                    st_obj.plotly_chart(fig, use_container_width=True)
            else:
                st_obj.info("图表创建器未提供")
                
        except Exception as e:
            st_obj.error(f"企业指标图表渲染失败: {str(e)}")
    
    def get_state_keys(self) -> list:
        """获取组件相关的状态键"""
        return []


class DownloadButtonComponent(UIComponent):
    """下载按钮组件"""
    
    def __init__(self):
        super().__init__()
    
    def render(self, st_obj, data: pd.DataFrame, filename: str, 
               button_label: str = "下载数据", **kwargs):
        """
        渲染下载按钮
        
        Args:
            st_obj: Streamlit对象
            data: 要下载的数据
            filename: 文件名
            button_label: 按钮标签
        """
        try:
            if not data.empty:
                # 创建Excel文件
                import io
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    data.to_excel(writer, sheet_name='数据', index=True)
                excel_data = excel_buffer.getvalue()

                col1, _ = st_obj.columns([1, 3])
                with col1:
                    st_obj.download_button(
                        label=button_label,
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_{filename.replace('.', '_')}"
                    )
            else:
                st_obj.warning("没有可下载的数据")
                
        except Exception as e:
            st_obj.error(f"下载按钮渲染失败: {str(e)}")
    
    def get_state_keys(self) -> list:
        """获取组件相关的状态键"""
        return ['download_*']


class DataFilterComponent(UIComponent):
    """数据筛选组件"""
    
    def __init__(self):
        super().__init__()
    
    def render(self, st_obj, data: pd.DataFrame, filter_type: str = "time_range", **kwargs):
        """
        渲染数据筛选组件
        
        Args:
            st_obj: Streamlit对象
            data: 数据DataFrame
            filter_type: 筛选类型
        """
        try:
            if filter_type == "time_range":
                self._render_time_filter(st_obj, data, **kwargs)
            elif filter_type == "column_select":
                self._render_column_filter(st_obj, data, **kwargs)
            else:
                st_obj.warning(f"不支持的筛选类型: {filter_type}")
                
        except Exception as e:
            st_obj.error(f"数据筛选组件渲染失败: {str(e)}")
    
    def _render_time_filter(self, st_obj, data: pd.DataFrame, **kwargs):
        """渲染时间筛选器"""
        if data.index.name and 'date' in data.index.name.lower():
            min_date = data.index.min()
            max_date = data.index.max()
            
            selected_range = st_obj.date_input(
                "选择时间范围",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="data_filter_time_range"
            )
            
            return selected_range
        else:
            st_obj.info("数据不包含时间索引")
            return None
    
    def _render_column_filter(self, st_obj, data: pd.DataFrame, **kwargs):
        """渲染列筛选器"""
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_columns:
            selected_columns = st_obj.multiselect(
                "选择要显示的列",
                options=numeric_columns,
                default=numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns,
                key="data_filter_columns"
            )
            
            return selected_columns
        else:
            st_obj.warning("数据中没有数值列")
            return []
    
    def get_state_keys(self) -> list:
        """获取组件相关的状态键"""
        return [
            'data_filter_time_range',
            'data_filter_columns'
        ]
