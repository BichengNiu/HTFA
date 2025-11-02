# -*- coding: utf-8 -*-
"""
可视化组件
整合原有的可视化UI组件功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional, Tuple
import logging

from dashboard.ui.components.analysis.base import AnalysisComponent

logger = logging.getLogger(__name__)


class VisualizationComponent(AnalysisComponent):
    """可视化组件"""
    
    def __init__(self):
        super().__init__("visualization", "数据可视化")
    
    def render_analysis_interface(self, st_obj, data: pd.DataFrame) -> Dict[str, Any]:
        """渲染可视化分析界面（实现抽象方法）"""
        return self.render_input_section(st_obj, data=data)

    def render(self, st_obj, **kwargs) -> Optional[Dict[str, Any]]:
        """渲染组件（实现抽象方法）"""
        data = kwargs.get('data')
        return self.render_input_section(st_obj, data=data)

    def get_state_keys(self) -> List[str]:
        """获取状态键列表（实现抽象方法）"""
        return [
            'last_chart',
            'chart_type',
            'chart_config',
            'selected_columns',
            'chart_data'
        ]
    
    def render_input_section(self, st_obj, **kwargs) -> Optional[Dict[str, Any]]:
        """渲染可视化输入部分"""
        
        data = kwargs.get('data')
        if data is None or data.empty:
            st_obj.error("请先提供数据")
            return None
        
        st_obj.markdown("#### 数据可视化")
        
        # 图表类型选择
        chart_types = {
            "线图": "line",
            "散点图": "scatter", 
            "柱状图": "bar",
            "直方图": "histogram",
            "箱线图": "box",
            "热力图": "heatmap",
            "相关性矩阵": "correlation"
        }
        
        col1, col2 = st_obj.columns(2)
        
        with col1:
            selected_chart_type = st_obj.selectbox(
                "选择图表类型:",
                options=list(chart_types.keys()),
                key=f"{self.component_name}_chart_type"
            )
            
            chart_type = chart_types[selected_chart_type]
        
        with col2:
            # 根据图表类型显示不同的选项
            if chart_type in ["line", "scatter", "bar"]:
                # 需要选择X轴和Y轴
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                all_cols = data.columns.tolist()
                
                x_column = st_obj.selectbox(
                    "X轴列:",
                    options=all_cols,
                    key=f"{self.component_name}_x_column"
                )
                
                y_columns = st_obj.multiselect(
                    "Y轴列:",
                    options=numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
                    key=f"{self.component_name}_y_columns"
                )
                
            elif chart_type == "histogram":
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                selected_column = st_obj.selectbox(
                    "选择列:",
                    options=numeric_cols,
                    key=f"{self.component_name}_hist_column"
                )
                
            elif chart_type == "box":
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                selected_columns = st_obj.multiselect(
                    "选择列:",
                    options=numeric_cols,
                    default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols,
                    key=f"{self.component_name}_box_columns"
                )
        
        # 生成图表按钮
        if st_obj.button("生成图表", key=f"{self.component_name}_generate"):
            try:
                fig = None
                
                if chart_type == "line" and y_columns:
                    fig = self._create_line_chart(data, x_column, y_columns)
                    
                elif chart_type == "scatter" and y_columns:
                    fig = self._create_scatter_chart(data, x_column, y_columns)
                    
                elif chart_type == "bar" and y_columns:
                    fig = self._create_bar_chart(data, x_column, y_columns)
                    
                elif chart_type == "histogram":
                    fig = self._create_histogram(data, selected_column)
                    
                elif chart_type == "box":
                    fig = self._create_box_plot(data, selected_columns)
                    
                elif chart_type == "heatmap":
                    fig = self._create_heatmap(data)
                    
                elif chart_type == "correlation":
                    fig = self._create_correlation_matrix(data)
                
                if fig is not None:
                    st_obj.plotly_chart(fig, use_container_width=True)
                    
                    # 保存图表到状态
                    self.set_state('last_chart', fig)
                    self.set_state('chart_type', chart_type)
                    
                    st_obj.success("图表生成成功！")
                    
                    return {
                        'chart': fig,
                        'chart_type': chart_type,
                        'data_shape': data.shape
                    }
                else:
                    st_obj.error("图表生成失败，请检查数据和参数设置")
                    
            except Exception as e:
                st_obj.error(f"图表生成失败: {e}")
                logger.error(f"图表生成失败: {e}")
        
        return None
    
    def _create_line_chart(self, data: pd.DataFrame, x_col: str, y_cols: List[str]) -> go.Figure:
        """创建线图"""
        fig = go.Figure()
        
        for y_col in y_cols:
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines+markers',
                name=y_col,
                line=dict(width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title=f"线图: {', '.join(y_cols)} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title="值",
            hovermode='x unified'
        )
        
        return fig
    
    def _create_scatter_chart(self, data: pd.DataFrame, x_col: str, y_cols: List[str]) -> go.Figure:
        """创建散点图"""
        fig = go.Figure()
        
        for y_col in y_cols:
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='markers',
                name=y_col,
                marker=dict(size=6, opacity=0.7)
            ))
        
        fig.update_layout(
            title=f"散点图: {', '.join(y_cols)} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title="值"
        )
        
        return fig
    
    def _create_bar_chart(self, data: pd.DataFrame, x_col: str, y_cols: List[str]) -> go.Figure:
        """创建柱状图"""
        fig = go.Figure()
        
        for y_col in y_cols:
            fig.add_trace(go.Bar(
                x=data[x_col],
                y=data[y_col],
                name=y_col
            ))
        
        fig.update_layout(
            title=f"柱状图: {', '.join(y_cols)} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title="值",
            barmode='group'
        )
        
        return fig
    
    def _create_histogram(self, data: pd.DataFrame, column: str) -> go.Figure:
        """创建直方图"""
        fig = px.histogram(
            data, 
            x=column,
            title=f"直方图: {column}",
            nbins=30
        )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="频数"
        )
        
        return fig
    
    def _create_box_plot(self, data: pd.DataFrame, columns: List[str]) -> go.Figure:
        """创建箱线图"""
        fig = go.Figure()
        
        for col in columns:
            fig.add_trace(go.Box(
                y=data[col],
                name=col,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title=f"箱线图: {', '.join(columns)}",
            yaxis_title="值"
        )
        
        return fig
    
    def _create_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """创建热力图"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("没有数值列可用于创建热力图")
        
        fig = px.imshow(
            numeric_data.T,
            title="数据热力图",
            aspect='auto',
            color_continuous_scale='RdYlBu_r'
        )
        
        fig.update_layout(
            xaxis_title="样本索引",
            yaxis_title="变量"
        )
        
        return fig
    
    def _create_correlation_matrix(self, data: pd.DataFrame) -> go.Figure:
        """创建相关性矩阵"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("没有数值列可用于创建相关性矩阵")
        
        corr_matrix = numeric_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            title="相关性矩阵",
            color_continuous_scale='RdBu_r',
            aspect='equal'
        )
        
        fig.update_layout(
            xaxis_title="变量",
            yaxis_title="变量"
        )
        
        return fig


__all__ = ['VisualizationComponent']
