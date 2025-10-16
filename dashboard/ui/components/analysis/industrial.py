# -*- coding: utf-8 -*-
"""
工业分析UI组件
从dashboard.analysis.industrial模块迁移的UI组件
"""

import streamlit as st
import pandas as pd
from typing import Optional, Tuple
from dashboard.ui.components.base import UIComponent
from dashboard.ui.components.data_input import UnifiedDataUploadComponent


class IndustrialFileUploadComponent(UIComponent):
    """
    工业分析文件上传组件

    完全基于UnifiedDataUploadComponent实现。
    """

    def __init__(self):
        super().__init__()
        # 使用UnifiedDataUploadComponent，返回文件对象模式
        self.upload_component = UnifiedDataUploadComponent(
            accepted_types=['xlsx', 'xls'],
            help_text="请上传包含'宏观运行'和相关工作表的Excel文件",
            show_data_source_selector=False,
            show_staging_data_option=False,
            component_id="industrial_unified_upload",
            return_file_object=True  # 工业分析需要文件对象
        )

    def render(self, st_obj, **kwargs) -> Optional[object]:
        """
        渲染文件上传功能

        Returns:
            uploaded_file: 上传的文件对象，如果没有上传则返回None
        """
        with st_obj.sidebar:
            uploaded_file = self.upload_component.render_file_upload_section(
                st_obj,
                upload_key="industrial_unified_file_uploader",
                show_overview=False,
                show_preview=False
            )

            if uploaded_file is None:
                st_obj.info("请上传Excel数据文件以开始工业分析")

        return uploaded_file

    def get_state_keys(self) -> list:
        """获取组件相关的状态键"""
        return self.upload_component.get_state_keys()


class IndustrialTimeRangeSelectorComponent(UIComponent):
    """工业分析时间范围选择器组件"""
    
    def __init__(self):
        super().__init__()
    
    def render(self, st_obj, key_suffix: str = "", default_index: int = 3, **kwargs) -> Tuple[str, Optional[str], Optional[str]]:
        """
        渲染时间范围选择器
        
        Args:
            st_obj: Streamlit对象
            key_suffix: 键后缀，用于区分不同的选择器实例
            default_index: 默认选择的索引
            
        Returns:
            Tuple[时间范围, 自定义开始时间, 自定义结束时间]
        """
        # Time range selector in top-left, horizontal layout
        time_range = st_obj.radio(
            "时间范围",
            ["1年", "3年", "5年", "全部", "自定义"],
            index=default_index,
            horizontal=True,
            key=f"macro_time_range_{key_suffix}"
        )

        # Custom date range inputs in same row when "自定义" is selected
        custom_start = None
        custom_end = None
        if time_range == "自定义":
            col_start, col_end = st_obj.columns([1, 1])
            with col_start:
                custom_start = st_obj.text_input("开始年月", placeholder="2020-01", key=f"custom_start_{key_suffix}")
            with col_end:
                custom_end = st_obj.text_input("结束年月", placeholder="2024-12", key=f"custom_end_{key_suffix}")

        return time_range, custom_start, custom_end
    
    def get_state_keys(self) -> list:
        """获取组件相关的状态键"""
        return [
            'macro_time_range_*',
            'custom_start_*',
            'custom_end_*'
        ]


class IndustrialGroupDetailsComponent(UIComponent):
    """工业分析分组详情组件"""
    
    def __init__(self):
        super().__init__()
    
    def render(self, st_obj, df_weights: pd.DataFrame, group_type: str, title: str, **kwargs):
        """
        渲染分组详情展开器
        
        Args:
            st_obj: Streamlit对象
            df_weights: 权重数据DataFrame
            group_type: 分组类型
            title: 展开器标题
        """
        with st_obj.expander(title):
            group_details = self._get_group_details(df_weights, group_type)
            if not group_details.empty:
                st_obj.dataframe(group_details, use_container_width=True, hide_index=True)
                st_obj.caption("权重基于2018年投入产出表增加值占比加权计算")
            else:
                st_obj.write("暂无分组数据")
    
    def _get_group_details(self, df_weights: pd.DataFrame, group_type: str) -> pd.DataFrame:
        """
        获取分组详情数据
        
        Args:
            df_weights: 权重数据
            group_type: 分组类型
            
        Returns:
            分组详情DataFrame
        """
        try:
            if group_type in df_weights.columns:
                # 按分组类型分组并计算权重
                group_details = df_weights.groupby(group_type).agg({
                    '指标名称': 'count',
                    '权重_2020': 'sum'
                }).rename(columns={
                    '指标名称': '指标数量',
                    '权重_2020': '总权重'
                }).reset_index()
                return group_details
            else:
                return pd.DataFrame()
        except Exception:
            logger.error(f"获取分组详情失败: {group_type}")
            return pd.DataFrame()
    
    def get_state_keys(self) -> list:
        """获取组件相关的状态键"""
        return []


class IndustrialWelcomeComponent(UIComponent):
    """工业分析欢迎页面组件"""
    
    def __init__(self):
        super().__init__()
    
    def render(self, st_obj, **kwargs):
        """渲染工业分析欢迎页面"""
        # 显示欢迎信息
        st_obj.markdown("""
        欢迎使用工业监测分析模块！此模块提供：

        **工业增加值分析**
        - PMI-工业增加值分析
        - 出口依赖度分组分析
        - 上中下游行业分析

        **工业企业利润拆解分析**
        - 盈利能力评估
        - 财务指标监测
        - 行业对比分析
        
        **开始使用：**
        请在左侧侧边栏上传包含工业数据的Excel模板文件。
        """)
        
        with st_obj.expander("数据格式要求"):
            st_obj.markdown("""
            **Excel文件应包含以下工作表：**
            
            1. **'分行业工业增加值同比增速'工作表** - 包含工业增加值等宏观指标数据
            2. **'工业增加值分行业指标权重'工作表** - 包含以下列：
               - `指标名称`: 与分行业工业增加值同比增速数据中的列名完全匹配
               - `出口依赖`: 分类标签（如：高出口依赖、低出口依赖）
               - `上中下游`: 分类标签（如：上游行业、中游行业、下游行业）
               - `权重_2012`: 2012年权重值（用于2012-2017年数据）
               - `权重_2018`: 2018年权重值（用于2018-2019年数据）
               - `权重_2020`: 2020年权重值（用于2020年及以后数据）

               注：至少需要包含上述权重列中的一列
            """)
    
    def get_state_keys(self) -> list:
        """获取组件相关的状态键"""
        return []
