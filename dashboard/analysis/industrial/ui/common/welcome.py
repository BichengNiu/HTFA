# -*- coding: utf-8 -*-
"""
工业分析欢迎页面组件
"""

import streamlit as st
from dashboard.core.ui.components.base import UIComponent


class IndustrialWelcomeComponent(UIComponent):
    """工业分析欢迎页面组件"""

    def __init__(self):
        super().__init__()

    def render(self, st_obj, **kwargs):
        """渲染工业分析欢迎页面"""
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
