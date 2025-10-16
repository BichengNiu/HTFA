# -*- coding: utf-8 -*-
"""
统一相关分析组件
提供多变量领先滞后筛选分析功能，包含相关性和KL散度双重评估
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging
from typing import List, Dict, Any, Optional, Tuple

# 配置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from dashboard.ui.components.analysis.timeseries.base import TimeSeriesAnalysisComponent
from dashboard.ui.components.analysis.timeseries.dtw import DTWAnalysisComponent

logger = logging.getLogger(__name__)


class UnifiedCorrelationAnalysisComponent(TimeSeriesAnalysisComponent):
    """统一相关分析组件"""
    
    def __init__(self):
        super().__init__("time_lag_corr", "相关分析")

    def render(self, st_obj, **kwargs) -> Any:
        """
        重写渲染方法，跳过数据状态显示

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

            # 直接获取数据，不显示数据状态信息
            data, data_source, data_name = self.get_module_data()

            if data is None:
                st_obj.info("请在左侧侧边栏上传数据文件以进行分析")
                st_obj.markdown(f"""
                **使用说明：**
                1. **数据上传**：在左侧侧边栏上传数据文件（唯一上传入口）
                2. **数据格式**：第一列为时间戳，其余列为变量数据
                3. **支持格式**：CSV、Excel (.xlsx, .xls)
                4. **编码支持**：UTF-8、GBK、GB2312等

                **数据共享说明：**
                - 侧边栏上传的数据在三个分析模块间自动共享
                - 平稳性分析、相关性分析、领先滞后分析使用同一数据源
                - 无需重复上传，一次上传即可在所有模块中使用
                """)
                return None

            # 渲染分析界面
            return self.render_analysis_interface(st_obj, data, data_name)

        except Exception as e:
            self.handle_error(st_obj, e, f"渲染{self.title}组件")
            return None
    
    def render_analysis_interface(self, st_obj, data: pd.DataFrame, data_name: str) -> Any:
        """
        渲染统一相关分析界面

        Args:
            st_obj: Streamlit对象
            data: 分析数据
            data_name: 数据名称

        Returns:
            Any: 分析结果
        """
        try:
            results = {}

            # 第一部分：DTW分析
            st_obj.markdown("#### DTW分析")
            dtw_component = DTWAnalysisComponent()
            dtw_result = dtw_component.render_analysis_interface(st_obj, data, data_name)
            results["dtw"] = dtw_result

            return results

        except Exception as e:
            logger.error(f"渲染统一相关分析界面时出错: {e}")
            st_obj.error(f"渲染分析界面时出错: {e}")
            return None
    
# 领先滞后分析功能已移至独立的 LeadLagAnalysisComponent 组件
