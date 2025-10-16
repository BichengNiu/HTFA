# -*- coding: utf-8 -*-
"""
胜率分析组件
迁移自 dashboard/explore/win_rate_frontend.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

from dashboard.ui.components.analysis.timeseries.base import TimeSeriesAnalysisComponent
from dashboard.explore import win_rate_backend

logger = logging.getLogger(__name__)


class WinRateAnalysisComponent(TimeSeriesAnalysisComponent):
    """胜率分析组件"""
    
    def __init__(self):
        super().__init__("win_rate", "胜率分析")
    
    def render_analysis_parameters(self, st_obj, data: pd.DataFrame, data_name: str) -> Tuple[str, str, int]:
        """
        渲染分析参数设置界面
        
        Args:
            st_obj: Streamlit对象
            data: 分析数据
            data_name: 数据名称
            
        Returns:
            Tuple[str, str, int]: (基准序列, 比较序列, 预测期数)
        """
        numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        
        if len(numeric_cols) < 2:
            st_obj.warning("胜率分析需要至少两个数值列")
            return None, None, 0
        
        col1, col2 = st_obj.columns(2)
        
        with col1:
            # 基准序列选择
            baseline_series = st_obj.selectbox(
                "选择基准序列:",
                options=numeric_cols,
                key=f"win_rate_{data_name}_baseline_series"
            )
            
            # 预测期数
            forecast_periods = st_obj.number_input(
                "预测期数:",
                min_value=1,
                max_value=min(20, len(data) // 10),
                value=5,
                key=f"win_rate_{data_name}_forecast_periods"
            )
        
        with col2:
            # 比较序列选择
            comparison_options = [col for col in numeric_cols if col != baseline_series]
            comparison_series = st_obj.selectbox(
                "选择比较序列:",
                options=comparison_options,
                key=f"win_rate_{data_name}_comparison_series"
            )
        
        return baseline_series, comparison_series, forecast_periods
    
    def render_analysis_results(self, st_obj, results: Dict[str, Any]):
        """渲染分析结果"""
        
        if not results:
            st_obj.warning("没有分析结果可显示")
            return
        
        # 显示胜率指标
        win_rate = results.get('win_rate', 'N/A')
        total_comparisons = results.get('total_comparisons', 'N/A')
        wins = results.get('wins', 'N/A')
        
        col1, col2, col3 = st_obj.columns(3)
        
        with col1:
            if win_rate != 'N/A':
                st_obj.metric("胜率", f"{win_rate:.2%}")
        
        with col2:
            if wins != 'N/A':
                st_obj.metric("胜利次数", wins)
        
        with col3:
            if total_comparisons != 'N/A':
                st_obj.metric("总比较次数", total_comparisons)
        
        # 显示详细结果
        detailed_results = results.get('detailed_results')
        if detailed_results is not None and not detailed_results.empty:
            st_obj.markdown("**详细胜率分析结果:**")
            st_obj.dataframe(detailed_results, use_container_width=True)
            
            # 提供下载功能，使用utf-8-sig编码避免中文乱码
            csv_string = detailed_results.to_csv(index=False, encoding='utf-8-sig')
            csv_data = csv_string.encode('utf-8-sig')
            st_obj.download_button(
                label="下载胜率分析结果",
                data=csv_data,
                file_name=f"win_rate_analysis_{results.get('baseline_series', 'data')}.csv",
                mime="text/csv",
                key="download_win_rate_data"
            )
        
        # 显示参数信息
        parameters = results.get('parameters', {})
        if parameters:
            st_obj.markdown("**分析参数:**")
            for key, value in parameters.items():
                st_obj.write(f"- {key}: {value}")
        
        # 解释结果
        if win_rate != 'N/A':
            if win_rate > 0.6:
                interpretation = "比较序列表现显著优于基准序列"
                st_obj.success(f"[SUCCESS] {interpretation}")
            elif win_rate > 0.4:
                interpretation = "比较序列与基准序列表现相当"
                st_obj.info(f"[INFO] {interpretation}")
            else:
                interpretation = "比较序列表现不如基准序列"
                st_obj.warning(f"[WARNING] {interpretation}")
    
    def render_analysis_interface(self, st_obj, data: pd.DataFrame, data_name: str) -> Any:
        """渲染胜率分析界面"""
        
        try:
            # 渲染参数设置
            baseline_series, comparison_series, forecast_periods = self.render_analysis_parameters(st_obj, data, data_name)
            
            if baseline_series is None or comparison_series is None:
                return None
            
            # 分析按钮
            analysis_key = f"win_rate_analyze_btn_{data_name}"
            if st_obj.button("开始胜率分析", key=analysis_key, type="primary"):
                
                with st_obj.spinner("正在进行胜率分析..."):
                    # 准备数据
                    baseline_data = data[baseline_series].dropna()
                    comparison_data = data[comparison_series].dropna()

                    # 调用后端函数
                    win_rate, wins, total_comparisons, detailed_results = win_rate_backend.compute_win_rate(
                        baseline_data, comparison_data, forecast_periods
                    )

                    # 构建结果
                    results = {
                        'win_rate': win_rate,
                        'wins': wins,
                        'total_comparisons': total_comparisons,
                        'detailed_results': detailed_results,
                        'baseline_series': baseline_series,
                        'comparison_series': comparison_series,
                        'parameters': {
                            '基准序列': baseline_series,
                            '比较序列': comparison_series,
                            '预测期数': forecast_periods
                        }
                    }

                    # 保存结果到状态
                    self.set_state('results', results)

                    st_obj.success("胜率分析完成！")

                    # 显示结果
                    self.render_analysis_results(st_obj, results)

                    return results
            
            # 显示之前的分析结果（如果有）
            previous_results = self.get_state('results')
            
            if previous_results:
                st_obj.markdown("---")
                st_obj.markdown("#### 上次分析结果")
                self.render_analysis_results(st_obj, previous_results)
                return previous_results
            
            return None
            
        except Exception as e:
            self.handle_error(st_obj, e, "渲染胜率分析界面")
            return None


__all__ = ['WinRateAnalysisComponent']
