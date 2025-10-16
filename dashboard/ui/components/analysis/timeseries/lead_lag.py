# -*- coding: utf-8 -*-
"""
领先滞后分析组件
迁移自 dashboard/explore/combined_lead_lag_frontend.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from typing import List, Dict, Any, Optional, Tuple

from dashboard.ui.components.analysis.timeseries.base import TimeSeriesAnalysisComponent
from dashboard.explore import perform_combined_lead_lag_analysis, get_detailed_lag_data_for_candidate

logger = logging.getLogger(__name__)


class LeadLagAnalysisComponent(TimeSeriesAnalysisComponent):
    """领先滞后分析组件"""
    
    def __init__(self):
        super().__init__("lead_lag", "领先滞后分析")

    def render_input_section(self, st_obj, data: pd.DataFrame, data_name: str = "数据") -> Dict[str, Any]:
        """渲染领先滞后分析输入界面"""
        st_obj.markdown("#### 领先滞后分析配置")

        # 显示数据基本信息
        st_obj.info(f"正在分析数据: {data_name}，形状: {data.shape}")

        # 检查数值列
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st_obj.error("数据中至少需要2个数值列才能进行领先滞后分析")
            return {'error': '数值列不足'}

        # 获取分析参数
        params = self.render_analysis_parameters(st_obj, data, data_name)
        if params[0] is None:  # 检查是否有错误
            return {'error': '参数设置失败'}

        target_var, candidate_vars, max_lag, kl_bins = params

        return {
            'data': data,
            'data_name': data_name,
            'target_var': target_var,
            'candidate_vars': candidate_vars,
            'max_lag': max_lag,
            'kl_bins': kl_bins
        }

    def render_analysis_parameters(self, st_obj, data: pd.DataFrame, data_name: str) -> Tuple[str, List[str], int, int]:
        """
        渲染分析参数设置界面
        
        Args:
            st_obj: Streamlit对象
            data: 分析数据
            data_name: 数据名称
            
        Returns:
            Tuple[str, List[str], int, int]: (目标变量, 候选变量列表, 最大滞后期数, KL分箱数)
        """
        st_obj.markdown("##### 参数设置")
        col_param1, col_param2 = st_obj.columns(2)
        
        with col_param1:
            # 目标变量
            numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
            if not numeric_cols:
                st_obj.warning("选定的数据集中没有可用的数值列进行分析。")
                return None, [], 0, 0
            
            current_target = self.get_state(f'{data_name}_target_var', numeric_cols[0])
            if current_target not in numeric_cols and numeric_cols:
                current_target = numeric_cols[0]
            
            target_var = st_obj.selectbox(
                "选择目标变量 (A):",
                numeric_cols,
                index=numeric_cols.index(current_target) if current_target in numeric_cols else 0,
                key=f"{data_name}_sb_target"
            )
            self.set_state(f'{data_name}_target_var', target_var)
            
            # 最大滞后期数
            max_df_len = len(data) if data is not None and not data.empty else 0
            max_lags_val = st_obj.number_input(
                "最大领先/滞后周期数:",
                min_value=1,
                max_value=max(24, max_df_len//2 if max_df_len > 0 else 24),
                value=self.get_state(f'{data_name}_max_lags', 12),
                key=f"{data_name}_ni_maxlags"
            )
            self.set_state(f'{data_name}_max_lags', max_lags_val)
        
        with col_param2:
            # 候选变量
            candidate_options = [col for col in numeric_cols if col != target_var]
            current_candidates_selection = [c for c in self.get_state(f'{data_name}_candidate_vars', []) if c in candidate_options]
            
            # 全选/清空按钮
            col_all, col_none = st_obj.columns(2)
            with col_all:
                if st_obj.button("全选候选变量", key=f"{data_name}_btn_all_candidates"):
                    self.set_state(f'{data_name}_candidate_vars', candidate_options[:])
            with col_none:
                if st_obj.button("清空候选变量", key=f"{data_name}_btn_none_candidates"):
                    self.set_state(f'{data_name}_candidate_vars', [])
            
            candidate_vars = st_obj.multiselect(
                "选择候选变量 (B们):",
                candidate_options,
                default=current_candidates_selection,
                key=f"{data_name}_ms_candidates"
            )
            self.set_state(f'{data_name}_candidate_vars', candidate_vars)
            
            # KL分箱数
            kl_bins_val = st_obj.number_input(
                "K-L分析分箱数:",
                min_value=5,
                max_value=50,
                value=self.get_state(f'{data_name}_kl_bins', 10),
                key=f"{data_name}_ni_klbins"
            )
            self.set_state(f'{data_name}_kl_bins', kl_bins_val)
        
        return target_var, candidate_vars, max_lags_val, kl_bins_val
    
    def interpret_lead_lag(self, k_corr, corr_value, k_kl, kl_value, 
                          corr_threshold=0.3, kl_significant_change_threshold=0.1, lag_agreement_tolerance=1):
        """
        解释领先滞后分析结果
        
        Args:
            k_corr: 相关性最优滞后期
            corr_value: 最大相关系数
            k_kl: KL散度最优滞后期
            kl_value: 最小KL散度
            corr_threshold: 相关性显著性阈值
            kl_significant_change_threshold: KL显著性变化阈值
            lag_agreement_tolerance: 滞后期一致性容忍度
            
        Returns:
            str: 解释文本
        """
        # 显著性检查
        corr_significant = pd.notna(corr_value) and abs(corr_value) >= corr_threshold
        kl_significant = pd.notna(kl_value) and kl_value != np.inf
        
        # 滞后期一致性
        lags_agree = pd.notna(k_corr) and pd.notna(k_kl) and abs(k_corr - k_kl) <= lag_agreement_tolerance
        
        def get_lead_lag_str(k_val, method_name):
            if k_val > 0:
                return f"Candidate leads Target by {int(k_val)} periods (via {method_name})"
            elif k_val < 0:
                return f"Candidate lags Target by {int(abs(k_val))} periods (via {method_name})"
            else:
                return f"Synchronous relationship (via {method_name})"
        
        if corr_significant and kl_significant and lags_agree:
            if k_corr == 0:
                return f"Strong Synchronous Agreement: Corr={corr_value:.2f}, KL={kl_value:.2f}. {get_lead_lag_str(k_corr, 'Both')}."
            lead_lag_description = get_lead_lag_str(k_corr, 'Both')
            return f"Strong Agreement: {lead_lag_description}. Corr={corr_value:.2f}, KL={kl_value:.2f}."
        
        # 部分一致或单一方法显著
        interpretations = []
        if corr_significant:
            interpretations.append(f"{get_lead_lag_str(k_corr, 'Correlation')} (Corr={corr_value:.2f})")
        if kl_significant:
            interpretations.append(f"{get_lead_lag_str(k_kl, 'KL Div.')} (KL={kl_value:.2f})")
        
        if interpretations:
            if len(interpretations) > 1 and not lags_agree:
                return f"Mixed Signals: {'; '.join(interpretations)}."
            return f"Potential Relationship: {'; '.join(interpretations)}."
        
        if pd.notna(k_corr) or pd.notna(k_kl):
            return "Weak or Unclear Relationship: Metrics did not meet significance thresholds or provide a clear signal."
        
        return "No significant relationship found or unable to compute."
    
    def plot_combined_lead_lag_charts(self, st_obj, full_correlogram_df, full_kl_divergence_df, target_var, candidate_var):
        """绘制综合领先滞后图表"""
        
        # 检查数据是否为空
        corr_is_empty = full_correlogram_df is None or full_correlogram_df.empty
        kl_is_empty = full_kl_divergence_df is None or full_kl_divergence_df.empty
        
        if corr_is_empty and kl_is_empty:
            st_obj.caption("No data available for plotting.")
            return
        
        # 相关性图表
        if not corr_is_empty and 'Lag' in full_correlogram_df.columns and 'Correlation' in full_correlogram_df.columns:
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Bar(
                x=full_correlogram_df['Lag'],
                y=full_correlogram_df['Correlation'],
                name='Correlation',
                marker_color='#1f77b4'
            ))
            
            if full_correlogram_df['Correlation'].notna().any():
                optimal_corr_idx = full_correlogram_df['Correlation'].abs().idxmax()
                k_corr_val = full_correlogram_df.loc[optimal_corr_idx, 'Lag']
                fig_corr.add_vline(
                    x=k_corr_val, line_width=2, line_dash="dash", line_color="red",
                    annotation_text=f"Optimal Lag (Corr): {k_corr_val}", annotation_position="top left"
                )
            
            fig_corr.update_layout(
                title_text=f'Time-Lagged Correlation: {target_var} vs {candidate_var}',
                xaxis_title_text='Lag of Candidate relative to Target (periods)',
                yaxis_title_text='Pearson Correlation',
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f0f0f0')
            )
            fig_corr.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#555555')
            fig_corr.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#555555')
            st_obj.plotly_chart(fig_corr, use_container_width=True)
        else:
            st_obj.caption(f"Correlation data for {candidate_var} is not available or is empty.")
        
        # KL散度图表
        if not kl_is_empty and 'Lag' in full_kl_divergence_df.columns and 'KL_Divergence' in full_kl_divergence_df.columns:
            plot_kl_df = full_kl_divergence_df.copy()
            plot_kl_df['KL_Divergence_Plot'] = plot_kl_df['KL_Divergence'].replace([np.inf, -np.inf], np.nan)
            
            if plot_kl_df['KL_Divergence_Plot'].notna().any():
                fig_kl = go.Figure()
                fig_kl.add_trace(go.Scatter(
                    x=plot_kl_df['Lag'],
                    y=plot_kl_df['KL_Divergence_Plot'],
                    mode='lines+markers',
                    name='K-L Divergence',
                    marker_color='#ff7f0e'
                ))
                
                valid_kl_for_min = full_kl_divergence_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['KL_Divergence'])
                if not valid_kl_for_min.empty:
                    optimal_kl_idx = valid_kl_for_min['KL_Divergence'].idxmin()
                    k_kl_val = full_kl_divergence_df.loc[optimal_kl_idx, 'Lag']
                    fig_kl.add_vline(
                        x=k_kl_val, line_width=2, line_dash="dash", line_color="green",
                        annotation_text=f"Optimal Lag (KL): {k_kl_val}", annotation_position="top right"
                    )
                
                fig_kl.update_layout(
                    title_text=f'Time-Lagged K-L Divergence: D(P_{target_var} || P_{candidate_var}@lag)',
                    xaxis_title_text='Lag of Candidate relative to Target (periods)',
                    yaxis_title_text='K-L Divergence (nats)',
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f0f0f0')
                )
                fig_kl.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#555555')
                fig_kl.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#555555', type="log")
                st_obj.plotly_chart(fig_kl, use_container_width=True)
            else:
                st_obj.caption(f"K-L Divergence data for {candidate_var} (after handling Inf/NaN) is not plottable.")
        else:
            st_obj.caption(f"K-L Divergence data for {candidate_var} is not available or is empty.")
    
    def render_analysis_results(self, st_obj, results: pd.DataFrame, target_var: str, candidate_vars: List[str], max_lags: int, kl_bins: int):
        """渲染分析结果"""
        
        if results is None or results.empty:
            st_obj.warning("没有分析结果可显示")
            return
        
        st_obj.markdown("##### 分析结果")
        
        # 格式化结果表格
        display_results = results.copy()
        
        # 移除不能序列化的列
        columns_to_remove = ['full_correlogram_df', 'full_kl_divergence_df']
        for col in columns_to_remove:
            if col in display_results.columns:
                display_results = display_results.drop(columns=[col])
        
        # 列名映射
        column_mapping = {
            'target_variable': '目标变量',
            'candidate_variable': '候选变量',
            'k_corr': '最优滞后(相关)',
            'corr_at_k_corr': '最大相关系数',
            'k_kl': '最优滞后(KL)',
            'kl_at_k_kl': '最小KL散度',
            'notes': '备注'
        }
        
        display_results = display_results.rename(columns=column_mapping)
        
        # 数值列四舍五入
        numeric_columns = display_results.select_dtypes(include=[np.number]).columns
        display_results[numeric_columns] = display_results[numeric_columns].round(3)
        
        # 添加解释列
        display_results['解释'] = display_results.apply(
            lambda row: self.interpret_lead_lag(
                row['最优滞后(相关)'], row['最大相关系数'],
                row['最优滞后(KL)'], row['最小KL散度']
            ), axis=1
        )
        
        st_obj.dataframe(display_results, use_container_width=True)
        
        # 详细图表选择
        st_obj.markdown("##### 详细图表")
        candidate_var_for_plot = st_obj.selectbox(
            "选择候选变量查看详细滞后图表:",
            [''] + candidate_vars,
            key=f"lead_lag_plot_candidate_selector"
        )
        
        if candidate_var_for_plot:
            # 获取原始数据
            data, _, _ = self.get_module_data()
            if data is not None:
                detailed_corr_df, detailed_kl_df = get_detailed_lag_data_for_candidate(
                    data, target_var, candidate_var_for_plot, max_lags, kl_bins
                )
                self.plot_combined_lead_lag_charts(st_obj, detailed_corr_df, detailed_kl_df, target_var, candidate_var_for_plot)
    
    def render_analysis_interface(self, st_obj, data: pd.DataFrame, data_name: str) -> Any:
        """渲染领先滞后分析界面"""
        
        try:
            # 渲染参数设置
            target_var, candidate_vars, max_lags_val, kl_bins_val = self.render_analysis_parameters(st_obj, data, data_name)
            
            if target_var is None:
                return None
            
            # 分析按钮
            analysis_key = f"lead_lag_analyze_btn_{data_name}"
            if st_obj.button("开始分析", key=analysis_key, type="primary"):
                if not candidate_vars:
                    st_obj.warning("请至少选择一个候选变量进行分析。")
                    return None
                
                with st_obj.spinner("正在进行时差相关性和K-L信息量分析..."):
                    # 调用后端函数
                    results_list, errors, warnings = perform_combined_lead_lag_analysis(
                        data, target_var, candidate_vars, max_lags_val, kl_bins_val
                    )

                    # 转换结果为DataFrame
                    if results_list:
                        results_df = pd.DataFrame(results_list)
                    else:
                        results_df = pd.DataFrame()

                    # 保存结果到状态
                    self.set_state('results', results_df)
                    self.set_state('errors', errors)
                    self.set_state('warnings', warnings)
                    self.set_state('parameters', {
                        'target_var': target_var,
                        'candidate_vars': candidate_vars,
                        'max_lags': max_lags_val,
                        'kl_bins': kl_bins_val
                    })

                    # 显示错误和警告
                    if errors:
                        for error in errors:
                            st_obj.error(error)
                    if warnings:
                        for warning in warnings:
                            st_obj.warning(warning)

                    if not results_df.empty:
                        st_obj.success("领先滞后分析完成！")
                        self.render_analysis_results(st_obj, results_df, target_var, candidate_vars, max_lags_val, kl_bins_val)
                    else:
                        st_obj.warning("分析未生成有效结果")

                    return {
                        'results': results_df,
                        'errors': errors,
                        'warnings': warnings,
                        'parameters': {
                            'target_var': target_var,
                            'candidate_vars': candidate_vars,
                            'max_lags': max_lags_val,
                            'kl_bins': kl_bins_val
                        }
                    }

            # 添加分隔线区分参数设置和结果显示区域
            st_obj.markdown("---")

            # 显示之前的分析结果（如果有）
            previous_results = self.get_state('results')
            previous_params = self.get_state('parameters')
            
            if previous_results is not None and not previous_results.empty and previous_params:
                st_obj.markdown("---")
                st_obj.markdown("#### 上次分析结果")
                self.render_analysis_results(
                    st_obj, previous_results,
                    previous_params.get('target_var', ''),
                    previous_params.get('candidate_vars', []),
                    previous_params.get('max_lags', 12),
                    previous_params.get('kl_bins', 10)
                )
                
                return {
                    'results': previous_results,
                    'parameters': previous_params
                }
            
            return None
            
        except Exception as e:
            self.handle_error(st_obj, e, "渲染领先滞后分析界面")
            return None


__all__ = ['LeadLagAnalysisComponent']
