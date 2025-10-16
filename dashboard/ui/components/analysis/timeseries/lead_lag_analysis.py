# -*- coding: utf-8 -*-
"""
领先滞后分析组件
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
from dashboard.explore import perform_combined_lead_lag_analysis, get_detailed_lag_data_for_candidate
from dashboard.explore.preprocessing.frequency_alignment import align_series_for_analysis
from dashboard.explore.preprocessing.standardization import standardize_series

logger = logging.getLogger(__name__)


class LeadLagAnalysisComponent(TimeSeriesAnalysisComponent):
    """领先滞后分析组件"""
    
    def __init__(self):
        super().__init__("lead_lag", "领先滞后分析")

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
        渲染领先滞后分析界面

        Args:
            st_obj: Streamlit对象
            data: 分析数据
            data_name: 数据名称

        Returns:
            Any: 分析结果
        """
        try:
                       
            # 渲染领先滞后分析
            result = self.render_multivariate_screening(st_obj, data, data_name)
            return result

        except Exception as e:
            logger.error(f"渲染领先滞后分析界面时出错: {e}")
            st_obj.error(f"渲染分析界面时出错: {e}")
            return None
    
    def render_multivariate_screening(self, st_obj, data: pd.DataFrame, data_name: str) -> Any:
        """渲染多变量领先滞后筛选界面"""
        try:
            # 参数设置区域
            col1, col2 = st_obj.columns(2)
            
            with col1:
                st_obj.markdown("**目标变量设置**")
                
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_columns) < 2:
                    st_obj.warning("数据中的数值型变量少于2个，无法进行分析")
                    return None
                
                target_var = st_obj.selectbox(
                    "选择目标变量",
                    numeric_columns,
                    key="lead_lag_target_var",
                    help="选择要预测的目标变量"
                )
                
                # 获取所有可选的候选变量
                available_candidates = [col for col in numeric_columns if col != target_var]

                candidate_vars = st_obj.multiselect(
                    "选择候选变量",
                    available_candidates,
                    key="lead_lag_candidate_vars",
                    help="选择用于筛选的候选预测变量"
                )

                # 添加分析按钮
                if st_obj.button(
                    "开始分析",
                    key="lead_lag_analyze_button",
                    type="primary",
                    use_container_width=False
                ):
                    analyze_button = True
                else:
                    analyze_button = False

            with col2:
                st_obj.markdown("**分析配置**")
                
                max_lags_val = st_obj.number_input(
                    "最大滞后期数",
                    min_value=1,
                    max_value=50,
                    value=20,
                    key="lead_lag_max_lags",
                    help="设置要分析的最大滞后期数"
                )
                
                kl_bins_val = st_obj.number_input(
                    "KL散度分箱数",
                    min_value=5,
                    max_value=50,
                    value=10,
                    key="lead_lag_kl_bins",
                    help="设置KL散度计算的分箱数量"
                )
                
                # 标准化配置
                st_obj.markdown("**标准化设置**")
                standardize_for_kl = st_obj.checkbox(
                    "KL散度计算标准化",
                    value=True,
                    key="lead_lag_standardize_kl",
                    help="是否对数据进行标准化以提高KL散度计算的可比性"
                )
                
                standardization_method = st_obj.selectbox(
                    "标准化方法",
                    options=['zscore', 'minmax', 'none'],
                    index=0,
                    key="lead_lag_standardization_method",
                    help="选择标准化方法：Z-Score标准化、Min-Max标准化或不标准化",
                    disabled=not standardize_for_kl
                )
                
                # 频率对齐配置
                st_obj.markdown("**频率对齐设置**")
                enable_frequency_alignment = st_obj.checkbox(
                    "启用时间频率对齐",
                    value=True,
                    key="lead_lag_enable_freq_align",
                    help="自动检测并对齐不同时间频率的变量（如月度vs周度）"
                )
                
                col_freq1, col_freq2 = st_obj.columns(2)
                with col_freq1:
                    target_frequency = st_obj.selectbox(
                        "目标频率",
                        options=[None, 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Annual'],
                        index=0,
                        key="lead_lag_target_frequency",
                        help="指定对齐目标频率，None为自动选择最低频率",
                        disabled=not enable_frequency_alignment
                    )
                
                with col_freq2:
                    freq_agg_method = st_obj.selectbox(
                        "聚合方法",
                        options=['mean', 'last', 'first', 'sum', 'median'],
                        index=0,
                        key="lead_lag_freq_agg_method",
                        help="频率对齐时的数据聚合方法",
                        disabled=not enable_frequency_alignment
                    )
            
            if analyze_button and target_var and candidate_vars:
                return self.perform_multivariate_screening(st_obj, data, target_var, candidate_vars, max_lags_val, kl_bins_val, standardize_for_kl, standardization_method, enable_frequency_alignment, target_frequency, freq_agg_method)
            
            # 显示之前的结果（如果有）
            results = self.get_state('multivariate_results')
            if results:
                self.render_multivariate_results(st_obj, results)
                
            return results
            
        except Exception as e:
            logger.error(f"渲染多变量领先滞后筛选界面时出错: {e}")
            st_obj.error(f"渲染多变量领先滞后筛选界面时出错: {e}")
            return None
    
    def perform_multivariate_screening(self, st_obj, data: pd.DataFrame, target_var: str, candidate_vars: List[str], max_lags: int, kl_bins: int, standardize_for_kl: bool = True, standardization_method: str = 'zscore', enable_frequency_alignment: bool = True, target_frequency: str = None, freq_agg_method: str = 'mean'):
        """执行多变量领先滞后筛选分析"""
        with st_obj.spinner("正在进行多变量领先滞后筛选分析..."):
            # 构建配置字典
            config = {
                'max_lags': max_lags,
                'kl_bins': kl_bins,
                'standardize_for_kl': standardize_for_kl,
                'standardization_method': standardization_method,
                'enable_frequency_alignment': enable_frequency_alignment,
                'target_frequency': target_frequency,
                'freq_agg_method': freq_agg_method
            }

            # 调用后端函数
            results_list, errors, warnings = perform_combined_lead_lag_analysis(
                data, target_var, candidate_vars, config
            )

            # 转换结果为DataFrame
            if results_list:
                results_df = pd.DataFrame(results_list)
            else:
                results_df = pd.DataFrame()

            results = {
                'target_var': target_var,
                'candidate_vars': candidate_vars,
                'max_lags': max_lags,
                'kl_bins': kl_bins,
                'standardize_for_kl': standardize_for_kl,
                'standardization_method': standardization_method,
                'enable_frequency_alignment': enable_frequency_alignment,
                'target_frequency': target_frequency,
                'freq_agg_method': freq_agg_method,
                'results_df': results_df,
                'errors': errors,
                'warnings': warnings
            }

            # 保存结果
            self.set_state('multivariate_results', results)

            # 渲染结果
            self.render_multivariate_results(st_obj, results)

            return results

    def render_multivariate_results(self, st_obj, results: Dict[str, Any]):
        """渲染多变量领先滞后筛选结果"""
        try:
            results_df = results['results_df']
            errors = results['errors']
            warnings = results['warnings']

            # 显示错误和警告
            if errors:
                for error in errors:
                    st_obj.error(f"{error}")

            if warnings:
                for warning in warnings:
                    # 过滤掉冗长的频率对齐详细信息，只保留简洁提示
                    if "频率对齐:" in warning or "频率检查:" in warning:
                        if "[成功]" in warning or "频率对齐成功" in warning:
                            st_obj.info("已完成时间频率对齐")
                        elif "无需对齐" in warning or "频率一致" in warning:
                            st_obj.info("时间频率一致，无需对齐")
                        elif "已禁用" in warning:
                            st_obj.info("频率对齐功能已禁用")
                        elif "[错误]" in warning or "失败" in warning:
                            st_obj.error(f"频率对齐失败: {warning}")
                        elif "[信息]" in warning:
                            st_obj.info(warning.split(':', 1)[-1].strip() if ':' in warning else warning)
                        else:
                            st_obj.warning(f"频率对齐: {warning}")
                    else:
                        st_obj.warning(f"{warning}")

            if results_df is None or results_df.empty:
                st_obj.warning("没有分析结果可显示")
                return

            st_obj.markdown("##### 筛选结果")

            # 格式化结果表格
            display_results = results_df.copy()

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

            # 数值格式化
            if '最大相关系数' in display_results.columns:
                display_results['最大相关系数'] = display_results['最大相关系数'].round(4)
            if '最小KL散度' in display_results.columns:
                display_results['最小KL散度'] = display_results['最小KL散度'].round(4)

            # 结果排序：1.首先按相关系数从1到-1，2.最优滞后（相关）绝对值从小到大
            if not display_results.empty:
                sort_columns = []
                sort_ascending = []
                
                if '最大相关系数' in display_results.columns:
                    # 按相关系数原始值降序排列（从1到-1）
                    sort_columns.append('最大相关系数')
                    sort_ascending.append(False)  # 降序：1, 0.8, 0.5, 0, -0.2, -0.5, -1
                
                if '最优滞后(相关)' in display_results.columns:
                    # 按最优滞后期绝对值升序排列（滞后期短的在前）
                    display_results['_sort_lag_abs'] = display_results['最优滞后(相关)'].abs()
                    sort_columns.append('_sort_lag_abs')
                    sort_ascending.append(True)  # 升序：0, 1, 2, 3...
                
                if sort_columns:
                    display_results = display_results.sort_values(
                        by=sort_columns, 
                        ascending=sort_ascending,
                        na_position='last'  # NaN值排在最后
                    )
                    
                    # 移除排序辅助列
                    if '_sort_lag_abs' in display_results.columns:
                        display_results = display_results.drop(columns=['_sort_lag_abs'])
                    
                    # 重置索引
                    display_results = display_results.reset_index(drop=True)

            st_obj.dataframe(display_results, use_container_width=True)

            # 显示分析配置信息
            standardize_info = "已启用" if results['standardize_for_kl'] else "未启用"
            method_info = results['standardization_method']
            freq_align_info = "已启用" if results['enable_frequency_alignment'] else "未启用"
            target_freq_info = results['target_frequency'] or '自动'
            agg_info = results['freq_agg_method']
            
            st_obj.info(f"📊 **分析配置**: 最大滞后期: {results['max_lags']}, KL分箱数: {results['kl_bins']}, "
                       f"KL标准化: {standardize_info} ({method_info}), "
                       f"频率对齐: {freq_align_info} (目标频率: {target_freq_info}, 聚合: {agg_info})")

            # 提供下载功能
            csv_string = display_results.to_csv(index=False, encoding='utf-8-sig')
            csv_data = csv_string.encode('utf-8-sig')
            st_obj.download_button(
                label="下载结果",
                data=csv_data,
                file_name=f"lead_lag_analysis_{results['target_var']}.csv",
                mime="text/csv",
                key="download_lead_lag_data"
            )

            st_obj.divider()

            # 详细图表展示
            candidate_var_for_plot = st_obj.selectbox(
                "选择变量查看详细图表",
                results['candidate_vars'],
                key="lead_lag_plot_var",
                help="选择一个候选变量查看其详细的相关性和KL散度图表"
            )

            if candidate_var_for_plot:
                self.render_detailed_multivariate_charts(st_obj, results, candidate_var_for_plot)

        except Exception as e:
            logger.error(f"渲染多变量结果时出错: {e}")
            st_obj.error(f"显示结果时出错: {e}")

    def render_detailed_multivariate_charts(self, st_obj, results: Dict[str, Any], candidate_var: str):
        """渲染多变量分析的详细图表"""
        # 获取原始数据
        data, _, _ = self.get_module_data()
        if data is None:
            st_obj.warning("无法获取原始数据")
            return

        # 构建配置字典
        config = {
            'max_lags': results['max_lags'],
            'kl_bins': results['kl_bins'],
            'standardize_for_kl': results['standardize_for_kl'],
            'standardization_method': results['standardization_method'],
            'enable_frequency_alignment': results['enable_frequency_alignment'],
            'target_frequency': results['target_frequency'],
            'freq_agg_method': results['freq_agg_method']
        }

        detailed_corr_df, detailed_kl_df = get_detailed_lag_data_for_candidate(
            data, results['target_var'], candidate_var, config
        )

        if detailed_corr_df is not None and detailed_kl_df is not None:
            # 创建两列布局
            col1, col2 = st_obj.columns(2)

            with col1:
                st_obj.markdown(f"**{candidate_var} 相关性分析**")
                if not detailed_corr_df.empty:
                    # 配置matplotlib以禁用工具栏
                    plt.rcParams['toolbar'] = 'None'
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(detailed_corr_df['Lag'], detailed_corr_df['Correlation'],
                           marker='o', linewidth=2, markersize=4)
                    ax.set_xlabel('滞后期')
                    ax.set_ylabel('相关系数')
                    ax.set_title(f'{results["target_var"]} vs {candidate_var} 相关性')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.tight_layout()
                    st_obj.pyplot(fig, use_container_width=True)
                    plt.close()

            with col2:
                st_obj.markdown(f"**{candidate_var} KL散度分析**")
                if not detailed_kl_df.empty:
                    # 配置matplotlib以禁用工具栏
                    plt.rcParams['toolbar'] = 'None'
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(detailed_kl_df['Lag'], detailed_kl_df['KL_Divergence'],
                           marker='s', linewidth=2, markersize=4, color='orange')
                    ax.set_xlabel('滞后期')
                    ax.set_ylabel('KL散度')
                    ax.set_title(f'{results["target_var"]} vs {candidate_var} KL散度')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st_obj.pyplot(fig, use_container_width=True)
                    plt.close()

        # 添加时间序列对比图
        st_obj.markdown(f"**{results['target_var']} vs {candidate_var} 时间序列对比**")
        self.render_time_series_comparison(st_obj, data, results, candidate_var)

    def render_time_series_comparison(self, st_obj, data: pd.DataFrame, results: Dict[str, Any], candidate_var: str):
        """渲染目标变量与候选变量的时间序列对比图"""
        target_var = results['target_var']

        # 执行频率对齐
        df_aligned = data
        if results['enable_frequency_alignment']:
            df_aligned, alignment_report = align_series_for_analysis(
                data,
                target_var,
                [candidate_var],
                enable_frequency_alignment=True,
                target_frequency=results['target_frequency'],
                agg_method=results['freq_agg_method']
            )

            if alignment_report['status'] == 'error':
                raise ValueError(f"频率对齐失败: {alignment_report['error']}")

        # 获取对齐后的序列
        target_series = df_aligned[target_var].copy()
        candidate_series = df_aligned[candidate_var].copy()

        # 删除任一序列中的NaN值（保持索引对齐）
        valid_idx = target_series.notna() & candidate_series.notna()
        target_series_clean = target_series[valid_idx]
        candidate_series_clean = candidate_series[valid_idx]

        if len(target_series_clean) < 2:
            st_obj.warning("有效数据点太少，无法绘制时间序列图")
            return

        # 应用标准化处理（如果启用KL散度标准化）
        if results['standardize_for_kl']:
            standardization_method = results['standardization_method']
            if standardization_method != 'none':
                target_series_plot = standardize_series(target_series_clean, standardization_method)
                candidate_series_plot = standardize_series(candidate_series_clean, standardization_method)
                y_label = f"标准化值 ({standardization_method})"
                title_suffix = f" (已标准化 - {standardization_method})"
            else:
                target_series_plot = target_series_clean
                candidate_series_plot = candidate_series_clean
                y_label = "原始值"
                title_suffix = " (原始值)"
        else:
            target_series_plot = target_series_clean
            candidate_series_plot = candidate_series_clean
            y_label = "原始值"
            title_suffix = " (原始值)"

        # 配置matplotlib以禁用工具栏
        plt.rcParams['toolbar'] = 'None'

        # 创建时间序列对比图
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制时间序列
        ax.plot(target_series_plot.index, target_series_plot.values,
               label=target_var, linewidth=2, alpha=0.8)
        ax.plot(candidate_series_plot.index, candidate_series_plot.values,
               label=candidate_var, linewidth=2, alpha=0.8)

        # 设置图表样式
        ax.set_xlabel('时间')
        ax.set_ylabel(y_label)
        ax.set_title(f'{target_var} vs {candidate_var} 时间序列对比{title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 优化x轴标签显示
        if hasattr(target_series_plot.index, 'to_pydatetime'):
            # 如果是日期时间索引，优化显示
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(target_series_plot) // 10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        st_obj.pyplot(fig, use_container_width=True)
        plt.close()