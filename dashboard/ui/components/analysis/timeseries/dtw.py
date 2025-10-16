# -*- coding: utf-8 -*-
"""
DTW分析组件
迁移自 dashboard/explore/dtw_frontend.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging
import io
from typing import List, Dict, Any, Optional, Tuple

# 配置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from dashboard.ui.components.analysis.timeseries.base import TimeSeriesAnalysisComponent
from dashboard.explore import perform_batch_dtw_calculation

logger = logging.getLogger(__name__)


class DTWAnalysisComponent(TimeSeriesAnalysisComponent):
    """DTW分析组件"""
    
    def __init__(self):
        super().__init__("dtw", "DTW分析")
    
    def smart_window_selection(self, target_length: int, comparison_length: int, series_length: int) -> Tuple[str, int]:
        """
        智能选择窗口类型和大小

        Args:
            target_length: 目标序列长度
            comparison_length: 比较序列长度
            series_length: 数据总长度

        Returns:
            Tuple[str, int]: (窗口类型, 窗口大小)
        """
        length_diff_ratio = abs(target_length - comparison_length) / max(target_length, comparison_length)

        # 根据长度差异选择窗口类型
        if length_diff_ratio > 0.3:  # 长度差异超过30%
            window_type = "Itakura"
        elif series_length >= 100:  # 长序列使用约束窗口
            window_type = "Sakoe-Chiba"
        else:  # 短序列可以使用无约束
            window_type = "无约束"

        # 根据序列长度选择窗口大小
        if window_type == "无约束":
            window_size = 0
        else:
            # 使用10%的序列长度作为窗口大小，最小5，最大50
            window_size = max(5, min(50, series_length // 10))

        return window_type, window_size

    def add_similarity_labels(self, results: list) -> list:
        """
        基于分位数法为结果添加相似度标签

        Args:
            results: DTW分析结果列表

        Returns:
            添加了相似度标签的结果列表
        """
        if not results:
            return results

        # 提取有效的标准化DTW距离
        valid_distances = []
        for result in results:
            if result.get('标准化DTW距离') != 'Error' and isinstance(result.get('标准化DTW距离'), (int, float)):
                valid_distances.append(result['标准化DTW距离'])

        if not valid_distances:
            # 如果没有有效距离，所有结果标记为错误
            for result in results:
                if result.get('标准化DTW距离') == 'Error':
                    result['相似度'] = '错误'
                else:
                    result['相似度'] = '未知'
            return results

        # 特殊情况：只有一个有效距离
        if len(valid_distances) == 1:
            for result in results:
                distance = result.get('标准化DTW距离')
                if distance == 'Error' or not isinstance(distance, (int, float)):
                    result['相似度'] = '错误'
                else:
                    result['相似度'] = '高度相似'  # 单个结果默认为高度相似
            return results

        # 计算分位数
        valid_distances = np.array(valid_distances)
        q25 = np.percentile(valid_distances, 25)
        q75 = np.percentile(valid_distances, 75)
        q95 = np.percentile(valid_distances, 95)

        # 为每个结果添加相似度标签
        for result in results:
            distance = result.get('标准化DTW距离')

            if distance == 'Error' or not isinstance(distance, (int, float)):
                result['相似度'] = '错误'
            elif distance < q25:
                result['相似度'] = '高度相似'
            elif distance < q75:
                result['相似度'] = '中等相似'
            elif distance < q95:
                result['相似度'] = '低相似'
            else:
                result['相似度'] = '不相似'

        return results

    def render_frequency_alignment_options(self, st_obj, data: pd.DataFrame, data_name: str):
        """渲染频率对齐选项"""
              
        # 启用频率对齐
        enable_alignment = st_obj.checkbox(
            "启用频率对齐", 
            value=True,
            key=f"dtw_{data_name}_enable_freq_align",
            help="当不同频率的时间序列进行DTW分析时，自动将高频数据对齐到低频"
        )
        
        if enable_alignment:
            col1, col2 = st_obj.columns(2)
            
            with col1:
                # 对齐模式
                alignment_mode = st_obj.selectbox(
                    "对齐模式:",
                    options=["stat_align", "value_align"],
                    format_func=lambda x: {
                        "stat_align": "统计对齐 (聚合方法)",
                        "value_align": "值对齐 (保持原值)"
                    }[x],
                    key=f"dtw_{data_name}_freq_align_mode",
                    help="统计对齐使用聚合方法，值对齐保持原始数值特征"
                )
            
            with col2:
                # 聚合方法
                if alignment_mode == "stat_align":
                    agg_method = st_obj.selectbox(
                        "聚合方法:",
                        options=["mean", "last", "first", "sum", "median"],
                        format_func=lambda x: {
                            "mean": "平均值",
                            "last": "最后值", 
                            "first": "首个值",
                            "sum": "求和",
                            "median": "中位数"
                        }[x],
                        key=f"dtw_{data_name}_freq_agg_method"
                    )
                else:
                    agg_method = "last"  # value_align模式的默认值
                    st_obj.write("值对齐模式")
            
            # 显示说明
            with st_obj.expander("频率对齐说明"):
                st_obj.markdown("""
                **为什么需要频率对齐？**
                
                不同频率的时间序列直接进行DTW分析可能产生不合理的结果：
                - 日度数据 vs 月度数据：长度不匹配，对齐困难
                - 高频数据包含更多噪声，影响相似性判断
                
                **对齐策略：**
                - **统计对齐**：将高频数据按时间窗口聚合（如取平均值）
                - **值对齐**：保持数据原始特征，按时间对应关系对齐
                
                **聚合方法选择：**
                - **平均值**：适合大多数经济指标，平滑噪声
                - **最后值**：适合股价等以期末值为准的指标  
                - **求和**：适合流量型指标如销售额
                """)
            
            return {
                'enable_alignment': True,
                'alignment_mode': alignment_mode, 
                'agg_method': agg_method
            }
        
        return {'enable_alignment': False}
    
    def render_standardization_options(self, st_obj, data: pd.DataFrame, data_name: str):
        """渲染数据标准化选项"""
      
        # 启用数据标准化
        enable_standardization = st_obj.checkbox(
            "启用数据标准化", 
            value=True,
            key=f"dtw_{data_name}_enable_standardization",
            help="对数据进行标准化处理，消除不同量纲的影响，提高DTW相似性分析的准确性"
        )
        
        if enable_standardization:
            col1, col2 = st_obj.columns(2)
            
            with col1:
                # 标准化方法
                standardization_method = st_obj.selectbox(
                    "标准化方法:",
                    options=["zscore", "minmax", "none"],
                    format_func=lambda x: {
                        "zscore": "Z-Score标准化 (推荐)",
                        "minmax": "Min-Max归一化",
                        "none": "不标准化"
                    }[x],
                    key=f"dtw_{data_name}_standardization_method",
                    help="选择数据标准化方法"
                )
            
            with col2:
                # 显示标准化方法说明
                if standardization_method == "zscore":
                    st_obj.info("将数据转换为均值0、标准差1的分布")
                elif standardization_method == "minmax":
                    st_obj.info("将数据缩放到[0,1]区间")
                else:
                    st_obj.warning("不进行标准化，保持原始数据")
            
            # 显示说明
            with st_obj.expander("数据标准化说明"):
                st_obj.markdown("""
                **为什么需要数据标准化？**
                
                不同量纲的时间序列直接进行DTW分析会产生问题：
                - 数值量级大的序列主导DTW距离计算
                - 例如：GDP(万亿级) vs 通胀率(个位数)，DTW距离主要反映GDP变化
                - 导致相似性分析结果不客观
                
                **标准化方法对比：**
                - **Z-Score标准化**：(x-μ)/σ，适合正态分布数据，保持数据分布形状
                - **Min-Max归一化**：(x-min)/(max-min)，适合有界数据，保持数据相对关系
                - **不标准化**：保持原始数据，适合同量纲数据比较
                
                **推荐使用场景：**
                - **经济指标分析**：推荐Z-Score，消除量纲影响同时保持波动特征
                - **技术指标分析**：可使用Min-Max，关注相对变化趋势
                - **同类指标分析**：可选择不标准化，保持原始差异
                """)
            
            return {
                'enable_standardization': True,
                'standardization_method': standardization_method
            }
        
        return {'enable_standardization': False, 'standardization_method': 'none'}

    def render_analysis_parameters(self, st_obj, data: pd.DataFrame, data_name: str):
        """
        渲染分析参数设置界面（仅自动模式）

        Args:
            st_obj: Streamlit对象
            data: 分析数据
            data_name: 数据名称

        Returns:
            自动模式参数
        """
        numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]

        if len(numeric_cols) < 2:
            st_obj.warning("DTW分析需要至少两个数值列")
            return None

        # 添加频率对齐选项
        freq_options = self.render_frequency_alignment_options(st_obj, data, data_name)
        
        # 添加数据标准化选项
        std_options = self.render_standardization_options(st_obj, data, data_name)
        
        # 自动模式参数（移除模式选择）
        params = self.render_auto_mode_parameters(st_obj, data, data_name, numeric_cols)
        
        if params:
            params.update(freq_options)
            params.update(std_options)
        
        return params

    def collect_analysis_parameters(self, st_obj, data: pd.DataFrame, data_name: str) -> dict:
        """
        收集当前界面上的所有分析参数

        Args:
            st_obj: Streamlit对象
            data: 输入数据
            data_name: 数据名称

        Returns:
            dict: 当前界面参数字典
        """
        try:
            # 获取数值列
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

            # 直接从st.session_state读取UI控件的值（而不是从统一状态管理器）
            # 因为Streamlit控件使用key参数时，值会自动保存到st.session_state
            target_series = st.session_state.get(f"dtw_{data_name}_auto_target_series")

            # 获取窗口约束参数
            enable_window_constraint = st.session_state.get(f"dtw_{data_name}_enable_window_constraint", True)
            radius = st.session_state.get(f"dtw_{data_name}_radius", 10)

            # 获取频率对齐参数
            enable_alignment = st.session_state.get(f"dtw_{data_name}_enable_freq_align", True)
            alignment_mode = st.session_state.get(f"dtw_{data_name}_freq_align_mode", 'stat_align')
            agg_method = st.session_state.get(f"dtw_{data_name}_freq_agg_method", 'mean')

            # 获取标准化参数
            enable_standardization = st.session_state.get(f"dtw_{data_name}_enable_standardization", True)
            standardization_method = st.session_state.get(f"dtw_{data_name}_standardization_method", 'zscore')

            # 距离度量在后端固定为euclidean，界面上无需设置
            # dist_metric_key = f"dtw_{data_name}_distance_metric"
            # distance_metric = st.session_state.get(dist_metric_key, 'euclidean')

            # 获取比较序列（所有除目标序列外的数值列）
            comparison_series = [col for col in numeric_cols if col != target_series] if target_series else []

            return {
                'target_series': target_series,
                'comparison_series': comparison_series,
                'enable_window_constraint': enable_window_constraint,
                'radius': radius,
                'enable_alignment': enable_alignment,
                'alignment_mode': alignment_mode,
                'agg_method': agg_method,
                'enable_standardization': enable_standardization,
                'standardization_method': standardization_method
                # 'distance_metric': distance_metric  # 移除：界面上不存在此参数
            }

        except Exception as e:
            # 如果获取参数失败，返回空字典
            logger.debug(f"Failed to collect analysis parameters: {e}")
            return {}
    
    def render_auto_mode_parameters(self, st_obj, data: pd.DataFrame, data_name: str, numeric_cols: list):
        """渲染自动模式参数"""
        st_obj.markdown("**DTW批量分析**: 选择目标序列和DTW参数，自动计算所有对比序列")

        col1, col2 = st_obj.columns(2)

        with col1:
            # 目标序列选择
            target_series = st_obj.selectbox(
                "选择目标序列:",
                options=numeric_cols,
                key=f"dtw_{data_name}_auto_target_series"
            )

            # 窗口约束选择
            enable_window_constraint = st_obj.checkbox(
                "启用窗口约束",
                value=True,
                key=f"dtw_{data_name}_enable_window_constraint",
                help="限制DTW对齐路径在对角线附近，提高计算效率"
            )

            # 窗口约束说明
            with st_obj.expander("窗口约束说明"):
                st_obj.markdown("""
                **无约束**: 允许对齐路径自由移动，计算最精确但速度较慢
                
                **有约束**: 限制对齐路径在主对角线附近的带状区域内
                - 提高计算效率，适合大数据集
                - Radius参数决定约束强度
                - 例如：Radius=5 表示序列1的第i个点只能与序列2的第(i-5)到(i+5)个点对齐
                """)
                st_obj.info("提示：小的radius值计算更快但可能错过最优对齐；大的radius值更灵活但计算更慢")

        with col2:
            # 显示将要分析的序列
            comparison_series = [col for col in numeric_cols if col != target_series]

            # Radius参数设置
            if enable_window_constraint:
                radius = st_obj.number_input(
                    "Radius (窗口大小):",
                    min_value=1,
                    max_value=min(50, len(data) // 2),
                    value=max(5, len(data) // 10),
                    key=f"dtw_{data_name}_radius",
                    help="限制对齐路径在主对角线上下各radius个单位内"
                )
                
                # Radius选择指导
                with st_obj.expander("Radius参数说明"):
                    series_length = len(data)
                    current_percentage = (radius / series_length) * 100
                    
                    st_obj.markdown(f"""
                    **当前数据长度**: {series_length} 个时间点
                    **当前设置**: Radius = {radius} (约{current_percentage:.1f}%序列长度)

                    **Radius意义**:
                    - Radius = {radius} 表示序列1的第i个点只能与序列2的第(i-{radius})到(i+{radius})个点对齐
                    - 越小的radius: 计算更快，但对齐更严格
                    - 越大的radius: 对齐更灵活，但计算更慢

                    **推荐值**:
                    - **小数据集** (<50点): 5-10
                    - **中等数据集** (50-200点): 10-20 
                    - **大数据集** (>200点): 20-50
                    """)
                    
                    # 根据当前设置给出评价
                    if current_percentage < 3:
                        st_obj.warning("当前Radius较小，可能限制对齐灵活性")
                    elif current_percentage > 30:
                        st_obj.warning("当前Radius较大，计算时间可能较长")
                    else:
                        st_obj.success("当前Radius设置合理")
            else:
                radius = None
                st_obj.info("无约束模式: 允许任意对齐，计算时间较长")

        return {
            'mode': 'auto',
            'target_series': target_series,
            'comparison_series': comparison_series,
            'enable_window_constraint': enable_window_constraint,
            'radius': radius
        }


    def perform_auto_dtw_analysis(self, st_obj, data: pd.DataFrame, target_series: str, comparison_series: list, enable_window_constraint: bool, radius: int = None, freq_options: dict = None, std_options: dict = None):
        """执行自动DTW分析，包含频率处理"""
        results = []
        
        # 设置默认频率选项
        if freq_options is None:
            freq_options = {'enable_alignment': True, 'alignment_mode': 'stat_align', 'agg_method': 'mean'}
        
        # 设置默认标准化选项
        if std_options is None:
            std_options = {'enable_standardization': True, 'standardization_method': 'zscore'}

        # 准备目标序列数据
        target_data = data[target_series].dropna()
        target_length = len(target_data)

        # 设置窗口约束参数
        if enable_window_constraint and radius is not None:
            window_type_param = "固定大小窗口 (Radius约束)"
            window_size_param = radius
        else:
            window_type_param = "无限制"
            window_size_param = None

        # 执行DTW计算（现在包含频率处理和标准化）
        dtw_results, paths_dict, errors, warnings = perform_batch_dtw_calculation(
            df_input=data,
            target_series_name=target_series,
            comparison_series_names=comparison_series,
            window_type_param=window_type_param,
            window_size_param=window_size_param,
            dist_metric_name_param="euclidean",
            dist_metric_display_param="欧氏距离",
            enable_freq_alignment=freq_options.get('enable_alignment', True),
            freq_alignment_mode=freq_options.get('alignment_mode', 'stat_align'),
            freq_agg_method=freq_options.get('agg_method', 'mean'),
            standardization_method=std_options.get('standardization_method', 'zscore')
        )

        # 转换结果格式
        for result in dtw_results:
            comp_var = result.get('对比变量', 'Unknown')
            dtw_distance = result.get('DTW距离', 'Error')

            if isinstance(dtw_distance, (int, float)):
                # 获取路径长度
                path_length = 'N/A'
                if comp_var in paths_dict:
                    path_data = paths_dict[comp_var]
                    if 'path' in path_data and path_data['path']:
                        path_length = len(path_data['path'])

                # 计算标准化DTW距离（使用路径长度）
                if isinstance(path_length, int) and path_length > 0:
                    normalized_dtw = round(dtw_distance / path_length, 4)
                else:
                    normalized_dtw = 'Error'

                results.append({
                    '变量名': comp_var,
                    '窗口约束': '启用' if enable_window_constraint else '禁用',
                    'Radius': radius if enable_window_constraint else 'N/A',
                    'DTW距离': round(dtw_distance, 4),
                    '对齐路径长度': path_length,
                    '标准化DTW距离': normalized_dtw
                })
            else:
                results.append({
                    '变量名': comp_var,
                    '窗口约束': '启用' if enable_window_constraint else '禁用',
                    'Radius': radius if enable_window_constraint else 'N/A',
                    'DTW距离': 'Error',
                    '对齐路径长度': 'Error',
                    '标准化DTW距离': 'Error'
                })

        # 仅显示重要的错误信息
        if errors:
            for error in errors:
                st_obj.error(error)

        # 重要警告信息简化显示
        if warnings:
            important_warnings = [w for w in warnings if any(keyword in w for keyword in ['失败', '错误', '无效'])]
            if important_warnings:
                for warning in important_warnings:
                    st_obj.warning(warning)

        # 添加相似度标签
        results = self.add_similarity_labels(results)

        return results

    def render_download_button(self, st_obj, results, params, data_name):
        """渲染下载按钮"""
        try:
            # 自动模式：下载批量结果
            if isinstance(results, list) and results:
                # 转换为DataFrame
                df = pd.DataFrame(results)

                # 生成CSV数据 - 使用UTF-8 BOM编码确保Excel正确显示中文
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False, encoding='utf-8')
                csv_data = '\ufeff' + csv_buffer.getvalue()  # 添加BOM

                # 生成文件名
                target_name = params.get('target_series', 'unknown') if params else 'unknown'
                filename = f"DTW批量分析结果_{target_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

                st_obj.download_button(
                    label="下载结果",
                    data=csv_data.encode('utf-8-sig'),
                    file_name=filename,
                    mime="text/csv",
                    key=f"dtw_download_auto_{data_name}_{len(results)}",  # 添加结果长度确保唯一性
                    help="下载DTW批量分析结果为CSV文件"
                )
            else:
                # 没有结果时显示禁用的按钮
                st_obj.download_button(
                    label="下载结果",
                    data="",
                    file_name="no_results.csv",
                    mime="text/csv",
                    key=f"dtw_download_auto_disabled_{data_name}",
                    help="暂无分析结果可下载",
                    disabled=True
                )

        except Exception as e:
            logger.error(f"生成下载按钮失败: {e}")
            st_obj.error(f"下载功能暂时不可用: {str(e)}")

    def render_auto_results(self, st_obj, results: list, target_series: str):
        """渲染自动模式结果"""
        if not results:
            st_obj.warning("没有分析结果")
            return

        # 转换为DataFrame并排序
        results_df = pd.DataFrame(results)

        # 过滤掉错误的结果进行排序
        valid_results = results_df[results_df['标准化DTW距离'] != 'Error'].copy()
        error_results = results_df[results_df['标准化DTW距离'] == 'Error'].copy()

        if not valid_results.empty:
            # 按标准化DTW距离排序（从小到大，越小越相似）
            valid_results = valid_results.sort_values('标准化DTW距离')

            # 合并结果
            final_results = pd.concat([valid_results, error_results], ignore_index=True)
        else:
            final_results = results_df

        # 显示统计信息在表格上方
        if not valid_results.empty:
            col1, col2, col3, col4 = st_obj.columns(4)
            with col1:
                st_obj.metric("成功分析", len(valid_results))
            with col2:
                st_obj.metric("最小标准化DTW距离", f"{valid_results['标准化DTW距离'].min():.4f}")
            with col3:
                st_obj.metric("最大标准化DTW距离", f"{valid_results['标准化DTW距离'].max():.4f}")
            with col4:
                # 显示实际使用的窗口设置
                window_constraint = valid_results['窗口约束'].iloc[0] if '窗口约束' in valid_results.columns else '未知'
                radius = valid_results['Radius'].iloc[0] if 'Radius' in valid_results.columns else '未知'
                st_obj.metric("窗口约束", f"{window_constraint}")
                if str(radius) != 'N/A' and radius != '未知':
                    st_obj.caption(f"Radius: {radius}")

        # 显示结果表格
        st_obj.dataframe(
            final_results,
            use_container_width=True,
            hide_index=True
        )

    def render_comparison_selection_and_plot(self, st_obj, data: pd.DataFrame, results: list, target_series: str, comparison_series: list, data_name: str):
        """
        渲染比较序列选择和DTW对比图
        
        Args:
            st_obj: Streamlit对象
            data: 原始数据
            results: DTW分析结果列表
            target_series: 目标序列名称
            comparison_series: 比较序列列表
            data_name: 数据名称（用于读取当前界面参数）
        """
        if not results or not comparison_series:
            return
            
        st_obj.markdown("---")
        st_obj.markdown("#### DTW对比图")
        
        # 创建比较序列选择下拉菜单
        col1, col2 = st_obj.columns([1, 2])
        
        with col1:
            # 获取所有有效的比较序列（去除错误的结果）
            valid_comparison_series = []
            for result in results:
                var_name = result.get('变量名')
                dtw_distance = result.get('DTW距离')
                if var_name and dtw_distance != 'Error' and isinstance(dtw_distance, (int, float)):
                    valid_comparison_series.append(var_name)
            
            if valid_comparison_series:
                selected_comparison = st_obj.selectbox(
                    "选择比较序列:",
                    options=valid_comparison_series,
                    key="dtw_comparison_selection",
                    help="选择一个序列查看DTW对比图"
                )
            else:
                st_obj.warning("没有有效的比较结果可供显示")
                return
        
        with col2:
            # 显示选中比较序列的DTW信息
            selected_result = None
            for result in results:
                if result.get('变量名') == selected_comparison:
                    selected_result = result
                    break
            
            if selected_result:
                dtw_distance = selected_result.get('DTW距离')
                normalized_distance = selected_result.get('标准化DTW距离')
                similarity = selected_result.get('相似度', '未知')
                
                info_col1, info_col2, info_col3 = st_obj.columns(3)
                with info_col1:
                    st_obj.metric("DTW距离", f"{dtw_distance:.4f}" if isinstance(dtw_distance, (int, float)) else str(dtw_distance))
                with info_col2:
                    st_obj.metric("标准化DTW距离", f"{normalized_distance:.4f}" if isinstance(normalized_distance, (int, float)) else str(normalized_distance))
                with info_col3:
                    st_obj.metric("相似度", similarity)
        
        # 绘制DTW对比图
        # 重新读取当前界面参数（使用实时参数而不是保存的旧参数）
        # 直接从st.session_state读取当前界面的窗口约束设置
        enable_constraint = st.session_state.get(f"dtw_{data_name}_enable_window_constraint", True)
        radius = st.session_state.get(f"dtw_{data_name}_radius", 10)

        # 直接从st.session_state读取频率和标准化参数
        freq_options = {
            'enable_alignment': st.session_state.get(f"dtw_{data_name}_enable_freq_align", True),
            'alignment_mode': st.session_state.get(f"dtw_{data_name}_freq_align_mode", 'stat_align'),
            'agg_method': st.session_state.get(f"dtw_{data_name}_freq_agg_method", 'mean')
        }

        std_options = {
            'enable_standardization': st.session_state.get(f"dtw_{data_name}_enable_standardization", True),
            'standardization_method': st.session_state.get(f"dtw_{data_name}_standardization_method", 'zscore')
        }

        # 设置窗口约束参数
        if enable_constraint and radius is not None:
            window_type_param = "固定大小窗口 (Radius约束)"
            window_size_param = radius
        else:
            window_type_param = "无限制"
            window_size_param = None

        # 为单个比较序列重新计算DTW以获取路径信息
        dtw_results, paths_dict, errors, warnings = perform_batch_dtw_calculation(
            df_input=data,
            target_series_name=target_series,
            comparison_series_names=[selected_comparison],
            window_type_param=window_type_param,
            window_size_param=window_size_param,
            dist_metric_name_param="euclidean",
            dist_metric_display_param="欧氏距离",
            enable_freq_alignment=freq_options.get('enable_alignment', True),
            freq_alignment_mode=freq_options.get('alignment_mode', 'stat_align'),
            freq_agg_method=freq_options.get('agg_method', 'mean'),
            standardization_method=std_options.get('standardization_method', 'zscore')
        )

        # 获取路径数据
        if selected_comparison in paths_dict:
            path_data = paths_dict[selected_comparison]
            s1_data = path_data.get('target_np')
            s2_data = path_data.get('compare_np')
            path = path_data.get('path', [])

            if s1_data is not None and s2_data is not None and path:
                st_obj.markdown(f"**{target_series} vs {selected_comparison} DTW对齐路径图:**")
                # 显示当前使用的参数设置
                constraint_info = "启用" if enable_constraint else "禁用"
                radius_info = f"Radius={radius}" if enable_constraint else "无限制"
                st_obj.caption(f"当前参数: 窗口约束={constraint_info}, {radius_info}")
                self.plot_dtw_path(st_obj, s1_data, s2_data, path, target_series, selected_comparison)
            else:
                st_obj.warning(f"无法获取 {selected_comparison} 的DTW路径数据")
        else:
            st_obj.warning(f"无法找到 {selected_comparison} 的DTW计算结果")

    def plot_dtw_path(self, st_obj, s1_np, s2_np, path, s1_name, s2_name):
        """绘制DTW路径图"""
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(np.arange(len(s1_np)), s1_np, "o-", label=s1_name, markersize=4, linewidth=1.5)
        ax.plot(np.arange(len(s2_np)), s2_np, "s-", label=s2_name, markersize=4, linewidth=1.5)

        for idx1, idx2 in path:
            if idx1 < len(s1_np) and idx2 < len(s2_np):
                ax.plot([idx1, idx2], [s1_np[idx1], s2_np[idx2]],
                       color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

        ax.set_xlabel("时间索引")
        ax.set_ylabel("数值")
        ax.set_title(f"DTW对齐路径: {s1_name} vs {s2_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st_obj.pyplot(fig)
        plt.close(fig)
    
    
    def render_analysis_interface(self, st_obj, data: pd.DataFrame, data_name: str) -> Any:
        """渲染DTW分析界面"""

        try:
            # 渲染参数设置
            params = self.render_analysis_parameters(st_obj, data, data_name)

            if params is None:
                return None

            # 分析按钮和下载按钮并排显示
            col1, col2, col3 = st_obj.columns([1, 1, 2])

            with col1:
                analysis_key = f"dtw_analyze_btn_{data_name}"
                analyze_clicked = st_obj.button("开始分析", key=analysis_key, type="primary")

            with col2:
                # 检查是否有结果可下载
                download_results = self.get_state('auto_results')
                # 始终显示下载按钮，根据是否有结果来决定是否禁用
                self.render_download_button(st_obj, download_results, params, data_name)

            # 初始化分析结果变量
            current_results = None

            if analyze_clicked:
                # 提取频率选项
                freq_options = {
                    'enable_alignment': params.get('enable_alignment', True),
                    'alignment_mode': params.get('alignment_mode', 'stat_align'),
                    'agg_method': params.get('agg_method', 'mean')
                }
                
                # 提取标准化选项
                std_options = {
                    'enable_standardization': params.get('enable_standardization', True),
                    'standardization_method': params.get('standardization_method', 'zscore')
                }
                
                with st_obj.spinner("正在进行批量DTW分析..."):
                    try:
                        results = self.perform_auto_dtw_analysis(
                            st_obj, data, params['target_series'], params['comparison_series'], 
                            params['enable_window_constraint'], params['radius'], freq_options, std_options
                        )

                        # 保存结果到状态
                        self.set_state('auto_results', results)
                        self.set_state('target_series', params['target_series'])
                        self.set_state('comparison_series', params['comparison_series'])
                        self.set_state('dtw_params', params)
                        
                        # 设置当前结果
                        current_results = results

                        # 显示结果
                        self.render_auto_results(st_obj, results, params['target_series'])
                        
                        # 添加比较序列选择和DTW图显示
                        self.render_comparison_selection_and_plot(st_obj, data, results, params['target_series'], params['comparison_series'], data_name)

                        return results

                    except Exception as e:
                        error_msg = f"批量DTW分析失败: {str(e)}"
                        st_obj.error(error_msg)
                        logger.error(error_msg)
                        return None
            
            # 检查是否有之前的分析结果，并验证参数是否发生变化
            previous_auto_results = self.get_state('auto_results')
            previous_target = self.get_state('target_series')
            previous_comparison_series = self.get_state('comparison_series')
            previous_params = self.get_state('dtw_params')

            logger.info(f"DTW状态检查 - 是否有之前的结果: {previous_auto_results is not None}")

            if previous_auto_results and previous_target:
                # 获取当前界面参数
                current_params = self.collect_analysis_parameters(st_obj, data, data_name)
                logger.info(f"DTW参数收集 - 当前参数: {list(current_params.keys())}")

                # 检查关键参数是否发生变化
                params_changed = False
                if previous_params is None:
                    params_changed = True
                    logger.info("DTW参数检查 - 之前无参数记录，判定为参数已变化")
                else:
                    # 检查影响DTW计算的关键参数
                    key_params = ['target_series', 'enable_window_constraint', 'radius',
                                'enable_standardization', 'standardization_method',
                                'enable_alignment', 'alignment_mode', 'agg_method']

                    # 检查关键参数是否变化
                    for param in key_params:
                        current_val = current_params.get(param)
                        previous_val = previous_params.get(param)
                        if current_val != previous_val:
                            logger.info(f"DTW参数变化检测 - {param}: {previous_val} → {current_val}")
                            params_changed = True
                            break

                    # 添加调试信息
                    if not params_changed:
                        logger.info("DTW参数检查 - 无变化，保持现有结果")

                    # 检查比较序列是否变化
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    current_comparison = [col for col in numeric_cols if col != current_params.get('target_series')]
                    if set(current_comparison) != set(previous_comparison_series or []):
                        logger.info(f"DTW参数变化检测 - 比较序列发生变化")
                        params_changed = True
                
                if params_changed:
                    # 参数已变化，清除之前的结果状态
                    logger.info("DTW参数已变化 - 清除之前的分析结果")
                    self.set_state('auto_results', None)
                    self.set_state('target_series', None)
                    self.set_state('comparison_series', None)
                    self.set_state('dtw_params', None)
                else:
                    # 参数未变化，显示之前的结果
                    logger.info("DTW参数未变化 - 显示之前的分析结果")
                    st_obj.markdown("---")
                    st_obj.markdown("#### 批量分析结果")
                    self.render_auto_results(st_obj, previous_auto_results, previous_target)

                    # 添加比较序列选择和DTW图显示（使用当前界面参数）
                    self.render_comparison_selection_and_plot(st_obj, data, previous_auto_results, previous_target, previous_comparison_series, data_name)

                    return previous_auto_results

            return None
            
        except Exception as e:
            self.handle_error(st_obj, e, "渲染DTW分析界面")
            return None


__all__ = ['DTWAnalysisComponent']
