# -*- coding: utf-8 -*-
"""
平稳性分析组件
迁移自 dashboard/explore/stationarity_frontend.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import logging
from typing import List, Dict, Any, Optional, Tuple

from dashboard.ui.components.analysis.timeseries.base import TimeSeriesAnalysisComponent
from dashboard.explore import test_and_process_stationarity

logger = logging.getLogger(__name__)


class StationarityAnalysisComponent(TimeSeriesAnalysisComponent):
    """平稳性分析组件"""
    
    def __init__(self):
        super().__init__("stationarity", "平稳性分析")

    def render_input_section(self, st_obj, data: pd.DataFrame, data_name: str = "数据") -> Dict[str, Any]:
        """渲染平稳性分析输入界面"""
        st_obj.markdown("#### 平稳性分析配置")

        # 显示数据基本信息
        st_obj.info(f"正在分析数据: {data_name}，形状: {data.shape}")

        # 选择分析列
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st_obj.error("数据中没有数值列，无法进行平稳性分析")
            return {'error': '没有数值列'}

        selected_cols = st_obj.multiselect(
            "选择要分析的列",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
            key=f"{self.component_id}_selected_cols"
        )

        if not selected_cols:
            st_obj.warning("请选择至少一列进行分析")
            return {'selected_cols': []}

        # 获取分析参数
        alpha, method, order = self.render_analysis_parameters(st_obj, data_name)

        return {
            'data': data,
            'data_name': data_name,
            'selected_cols': selected_cols,
            'alpha': alpha,
            'method': method,
            'order': order
        }

    def render_analysis_parameters(self, st_obj, data_name: str) -> Tuple[float, str, int]:
        """
        渲染分析参数设置界面
        
        Args:
            st_obj: Streamlit对象
            data_name: 数据名称
            
        Returns:
            Tuple[float, str, int]: (显著性水平, 处理方法, 差分阶数)
        """
        # 分析设置
        settings_cols = st_obj.columns([1, 2, 1.5])  # Alpha | Method | Order
        
        with settings_cols[0]:  # Alpha
            alpha_key = f"stationarity_alpha_radio_{data_name}"
            selected_alpha = st_obj.radio(
                "显著性水平:",
                options=[0.01, 0.05, 0.10],
                index=1,
                horizontal=True,
                key=alpha_key,
                label_visibility="visible"
            )
        
        with settings_cols[1]:  # Method
            method_key = f"stationarity_method_radio_{data_name}"
            method_options = ["无处理", "一阶差分", "对数差分", "对数+一阶差分"]
            selected_method_idx = st_obj.radio(
                "平稳化处理方法:",
                options=list(range(len(method_options))),
                format_func=lambda x: method_options[x],
                index=0,
                horizontal=False,
                key=method_key,
                label_visibility="visible"
            )
            selected_method_label = method_options[selected_method_idx]
            
            # 映射到后端方法名
            method_mapping = {
                0: "none",
                1: "diff", 
                2: "log_diff",
                3: "log_then_diff"
            }
            processing_method = method_mapping[selected_method_idx]
        
        with settings_cols[2]:  # Order
            order_key = f"stationarity_order_number_{data_name}"
            diff_order = st_obj.number_input(
                "差分阶数:",
                min_value=1,
                max_value=3,
                value=1,
                step=1,
                key=order_key,
                disabled=processing_method not in ['diff', 'log_diff'],
                label_visibility="visible"
            )
        
        return selected_alpha, processing_method, diff_order
    
    def render_analysis_results(self, st_obj, results: Dict[str, Any], processed_data_dict: Dict[str, pd.DataFrame]):
        """
        渲染分析结果
        
        Args:
            st_obj: Streamlit对象
            results: 分析结果
            processed_data_dict: 处理后的数据字典
        """
        # 修复DataFrame布尔值判断错误 - 处理DataFrame格式的结果
        if results is None or (isinstance(results, pd.DataFrame) and len(results) == 0):
            st_obj.warning("没有分析结果可显示")
            return

        # 显示检验结果
        st_obj.markdown("#### 平稳性检验结果")

        # 如果results是DataFrame，直接显示
        if isinstance(results, pd.DataFrame):
            st_obj.dataframe(results, use_container_width=True)
        else:
            # 如果是字典格式，按原来的方式处理
            result_data = []
            for var_name, var_results in results.items():
                if isinstance(var_results, dict):
                    result_data.append({
                        '变量名': var_name,
                        'ADF统计量': var_results.get('adf_statistic', 'N/A'),
                        'ADF p值': var_results.get('adf_pvalue', 'N/A'),
                        'KPSS统计量': var_results.get('kpss_statistic', 'N/A'),
                        'KPSS p值': var_results.get('kpss_pvalue', 'N/A'),
                        '平稳性结论': var_results.get('stationarity_conclusion', 'N/A')
                    })

            if result_data:
                results_df = pd.DataFrame(result_data)

                # 格式化数值列
                numeric_cols = ['ADF统计量', 'ADF p值', 'KPSS统计量', 'KPSS p值']
                for col in numeric_cols:
                    if col in results_df.columns:
                        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
                        results_df[col] = results_df[col].round(4)

                st_obj.dataframe(results_df, use_container_width=True)
        
        # 显示处理后的数据预览
        if processed_data_dict:
            st_obj.markdown("#### 处理后数据预览")

            # 获取第一个DataFrame（通常是'processed_data'）
            first_key = next(iter(processed_data_dict.keys()))
            main_df = processed_data_dict[first_key]

            if isinstance(main_df, pd.DataFrame) and len(main_df) > 0:
                # 智能识别时间列
                available_columns = main_df.columns.tolist()

                # 智能识别时间列
                time_columns = []
                for col in available_columns:
                    # 检查列名是否包含时间相关关键词
                    if any(keyword in col.lower() for keyword in ['时间', 'date', 'time', '日期', 'datetime', 'timestamp']):
                        time_columns.append(col)
                        break  # 找到第一个时间列就停止
                    # 如果没有找到明显的时间列名，检查第一列的内容
                    elif col == available_columns[0]:
                        try:
                            # 尝试转换第一列为时间格式
                            sample_data = main_df[col].dropna().head(5)
                            if len(sample_data) > 0:
                                pd.to_datetime(sample_data, errors='raise')
                                time_columns.append(col)
                                break
                        except:
                            pass

                # 如果还是没有找到时间列，使用第一列作为备选
                if not time_columns and available_columns:
                    time_columns = [available_columns[0]]

                # 其余列中的数值列作为可选变量
                numeric_columns = []
                for col in available_columns:
                    if col not in time_columns and pd.api.types.is_numeric_dtype(main_df[col]):
                        numeric_columns.append(col)

                if numeric_columns:
                    # 使用多选框选择变量
                    selected_columns = st_obj.multiselect(
                        "选择变量查看处理后数据:",
                        options=numeric_columns,
                        default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns,  # 默认选择前3个
                        key=f"stationarity_preview_var_selector"
                    )

                    if selected_columns:
                        # 构建要显示的数据框（确保时间列在第一列）
                        if time_columns:
                            # 时间列 + 选中的变量列
                            display_columns = time_columns + selected_columns
                            display_df = main_df[display_columns]
                        else:
                            # 如果没有时间列，只显示选中的变量
                            display_df = main_df[selected_columns]

                        # 显示数据信息
                        st_obj.info(f"显示数据形状: {display_df.shape} | 时间戳列(第一列): {time_columns[0] if time_columns else '无'}")

                        # 显示选中变量的数据
                        st_obj.dataframe(display_df.head(20), use_container_width=True)

                        # 生成带时间戳的文件名
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        selected_vars_str = "_".join(selected_columns[:3])  # 最多显示前3个变量名
                        if len(selected_columns) > 3:
                            selected_vars_str += f"_等{len(selected_columns)}个变量"

                        # 准备下载数据（确保包含时间列和选中变量）
                        if time_columns:
                            # 如果有时间列，确保包含在下载数据中（时间列在第一列）
                            download_columns = time_columns + selected_columns
                            download_df = main_df[download_columns]
                            include_index = False  # 时间列已经在数据中，不需要index
                        else:
                            # 如果没有时间列，只包含选中的变量
                            download_df = main_df[selected_columns]
                            include_index = False

                        # 使用utf-8-sig编码避免乱码，直接返回字符串
                        csv_data = download_df.to_csv(index=include_index, encoding='utf-8-sig')

                        # 提供下载功能
                        st_obj.download_button(
                            label=f"下载选中变量数据 ({len(selected_columns)}个变量)",
                            data=csv_data.encode('utf-8-sig'),  # 使用utf-8-sig编码保持一致性
                            file_name=f"平稳性分析_{selected_vars_str}_{timestamp}.csv",
                            mime="text/csv",
                            key=f"download_selected_vars_{timestamp}"
                        )
                    else:
                        st_obj.info("请选择至少一个变量来查看数据")
                else:
                    st_obj.warning("处理后的数据没有可用的数值列")
            else:
                st_obj.warning("处理后的数据为空或格式无效")
    
    def render_analysis_interface(self, st_obj, data: pd.DataFrame, data_name: str) -> Any:
        """
        渲染平稳性分析界面
        
        Args:
            st_obj: Streamlit对象
            data: 分析数据
            data_name: 数据名称
            
        Returns:
            Any: 分析结果
        """
        try:
            # 渲染参数设置
            selected_alpha, processing_method, diff_order = self.render_analysis_parameters(st_obj, data_name)
            
            # 分析按钮
            analysis_key = f"stationarity_analyze_btn_{data_name}"
            if st_obj.button("开始平稳性分析", key=analysis_key, type="primary"):
                
                with st_obj.spinner(f"正在执行平稳性检验与处理 (方法: {processing_method}, Alpha: {selected_alpha}, 阶数: {diff_order if processing_method in ['diff', 'log_diff'] else 'N/A'})..."):
                    # 调用后端函数 - 修复数据格式处理
                    summary_df, final_processed_df = test_and_process_stationarity(
                        data.copy(),
                        alpha=selected_alpha,
                        processing_method=processing_method,
                        diff_order=diff_order
                    )

                    # 转换为UI期望的格式
                    results = summary_df  # 结果表格
                    processed_data_dict = {'processed_data': final_processed_df}  # 处理后的数据

                    # 保存结果到状态
                    self.set_state('results', results)
                    self.set_state('processed_data', processed_data_dict)
                    self.set_state('parameters', {
                        'alpha': selected_alpha,
                        'processing_method': processing_method,
                        'diff_order': diff_order
                    })

                    st_obj.success("平稳性分析完成！")

                    # 显示结果
                    self.render_analysis_results(st_obj, results, processed_data_dict)

                    return {
                        'results': results,
                        'processed_data': processed_data_dict,
                        'parameters': {
                            'alpha': selected_alpha,
                            'processing_method': processing_method,
                            'diff_order': diff_order
                        }
                    }
            
            # 显示之前的分析结果（如果有）
            previous_results = self.get_state('results')
            previous_processed_data = self.get_state('processed_data')
            
            # 修复DataFrame布尔值判断错误 - 支持DataFrame和dict格式
            if previous_results is not None and (
                (isinstance(previous_results, pd.DataFrame) and len(previous_results) > 0) or
                (isinstance(previous_results, dict) and len(previous_results) > 0)
            ):
                st_obj.markdown("---")
                st_obj.markdown("#### 上次分析结果")
                self.render_analysis_results(st_obj, previous_results, previous_processed_data)
                
                return {
                    'results': previous_results,
                    'processed_data': previous_processed_data,
                    'parameters': self.get_state('parameters')
                }
            
            return None
            
        except Exception as e:
            self.handle_error(st_obj, e, "渲染平稳性分析界面")
            return None


__all__ = ['StationarityAnalysisComponent']
