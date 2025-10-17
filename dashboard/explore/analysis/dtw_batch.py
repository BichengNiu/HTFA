# -*- coding: utf-8 -*-
"""
DTW批量分析模块

重构自根目录的dtw_backend.py，整合到explore模块架构中
增强功能：支持频率对齐和数据标准化
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional

from dashboard.explore.metrics.dtw import calculate_dtw_distance, calculate_dtw_path
from dashboard.explore.preprocessing.frequency_alignment import align_multiple_series_frequencies
from dashboard.explore.preprocessing.standardization import standardize_series
from dashboard.explore.core.validation import validate_series
from dashboard.explore.core.constants import ERROR_MESSAGES

logger = logging.getLogger(__name__)


def perform_batch_dtw_calculation(
    df_input: pd.DataFrame,
    target_series_name: str,
    comparison_series_names: List[str],
    window_type_param: str,
    window_size_param: Optional[int],
    dist_metric_name_param: str,
    dist_metric_display_param: str,
    enable_freq_alignment: bool = True,
    freq_alignment_mode: str = 'stat_align',
    freq_agg_method: str = 'mean',
    strict_alignment: bool = True,
    standardization_method: str = 'zscore'
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[str], List[str]]:
    """
    执行批量DTW分析（增强版）

    重构自原dtw_backend.py，增加频率对齐和标准化功能

    Args:
        df_input: 输入DataFrame
        target_series_name: 目标序列名称
        comparison_series_names: 对比序列名称列表
        window_type_param: 窗口类型 ("无限制" 或 "固定大小窗口 (Radius约束)")
        window_size_param: 窗口大小（固定窗口时使用）
        dist_metric_name_param: 距离度量内部名称 ("euclidean", "manhattan", "sqeuclidean")
        dist_metric_display_param: 距离度量显示名称
        enable_freq_alignment: 是否启用频率对齐
        freq_alignment_mode: 频率对齐模式 ('stat_align' 或 'value_align')
        freq_agg_method: 聚合方法 ('mean', 'last', 'first', 'sum', 'median')
        strict_alignment: 是否严格对齐（成对删除NA值）
        standardization_method: 标准化方法 ('zscore', 'minmax', 'none')

    Returns:
        Tuple[结果列表, 路径字典, 错误列表, 警告列表]
    """
    df = df_input.copy()
    results_list = []
    paths_dict = {}
    error_messages = []
    warning_messages = []

    # 输入验证
    if not target_series_name:
        error_messages.append("目标变量未选择")
        return results_list, paths_dict, error_messages, warning_messages

    if not comparison_series_names:
        error_messages.append("对比变量未选择")
        return results_list, paths_dict, error_messages, warning_messages

    if target_series_name not in df.columns:
        error_messages.append(f"目标序列 '{target_series_name}' 不存在于数据中")
        return results_list, paths_dict, error_messages, warning_messages

    # 频率对齐（如果启用）- 修复：必须在验证目标序列之前执行
    if enable_freq_alignment:
        logger.info(f"执行频率对齐: 模式={freq_alignment_mode}, 聚合方法={freq_agg_method}")
        try:
            # 构建包含目标序列和所有对比序列的DataFrame
            all_series_names = [target_series_name] + comparison_series_names

            # 简化逻辑：数据第一列固定是Date(日期格式)
            if isinstance(df.index, pd.DatetimeIndex):
                # 如果已经是DatetimeIndex，直接使用
                df_for_alignment = df[all_series_names].copy()
            elif 'Date' in df.columns:
                # 数据第一列是Date，将其设为索引
                df_for_alignment = df[['Date'] + all_series_names].copy()
                df_for_alignment['Date'] = pd.to_datetime(df_for_alignment['Date'])
                df_for_alignment = df_for_alignment.set_index('Date')
                logger.info("[频率对齐] 使用Date列作为时间索引")
            else:
                # 找不到Date列，跳过对齐
                df_for_alignment = None
                warning_messages.append("数据中没有Date列，频率对齐已跳过")
                logger.warning("[频率对齐] 未找到Date列")


            if df_for_alignment is not None:
                logger.info(f"[频率对齐] 调用align_multiple_series_frequencies，输入形状: {df_for_alignment.shape}")

                # 执行频率对齐
                df_aligned = align_multiple_series_frequencies(
                    df_for_alignment,
                    mode=freq_alignment_mode,
                    agg_method=freq_agg_method
                )

                # 更新数据（不对所有列一起dropna，而是在计算每对DTW时单独处理）
                df = df_aligned
                logger.info(f"[频率对齐] 完成: 数据形状 {df.shape}")
            else:
                logger.warning("[频率对齐] 跳过对齐，继续使用原始数据")

        except Exception as e:
            warning_messages.append(f"频率对齐失败: {str(e)}，使用原始数据")
            logger.warning(f"[频率对齐] 异常: {e}")

    # 验证并清洗目标序列（在频率对齐之后）
    # 注意：不在这里标准化，因为需要针对每对序列单独对齐后再标准化
    target_series = df[target_series_name]
    target_validation = validate_series(target_series, min_samples=10, series_name=target_series_name)

    if not target_validation.is_valid:
        error_messages.append(f"目标序列 '{target_series_name}' 无效: {target_validation.error_message}")
        return results_list, paths_dict, error_messages, warning_messages

    # 准备目标序列数据（保留原始数据，每对序列计算时单独处理）
    target_data_clean = target_validation.cleaned_data
    logger.info(f"目标序列准备完成: 长度={len(target_data_clean)}")

    # 批量处理对比序列
    for compare_name in comparison_series_names:
        current_result = {
            '目标变量': target_series_name,
            '对比变量': compare_name,
            'DTW距离': np.nan,
            '原因': '-',
            '窗口类型': window_type_param,
            '窗口大小': window_size_param if window_type_param == "固定大小窗口 (Radius约束)" else 'N/A',
            '距离度量': dist_metric_display_param
        }

        # 验证对比序列
        if compare_name not in df.columns:
            warning_messages.append(f"对比序列 '{compare_name}' 不存在，已跳过")
            current_result['原因'] = '序列不存在'
            results_list.append(current_result)
            continue

        compare_series = df[compare_name]
        compare_validation = validate_series(compare_series, min_samples=10, series_name=compare_name)

        if not compare_validation.is_valid:
            warning_messages.append(f"对比序列 '{compare_name}' 无效: {compare_validation.error_message}")
            current_result['原因'] = '对比序列无效'
            results_list.append(current_result)
            continue

        # 准备对比序列数据
        compare_data_clean = compare_validation.cleaned_data

        # 严格对齐模式：成对删除NA值
        if strict_alignment:
            # 创建包含目标序列和对比序列的DataFrame
            pair_df = pd.DataFrame({
                'target': target_data_clean,
                'compare': compare_data_clean
            })

            # 同时删除两列中任一有NA的行
            pair_df_clean = pair_df.dropna()

            # 提取清洗后的序列
            target_data_for_dtw = pair_df_clean['target']
            compare_data_for_dtw = pair_df_clean['compare']

            logger.debug(f"[严格对齐] {target_series_name} vs {compare_name}: "
                        f"对齐前=({len(target_data_clean)}, {len(compare_data_clean)}), "
                        f"对齐后={len(pair_df_clean)}")
        else:
            # 非严格对齐模式：DTW可以比较不同长度的序列，每个序列分别去除NaN
            target_data_for_dtw = target_data_clean.dropna()
            compare_data_for_dtw = compare_data_clean.dropna()

            logger.debug(f"[非严格对齐] {target_series_name} vs {compare_name}: "
                        f"序列长度=({len(target_data_for_dtw)}, {len(compare_data_for_dtw)})")

        # 检查序列是否有足够的样本
        if len(target_data_for_dtw) < 10:
            warning_messages.append(f"目标序列 '{target_series_name}' 有效样本不足10个 ({len(target_data_for_dtw)})")
            current_result['原因'] = f'目标序列样本不足: {len(target_data_for_dtw)}'
            results_list.append(current_result)
            continue

        if len(compare_data_for_dtw) < 10:
            warning_messages.append(f"对比序列 '{compare_name}' 有效样本不足10个 ({len(compare_data_for_dtw)})")
            current_result['原因'] = f'对比序列样本不足: {len(compare_data_for_dtw)}'
            results_list.append(current_result)
            continue

        # 标准化（各自独立标准化）
        if standardization_method and standardization_method != 'none':
            try:
                target_data_for_dtw = standardize_series(target_data_for_dtw, method=standardization_method)
                compare_data_for_dtw = standardize_series(compare_data_for_dtw, method=standardization_method)
            except Exception as e:
                warning_messages.append(f"序列对 '{target_series_name}' vs '{compare_name}' 标准化失败: {str(e)}")

        target_np = target_data_for_dtw.to_numpy()
        compare_np = compare_data_for_dtw.to_numpy()

        # 计算DTW距离
        try:
            # 确定窗口约束
            if window_type_param == "固定大小窗口 (Radius约束)":
                if window_size_param is not None and window_size_param > 0:
                    radius = int(window_size_param)
                else:
                    radius = 10
                    warning_messages.append(f"窗口大小参数无效 ({window_size_param})，已默认为10")

                logger.debug(f"[DTW计算] 启用窗口约束: radius={radius}")
            else:  # "无限制"
                radius = None  # 修复: 无约束时radius应为None,而非序列最大长度
                logger.debug(f"[DTW计算] 无窗口约束模式")

            # 调用DTW计算（使用explore模块的metrics.dtw）
            distance, path = calculate_dtw_path(
                target_np,
                compare_np,
                radius=radius,
                dist_metric=dist_metric_name_param
            )

            current_result['DTW距离'] = distance
            paths_dict[compare_name] = {
                'target_np': target_np,
                'compare_np': compare_np,
                'path': path
            }

            logger.debug(f"DTW计算成功: {target_series_name} vs {compare_name}, 距离={distance:.4f}")

        except Exception as e:
            error_msg = f"计算 '{target_series_name}' vs '{compare_name}' DTW时出错: {str(e)[:200]}"
            error_messages.append(error_msg)
            current_result['原因'] = f'计算错误: {str(e)[:100]}'
            logger.error(error_msg)

        results_list.append(current_result)

    logger.info(f"批量DTW分析完成: {len(results_list)} 个对比序列")
    return results_list, paths_dict, error_messages, warning_messages
