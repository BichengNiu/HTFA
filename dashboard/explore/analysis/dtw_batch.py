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

    # 验证并清洗目标序列
    target_series = df[target_series_name]
    target_validation = validate_series(target_series, min_samples=10, series_name=target_series_name)

    if not target_validation.is_valid:
        error_messages.append(f"目标序列 '{target_series_name}' 无效: {target_validation.error_message}")
        return results_list, paths_dict, error_messages, warning_messages

    # 准备目标序列数据
    target_data_clean = target_validation.cleaned_data

    # 频率对齐（如果启用）
    if enable_freq_alignment:
        logger.info(f"执行频率对齐: 模式={freq_alignment_mode}, 聚合方法={freq_agg_method}")
        try:
            # 构建包含目标序列和所有对比序列的DataFrame
            all_series_names = [target_series_name] + comparison_series_names

            # 修复：检查df是否已经有DatetimeIndex
            if isinstance(df.index, pd.DatetimeIndex):
                # 如果已经是DatetimeIndex，直接使用
                logger.info("[频率对齐] 输入数据已有DatetimeIndex")
                df_for_alignment = df[all_series_names].copy()
            else:
                # 如果不是DatetimeIndex，需要保留时间列
                logger.info("[频率对齐] 输入数据没有DatetimeIndex，尝试查找时间列")
                time_col = None

                # 方法1: 检查已经是datetime类型的列
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        time_col = col
                        logger.info(f"[频率对齐] 找到datetime类型时间列: {time_col}")
                        break

                # 方法2: 如果方法1失败，尝试找能转换为datetime的字符串列
                if time_col is None:
                    for col in df.columns:
                        try:
                            pd.to_datetime(df[col], errors='raise')
                            time_col = col
                            logger.info(f"[频率对齐] 找到可转换为datetime的列: {time_col}")
                            break
                        except:
                            continue

                if time_col:
                    # 包含时间列和数值列
                    df_for_alignment = df[[time_col] + all_series_names].copy()
                    # 确保时间列是datetime类型
                    if not pd.api.types.is_datetime64_any_dtype(df_for_alignment[time_col]):
                        df_for_alignment[time_col] = pd.to_datetime(df_for_alignment[time_col])
                        logger.info("[频率对齐] 已将时间列转换为datetime类型")
                    # 设置时间列为索引
                    df_for_alignment = df_for_alignment.set_index(time_col)
                    logger.info("[频率对齐] 已设置时间列为索引")
                else:
                    # 如果找不到时间列，记录警告并跳过对齐
                    warning_messages.append("无法找到时间列，频率对齐已跳过")
                    logger.warning("[频率对齐] 失败: 无法找到时间列")
                    df_for_alignment = None

            if df_for_alignment is not None:
                logger.info(f"[频率对齐] 调用align_multiple_series_frequencies，输入形状: {df_for_alignment.shape}")

                # 执行频率对齐
                df_aligned = align_multiple_series_frequencies(
                    df_for_alignment,
                    mode=freq_alignment_mode,
                    agg_method=freq_agg_method
                )

                # 更新数据
                df = df_aligned
                target_data_clean = df[target_series_name].dropna()

                logger.info(f"[频率对齐] 完成: 数据形状 {df.shape}")
                logger.info(f"[频率对齐] 对齐后目标序列长度: {len(target_data_clean)}")
            else:
                logger.warning("[频率对齐] 跳过对齐，继续使用原始数据")

        except Exception as e:
            warning_messages.append(f"频率对齐失败: {str(e)}，使用原始数据")
            logger.warning(f"[频率对齐] 异常: {e}")

    # 标准化目标序列（如果需要）
    if standardization_method and standardization_method != 'none':
        try:
            target_data_clean = standardize_series(target_data_clean, method=standardization_method)
            logger.info(f"目标序列已标准化: 方法={standardization_method}")
        except Exception as e:
            warning_messages.append(f"目标序列标准化失败: {str(e)}")
            logger.warning(f"标准化失败: {e}")

    target_np = target_data_clean.to_numpy()

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

        # 标准化对比序列（如果需要）
        if standardization_method and standardization_method != 'none':
            try:
                compare_data_clean = standardize_series(compare_data_clean, method=standardization_method)
            except Exception as e:
                warning_messages.append(f"对比序列 '{compare_name}' 标准化失败: {str(e)}")

        compare_np = compare_data_clean.to_numpy()

        # 计算DTW距离
        try:
            # 确定窗口约束
            if window_type_param == "固定大小窗口 (Radius约束)":
                if window_size_param is not None and window_size_param > 0:
                    radius = int(window_size_param)
                else:
                    radius = 10
                    warning_messages.append(f"窗口大小参数无效 ({window_size_param})，已默认为10")
            else:  # "无限制"
                radius = max(len(target_np), len(compare_np))
                radius = max(radius, 1)  # 确保至少为1

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
