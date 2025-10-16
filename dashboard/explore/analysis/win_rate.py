# -*- coding: utf-8 -*-
"""
胜率分析模块

从win_rate_backend.py重构而来，使用统一的core模块简化代码
"""

import logging
from typing import Tuple, List, Callable
from collections import defaultdict
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

from dashboard.explore.core.validation import validate_series
from dashboard.explore.core.constants import MIN_SAMPLES_WIN_RATE, ERROR_MESSAGES

logger = logging.getLogger(__name__)


def calculate_single_win_rate(
    target_series: pd.Series,
    ref_series: pd.Series
) -> Tuple[float | str, str]:
    """
    计算单个参考序列相对于目标序列的胜率

    Args:
        target_series: 目标序列
        ref_series: 参考序列

    Returns:
        Tuple[胜率值或"N/A", 备注信息]
    """
    # 验证序列（使用统一的错误消息）
    target_result = validate_series(target_series, min_samples=MIN_SAMPLES_WIN_RATE)
    if not target_result.is_valid:
        return "N/A", ERROR_MESSAGES['target_series_invalid']

    ref_result = validate_series(ref_series, min_samples=MIN_SAMPLES_WIN_RATE)
    if not ref_result.is_valid:
        return "N/A", ERROR_MESSAGES['ref_series_invalid']

    # 计算变化并对齐（优化：一次性完成对齐和清理）
    try:
        target_diff = target_result.cleaned_data.diff().iloc[1:]
        ref_diff = ref_result.cleaned_data.diff().iloc[1:]
    except Exception as e:
        logger.error(f"计算diff时出错: {e}")
        return "N/A", f"计算变化量时出错"

    if target_diff.empty or ref_diff.empty:
        return "N/A", "计算变化量后序列为空"

    # 优化：使用pandas的align方法一次性完成对齐和NaN处理
    aligned_target_diff, aligned_ref_diff = target_diff.align(ref_diff, join='inner')

    # 移除NaN值
    valid_mask = aligned_target_diff.notna() & aligned_ref_diff.notna()
    aligned_target_diff = aligned_target_diff[valid_mask]
    aligned_ref_diff = aligned_ref_diff[valid_mask]

    if aligned_target_diff.empty:
        return "N/A", ERROR_MESSAGES['no_common_data']

    # 计算胜率
    target_changed_mask = (aligned_target_diff != 0)
    num_target_changes = target_changed_mask.sum()

    if num_target_changes == 0:
        return f"N/A ({ERROR_MESSAGES['no_target_change']})", f"基于 {len(aligned_target_diff)} 个共同周期, {ERROR_MESSAGES['no_target_change']}"

    target_diff_when_changed = aligned_target_diff[target_changed_mask]
    ref_diff_when_target_changed = aligned_ref_diff[target_changed_mask]

    # 计算同方向的比例
    same_direction = np.logical_or(
        np.logical_and(target_diff_when_changed > 0, ref_diff_when_target_changed > 0),
        np.logical_and(target_diff_when_changed < 0, ref_diff_when_target_changed < 0)
    )

    win_rate_val = (same_direction.sum() / num_target_changes) * 100
    remark = f"基于 {num_target_changes} 个目标变化周期 (共 {len(aligned_target_diff)} 周期)"

    logger.debug(f"胜率计算完成: {win_rate_val:.2f}%, {remark}")
    return win_rate_val, remark


def filter_data_by_time_range(
    df: pd.DataFrame,
    time_range: str,
    get_current_time_func: Callable
) -> pd.DataFrame:
    """
    根据时间范围筛选数据

    Args:
        df: 输入DataFrame（必须有DatetimeIndex）
        time_range: 时间范围 ('全部时间', '近半年', '近1年', '近3年')
        get_current_time_func: 获取当前时间的函数

    Returns:
        筛选后的DataFrame
    """
    if time_range == "全部时间":
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame索引不是DatetimeIndex，无法按时间筛选")
        return df

    current_time = get_current_time_func()

    time_delta_map = {
        "近半年": relativedelta(months=6),
        "近1年": relativedelta(years=1),
        "近3年": relativedelta(years=3)
    }

    delta = time_delta_map.get(time_range)
    if delta is None:
        logger.warning(f"未知的时间范围: {time_range}")
        return df

    start_date = current_time - delta
    filtered_df = df[df.index >= start_date]

    logger.debug(f"时间筛选: {time_range}, 起始日期: {start_date}, 结果行数: {len(filtered_df)}")
    return filtered_df


def perform_batch_win_rate_calculation(
    df_input: pd.DataFrame,
    target_series_name: str,
    ref_series_names_list: List[str],
    selected_time_ranges: List[str],
    is_datetime_index_available: bool,
    get_current_time_for_filter: Callable
) -> Tuple[defaultdict, List[str], List[str]]:
    """
    执行批量胜率计算

    重构版本：使用core模块简化代码，优化性能
    性能优化：预先筛选所有时间范围，避免重复操作

    Args:
        df_input: 输入DataFrame
        target_series_name: 目标序列名称
        ref_series_names_list: 参考序列名称列表
        selected_time_ranges: 选择的时间范围列表
        is_datetime_index_available: 是否有DatetimeIndex
        get_current_time_for_filter: 获取当前时间的函数

    Returns:
        Tuple[结果累加器, 错误消息列表, 警告消息列表]
    """
    df_original = df_input.copy()
    results_accumulator = defaultdict(dict)
    error_messages = []
    warning_messages = []

    # 输入验证
    if not target_series_name:
        error_messages.append("目标序列未选择")
        return results_accumulator, error_messages, warning_messages

    if not ref_series_names_list:
        warning_messages.append("没有选择任何参考序列")
        return results_accumulator, error_messages, warning_messages

    if not selected_time_ranges:
        warning_messages.append("没有选择任何时间范围")
        return results_accumulator, error_messages, warning_messages

    # 验证目标序列
    if target_series_name not in df_original.columns:
        error_messages.append(f"目标序列 '{target_series_name}' 在数据中未找到")
        return results_accumulator, error_messages, warning_messages

    target_result = validate_series(
        df_original[target_series_name],
        min_samples=MIN_SAMPLES_WIN_RATE,
        series_name=target_series_name
    )

    if not target_result.is_valid:
        warning_messages.append(f"目标序列验证失败: {target_result.error_message}")

    # 性能优化：预先筛选所有时间范围（只执行一次）
    time_range_dfs = {}
    for time_range in selected_time_ranges:
        if time_range != "全部时间" and is_datetime_index_available:
            try:
                df_for_range = filter_data_by_time_range(
                    df_original,
                    time_range,
                    get_current_time_for_filter
                )
                time_range_dfs[time_range] = df_for_range

                if df_for_range.empty:
                    logger.warning(f"时间范围 '{time_range}' 筛选后无数据")
            except Exception as e:
                logger.error(f"时间范围 '{time_range}' 筛选失败: {e}")
                time_range_dfs[time_range] = None
                warning_messages.append(f"无法应用时间范围 '{time_range}': {str(e)}")
        else:
            time_range_dfs[time_range] = df_original

    # 批量处理参考序列
    for ref_s_name in ref_series_names_list:
        # 验证参考序列
        if ref_s_name not in df_original.columns:
            warning_messages.append(f"参考序列 '{ref_s_name}' 在数据中未找到，已跳过")
            for time_range_key in selected_time_ranges:
                results_accumulator[ref_s_name][time_range_key] = "N/A (序列不存在)"
            continue

        ref_result = validate_series(
            df_original[ref_s_name],
            min_samples=MIN_SAMPLES_WIN_RATE,
            series_name=ref_s_name
        )

        if not ref_result.is_valid:
            warning_messages.append(f"参考序列 '{ref_s_name}' 验证失败: {ref_result.error_message}")
            for time_range_key in selected_time_ranges:
                results_accumulator[ref_s_name][time_range_key] = "N/A (参考序列数据不足)"
            continue

        # 处理每个时间范围（直接使用预先筛选的结果）
        for time_range in selected_time_ranges:
            df_for_range = time_range_dfs.get(time_range)

            # 处理筛选失败或无数据的情况
            if df_for_range is None:
                results_accumulator[ref_s_name][time_range] = f"N/A (时间筛选失败)"
                continue

            if df_for_range.empty:
                results_accumulator[ref_s_name][time_range] = f"N/A (无数据 @ {time_range})"
                continue

            # 计算胜率
            current_target_data = df_for_range.get(target_series_name, pd.Series(dtype=float))
            current_ref_data = df_for_range.get(ref_s_name, pd.Series(dtype=float))

            win_rate_val, remark = calculate_single_win_rate(current_target_data, current_ref_data)

            # 格式化显示值
            if isinstance(win_rate_val, float):
                final_display_value = f"{win_rate_val:.2f}% ({remark})"
            else:
                # win_rate_val是字符串（如"N/A"）
                if remark and remark not in win_rate_val:
                    final_display_value = f"{win_rate_val} ({remark})"
                else:
                    final_display_value = str(win_rate_val)

            results_accumulator[ref_s_name][time_range] = final_display_value

    logger.info(f"批量胜率计算完成: {len(ref_series_names_list)} 个参考序列, {len(selected_time_ranges)} 个时间范围")
    return results_accumulator, error_messages, warning_messages
