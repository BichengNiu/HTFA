# -*- coding: utf-8 -*-
"""
平稳性分析模块

重构自stationarity_backend.py，将超长函数拆分为多个职责明确的小函数
"""

import logging
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.tools import diff as statespace_diff
import warnings

from dashboard.explore.core.constants import MIN_SAMPLES_ADF
from dashboard.explore.core.series_utils import prepare_time_index

logger = logging.getLogger(__name__)


def run_adf_test(series: pd.Series, alpha: float = 0.05) -> Tuple[Optional[float], str]:
    """
    执行ADF平稳性检验

    Args:
        series: 输入序列
        alpha: 显著性水平

    Returns:
        Tuple[p值, 平稳性状态]
    """
    series_cleaned = series.dropna()

    if series_cleaned.empty or len(series_cleaned) < MIN_SAMPLES_ADF:
        logger.warning(f"序列 '{series.name}' 样本数不足，无法执行ADF检验")
        return None, '数据不足'

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(series_cleaned, regression='ct')

        p_value = result[1]
        is_stationary = '是' if p_value < alpha else '否'
        logger.debug(f"ADF检验: p={p_value:.4f}, 平稳={is_stationary}")

        return p_value, is_stationary

    except Exception as e:
        logger.error(f"ADF检验失败: {e}")

        # 提供更具体的错误信息
        if "Ensure that you have included enough lag variables" in str(e):
            return None, '计算失败(滞后阶数不足)'
        elif "degrees of freedom" in str(e):
            return None, '计算失败(自由度不足)'
        else:
            return None, f'计算失败({type(e).__name__})'


def run_kpss_test(residuals: pd.Series, alpha: float = 0.05) -> Tuple[Optional[float], str]:
    """
    对残差执行KPSS平稳性检验

    H0: 序列围绕趋势平稳
    HA: 序列有单位根（非平稳）

    Args:
        residuals: 残差序列
        alpha: 显著性水平

    Returns:
        Tuple[p值, 平稳性状态]
    """
    if residuals is None or residuals.empty:
        return None, "无残差数据"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, p_value, lags, crit = kpss(residuals.dropna(), regression='ct')

        is_stationary = '是' if p_value > alpha else '否'
        logger.debug(f"KPSS检验: p={p_value:.4f}, 平稳={is_stationary}")

        return p_value, is_stationary

    except Exception as e:
        logger.error(f"KPSS检验失败: {e}")
        return None, f'计算失败({type(e).__name__})'


def apply_differencing(
    series: pd.Series,
    order: int = 1,
    use_log: bool = False
) -> Tuple[Optional[pd.Series], str]:
    """
    对序列应用差分处理

    Args:
        series: 输入序列
        order: 差分阶数
        use_log: 是否先取对数

    Returns:
        Tuple[差分后的序列, 方法描述]
    """
    try:
        temp_series = series.copy()
        method_desc = f"{order}阶差分" if not use_log else f"{order}阶对数差分"

        # 对数转换
        if use_log:
            temp_series_clean = temp_series.dropna()
            if len(temp_series_clean) == 0 or (temp_series_clean <= 0).any():
                logger.warning(f"序列 '{series.name}' 包含非正值，无法应用对数转换")
                return None, method_desc + " (无法应用对数)"

            temp_series = np.log(temp_series)

        # 差分
        diff_series = statespace_diff(temp_series, k_diff=order)

        logger.info(f"应用 {method_desc} 成功")
        return diff_series, method_desc

    except Exception as e:
        logger.error(f"差分处理失败: {e}")
        return None, method_desc + " (处理失败)"


def process_nonstationary_series(
    series: pd.Series,
    processing_method: str,
    diff_order: int,
    alpha: float
) -> Dict[str, Any]:
    """
    处理非平稳序列

    Args:
        series: 非平稳序列
        processing_method: 处理方法 ('diff', 'log_diff', 'keep')
        diff_order: 差分阶数
        alpha: ADF检验显著性水平

    Returns:
        处理结果字典
    """
    result = {
        'processed_series': None,
        'new_column_name': None,
        'method_description': '原始序列',
        'p_value': None,
        'is_stationary': '否'
    }

    if processing_method == 'keep':
        result['method_description'] = '保留原始 (非平稳)'
        return result

    # 应用差分
    use_log = (processing_method == 'log_diff')
    diff_series, method_desc = apply_differencing(series, diff_order, use_log)

    if diff_series is None:
        result['method_description'] = method_desc
        return result

    # 检验差分后的平稳性
    p_value, is_stationary = run_adf_test(diff_series, alpha)

    result.update({
        'method_description': method_desc,
        'p_value': p_value,
        'is_stationary': is_stationary
    })

    if is_stationary == '是':
        # 差分成功使序列平稳
        suffix = f"_log_diff{diff_order}" if use_log else f"_diff{diff_order}"
        result['processed_series'] = diff_series
        result['new_column_name'] = f"{series.name}{suffix}"
        logger.info(f"序列 '{series.name}' 经 {method_desc} 后平稳")

    return result


def test_and_process_stationarity(
    df_in: pd.DataFrame,
    alpha: float = 0.05,
    processing_method: str = 'keep',
    diff_order: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    测试并处理序列平稳性

    重构版本：将原260行函数拆分为多个子函数

    Args:
        df_in: 输入DataFrame
        alpha: 显著性水平
        processing_method: 处理方法 ('keep', 'diff', 'log_diff')
        diff_order: 差分阶数

    Returns:
        Tuple[摘要DataFrame, 处理后的数据DataFrame]
    """
    logger.info(f"开始平稳性检验: shape={df_in.shape}, method={processing_method}, order={diff_order}")

    df = df_in.copy()
    summary_results = []
    processed_data = {}

    # 1. 准备时间索引
    df_work, time_column = prepare_time_index(df, time_column=None, set_as_index=True, keep_column=True)

    # 记录时间列信息
    if time_column:
        summary_results.append({
            '指标名称': time_column,
            '原始P值': None,
            '原始是否平稳': '时间戳列',
            '处理方法': '保留',
            '处理后P值': None,
            '最终是否平稳': '时间戳列'
        })
        processed_data[time_column] = df_work[time_column] if time_column in df_work.columns else df_work.index
        logger.info(f"时间列: '{time_column}'")

    # 2. 处理每个数值列
    for col_name in df_work.columns:
        # 跳过已处理的时间列
        if col_name == time_column:
            continue

        if not pd.api.types.is_numeric_dtype(df_work[col_name]):
            logger.debug(f"跳过非数值列: '{col_name}'")
            continue

        logger.info(f"\n处理列: '{col_name}'")
        series = df_work[col_name]

        # 保留原始数值序列
        processed_data[col_name] = series.copy()

        # 初始ADF检验
        original_p_value, original_status = run_adf_test(series, alpha)
        logger.info(f"  初始ADF: p={original_p_value}, 平稳={original_status}")

        # 准备摘要条目
        summary_entry = {
            '指标名称': col_name,
            '原始P值': original_p_value,
            '原始是否平稳': original_status,
            '处理方法': '原始序列',
            '处理后P值': None,
            '最终是否平稳': original_status
        }

        # 如果非平稳且需要处理
        if original_status == '否' and processing_method != 'keep':
            process_result = process_nonstationary_series(
                series, processing_method, diff_order, alpha
            )

            summary_entry['处理方法'] = process_result['method_description']
            summary_entry['处理后P值'] = process_result['p_value']
            summary_entry['最终是否平稳'] = process_result['is_stationary']

            # 添加新的处理后序列
            if process_result['processed_series'] is not None:
                new_col_name = process_result['new_column_name']
                processed_data[new_col_name] = process_result['processed_series']
                logger.info(f"  生成新列: '{new_col_name}'")

        elif original_status == '否' and processing_method == 'keep':
            summary_entry['处理方法'] = '保留原始 (非平稳)'

        summary_results.append(summary_entry)

    # 3. 组装结果
    summary_df = pd.DataFrame(summary_results)

    # 将时间列排在第一行
    if time_column:
        is_time_col = summary_df['指标名称'] == time_column
        summary_df = pd.concat([summary_df[is_time_col], summary_df[~is_time_col]], ignore_index=True)

    # 创建最终DataFrame
    if time_column and time_column in processed_data:
        # 时间列在第一列
        ordered_columns = [time_column] + [col for col in processed_data.keys() if col != time_column]
        final_df = pd.DataFrame({col: processed_data[col] for col in ordered_columns})
    else:
        final_df = pd.DataFrame(processed_data)

    # 设置时间索引
    if time_column and time_column in final_df.columns:
        try:
            if isinstance(df_work.index, pd.DatetimeIndex):
                final_df.index = df_work.index
            logger.info(f"已设置时间索引")
        except Exception as e:
            logger.warning(f"设置时间索引失败: {e}")

    logger.info(f"完成: 摘要 {summary_df.shape}, 数据 {final_df.shape}")

    return summary_df, final_df
