# -*- coding: utf-8 -*-
"""
指标计算模块

计算模型评估指标
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dashboard.models.DFM.train.utils.logger import get_logger


logger = get_logger(__name__)


def calculate_rmse(y_true, y_pred) -> float:
    """
    统一的RMSE计算函数

    Args:
        y_true: 真实值（array-like或Series）
        y_pred: 预测值（array-like或Series）

    Returns:
        float: RMSE值，无有效数据时返回np.inf
    """
    # 转换为numpy数组
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # 移除NaN值
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not valid_mask.any():
        logger.warning("[RMSE] 无有效数据，返回np.inf")
        return np.inf

    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]

    # 使用sklearn计算RMSE
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))

    return float(rmse)


def calculate_combined_score(
    is_rmse: float,
    oos_rmse: float,
    is_hit_rate: float,
    oos_hit_rate: float
) -> Tuple[float, float]:
    """
    计算组合得分（用于变量选择）

    评分标准：
    - 仅使用验证期（样本外）RMSE作为评估标准
    - Hit Rate已弃用，不再作为评估标准

    Args:
        is_rmse: 样本内RMSE（不参与评分）
        oos_rmse: 样本外RMSE（唯一评分标准）
        is_hit_rate: 样本内命中率（已弃用）
        oos_hit_rate: 样本外命中率（已弃用）

    Returns:
        Tuple[float, float]: (0, -oos_rmse)
            - 第一个元素：固定为0（Hit Rate已弃用）
            - 第二个元素：负验证期RMSE（越大越好，即RMSE越小越好）
    """
    # 仅使用验证期RMSE作为评估标准
    validation_rmse = oos_rmse if np.isfinite(oos_rmse) else np.inf

    # Hit Rate已弃用，固定返回0
    combined_hr = 0.0

    return (combined_hr, -validation_rmse)


# ==================== 下月配对评估函数（新定义）====================

def align_next_month_weekly_data(
    nowcast_series: pd.Series,
    target_series: pd.Series
) -> pd.DataFrame:
    """对齐m月所有周的nowcast与m+1月target（用于变量筛选RMSE计算）

    Args:
        nowcast_series: 周度nowcast序列
        target_series: 月度target序列

    Returns:
        pd.DataFrame: 对齐后数据，列['month', 'week_date', 'nowcast', 'next_month_target']
    """
    # 确保索引是DatetimeIndex
    if not isinstance(nowcast_series.index, pd.DatetimeIndex):
        nowcast_series.index = pd.to_datetime(nowcast_series.index)
    if not isinstance(target_series.index, pd.DatetimeIndex):
        target_series.index = pd.to_datetime(target_series.index)

    # 参考老代码，使用DataFrame方式处理（避免Grouper错误）
    # 1. 转换为DataFrame并添加月份列
    nowcast_df = nowcast_series.to_frame('Nowcast').copy()
    nowcast_df['NowcastMonth'] = nowcast_df.index.to_period('M')

    target_df = target_series.to_frame('Target').copy()
    target_df['TargetMonth'] = target_df.index.to_period('M')
    # 确保每月只有一个target值
    target_df = target_df.groupby('TargetMonth').last()

    weekly_data = []

    # 2. 按月遍历nowcast数据
    for period, group in nowcast_df.groupby('NowcastMonth'):
        # 获取下个月的period
        next_period = period + 1

        # 检查下个月是否有target数据
        if next_period in target_df.index:
            next_month_target = target_df.loc[next_period, 'Target']

            # 该月所有周的nowcast都与下月target配对
            for date, row in group.iterrows():
                weekly_data.append({
                    'month': period,
                    'week_date': date,
                    'nowcast': row['Nowcast'],
                    'next_month_target': next_month_target
                })

    if not weekly_data:
        logger.warning("[align_next_month_weekly] 未找到有效的周度-下月配对数据")
        return pd.DataFrame(columns=['month', 'week_date', 'nowcast', 'next_month_target'])

    df = pd.DataFrame(weekly_data)

    return df


def align_next_month_last_friday(
    nowcast_series: pd.Series,
    target_series: pd.Series
) -> pd.DataFrame:
    """对齐m月最后周五nowcast、m月target与m+1月target（用于Hit Rate和MAE计算）

    Args:
        nowcast_series: 周度nowcast序列
        target_series: 月度target序列

    Returns:
        pd.DataFrame: 对齐后数据，列['month', 'last_friday_date', 'nowcast', 'current_target', 'next_target']
    """
    # 确保索引是DatetimeIndex
    if not isinstance(nowcast_series.index, pd.DatetimeIndex):
        nowcast_series.index = pd.to_datetime(nowcast_series.index)
    if not isinstance(target_series.index, pd.DatetimeIndex):
        target_series.index = pd.to_datetime(target_series.index)

    # 参考老代码，使用DataFrame方式处理（避免Grouper错误）
    # 1. 转换为DataFrame并添加月份列
    nowcast_df = nowcast_series.to_frame('Nowcast').copy()
    nowcast_df['NowcastMonth'] = nowcast_df.index.to_period('M')

    target_df = target_series.to_frame('Target').copy()
    target_df['TargetMonth'] = target_df.index.to_period('M')
    # 确保每月只有一个target值
    target_df = target_df.groupby('TargetMonth').last()

    monthly_friday_data = []

    # 2. 按月遍历nowcast数据
    for period, group in nowcast_df.groupby('NowcastMonth'):
        # 找到该月的所有周五 (weekday=4)
        fridays = group[group.index.weekday == 4]
        if fridays.empty:
            continue

        # 取最后一个周五
        last_friday_date = fridays.index.max()
        last_friday_nowcast = fridays.loc[last_friday_date, 'Nowcast']

        # 获取当月和下月的target
        next_period = period + 1

        if period in target_df.index and next_period in target_df.index:
            current_target = target_df.loc[period, 'Target']
            next_target = target_df.loc[next_period, 'Target']

            monthly_friday_data.append({
                'month': period,
                'last_friday_date': last_friday_date,
                'nowcast': last_friday_nowcast,
                'current_target': current_target,
                'next_target': next_target
            })

    if not monthly_friday_data:
        logger.warning("[align_next_month_last_friday] 未找到有效的月度最后周五配对数据")
        return pd.DataFrame(columns=['month', 'last_friday_date', 'nowcast', 'current_target', 'next_target'])

    df = pd.DataFrame(monthly_friday_data)
    df = df.set_index('last_friday_date').sort_index()

    return df


def calculate_next_month_rmse(
    nowcast_series: pd.Series,
    target_series: pd.Series
) -> float:
    """计算m月所有周nowcast与m+1月target配对的RMSE（用于变量筛选）

    Args:
        nowcast_series: 周度nowcast序列
        target_series: 月度target序列

    Returns:
        float: RMSE值，失败返回np.inf
    """
    try:
        aligned_df = align_next_month_weekly_data(nowcast_series, target_series)

        if aligned_df.empty or len(aligned_df) < 2:
            logger.warning(f"[next_month_rmse] 配对数据不足: {len(aligned_df)}个数据点")
            return np.inf

        # 计算RMSE
        squared_errors = (aligned_df['nowcast'] - aligned_df['next_month_target']) ** 2
        rmse = np.sqrt(squared_errors.mean())
        return float(rmse)

    except Exception as e:
        logger.error(f"[next_month_rmse] 计算失败: {e}")
        return np.inf


def calculate_next_month_mae(
    nowcast_series: pd.Series,
    target_series: pd.Series
) -> float:
    """计算m月最后周五nowcast与m+1月target配对的MAE

    Args:
        nowcast_series: 周度nowcast序列
        target_series: 月度target序列

    Returns:
        float: MAE值，失败返回np.inf
    """
    try:
        aligned_df = align_next_month_last_friday(nowcast_series, target_series)

        if aligned_df.empty or len(aligned_df) < 2:
            logger.warning(f"[next_month_mae] 配对数据不足: {len(aligned_df)}个数据点")
            return np.inf

        # 计算MAE
        abs_errors = np.abs(aligned_df['nowcast'] - aligned_df['next_target'])
        mae = abs_errors.mean()
        return float(mae)

    except Exception as e:
        logger.error(f"[next_month_mae] 计算失败: {e}")
        return np.inf


def calculate_next_month_hit_rate(
    nowcast_series: pd.Series,
    target_series: pd.Series
) -> float:
    """计算新定义的Hit Rate

    对每个月m：
    - 预测变化：target_{m+1} - nowcast_m（m月最后周五）
    - 实际变化：target_{m+1} - target_m
    - 命中条件：两个变化符号相同

    Args:
        nowcast_series: 周度nowcast序列
        target_series: 月度target序列

    Returns:
        float: Hit Rate百分比（0-100），失败返回np.nan
    """
    try:
        aligned_df = align_next_month_last_friday(nowcast_series, target_series)

        if aligned_df.empty or len(aligned_df) < 2:
            logger.warning(f"[next_month_hit_rate] 配对数据不足: {len(aligned_df)}个数据点")
            return np.nan

        # 计算预测变化：下月target - 当月最后周五nowcast
        aligned_df['pred_change'] = aligned_df['next_target'] - aligned_df['nowcast']

        # 计算实际变化：下月target - 当月target
        aligned_df['actual_change'] = aligned_df['next_target'] - aligned_df['current_target']

        # 判断符号是否相同
        aligned_df['hit'] = (
            np.sign(aligned_df['pred_change']) ==
            np.sign(aligned_df['actual_change'])
        )

        # 计算命中率
        hits = aligned_df['hit'].sum()
        total = len(aligned_df)
        hit_rate = (hits / total) * 100.0

        return float(hit_rate)

    except Exception as e:
        logger.error(f"[next_month_hit_rate] 计算失败: {e}")
        return np.nan


# ==================== 本月配对评估函数（2025-12新增）====================

def align_current_month_weekly_data(
    nowcast_series: pd.Series,
    target_series: pd.Series
) -> pd.DataFrame:
    """对齐m月所有周的nowcast与m月target（用于本月配对RMSE计算）

    Args:
        nowcast_series: 周度nowcast序列
        target_series: 月度target序列

    Returns:
        pd.DataFrame: 对齐后数据，列['month', 'week_date', 'nowcast', 'current_month_target']
    """
    # 确保索引是DatetimeIndex
    if not isinstance(nowcast_series.index, pd.DatetimeIndex):
        nowcast_series.index = pd.to_datetime(nowcast_series.index)
    if not isinstance(target_series.index, pd.DatetimeIndex):
        target_series.index = pd.to_datetime(target_series.index)

    # 转换为DataFrame并添加月份列
    nowcast_df = nowcast_series.to_frame('Nowcast').copy()
    nowcast_df['NowcastMonth'] = nowcast_df.index.to_period('M')

    target_df = target_series.to_frame('Target').copy()
    target_df['TargetMonth'] = target_df.index.to_period('M')
    target_df = target_df.groupby('TargetMonth').last()

    weekly_data = []

    # 按月遍历nowcast数据
    for period, group in nowcast_df.groupby('NowcastMonth'):
        # 检查当月是否有target数据
        if period in target_df.index:
            current_month_target = target_df.loc[period, 'Target']

            # 该月所有周的nowcast都与当月target配对
            for date, row in group.iterrows():
                weekly_data.append({
                    'month': period,
                    'week_date': date,
                    'nowcast': row['Nowcast'],
                    'current_month_target': current_month_target
                })

    if not weekly_data:
        logger.warning("[align_current_month_weekly] 未找到有效的周度-当月配对数据")
        return pd.DataFrame(columns=['month', 'week_date', 'nowcast', 'current_month_target'])

    return pd.DataFrame(weekly_data)


def align_current_month_last_friday(
    nowcast_series: pd.Series,
    target_series: pd.Series
) -> pd.DataFrame:
    """对齐m月最后周五nowcast与m月target、m-1月target（用于本月配对Hit Rate和MAE计算）

    Args:
        nowcast_series: 周度nowcast序列
        target_series: 月度target序列

    Returns:
        pd.DataFrame: 对齐后数据，列['month', 'last_friday_date', 'nowcast', 'prev_target', 'current_target']
    """
    # 确保索引是DatetimeIndex
    if not isinstance(nowcast_series.index, pd.DatetimeIndex):
        nowcast_series.index = pd.to_datetime(nowcast_series.index)
    if not isinstance(target_series.index, pd.DatetimeIndex):
        target_series.index = pd.to_datetime(target_series.index)

    nowcast_df = nowcast_series.to_frame('Nowcast').copy()
    nowcast_df['NowcastMonth'] = nowcast_df.index.to_period('M')

    target_df = target_series.to_frame('Target').copy()
    target_df['TargetMonth'] = target_df.index.to_period('M')
    target_df = target_df.groupby('TargetMonth').last()

    monthly_friday_data = []

    for period, group in nowcast_df.groupby('NowcastMonth'):
        # 找到该月的所有周五 (weekday=4)
        fridays = group[group.index.weekday == 4]
        if fridays.empty:
            continue

        last_friday_date = fridays.index.max()
        last_friday_nowcast = fridays.loc[last_friday_date, 'Nowcast']

        # 获取上月��当月的target
        prev_period = period - 1

        if prev_period in target_df.index and period in target_df.index:
            prev_target = target_df.loc[prev_period, 'Target']
            current_target = target_df.loc[period, 'Target']

            monthly_friday_data.append({
                'month': period,
                'last_friday_date': last_friday_date,
                'nowcast': last_friday_nowcast,
                'prev_target': prev_target,
                'current_target': current_target
            })

    if not monthly_friday_data:
        logger.warning("[align_current_month_last_friday] 未找到有效的月度最后周五配对数据")
        return pd.DataFrame(columns=['month', 'last_friday_date', 'nowcast', 'prev_target', 'current_target'])

    df = pd.DataFrame(monthly_friday_data)
    df = df.set_index('last_friday_date').sort_index()

    return df


def calculate_current_month_rmse(
    nowcast_series: pd.Series,
    target_series: pd.Series
) -> float:
    """计算m月所有周nowcast与m月target配对的RMSE（本月配对）

    Args:
        nowcast_series: 周度nowcast序列
        target_series: 月度target序列

    Returns:
        float: RMSE值，失败返回np.inf
    """
    try:
        aligned_df = align_current_month_weekly_data(nowcast_series, target_series)

        if aligned_df.empty or len(aligned_df) < 2:
            logger.warning(f"[current_month_rmse] 配对数据不足: {len(aligned_df)}个数据点")
            return np.inf

        squared_errors = (aligned_df['nowcast'] - aligned_df['current_month_target']) ** 2
        rmse = np.sqrt(squared_errors.mean())
        return float(rmse)

    except Exception as e:
        logger.error(f"[current_month_rmse] 计算失败: {e}")
        return np.inf


def calculate_current_month_mae(
    nowcast_series: pd.Series,
    target_series: pd.Series
) -> float:
    """计算m月最后周五nowcast与m月target配对的MAE（本月配对）

    Args:
        nowcast_series: 周度nowcast序列
        target_series: 月度target序列

    Returns:
        float: MAE值，失败返回np.inf
    """
    try:
        aligned_df = align_current_month_last_friday(nowcast_series, target_series)

        if aligned_df.empty or len(aligned_df) < 2:
            logger.warning(f"[current_month_mae] 配对数据不足: {len(aligned_df)}个数据点")
            return np.inf

        abs_errors = np.abs(aligned_df['nowcast'] - aligned_df['current_target'])
        mae = abs_errors.mean()
        return float(mae)

    except Exception as e:
        logger.error(f"[current_month_mae] 计算失败: {e}")
        return np.inf


def calculate_current_month_hit_rate(
    nowcast_series: pd.Series,
    target_series: pd.Series
) -> float:
    """计算本月配对Hit Rate

    对每个月m：
    - 预测变化：target_m - nowcast_m（m月最后周五）
    - 实际变化：target_m - target_{m-1}
    - 命中条件：两个变化符号相同

    Args:
        nowcast_series: 周度nowcast序列
        target_series: 月度target序列

    Returns:
        float: Hit Rate百分比（0-100），失败返回np.nan
    """
    try:
        aligned_df = align_current_month_last_friday(nowcast_series, target_series)

        if aligned_df.empty or len(aligned_df) < 2:
            logger.warning(f"[current_month_hit_rate] 配对数据不足: {len(aligned_df)}个数据点")
            return np.nan

        # 计算预测变化：当月target - 当月最后周五nowcast
        aligned_df['pred_change'] = aligned_df['current_target'] - aligned_df['nowcast']

        # 计算实际变化：当月target - 上月target
        aligned_df['actual_change'] = aligned_df['current_target'] - aligned_df['prev_target']

        # 判断符号是否相同
        aligned_df['hit'] = (
            np.sign(aligned_df['pred_change']) ==
            np.sign(aligned_df['actual_change'])
        )

        hits = aligned_df['hit'].sum()
        total = len(aligned_df)
        hit_rate = (hits / total) * 100.0

        return float(hit_rate)

    except Exception as e:
        logger.error(f"[current_month_hit_rate] 计算失败: {e}")
        return np.nan


# ==================== 统一调度函数 ====================

def calculate_aligned_rmse(
    nowcast_series: pd.Series,
    target_series: pd.Series,
    alignment_mode: str = 'next_month'
) -> float:
    """根据配对模式计算RMSE

    Args:
        nowcast_series: 周度nowcast序列
        target_series: 月度target序列
        alignment_mode: 配对模式 ('current_month' 或 'next_month')

    Returns:
        float: RMSE值
    """
    if alignment_mode == 'current_month':
        return calculate_current_month_rmse(nowcast_series, target_series)
    else:
        return calculate_next_month_rmse(nowcast_series, target_series)


def calculate_aligned_mae(
    nowcast_series: pd.Series,
    target_series: pd.Series,
    alignment_mode: str = 'next_month'
) -> float:
    """根据配对模式计算MAE

    Args:
        nowcast_series: 周度nowcast序列
        target_series: 月度target序列
        alignment_mode: 配对模式 ('current_month' 或 'next_month')

    Returns:
        float: MAE值
    """
    if alignment_mode == 'current_month':
        return calculate_current_month_mae(nowcast_series, target_series)
    else:
        return calculate_next_month_mae(nowcast_series, target_series)


def calculate_aligned_hit_rate(
    nowcast_series: pd.Series,
    target_series: pd.Series,
    alignment_mode: str = 'next_month'
) -> float:
    """根据配对模式计算Hit Rate

    Args:
        nowcast_series: 周度nowcast序列
        target_series: 月度target序列
        alignment_mode: 配对模式 ('current_month' 或 'next_month')

    Returns:
        float: Hit Rate百分比
    """
    if alignment_mode == 'current_month':
        return calculate_current_month_hit_rate(nowcast_series, target_series)
    else:
        return calculate_next_month_hit_rate(nowcast_series, target_series)
