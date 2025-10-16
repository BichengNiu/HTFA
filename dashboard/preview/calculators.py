# -*- coding: utf-8 -*-
"""
Preview模块统一计算组件
通过配置驱动,一个函数支持所有频率的摘要计算
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st

from dashboard.preview.config import SUMMARY_CONFIGS


@st.cache_data(show_spinner=False, max_entries=30, ttl=3600)
def calculate_summary(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """通用摘要计算函数

    根据频率自动选择合适的计算策略

    Args:
        df: 数据DataFrame
        frequency: 数据频率 ('weekly'/'monthly'/'daily'/'ten_day'/'yearly')

    Returns:
        摘要DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"输入的 {frequency} DataFrame 索引无效,无法计算。")
        return pd.DataFrame()

    df = df.sort_index()
    config = SUMMARY_CONFIGS[frequency]

    summary_data = []

    for indicator in df.columns:
        series = df[indicator].dropna()
        if series.empty:
            continue

        # 1. 获取最新值
        current_date = series.index.max()
        current_value = series.loc[current_date]

        # 2. 计算参考值(根据频率调用不同策略)
        reference_values = _calculate_reference_values(
            df, series, current_date, frequency
        )

        # 3. 计算增长率
        growth_rates = _calculate_growth_rates(
            current_value, reference_values, frequency
        )

        # 4. 计算历史统计
        hist_stats = _calculate_historical_stats(
            series, current_date, config['lookback_years']
        )

        # 5. 构建行数据
        row_data = {
            config['indicator_name_column']: indicator,
            '最新值': current_value,
            config['date_column']: (
                current_date.strftime('%Y-%m-%d') if frequency != 'yearly'
                else current_date.year
            ),
            **reference_values,
            **growth_rates,
            **hist_stats
        }

        summary_data.append(row_data)

    # 构建DataFrame并排序列
    summary_df = pd.DataFrame(summary_data)
    existing_cols = [c for c in config['column_order'] if c in summary_df.columns]
    return summary_df[existing_cols]


def _calculate_reference_values(
    df: pd.DataFrame,
    series: pd.Series,
    current_date: pd.Timestamp,
    frequency: str
) -> Dict[str, Any]:
    """计算参考值

    这是唯一因频率而异的部分
    """
    if frequency == 'weekly':
        last_week_date = current_date - pd.Timedelta(weeks=1)
        return {
            '上周值': _get_value_at_date(df, series.name, last_week_date)
        }

    elif frequency == 'monthly':
        current_year = current_date.year
        current_month = current_date.month

        last_month_year = current_year if current_month > 1 else current_year - 1
        last_month = current_month - 1 if current_month > 1 else 12
        val_last_month = _get_value_by_year_month(series, last_month_year, last_month)

        last_year_target_year = current_year - 1
        val_last_year = _get_value_by_year_month(series, last_year_target_year, current_month)

        return {
            '上月值': val_last_month,
            '上年同月值': val_last_year
        }

    elif frequency == 'daily':
        yesterday_date = current_date - pd.Timedelta(days=1)
        last_week_date = current_date - pd.Timedelta(days=7)
        last_month_date = current_date - pd.Timedelta(days=30)
        last_year_date = current_date - pd.Timedelta(days=365)

        # 计算时间段均值
        week_mean = _calculate_period_mean(series, current_date, days_back=7)
        month_mean = _calculate_period_mean(series, current_date, days_back=30)
        year_mean = _calculate_period_mean(series, current_date, days_back=365)

        return {
            '昨日值': _get_value_at_date(df, series.name, yesterday_date),
            '上周值': _get_value_at_date(df, series.name, last_week_date),
            '上月值': _get_value_at_date(df, series.name, last_month_date),
            '上年值': _get_value_at_date(df, series.name, last_year_date),
            '上周均值': week_mean,
            '上月均值': month_mean,
            '上年均值': year_mean
        }

    elif frequency == 'ten_day':
        last_ten_day_date = current_date - pd.Timedelta(days=10)
        return {
            '上旬值': _get_value_at_date(df, series.name, last_ten_day_date)
        }

    elif frequency == 'yearly':
        current_year = current_date.year
        val_last_year = _get_value_by_year_end(series, current_year - 1)
        val_two_years_ago = _get_value_by_year_end(series, current_year - 2)
        val_three_years_ago = _get_value_by_year_end(series, current_year - 3)

        return {
            '上年值': val_last_year,
            '两年前值': val_two_years_ago,
            '三年前值': val_three_years_ago
        }

    return {}


def _calculate_growth_rates(
    current_value: Any,
    reference_values: Dict[str, Any],
    frequency: str
) -> Dict[str, float]:
    """计算增长率(所有频率通用)"""
    growth_rates = {}

    # 根据频率选择use_abs参数
    use_abs = frequency != 'monthly'

    # 环比增长率 - 根据频率类型分别处理
    if frequency == 'daily':
        # 日度数据：计算环比昨日、环比上周、环比上月
        if '昨日值' in reference_values:
            growth_rates['环比昨日'] = _growth_rate(
                current_value, reference_values['昨日值'], True
            )
        if '上周值' in reference_values:
            growth_rates['环比上周'] = _growth_rate(
                current_value, reference_values['上周值'], True
            )
        if '上月值' in reference_values:
            growth_rates['环比上月'] = _growth_rate(
                current_value, reference_values['上月值'], True
            )
    elif frequency == 'weekly':
        # 周度数据：只计算环比上周
        if '上周值' in reference_values:
            growth_rates['环比上周'] = _growth_rate(
                current_value, reference_values['上周值'], True
            )
    elif frequency == 'monthly':
        # 月度数据：只计算环比上月
        if '上月值' in reference_values:
            growth_rates['环比上月'] = _growth_rate(
                current_value, reference_values['上月值'], False
            )
    elif frequency == 'ten_day':
        # 旬度数据：只计算环比上旬
        if '上旬值' in reference_values:
            growth_rates['环比上旬'] = _growth_rate(
                current_value, reference_values['上旬值'], True
            )

    # 同比增长率
    if '上年同月值' in reference_values:
        growth_rates['同比上年'] = _growth_rate(
            current_value, reference_values['上年同月值'], False
        )
    elif '上年值' in reference_values and frequency in ('daily', 'yearly'):
        growth_rates['同比上年'] = _growth_rate(
            current_value, reference_values['上年值'], True
        )

    return growth_rates


def _calculate_historical_stats(
    series: pd.Series,
    current_date: pd.Timestamp,
    lookback_years: int
) -> Dict[str, float]:
    """计算历史统计(所有频率通用)"""
    current_year = current_date.year
    start_year = current_year - lookback_years
    end_year = current_year - 1

    start_date = pd.Timestamp(f'{start_year}-01-01')
    end_date = pd.Timestamp(f'{end_year}-12-31')

    historical_data = series.loc[
        (series.index >= start_date) & (series.index <= end_date)
    ]

    if not historical_data.empty:
        return {
            f'近{lookback_years}年最大值': historical_data.max(),
            f'近{lookback_years}年最小值': historical_data.min(),
            f'近{lookback_years}年平均值': historical_data.mean()
        }
    else:
        return {
            f'近{lookback_years}年最大值': np.nan,
            f'近{lookback_years}年最小值': np.nan,
            f'近{lookback_years}年平均值': np.nan
        }


# 辅助函数


def _get_value_at_date(df: pd.DataFrame, indicator: str, date: pd.Timestamp) -> Any:
    """获取指定日期的值"""
    try:
        return df.loc[date, indicator] if date in df.index else np.nan
    except (KeyError, IndexError):
        return np.nan


def _get_value_by_year_month(series: pd.Series, year: int, month: int) -> Any:
    """通过年月获取值"""
    mask = (series.index.year == year) & (series.index.month == month)
    data = series[mask]
    return data.iloc[-1] if not data.empty else np.nan


def _get_value_by_year_end(series: pd.Series, target_year: int) -> Any:
    """获取指定年份年末前的最新值"""
    target_date = pd.Timestamp(f'{target_year}-12-31')
    try:
        data = series.loc[series.index <= target_date]
        return data.iloc[-1] if len(data) > 0 else np.nan
    except (KeyError, IndexError):
        return np.nan


def _calculate_period_mean(
    series: pd.Series,
    current_date: pd.Timestamp,
    days_back: int
) -> float:
    """计算指定时间段的均值"""
    start_date = current_date - pd.Timedelta(days=days_back)
    period_data = series.loc[
        (series.index >= start_date) & (series.index < current_date)
    ]
    return period_data.mean() if not period_data.empty else np.nan


def _growth_rate(current: Any, reference: Any, use_abs: bool = True) -> float:
    """计算增长率"""
    if pd.notna(current) and pd.notna(reference) and reference != 0:
        denominator = abs(reference) if use_abs else reference
        return (current - reference) / denominator
    return np.nan
