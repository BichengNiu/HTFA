# -*- coding: utf-8 -*-
"""
Preview模块统一计算组件
通过配置驱动,一个函数支持所有频率的摘要计算
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st

from dashboard.preview.modules.industrial.config import SUMMARY_CONFIGS


@st.cache_data(show_spinner=False, max_entries=30, ttl=3600)
def calculate_summary(
    df: pd.DataFrame,
    frequency: str,
    indicator_unit_map: Dict[str, str] = None,
    indicator_type_map: Dict[str, str] = None,
    indicator_industry_map: Dict[str, str] = None
) -> pd.DataFrame:
    """通用摘要计算函数

    根据频率自动选择合适的计算策略

    Args:
        df: 数据DataFrame
        frequency: 数据频率 ('weekly'/'monthly'/'daily'/'ten_day'/'yearly')
        indicator_unit_map: 指标单位映射字典
        indicator_type_map: 指标类型映射字典
        indicator_industry_map: 指标行业映射字典

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

    if indicator_unit_map is None:
        indicator_unit_map = {}
    if indicator_type_map is None:
        indicator_type_map = {}
    if indicator_industry_map is None:
        indicator_industry_map = {}

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

        # 3. 计算增长率(传入单位和类型信息)
        indicator_unit = indicator_unit_map.get(indicator, '')
        indicator_type = indicator_type_map.get(indicator, '')
        indicator_industry = indicator_industry_map.get(indicator, '未分类')
        growth_rates = _calculate_growth_rates(
            current_value, reference_values, frequency, indicator_unit, indicator_type
        )

        # 4. 构建行数据
        row_data = {
            config['indicator_name_column']: indicator,
            '行业': indicator_industry,  # 添加行业列
            '单位': indicator_unit,  # 添加单位列,用于显示时格式化判断
            '类型': indicator_type,  # 添加类型列,用于显示时格式化判断
            '最新值': current_value,
            config['date_column']: (
                current_date.strftime('%Y-%m-%d') if frequency != 'yearly'
                else current_date.year
            ),
            **reference_values,
            **growth_rates
        }

        summary_data.append(row_data)

    # 构建DataFrame并排序列(添加单位、行业和类型到列顺序中)
    summary_df = pd.DataFrame(summary_data)
    # 确保单位、行业和类型列在指标名称后面，顺序为：单位 -> 行业 -> 类型
    column_order_with_meta = []
    for col in config['column_order']:
        if col in summary_df.columns:
            column_order_with_meta.append(col)
            # 在指标名称列后面插入单位、行业和类型
            if col == config['indicator_name_column']:
                if '单位' in summary_df.columns:
                    column_order_with_meta.append('单位')
                if '行业' in summary_df.columns:
                    column_order_with_meta.append('行业')
                if '类型' in summary_df.columns:
                    column_order_with_meta.append('类型')

    return summary_df[column_order_with_meta]


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

        # 计算时间段均值
        week_mean = _calculate_period_mean(series, current_date, days_back=7)
        month_mean = _calculate_period_mean(series, current_date, days_back=30)

        return {
            '昨日值': _get_value_at_date(df, series.name, yesterday_date),
            '上周均值': week_mean,
            '上月均值': month_mean
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
    frequency: str,
    indicator_unit: str = '',
    indicator_type: str = ''
) -> Dict[str, float]:
    """计算增长率(所有频率通用)

    Args:
        current_value: 当前值
        reference_values: 参考值字典
        frequency: 数据频率
        indicator_unit: 指标单位
        indicator_type: 指标类型

    Returns:
        增长率字典
    """
    growth_rates = {}

    # 判断是否使用差值计算
    # 规则：单位为"%"且类型不是"开工率"时,使用差值而不是比率
    use_difference = (indicator_unit == '%' and indicator_type != '开工率')

    # 根据频率选择use_abs参数
    use_abs = frequency != 'monthly'

    # 环比增长率 - 根据频率类型分别处理
    if frequency == 'daily':
        # 日度数据：只计算环比昨日
        if '昨日值' in reference_values:
            growth_rates['环比昨日'] = _growth_rate(
                current_value, reference_values['昨日值'], True, use_difference
            )
    elif frequency == 'weekly':
        # 周度数据：只计算环比上周
        if '上周值' in reference_values:
            growth_rates['环比上周'] = _growth_rate(
                current_value, reference_values['上周值'], True, use_difference
            )
    elif frequency == 'monthly':
        # 月度数据：只计算环比上月
        if '上月值' in reference_values:
            growth_rates['环比上月'] = _growth_rate(
                current_value, reference_values['上月值'], False, use_difference
            )
    elif frequency == 'ten_day':
        # 旬度数据：只计算环比上旬
        if '上旬值' in reference_values:
            growth_rates['环比上旬'] = _growth_rate(
                current_value, reference_values['上旬值'], True, use_difference
            )

    # 同比增长率
    if '上年同月值' in reference_values:
        growth_rates['同比上年'] = _growth_rate(
            current_value, reference_values['上年同月值'], False, use_difference
        )
    elif '上年值' in reference_values and frequency == 'yearly':
        growth_rates['同比上年'] = _growth_rate(
            current_value, reference_values['上年值'], True, use_difference
        )

    return growth_rates


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


def _growth_rate(current: Any, reference: Any, use_abs: bool = True, use_difference: bool = False) -> float:
    """计算增长率或差值

    Args:
        current: 当前值
        reference: 参考值
        use_abs: 是否使用绝对值作为分母
        use_difference: 是否使用差值而不是比率

    Returns:
        增长率或差值
    """
    if pd.notna(current) and pd.notna(reference):
        if use_difference:
            # 使用差值计算
            return current - reference
        elif reference != 0:
            # 使用比率计算
            denominator = abs(reference) if use_abs else reference
            return (current - reference) / denominator
    return np.nan
