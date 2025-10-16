"""
Data Converter Utility
数据转换工具 - 统一的数据转换函数

主要功能:
- convert_cumulative_to_yoy: 将累计值转换为年同比
"""

import pandas as pd
import streamlit as st


def convert_cumulative_to_yoy(data_series: pd.Series, periods: int = 12) -> pd.Series:
    """
    将累计值转换为年同比数据，并将1、2月设为缺失值（向量化优化版本）

    该函数用于处理累计值指标(如累计利润总额)，计算其年同比增长率。
    由于1月和2月的累计值不具有可比性，这两个月的值将设为NaN。

    性能优化：使用pandas向量化操作替代循环，预期性能提升5-10倍
    - 消除了for循环遍历每个日期点
    - 使用reindex()方法一次性处理所有日期的匹配（带容差）
    - 使用向量化操作计算百分比变化

    Args:
        data_series: 累计值数据序列，索引应为DatetimeIndex
        periods: 年同比的期数，默认12个月

    Returns:
        年同比数据序列 (单位: 百分比)

    Example:
        >>> cumulative_profit = pd.Series([100, 150, 200],
        ...     index=pd.date_range('2023-01', periods=3, freq='M'))
        >>> yoy = convert_cumulative_to_yoy(cumulative_profit)
    """
    try:
        # 确保数据是数值型
        data_series = pd.to_numeric(data_series, errors='coerce')

        # 确保索引已排序
        data_series = data_series.sort_index()

        # 创建12个月前的日期索引
        prev_dates = data_series.index - pd.DateOffset(months=periods)

        # 使用reindex获取12个月前的值（带容差匹配，30天内）
        # reindex的nearest方法会自动找到最接近的日期
        tolerance = pd.Timedelta(days=30)
        prev_values = data_series.reindex(prev_dates, method='nearest', tolerance=tolerance)

        # 将prev_values的索引对齐到当前索引（用于向量化计算）
        prev_values.index = data_series.index

        # 向量化计算年同比：((current - prev) / prev) * 100
        # pandas会自动处理NaN和除零情况
        yoy_series = ((data_series - prev_values) / prev_values) * 100

        # 将1月和2月的值设为NaN（因为累计值在这两个月不具有可比性）
        jan_feb_mask = (yoy_series.index.month == 1) | (yoy_series.index.month == 2)
        yoy_series.loc[jan_feb_mask] = float('nan')

        return yoy_series

    except Exception:
        return pd.Series()


# 已删除convert_cumulative_to_yoy_cached函数
# 原因：定义了缓存版本但从未使用（违反YAGNI原则）
# 如果将来需要缓存，可以直接在convert_cumulative_to_yoy函数上添加@st.cache_data装饰器
# 或从git历史中恢复此函数
