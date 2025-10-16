"""
Weight Calculator Utility
权重计算工具 - 统一的权重计算函数

消除重复代码:
- 原 macro_operations.py:233-252 (20行)
- 原 data_processor.py:13-32 (20行)
- 原 macro_operations.py:255-277 (23行)
- 原 data_processor.py:35-57 (23行)
"""

import pandas as pd


def get_weight_for_year(df_weights_row: pd.Series, year: int) -> float:
    """
    根据年份选择合适的权重

    权重选择规则:
    - 2020年及以后: 使用权重_2020
    - 2018-2019年: 使用权重_2018
    - 2012-2017年: 使用权重_2012
    - 2012年之前: 返回0 (不计算加权增速)

    Args:
        df_weights_row: 权重数据行 (Series)，应包含权重_2012、权重_2018、权重_2020列
        year: 年份

    Returns:
        对应年份的权重值，如果无可用权重则返回0.0
    """
    if year >= 2020 and '权重_2020' in df_weights_row.index and pd.notna(df_weights_row['权重_2020']):
        return df_weights_row['权重_2020']
    elif year >= 2018 and '权重_2018' in df_weights_row.index and pd.notna(df_weights_row['权重_2018']):
        return df_weights_row['权重_2018']
    elif year >= 2012 and '权重_2012' in df_weights_row.index and pd.notna(df_weights_row['权重_2012']):
        return df_weights_row['权重_2012']
    else:
        # 2012年之前不计算加权增速
        return 0.0


def filter_data_from_2012(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤数据，只保留2012年及以后的数据

    由于权重数据从2012年开始，2012年之前的数据无法进行加权计算，
    因此需要过滤掉这部分数据。

    Args:
        df: 输入的DataFrame，索引应为DatetimeIndex

    Returns:
        过滤后只包含2012年及以后数据的DataFrame
    """
    if df.empty:
        return df

    # 筛选2012年及以后的日期索引
    date_indices = []
    for idx in df.index:
        if hasattr(idx, 'year') and idx.year >= 2012:
            date_indices.append(idx)

    if date_indices:
        return df.loc[date_indices]
    else:
        return pd.DataFrame()


# 已删除create_year_to_weight_mapping函数
# 原因：该函数从未被使用（违反YAGNI原则）
# 如果将来需要，可以从git历史中恢复
