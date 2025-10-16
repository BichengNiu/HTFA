"""
日期处理工具模块

提供统一的日期处理功能，包括日期标准化、解析和验证
"""

import pandas as pd
from typing import Union, Optional, Tuple
from datetime import datetime, date


def standardize_date(
    date_param: Union[str, datetime, date, pd.Timestamp, None]
) -> Optional[pd.Timestamp]:
    """
    标准化日期参数，确保在整个流程中保持一致的格式

    Args:
        date_param: 日期参数，可以是字符串、datetime、date或pd.Timestamp

    Returns:
        pd.Timestamp或None: 标准化后的时间戳，解析失败返回None

    Examples:
        >>> standardize_date('2023-01-01')
        Timestamp('2023-01-01 00:00:00')
        >>> standardize_date(datetime(2023, 1, 1))
        Timestamp('2023-01-01 00:00:00')
        >>> standardize_date(None)
        None
    """
    if date_param is None:
        return None

    if isinstance(date_param, str):
        try:
            return pd.to_datetime(date_param)
        except (ValueError, TypeError) as e:
            print(f"警告: 无法解析日期字符串: {date_param}, 错误: {e}")
            return None
    elif isinstance(date_param, (datetime, date)):
        return pd.to_datetime(date_param)
    elif isinstance(date_param, pd.Timestamp):
        return date_param
    else:
        print(f"警告: 不支持的日期类型: {type(date_param)}")
        return None


def parse_date_range(
    start_date: Union[str, datetime, date, pd.Timestamp, None],
    end_date: Union[str, datetime, date, pd.Timestamp, None],
    default_start: Optional[pd.Timestamp] = None,
    default_end: Optional[pd.Timestamp] = None
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    解析并标准化日期范围

    Args:
        start_date: 开始日期
        end_date: 结束日期
        default_start: 默认开始日期
        default_end: 默认结束日期

    Returns:
        Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
            (标准化后的开始日期, 标准化后的结束日期)

    Examples:
        >>> parse_date_range('2023-01-01', '2023-12-31')
        (Timestamp('2023-01-01 00:00:00'), Timestamp('2023-12-31 00:00:00'))
        >>> parse_date_range(None, None, pd.Timestamp('2020-01-01'), pd.Timestamp('2025-01-01'))
        (Timestamp('2020-01-01 00:00:00'), Timestamp('2025-01-01 00:00:00'))
    """
    parsed_start = standardize_date(start_date)
    parsed_end = standardize_date(end_date)

    # 如果解析失败，使用默认值
    if parsed_start is None:
        parsed_start = default_start
    if parsed_end is None:
        parsed_end = default_end

    # 验证日期范围的合理性
    if parsed_start and parsed_end and parsed_start > parsed_end:
        print(f"警告: 开始日期 ({parsed_start}) 晚于结束日期 ({parsed_end})")

    return parsed_start, parsed_end


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: Union[str, datetime, date, pd.Timestamp, None] = None,
    end_date: Union[str, datetime, date, pd.Timestamp, None] = None
) -> pd.DataFrame:
    """
    根据日期范围筛选DataFrame

    Args:
        df: 输入DataFrame，索引应为DatetimeIndex
        start_date: 开始日期（可选）
        end_date: 结束日期（可选）

    Returns:
        pd.DataFrame: 筛选后的DataFrame

    Examples:
        >>> df = pd.DataFrame({'value': [1, 2, 3]},
        ...                   index=pd.date_range('2023-01-01', periods=3))
        >>> filtered = filter_by_date_range(df, '2023-01-02', None)
        >>> len(filtered)
        2
    """
    if df.empty:
        return df

    result = df.copy()

    try:
        if start_date:
            start_dt = standardize_date(start_date)
            if start_dt:
                result = result[result.index >= start_dt]

        if end_date:
            end_dt = standardize_date(end_date)
            if end_dt:
                result = result[result.index <= end_dt]

    except Exception as e:
        print(f"警告: 时间范围筛选失败: {e}，返回原始数据")
        return df

    return result


def get_date_range_info(df: pd.DataFrame) -> dict:
    """
    获取DataFrame的日期范围信息

    Args:
        df: 输入DataFrame，索引应为DatetimeIndex

    Returns:
        dict: 包含日期范围信息的字典
            - min_date: 最小日期
            - max_date: 最大日期
            - date_count: 日期数量
            - frequency: 频率（如果可检测）
            - has_gaps: 是否有间隙

    Examples:
        >>> df = pd.DataFrame({'value': [1, 2, 3]},
        ...                   index=pd.date_range('2023-01-01', periods=3, freq='D'))
        >>> info = get_date_range_info(df)
        >>> info['date_count']
        3
    """
    if df.empty:
        return {
            'min_date': None,
            'max_date': None,
            'date_count': 0,
            'frequency': None,
            'has_gaps': False
        }

    try:
        return {
            'min_date': df.index.min(),
            'max_date': df.index.max(),
            'date_count': len(df.index),
            'frequency': pd.infer_freq(df.index) if len(df.index) > 1 else None,
            'has_gaps': len(df.index) != len(df.index.unique())
        }
    except Exception as e:
        print(f"警告: 获取日期范围信息失败: {e}")
        return {
            'min_date': None,
            'max_date': None,
            'date_count': len(df.index),
            'frequency': None,
            'has_gaps': False
        }


__all__ = [
    'standardize_date',
    'parse_date_range',
    'filter_by_date_range',
    'get_date_range_info'
]
