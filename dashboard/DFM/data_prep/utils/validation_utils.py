"""
数据验证工具模块

提供统一的数据验证功能，用于检查数据质量
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple


def is_empty_data(data: Union[pd.Series, pd.DataFrame]) -> bool:
    """
    检查数据是否为空

    Args:
        data: Series或DataFrame

    Returns:
        bool: 如果数据为空或全为NaN返回True

    Examples:
        >>> s = pd.Series([np.nan, np.nan, np.nan])
        >>> is_empty_data(s)
        True
        >>> s = pd.Series([1, 2, 3])
        >>> is_empty_data(s)
        False
    """
    if data is None:
        return True

    if isinstance(data, pd.DataFrame):
        return data.empty or data.isna().all().all()
    elif isinstance(data, pd.Series):
        return data.empty or data.isna().all()
    else:
        return False


def is_constant_data(data: Union[pd.Series, pd.DataFrame]) -> bool:
    """
    检查数据是否为常量（所有非NaN值相同）

    Args:
        data: Series或DataFrame

    Returns:
        bool: 如果数据为常量返回True

    Examples:
        >>> s = pd.Series([1, 1, 1, np.nan])
        >>> is_constant_data(s)
        True
        >>> s = pd.Series([1, 2, 3])
        >>> is_constant_data(s)
        False
    """
    if data is None or is_empty_data(data):
        return False

    if isinstance(data, pd.Series):
        non_nan_data = data.dropna()
        if non_nan_data.empty:
            return False
        return non_nan_data.nunique() == 1

    elif isinstance(data, pd.DataFrame):
        # 对DataFrame，检查每列是否都是常量
        return all(is_constant_data(data[col]) for col in data.columns)

    return False


def has_sufficient_valid_data(
    data: Union[pd.Series, pd.DataFrame],
    min_valid_ratio: float = 0.5,
    min_valid_count: int = 10
) -> bool:
    """
    检查数据是否有足够的有效值

    Args:
        data: Series或DataFrame
        min_valid_ratio: 最小有效值比例，默认0.5（50%）
        min_valid_count: 最小有效值数量，默认10

    Returns:
        bool: 如果有效数据足够返回True

    Examples:
        >>> s = pd.Series([1, 2, 3, np.nan, np.nan])
        >>> has_sufficient_valid_data(s, min_valid_ratio=0.5, min_valid_count=3)
        True
        >>> has_sufficient_valid_data(s, min_valid_ratio=0.8, min_valid_count=3)
        False
    """
    if data is None or is_empty_data(data):
        return False

    if isinstance(data, pd.Series):
        valid_count = data.notna().sum()
        total_count = len(data)

        if total_count == 0:
            return False

        valid_ratio = valid_count / total_count
        return valid_count >= min_valid_count and valid_ratio >= min_valid_ratio

    elif isinstance(data, pd.DataFrame):
        # 对DataFrame，检查每列是否都有足够的有效数据
        return all(
            has_sufficient_valid_data(data[col], min_valid_ratio, min_valid_count)
            for col in data.columns
        )

    return False


def calculate_data_quality_score(data: Union[pd.Series, pd.DataFrame]) -> float:
    """
    计算数据质量得分（0-1之间）

    考虑因素：
    - 有效值比例
    - 是否为常量
    - 数据量大小

    Args:
        data: Series或DataFrame

    Returns:
        float: 数据质量得分，0表示质量最差，1表示质量最好

    Examples:
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> score = calculate_data_quality_score(s)
        >>> score >= 0.9
        True
    """
    if data is None or is_empty_data(data):
        return 0.0

    if isinstance(data, pd.Series):
        # 计算有效值比例
        valid_ratio = data.notna().sum() / len(data)

        # 检查是否为常量（常量数据质量较低）
        is_const = is_constant_data(data)
        constant_penalty = 0.5 if is_const else 0.0

        # 计算数据量得分（数据越多越好，但有上限）
        size_score = min(len(data) / 100, 1.0)

        # 综合得分
        score = (valid_ratio * 0.6 + size_score * 0.4) * (1.0 - constant_penalty)
        return max(0.0, min(1.0, score))

    elif isinstance(data, pd.DataFrame):
        # 对DataFrame，计算所有列的平均得分
        if data.empty or len(data.columns) == 0:
            return 0.0

        scores = [calculate_data_quality_score(data[col]) for col in data.columns]
        return sum(scores) / len(scores)

    return 0.0


def get_data_quality_report(data: Union[pd.Series, pd.DataFrame]) -> dict:
    """
    生成数据质量报告

    Args:
        data: Series或DataFrame

    Returns:
        dict: 包含数据质量指标的字典

    Examples:
        >>> s = pd.Series([1, 2, 3, np.nan, np.nan])
        >>> report = get_data_quality_report(s)
        >>> report['total_count']
        5
        >>> report['valid_count']
        3
    """
    if data is None:
        return {
            'total_count': 0,
            'valid_count': 0,
            'missing_count': 0,
            'missing_ratio': 0.0,
            'is_empty': True,
            'is_constant': False,
            'quality_score': 0.0
        }

    if isinstance(data, pd.Series):
        total_count = len(data)
        valid_count = data.notna().sum()
        missing_count = data.isna().sum()

        return {
            'total_count': total_count,
            'valid_count': valid_count,
            'missing_count': missing_count,
            'missing_ratio': missing_count / total_count if total_count > 0 else 0.0,
            'is_empty': is_empty_data(data),
            'is_constant': is_constant_data(data),
            'unique_values': data.nunique() if not is_empty_data(data) else 0,
            'quality_score': calculate_data_quality_score(data)
        }

    elif isinstance(data, pd.DataFrame):
        total_count = len(data)
        column_count = len(data.columns)

        return {
            'total_count': total_count,
            'column_count': column_count,
            'is_empty': is_empty_data(data),
            'quality_score': calculate_data_quality_score(data),
            'per_column_quality': {
                col: get_data_quality_report(data[col])
                for col in data.columns
            }
        }

    return {}


__all__ = [
    'is_empty_data',
    'is_constant_data',
    'has_sufficient_valid_data',
    'calculate_data_quality_score',
    'get_data_quality_report'
]
