"""
文本处理工具模块

提供统一的文本标准化功能，用于处理中文变量名、列名等
"""

import unicodedata
import pandas as pd
from typing import Union, Optional


def normalize_text(text: Union[str, float, None], to_lower: bool = True) -> str:
    """
    标准化文本，移除特殊字符和空格

    此函数适用于中文和英文混合文本的标准化处理，
    主要用于变量名、列名的规范化以确保匹配一致性。

    Args:
        text: 待标准化的文本，可以是字符串、浮点数或None
        to_lower: 是否转换为小写，默认True

    Returns:
        str: 标准化后的文本，如果输入为空则返回空字符串

    Examples:
        >>> normalize_text('  工业增加值  ')
        '工业增加值'
        >>> normalize_text('GDP Growth Rate', to_lower=True)
        'gdp growth rate'
        >>> normalize_text(None)
        ''
    """
    if pd.isna(text) or text == '':
        return ''

    # 转换为字符串并标准化Unicode（NFKC规范化）
    text = str(text)
    text = unicodedata.normalize('NFKC', text)

    # 移除前后空格
    text = text.strip()

    # 根据参数决定是否转换为小写
    # 对于包含ASCII字符的文本进行小写转换
    if to_lower:
        if any(ord(char) < 128 for char in text):
            # 包含ASCII字符，可能有英文，进行小写转换
            text = text.lower()

    return text


def normalize_column_name(column_name: Union[str, float, None]) -> str:
    """
    标准化列名

    这是 normalize_text 的便利包装函数，专门用于列名标准化。
    默认转换为小写并去除空格。

    Args:
        column_name: 列名

    Returns:
        str: 标准化后的列名

    Examples:
        >>> normalize_column_name('  规模以上工业增加值  ')
        '规模以上工业增加值'
        >>> normalize_column_name('Total_Revenue')
        'total_revenue'
    """
    return normalize_text(column_name, to_lower=True)


def create_normalized_mapping(
    keys: list,
    values: list,
    normalize_keys: bool = True,
    normalize_values: bool = False
) -> dict:
    """
    创建标准化后的映射字典

    Args:
        keys: 键列表
        values: 值列表
        normalize_keys: 是否标准化键，默认True
        normalize_values: 是否标准化值，默认False

    Returns:
        dict: 标准化后的映射字典，过滤掉空键或空值

    Examples:
        >>> keys = ['  工业  ', 'GDP', None]
        >>> values = ['Industry', 'gdp', 'empty']
        >>> create_normalized_mapping(keys, values)
        {'工业': 'Industry', 'gdp': 'gdp'}
    """
    if len(keys) != len(values):
        raise ValueError(f"键和值的数量不匹配：{len(keys)} vs {len(values)}")

    result = {}
    for k, v in zip(keys, values):
        # 标准化键
        normalized_key = normalize_text(k) if normalize_keys else k

        # 标准化值
        normalized_value = normalize_text(v, to_lower=False) if normalize_values else v

        # 过滤空键或空值
        if normalized_key and normalized_value and str(normalized_value).strip().lower() not in ['nan', '']:
            result[normalized_key] = normalized_value

    return result


__all__ = [
    'normalize_text',
    'normalize_column_name',
    'create_normalized_mapping'
]
