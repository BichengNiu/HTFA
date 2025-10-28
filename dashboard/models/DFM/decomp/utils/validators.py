# -*- coding: utf-8 -*-
"""
数据验证器

提供各种数据验证功能，确保影响分析计算的数据质量和格式正确性。
"""

import numpy as np
import pandas as pd
from typing import Any, Optional, Tuple, List, Union
from datetime import datetime

from .exceptions import ValidationError, DataFormatError


def validate_model_data(model: Any, metadata: Any) -> Tuple[bool, List[str]]:
    """
    验证模型数据的完整性和兼容性

    Args:
        model: DFM模型对象
        metadata: 模型元数据字典

    Returns:
        Tuple[bool, List[str]]: (是否有效, 错误信息列表)
    """
    errors = []

    try:
        # 验证模型对象
        if model is None:
            errors.append("模型对象为空")
        elif not hasattr(model, '__dict__'):
            errors.append("模型对象格式无效")
        else:
            # 检查必要的属性
            required_attrs = ['factors', 'H', 'A', 'Q', 'R']
            for attr in required_attrs:
                if not hasattr(model, attr):
                    errors.append(f"模型缺少必要属性: {attr}")
                else:
                    value = getattr(model, attr)
                    if not isinstance(value, np.ndarray):
                        errors.append(f"属性 {attr} 不是numpy数组")
                    elif value.size == 0:
                        errors.append(f"属性 {attr} 为空数组")

        # 验证元数据
        if metadata is None:
            errors.append("元数据为空")
        elif not isinstance(metadata, dict):
            errors.append("元数据不是字典格式")
        else:
            # 检查必要的元数据键
            required_keys = ['complete_aligned_table', 'factor_loadings_df', 'target_variable']
            for key in required_keys:
                if key not in metadata:
                    errors.append(f"元数据缺少必要键: {key}")

        # 检查nowcast数据
        if 'complete_aligned_table' in metadata:
            nowcast_data = metadata['complete_aligned_table']
            if not isinstance(nowcast_data, (pd.DataFrame, pd.Series)):
                errors.append("nowcast数据格式无效")
            elif isinstance(nowcast_data, pd.DataFrame):
                if 'Nowcast (Original Scale)' not in nowcast_data.columns:
                    errors.append("nowcast数据中缺少目标列")
                elif nowcast_data.empty:
                    errors.append("nowcast数据为空")

        return len(errors) == 0, errors

    except Exception as e:
        return False, [f"验证过程发生异常: {str(e)}"]


def validate_impact_parameters(
    target_month: str,
    kalman_gains: np.ndarray,
    data_releases: List[Any]
) -> Tuple[bool, List[str]]:
    """
    验证影响分析参数的有效性

    Args:
        target_month: 目标月份字符串
        kalman_gains: 卡尔曼增益矩阵
        data_releases: 数据发布列表

    Returns:
        Tuple[bool, List[str]]: (是否有效, 错误信息列表)
    """
    errors = []

    try:
        # 验证目标月份格式
        if not isinstance(target_month, str):
            errors.append("目标月份必须是字符串")
        else:
            try:
                datetime.strptime(target_month, '%Y-%m')
            except ValueError:
                errors.append("目标月份格式错误，应为YYYY-MM")

        # 验证卡尔曼增益矩阵
        if not isinstance(kalman_gains, np.ndarray):
            errors.append("卡尔曼增益矩阵必须是numpy数组")
        elif kalman_gains.ndim != 2:
            errors.append("卡尔曼增益矩阵必须是二维数组")
        elif kalman_gains.shape[0] != kalman_gains.shape[1]:
            errors.append("卡尔曼增益矩阵应为方阵")
        elif np.any(np.isnan(kalman_gains)):
            errors.append("卡尔曼增益矩阵包含NaN值")
        elif np.any(np.isinf(kalman_gains)):
            errors.append("卡尔曼增益矩阵包含无穷值")

        # 验证数据发布列表
        if not isinstance(data_releases, list):
            errors.append("数据发布必须是列表")
        elif len(data_releases) == 0:
            errors.append("数据发布列表不能为空")
        else:
            for i, release in enumerate(data_releases):
                if not hasattr(release, 'observed_value'):
                    errors.append(f"数据发布 {i} 缺少观测值")
                elif not hasattr(release, 'expected_value'):
                    errors.append(f"数据发布 {i} 缺少期望值")
                elif not np.isfinite(release.observed_value):
                    errors.append(f"数据发布 {i} 观测值无效")
                elif not np.isfinite(release.expected_value):
                    errors.append(f"数据发布 {i} 期望值无效")

        return len(errors) == 0, errors

    except Exception as e:
        return False, [f"参数验证过程发生异常: {str(e)}"]


def validate_time_series(data: Union[pd.DataFrame, pd.Series], name: str = "时间序列") -> Tuple[bool, List[str]]:
    """
    验证时间序列数据的质量

    Args:
        data: 时间序列数据
        name: 数据名称（用于错误信息）

    Returns:
        Tuple[bool, List[str]]: (是否有效, 错误信息列表)
    """
    errors = []

    try:
        # 检查数据类型
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            errors.append(f"{name}必须是pandas DataFrame或Series")
            return False, errors

        # 检查数据是否为空
        if data.empty:
            errors.append(f"{name}为空")
            return False, errors

        # 检查索引是否为时间类型
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception:
                errors.append(f"{name}的索引无法转换为时间类型")

        # 检查时间序列的连续性
        if len(data) > 1:
            time_diffs = data.index.to_series().diff().dropna()
            if len(time_diffs.unique()) > 3:  # 允许少量不一致
                errors.append(f"{name}的时间间隔不规律")

        # 检查数据中的异常值
        if isinstance(data, pd.Series):
            if data.isnull().any():
                null_count = data.isnull().sum()
                errors.append(f"{name}包含 {null_count} 个缺失值")
            if np.any(np.isinf(data)):
                inf_count = np.isinf(data).sum()
                errors.append(f"{name}包含 {inf_count} 个无穷值")
        elif isinstance(data, pd.DataFrame):
            for col in data.columns:
                if data[col].isnull().any():
                    null_count = data[col].isnull().sum()
                    errors.append(f"{name}的列 {col} 包含 {null_count} 个缺失值")
                if np.any(np.isinf(data[col])):
                    inf_count = np.isinf(data[col]).sum()
                    errors.append(f"{name}的列 {col} 包含 {inf_count} 个无穷值")

        return len(errors) == 0, errors

    except Exception as e:
        return False, [f"{name}验证过程发生异常: {str(e)}"]


def validate_matrix_dimensions(
    matrix: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_dims: int = 1,
    max_dims: int = 2,
    name: str = "矩阵"
) -> Tuple[bool, List[str]]:
    """
    验证矩阵的维度和形状

    Args:
        matrix: 要验证的矩阵
        expected_shape: 期望的形状
        min_dims: 最小维度数
        max_dims: 最大维度数
        name: 矩阵名称

    Returns:
        Tuple[bool, List[str]]: (是否有效, 错误信息列表)
    """
    errors = []

    try:
        # 检查类型
        if not isinstance(matrix, np.ndarray):
            errors.append(f"{name}必须是numpy数组")
            return False, errors

        # 检查维度数
        if matrix.ndim < min_dims:
            errors.append(f"{name}维度数不足，至少需要 {min_dims} 维")
        elif matrix.ndim > max_dims:
            errors.append(f"{name}维度数过多，最多允许 {max_dims} 维")

        # 检查形状
        if expected_shape is not None:
            if len(expected_shape) != matrix.ndim:
                errors.append(f"{name}的维度数与期望不符")
            else:
                for i, (actual, expected) in enumerate(zip(matrix.shape, expected_shape)):
                    if expected != -1 and actual != expected:  # -1 表示任意大小
                        errors.append(f"{name}的第 {i+1} 维大小不匹配，实际: {actual}, 期望: {expected}")

        # 检查数值有效性
        if np.any(np.isnan(matrix)):
            errors.append(f"{name}包含NaN值")
        if np.any(np.isinf(matrix)):
            errors.append(f"{name}包含无穷值")

        # 检查是否为空
        if matrix.size == 0:
            errors.append(f"{name}为空数组")

        return len(errors) == 0, errors

    except Exception as e:
        return False, [f"{name}维度验证发生异常: {str(e)}"]


def validate_date_range(start_date: str, end_date: str, data_period: Tuple[str, str]) -> Tuple[bool, List[str]]:
    """
    验证日期范围的有效性

    Args:
        start_date: 开始日期
        end_date: 结束日期
        data_period: 数据可用期间 (开始, 结束)

    Returns:
        Tuple[bool, List[str]]: (是否有效, 错误信息列表)
    """
    errors = []

    try:
        # 解析日期
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        data_start, data_end = data_period
        data_start_dt = pd.to_datetime(data_start)
        data_end_dt = pd.to_datetime(data_end)

        # 检查日期格式和有效性
        if pd.isna(start_dt):
            errors.append("开始日期格式无效")
        if pd.isna(end_dt):
            errors.append("结束日期格式无效")

        # 检查日期逻辑
        if start_dt >= end_dt:
            errors.append("开始日期必须早于结束日期")

        # 检查日期是否在数据范围内
        if start_dt < data_start_dt:
            errors.append(f"开始日期早于数据起始日期 {data_start}")
        if end_dt > data_end_dt:
            errors.append(f"结束日期晚于数据结束日期 {data_end}")

        return len(errors) == 0, errors

    except Exception as e:
        return False, [f"日期范围验证发生异常: {str(e)}"]