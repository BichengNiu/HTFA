# -*- coding: utf-8 -*-
"""
辅助函数模块

提供数值计算优化、矩阵运算和时间序列处理等辅助功能。
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List, Any
from scipy import linalg
from scipy.stats import zscore
import warnings

from .exceptions import ComputationError


def ensure_numerical_stability(
    matrix: np.ndarray,
    epsilon: float = 1e-10,
    method: str = "regularize"
) -> np.ndarray:
    """
    确保矩阵的数值稳定性

    Args:
        matrix: 输入矩阵
        epsilon: 正则化参数
        method: 稳定化方法 ("regularize", "clip", "cholesky")

    Returns:
        np.ndarray: 稳定化后的矩阵
    """
    try:
        if method == "regularize":
            # 添加正则化项到对角线 - 修复矩阵广播问题
            if matrix.ndim == 2:
                if matrix.shape[0] == matrix.shape[1]:
                    # 方阵：直接添加到对角线
                    stabilized = matrix + epsilon * np.eye(matrix.shape[0])
                else:
                    # 非方阵：只对存在的对角元素添加正则化
                    min_dim = min(matrix.shape)
                    stabilized = matrix.copy()
                    for i in range(min_dim):
                        stabilized[i, i] += epsilon
            else:
                # 1D数组或其他情况
                stabilized = matrix + epsilon
        elif method == "clip":
            # 裁剪极端值
            stabilized = np.clip(matrix, -1e10, 1e10)
        elif method == "cholesky":
            # 确保矩阵正定 - 只对方阵操作
            if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                try:
                    stabilized = nearest_positive_definite(matrix)
                except:
                    # 方阵情况下可以安全使用eye
                    stabilized = matrix + epsilon * np.eye(matrix.shape[0])
            else:
                # 非方阵不进行cholesky分解，直接返回原矩阵
                print(f"[Warning] 矩阵形状 {matrix.shape} 不适用于Cholesky分解，跳过正定化")
                stabilized = matrix
        else:
            stabilized = matrix

        return stabilized

    except Exception as e:
        raise ComputationError(f"数值稳定化失败: {str(e)}", "numerical_stability")


def safe_matrix_division(
    numerator: Union[np.ndarray, float],
    denominator: Union[np.ndarray, float],
    default_value: float = 0.0
) -> Union[np.ndarray, float]:
    """
    安全的矩阵除法，处理除零情况

    Args:
        numerator: 分子
        denominator: 分母
        default_value: 除零时的默认值

    Returns:
        除法结果
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if isinstance(denominator, (int, float)):
                if abs(denominator) < 1e-15:
                    return default_value if isinstance(numerator, (int, float)) else np.full_like(numerator, default_value)
                return numerator / denominator

            elif isinstance(denominator, np.ndarray):
                # 创建结果数组
                result = np.full_like(numerator, default_value, dtype=float)

                # 找到非零元素
                non_zero_mask = np.abs(denominator) > 1e-15

                # 只对非零元素进行除法
                if np.any(non_zero_mask):
                    if isinstance(numerator, (int, float)):
                        result[non_zero_mask] = numerator / denominator[non_zero_mask]
                    else:
                        result[non_zero_mask] = numerator[non_zero_mask] / denominator[non_zero_mask]

                return result

            else:
                return default_value

    except Exception as e:
        raise ComputationError(f"安全除法失败: {str(e)}", "matrix_division")


def align_time_series(
    *series: Union[pd.Series, pd.DataFrame],
    method: str = "inner",
    fill_method: Optional[str] = None
) -> Tuple[Union[pd.Series, pd.DataFrame], ...]:
    """
    对齐多个时间序列的索引

    Args:
        *series: 要对齐的时间序列
        method: 对齐方法 ("inner", "outer", "left", "right")
        fill_method: 填充方法 (None, "ffill", "bfill", "interpolate")

    Returns:
        对齐后的时间序列元组
    """
    try:
        if len(series) < 2:
            return series

        # 获取共同的索引
        if method == "inner":
            common_index = series[0].index
            for s in series[1:]:
                common_index = common_index.intersection(s.index)
        elif method == "outer":
            common_index = series[0].index
            for s in series[1:]:
                common_index = common_index.union(s.index)
        elif method == "left":
            common_index = series[0].index
        elif method == "right":
            common_index = series[-1].index
        else:
            raise ValueError(f"未知的对齐方法: {method}")

        # 对齐所有序列
        aligned_series = []
        for s in series:
            aligned = s.reindex(common_index)

            # 应用填充方法
            if fill_method is not None:
                if isinstance(aligned, pd.Series):
                    if fill_method == "ffill":
                        aligned = aligned.fillna(method='ffill')
                    elif fill_method == "bfill":
                        aligned = aligned.fillna(method='bfill')
                    elif fill_method == "interpolate":
                        aligned = aligned.interpolate()
                elif isinstance(aligned, pd.DataFrame):
                    if fill_method == "ffill":
                        aligned = aligned.fillna(method='ffill')
                    elif fill_method == "bfill":
                        aligned = aligned.fillna(method='bfill')
                    elif fill_method == "interpolate":
                        aligned = aligned.interpolate()

            aligned_series.append(aligned)

        return tuple(aligned_series)

    except Exception as e:
        raise ComputationError(f"时间序列对齐失败: {str(e)}", "time_series_alignment")


def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """
    找到最近的正定矩阵

    Args:
        matrix: 输入矩阵

    Returns:
        最近的正定矩阵
    """
    try:
        # 确保矩阵是对称的
        symmetric_matrix = (matrix + matrix.T) / 2

        # 特征值分解
        eigenvalues, eigenvectors = linalg.eigh(symmetric_matrix)

        # 将负特征值设置为小的正数
        min_eigenvalue = np.min(eigenvalues)
        if min_eigenvalue < 0:
            eigenvalues = np.maximum(eigenvalues, -min_eigenvalue * 0.5 + 1e-10)

        # 重构矩阵
        positive_definite = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        return positive_definite

    except Exception as e:
        raise ComputationError(f"正定矩阵计算失败: {str(e)}", "positive_definite")


def detect_outliers(
    data: Union[pd.Series, np.ndarray],
    method: str = "zscore",
    threshold: float = 3.0
) -> np.ndarray:
    """
    检测异常值

    Args:
        data: 输入数据
        method: 检测方法 ("zscore", "iqr", "modified_zscore")
        threshold: 异常值阈值

    Returns:
        布尔数组，True表示异常值
    """
    try:
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = data

        if method == "zscore":
            z_scores = np.abs(zscore(values))
            outliers = z_scores > threshold

        elif method == "iqr":
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (values < lower_bound) | (values > upper_bound)

        elif method == "modified_zscore":
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            modified_z_scores = 0.6745 * (values - median) / mad
            outliers = np.abs(modified_z_scores) > threshold

        else:
            raise ValueError(f"未知的异常值检测方法: {method}")

        return outliers

    except Exception as e:
        raise ComputationError(f"异常值检测失败: {str(e)}", "outlier_detection")