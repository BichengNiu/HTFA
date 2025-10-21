# -*- coding: utf-8 -*-
"""
数据处理工具模块
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from pathlib import Path
from dashboard.DFM.train_ref.utils.logger import get_logger


logger = get_logger(__name__)


def load_data(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """加载数据文件

    Args:
        file_path: 文件路径
        sheet_name: Excel工作表名称

    Returns:
        pd.DataFrame: 加载的数据

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式不支持
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, sheet_name=sheet_name or 0, index_col=0, parse_dates=True)
    elif suffix == '.csv':
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("索引不是DatetimeIndex，尝试转换")
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(f"索引转换为DatetimeIndex失败: {e}")

    logger.info(f"成功加载数据: {df.shape}, 时间范围: {df.index.min()} 到 {df.index.max()}")

    return df


def prepare_data(
    df: pd.DataFrame,
    target_variable: str,
    selected_variables: Optional[List[str]] = None,
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """准备训练数据

    Args:
        df: 原始数据
        target_variable: 目标变量
        selected_variables: 已选变量列表
        end_date: 截止日期

    Returns:
        Tuple[pd.DataFrame, List[str]]: (准备好的数据, 预测变量列表)
    """
    data = df.copy()

    if end_date:
        try:
            data = data.loc[:end_date]
        except KeyError:
            logger.warning(f"截止日期{end_date}超出数据范围，使用全部数据")

    if selected_variables:
        variables = [v for v in selected_variables if v in data.columns]
        if target_variable not in variables:
            variables.append(target_variable)
        data = data[variables]
    else:
        if target_variable not in data.columns:
            raise ValueError(f"目标变量{target_variable}不在数据中")

    predictor_variables = [col for col in data.columns if col != target_variable]

    logger.info(f"数据准备完成: {data.shape}, 预测变量数: {len(predictor_variables)}")

    return data, predictor_variables


def split_train_validation(
    df: pd.DataFrame,
    train_end: str,
    validation_start: Optional[str] = None,
    validation_end: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """分割训练集和验证集

    Args:
        df: 完整数据
        train_end: 训练集结束日期
        validation_start: 验证集开始日期
        validation_end: 验证集结束日期

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (训练集, 验证集)
    """
    train_data = df.loc[:train_end]

    if validation_end:
        validation_data = df.loc[validation_start:validation_end] if validation_start else df.loc[train_end:validation_end]
    else:
        validation_data = df.loc[validation_start:] if validation_start else df.loc[train_end:]

    logger.info(f"训练集: {train_data.shape}, 验证集: {validation_data.shape}")

    return train_data, validation_data


def standardize_data(
    data: pd.DataFrame,
    fit_data: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """标准化数据

    Args:
        data: 要标准化的数据
        fit_data: 用于计算均值和标准差的数据，如果为None则使用data本身

    Returns:
        Tuple[pd.DataFrame, Dict]: (标准化后的数据, 统计量字典)
    """
    if fit_data is None:
        fit_data = data

    stats = {}
    standardized = pd.DataFrame(index=data.index)

    for col in data.columns:
        mean_val = fit_data[col].mean()
        std_val = fit_data[col].std()

        if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
            logger.warning(f"列{col}标准化失败: mean={mean_val}, std={std_val}，跳过标准化")
            standardized[col] = data[col]
            stats[col] = (mean_val, 1.0)
        else:
            standardized[col] = (data[col] - mean_val) / std_val
            stats[col] = (mean_val, std_val)

    logger.debug(f"数据标准化完成: {standardized.shape}")

    return standardized, stats


def destandardize_series(
    series: pd.Series,
    mean: float,
    std: float
) -> pd.Series:
    """反标准化序列

    Args:
        series: 标准化的序列
        mean: 均值
        std: 标准差

    Returns:
        pd.Series: 反标准化后的序列
    """
    return series * std + mean


def apply_seasonal_mask(
    data: pd.DataFrame,
    target_variable: str,
    months_to_mask: List[int] = None
) -> pd.DataFrame:
    """应用季节性掩码

    Args:
        data: 输入数据
        target_variable: 目标变量
        months_to_mask: 要掩码的月份列表

    Returns:
        pd.DataFrame: 应用掩码后的数据
    """
    if months_to_mask is None:
        months_to_mask = [1, 2]

    masked_data = data.copy()

    mask = masked_data.index.month.isin(months_to_mask)
    masked_count = mask.sum()

    masked_data.loc[mask, target_variable] = np.nan

    logger.info(f"对目标变量应用季节性掩码: {months_to_mask}月，掩码数量: {masked_count}")

    return masked_data
