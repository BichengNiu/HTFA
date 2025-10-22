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


def verify_alignment(
    data: pd.DataFrame,
    variables: List[str],
    strict: bool = True
) -> Tuple[bool, Dict]:
    """验证数据对齐情况

    检查指定变量的数据是否在时间维度上对齐，
    包括索引一致性、缺失值分布等。

    Args:
        data: 输入数据
        variables: 要验证的变量列表
        strict: 是否严格模式（要求完全无缺失值）

    Returns:
        Tuple[bool, Dict]: (是否对齐, 诊断信息)
    """
    diagnosis = {
        'aligned': True,
        'missing_variables': [],
        'missing_counts': {},
        'coverage': {},
        'common_dates': 0,
        'issues': []
    }

    # 检查变量是否存在
    missing_vars = [v for v in variables if v not in data.columns]
    if missing_vars:
        diagnosis['aligned'] = False
        diagnosis['missing_variables'] = missing_vars
        diagnosis['issues'].append(f"缺失变量: {missing_vars}")
        return False, diagnosis

    # 选择相关变量
    subset = data[variables]

    # 统计每个变量的缺失值
    for var in variables:
        missing_count = subset[var].isna().sum()
        total_count = len(subset)
        coverage = (total_count - missing_count) / total_count * 100

        diagnosis['missing_counts'][var] = int(missing_count)
        diagnosis['coverage'][var] = round(coverage, 2)

        if strict and missing_count > 0:
            diagnosis['aligned'] = False
            diagnosis['issues'].append(
                f"变量 {var} 有 {missing_count} 个缺失值 (覆盖率: {coverage:.1f}%)"
            )

    # 统计共同有效日期数量
    common_valid = subset.dropna()
    diagnosis['common_dates'] = len(common_valid)

    if diagnosis['common_dates'] == 0:
        diagnosis['aligned'] = False
        diagnosis['issues'].append("没有共同的完整观测点")
    elif not strict and diagnosis['common_dates'] < len(subset) * 0.5:
        diagnosis['aligned'] = False
        diagnosis['issues'].append(
            f"共同有效日期过少: {diagnosis['common_dates']}/{len(subset)} "
            f"({diagnosis['common_dates']/len(subset)*100:.1f}%)"
        )

    # 记录日志
    if diagnosis['aligned']:
        logger.info(
            f"数据对齐验证通过: {len(variables)}个变量, "
            f"共同有效日期: {diagnosis['common_dates']}/{len(subset)}"
        )
    else:
        logger.warning(
            f"数据对齐验证失败: {', '.join(diagnosis['issues'])}"
        )

    return diagnosis['aligned'], diagnosis


def check_data_quality(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    max_missing_ratio: float = 0.3,
    min_variance: float = 1e-8
) -> Tuple[bool, List[str], Dict]:
    """检查数据质量

    检查数据的基本质量指标，包括缺失值比例、方差等。

    Args:
        data: 输入数据
        variables: 要检查的变量列表，None表示检查所有列
        max_missing_ratio: 最大允许缺失值比例
        min_variance: 最小方差阈值（低于此值认为是常数）

    Returns:
        Tuple[bool, List[str], Dict]: (是否通过, 问题变量列表, 详细诊断)
    """
    if variables is None:
        variables = data.columns.tolist()

    diagnosis = {
        'passed': True,
        'total_variables': len(variables),
        'problematic_variables': [],
        'quality_metrics': {}
    }

    problematic = []

    for var in variables:
        if var not in data.columns:
            problematic.append(var)
            diagnosis['quality_metrics'][var] = {
                'issue': 'variable_not_found'
            }
            continue

        series = data[var]
        metrics = {}

        # 检查缺失值
        missing_count = series.isna().sum()
        missing_ratio = missing_count / len(series)
        metrics['missing_count'] = int(missing_count)
        metrics['missing_ratio'] = round(missing_ratio, 4)

        # 检查方差
        valid_data = series.dropna()
        if len(valid_data) > 0:
            variance = valid_data.var()
            metrics['variance'] = float(variance)
            metrics['is_constant'] = variance < min_variance
        else:
            metrics['variance'] = 0.0
            metrics['is_constant'] = True

        # 判断是否有问题
        issues = []
        if missing_ratio > max_missing_ratio:
            issues.append(f"缺失值过多: {missing_ratio*100:.1f}%")
        if metrics['is_constant']:
            issues.append(f"方差过低: {metrics['variance']:.2e}")

        if issues:
            problematic.append(var)
            metrics['issues'] = issues

        diagnosis['quality_metrics'][var] = metrics

    diagnosis['problematic_variables'] = problematic
    diagnosis['passed'] = len(problematic) == 0

    # 记录日志
    if diagnosis['passed']:
        logger.info(f"数据质量检查通过: {len(variables)}个变量均符合要求")
    else:
        logger.warning(
            f"数据质量检查发现问题: {len(problematic)}个变量有问题 - {problematic}"
        )

    return diagnosis['passed'], problematic, diagnosis
