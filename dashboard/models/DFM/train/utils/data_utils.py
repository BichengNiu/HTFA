# -*- coding: utf-8 -*-
"""
数据工具模块

提供数据加载、验证和处理相关功能
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Callable
from dashboard.models.DFM.train.utils.logger import get_logger
from dashboard.models.DFM.train.constants import MIN_REQUIRED_DATA_POINTS

logger = get_logger(__name__)


def load_and_validate_data(
    data_path: str,
    target_variable: str,
    selected_indicators: List[str],
    progress_callback: Optional[Callable] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    加载和验证训练数据

    支持Excel和CSV格式，自动进行数据质量检查和清理。

    Args:
        data_path: 数据文件路径 (支持 .xlsx, .xls, .csv)
        target_variable: 目标变量名
        selected_indicators: 选中的指标列表（空列表表示使用所有变量）
        progress_callback: 进度回调函数 (message: str) -> None

    Returns:
        (data, target_data, predictor_vars):
            - data: 完整数据 DataFrame
            - target_data: 目标变量 Series
            - predictor_vars: 有效预测变量列表

    Raises:
        ValueError: 如果文件格式不支持、目标变量不存在或有效变量不足
        FileNotFoundError: 如果文件不存在

    Examples:
        >>> data, target, predictors = load_and_validate_data(
        ...     data_path='data/经济数据.xlsx',
        ...     target_variable='GDP增速',
        ...     selected_indicators=['工业增加值', '消费', '投资']
        ... )
        >>> print(f"数据形状: {data.shape}, 预测变量数: {len(predictors)}")
    """
    if progress_callback:
        progress_callback("[DATA] 加载数据...")

    logger.info(f"加载数据: {data_path}")

    # 加载数据 - 支持Excel和CSV格式
    file_path = Path(data_path)
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    if file_path.suffix.lower() in ['.xlsx', '.xls']:
        data = pd.read_excel(data_path, index_col=0, parse_dates=True)
    elif file_path.suffix.lower() == '.csv':
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}，仅支持 .xlsx, .xls, .csv")

    # 验证目标变量
    if target_variable not in data.columns:
        raise ValueError(
            f"目标变量'{target_variable}'不在数据中. "
            f"可用列: {list(data.columns[:5])}..."
        )

    target_data = data[target_variable]

    # 确定预测变量
    if selected_indicators:
        predictor_vars = [
            v for v in selected_indicators
            if v != target_variable and v in data.columns
        ]
    else:
        predictor_vars = [
            v for v in data.columns
            if v != target_variable
        ]

    # 数据质量检查和清理
    valid_predictor_vars = []
    removed_vars = []

    for var in predictor_vars:
        var_data = data[var]
        valid_count = var_data.notna().sum()

        # 过滤全NaN或有效数据过少的列
        if valid_count < MIN_REQUIRED_DATA_POINTS:
            removed_vars.append((var, valid_count))
            logger.warning(
                f"移除变量'{var}': 有效数据点({valid_count}) < "
                f"最小要求({MIN_REQUIRED_DATA_POINTS})"
            )
        else:
            valid_predictor_vars.append(var)

    predictor_vars = valid_predictor_vars

    if removed_vars:
        logger.info(
            f"数据清理: 移除了{len(removed_vars)}个无效变量, "
            f"剩余{len(predictor_vars)}个有效变量"
        )
        if progress_callback:
            progress_callback(
                f"[DATA] 数据清理: 移除{len(removed_vars)}个无效变量"
            )

    # 验证是否还有足够的预测变量
    if len(predictor_vars) < 2:
        raise ValueError(
            f"有效预测变量不足({len(predictor_vars)}个), "
            f"至少需要2个有效变量进行DFM建模"
        )

    logger.info(
        f"数据加载完成: {data.shape}, "
        f"有效预测变量数: {len(predictor_vars)}"
    )

    if progress_callback:
        progress_callback(
            f"[DATA] 数据加载完成: {data.shape[0]}行, "
            f"{len(predictor_vars)}个有效预测变量"
        )

    return data, target_data, predictor_vars


__all__ = ['load_and_validate_data']
