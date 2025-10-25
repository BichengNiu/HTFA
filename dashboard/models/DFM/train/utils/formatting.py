# -*- coding: utf-8 -*-
"""
格式化工具模块

提供训练结果的格式化和打印功能
"""

import numpy as np
from typing import Optional, Callable
from dashboard.models.DFM.train.core.models import TrainingResult


def format_training_summary(result: TrainingResult) -> str:
    """
    格式化训练摘要

    将TrainingResult对象格式化为易读的文本摘要。

    Args:
        result: 训练结果对象

    Returns:
        str: 格式化的摘要字符串

    Examples:
        >>> summary = format_training_summary(training_result)
        >>> print(summary)
        ========== 训练摘要 ==========
        变量数: 15
        因子数: 3
        ...
    """
    # 格式化Hit Rate显示（处理无穷大和NaN）
    is_hit_rate_display = (
        f"{result.metrics.is_hit_rate:.2f}%"
        if np.isfinite(result.metrics.is_hit_rate)
        else "N/A (数据不足)"
    )
    oos_hit_rate_display = (
        f"{result.metrics.oos_hit_rate:.2f}%"
        if np.isfinite(result.metrics.oos_hit_rate)
        else "N/A (数据不足)"
    )

    summary = f"""
========== 训练摘要 ==========
变量数: {len(result.selected_variables) - 1}
因子数: {result.k_factors}
迭代次数: {result.model_result.iterations}
收敛: {result.model_result.converged}

样本内RMSE: {result.metrics.is_rmse:.4f}
样本外RMSE: {result.metrics.oos_rmse:.4f}
样本内命中率: {is_hit_rate_display}
样本外命中率: {oos_hit_rate_display}

总评估次数: {result.total_evaluations}
SVD错误: {result.svd_error_count}
训练时间: {result.training_time:.2f}秒
=============================
"""
    return summary


def print_training_summary(
    result: TrainingResult,
    progress_callback: Optional[Callable[[str], None]] = None,
    logger=None
):
    """
    打印训练摘要

    将训练摘要输出到日志和进度回调。

    Args:
        result: 训练结果对象
        progress_callback: 进度回调函数 (message: str) -> None
        logger: 日志对象（通常是logging.Logger实例）

    Examples:
        >>> from dashboard.models.DFM.train.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> print_training_summary(result, progress_callback, logger)
    """
    summary = format_training_summary(result)

    # 输出到日志
    if logger:
        logger.info(summary)

    # 输出到回调
    if progress_callback:
        progress_callback(f"[TRAIN] {summary}")


__all__ = [
    'format_training_summary',
    'print_training_summary',
]
