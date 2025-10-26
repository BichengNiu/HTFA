# -*- coding: utf-8 -*-
"""
格式化工具模块

提供训练结果的格式化和打印功能
"""

import numpy as np
import pandas as pd
from typing import Optional, Callable
from dashboard.models.DFM.train.core.models import TrainingResult


def format_training_config(
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    train_samples: int,
    validation_samples: int,
    initial_vars: int,
    k_factors: int
) -> str:
    """
    格式化训练配置摘要

    Args:
        train_start: 训练期开始日期
        train_end: 训练期结束日期
        validation_start: 验证期开始日期
        validation_end: 验证期结束日期
        train_samples: 训练样本数
        validation_samples: 验证样本数
        initial_vars: 初始变量数
        k_factors: 因子数

    Returns:
        str: 格式化的配置摘要字符串
    """
    config_summary = f"""
========== 训练配置 ==========
训练期: {train_start} ~ {train_end} (样本数: {train_samples})
验证期: {validation_start} ~ {validation_end} (样本数: {validation_samples})
初始变量数: {initial_vars}
因子数: {k_factors}
============================
"""
    return config_summary


def format_training_summary(result: TrainingResult) -> str:
    """
    格式化训练摘要（精简版）

    将TrainingResult对象格式化为精简的摘要。

    Args:
        result: 训练结果对象

    Returns:
        str: 格式化的摘要字符串
    """
    summary = f"""
========== 最终模型 ==========
最终变量数: {len(result.selected_variables) - 1}
因子数: {result.k_factors}
训练期RMSE: {result.metrics.is_rmse:.4f}
验证期RMSE: {result.metrics.oos_rmse:.4f}
训练时间: {result.training_time:.2f}秒
============================
"""
    return summary


def print_training_summary(
    result: TrainingResult,
    progress_callback: Optional[Callable[[str], None]] = None,
    logger=None
):
    """
    打印训练摘要（精简版）

    将训练摘要输出到日志和进度回调。

    Args:
        result: 训练结果对象
        progress_callback: 进度回调函数 (message: str) -> None
        logger: 日志对象（通常是logging.Logger实例）
    """
    summary = format_training_summary(result)

    # 输出到日志（保留详细信息）
    if logger:
        logger.info(summary)

    # 输出到回调（精简信息，移除[TRAIN]标签）
    if progress_callback:
        progress_callback(summary.strip())


__all__ = [
    'format_training_config',
    'format_training_summary',
    'print_training_summary',
]
