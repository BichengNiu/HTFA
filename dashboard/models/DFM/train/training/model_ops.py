# -*- coding: utf-8 -*-
"""
DFM模型操作模块

合并训练和评估功能，提供统一的模型操作接口
"""

import pandas as pd
import numpy as np
from typing import Optional, Callable
from dashboard.models.DFM.train.utils.logger import get_logger
from dashboard.models.DFM.train.core.models import DFMModelResult, EvaluationMetrics
from dashboard.models.DFM.train.core.factor_model import DFMModel
from dashboard.models.DFM.train.core.prediction import generate_target_forecast
from dashboard.models.DFM.train.evaluation.metrics import (
    calculate_rmse,
    calculate_next_month_mae,
    calculate_next_month_hit_rate
)

logger = get_logger(__name__)


# ==================== 训练功能 ====================

def train_dfm_with_forecast(
    predictor_data: pd.DataFrame,
    target_data: pd.Series,
    k_factors: int,
    train_end: str,
    validation_start: Optional[str] = None,
    validation_end: Optional[str] = None,
    max_iter: int = 30,
    max_lags: int = 1,
    tolerance: float = 1e-6,
    progress_callback: Optional[Callable[[str], None]] = None
) -> DFMModelResult:
    """
    统一的DFM训练和预测函数

    完成以下流程：
    1. 创建并配置DFM模型
    2. 在训练集上拟合模型 (EM算法)
    3. 生成目标变量的样本内和样本外预测

    Args:
        predictor_data: 预测变量数据 (DataFrame, columns=变量名, index=日期)
        target_data: 目标变量数据 (Series, index=日期)
        k_factors: 因子个数
        train_end: 训练集结束日期
        validation_start: 验证集开始日期 (可选)
        validation_end: 验证集结束日期 (可选)
        max_iter: EM算法最大迭代次数
        max_lags: 因子自回归最大滞后阶数
        tolerance: EM算法收敛容差
        progress_callback: 进度回调函数

    Returns:
        DFMModelResult: 包含模型参数、因子、预测值的完整结果

    Raises:
        ValueError: 如果数据格式不正确或参数无效
        RuntimeError: 如果模型训练失败

    Examples:
        >>> result = train_dfm_with_forecast(
        ...     predictor_data=df[indicator_cols],
        ...     target_data=df['target'],
        ...     k_factors=3,
        ...     train_end='2023-12-31',
        ...     validation_start='2024-01-01',
        ...     validation_end='2024-06-30',
        ...     max_iter=50
        ... )
        >>> print(f"收敛: {result.converged}, 迭代: {result.iterations}次")
    """
    # 参数验证
    if k_factors <= 0:
        raise ValueError(f"k_factors必须为正数，当前值: {k_factors}")
    if max_iter <= 0:
        raise ValueError(f"max_iter必须为正数，当前值: {max_iter}")
    if max_lags < 1:
        raise ValueError(f"max_lags必须>=1，当前值: {max_lags}")

    # logger.info(
    #     f"[ModelOps] 开始训练 - k={k_factors}, "
    #     f"变量数={predictor_data.shape[1]}, "
    #     f"样本数={predictor_data.shape[0]}"
    # )

    # 1. 创建并配置DFM模型
    dfm = DFMModel(
        n_factors=k_factors,
        max_lags=max_lags,
        max_iter=max_iter,
        tolerance=tolerance
    )

    # 2. 训练模型
    try:
        model_result = dfm.fit(
            data=predictor_data,
            train_end=train_end
        )

        # logger.info(
        #     f"[ModelOps] 模型训练完成 - "
        #     f"收敛={model_result.converged}, "
        #     f"迭代={model_result.iterations}次, "
        #     f"LogLik={model_result.log_likelihood:.2f}"
        # )

    except Exception as e:
        logger.error(f"[ModelOps] 模型训练失败: {e}")
        raise RuntimeError(f"DFM模型训练失败: {e}") from e

    # 3. 生成目标变量预测
    try:
        model_result = generate_target_forecast(
            model_result=model_result,
            target_data=target_data,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            progress_callback=progress_callback
        )

        # logger.info(
        #     f"[ModelOps] 预测生成完成 - "
        #     f"IS预测点数={len(model_result.forecast_is) if model_result.forecast_is is not None else 0}, "
        #     f"OOS预测点数={len(model_result.forecast_oos) if model_result.forecast_oos is not None else 0}"
        # )

    except Exception as e:
        logger.error(f"[ModelOps] 预测生成失败: {e}")
        raise RuntimeError(f"目标预测生成失败: {e}") from e

    return model_result


# ==================== 评估功能 ====================

def evaluate_model_performance(
    model_result: DFMModelResult,
    target_data: pd.Series,
    train_end: str,
    validation_start: Optional[str] = None,
    validation_end: Optional[str] = None
) -> EvaluationMetrics:
    """
    统一的模型性能评估函数

    计算样本内(IS)和样本外(OOS)的评估指标：
    - RMSE (均方根误差)
    - Hit Rate (方向命中率)
    - Correlation (相关系数)

    Args:
        model_result: DFM模型结果，包含预测值
        target_data: 目标变量的真实值 (Series with DatetimeIndex)
        train_end: 训练集结束日期
        validation_start: 验证集开始日期 (可选)
        validation_end: 验证集结束日期 (可选)

    Returns:
        EvaluationMetrics: 包含所有评估指标的对象

    Examples:
        >>> metrics = evaluate_model_performance(
        ...     model_result=dfm_result,
        ...     target_data=target_series,
        ...     train_end='2023-12-31',
        ...     validation_start='2024-01-01',
        ...     validation_end='2024-06-30'
        ... )
        >>> print(f"OOS RMSE: {metrics.oos_rmse:.4f}")
    """
    metrics = EvaluationMetrics()

    try:
        # 1. 分割样本内和样本外数据
        train_data = target_data.loc[:train_end]

        if validation_start and validation_end:
            val_data = target_data.loc[validation_start:validation_end]
        else:
            # 如果未指定验证期，使用训练期之后的所有数据
            val_data = target_data.loc[train_end:]
            if len(val_data) > 0:
                val_data = val_data.iloc[1:]  # 排除训练期最后一天

        # 2. 样本内评估
        if model_result.forecast_is is not None and len(train_data) > 0:
            _evaluate_in_sample(
                metrics=metrics,
                forecast=model_result.forecast_is,
                actual=train_data
            )

        # 3. 样本外评估
        if model_result.forecast_oos is not None and len(val_data) > 0:
            _evaluate_out_of_sample(
                metrics=metrics,
                forecast=model_result.forecast_oos,
                actual=val_data
            )

    except Exception as e:
        logger.error(f"[ModelOps] 评估过程出错: {e}")
        import traceback
        traceback.print_exc()

    return metrics


def _evaluate_in_sample(
    metrics: EvaluationMetrics,
    forecast: np.ndarray,
    actual: pd.Series
) -> None:
    """
    计算样本内评估指标

    Args:
        metrics: 要更新的EvaluationMetrics对象
        forecast: 预测值数组
        actual: 真实值Series
    """
    # 对齐长度
    min_len = min(len(forecast), len(actual))
    forecast_aligned = forecast[:min_len]
    actual_aligned = actual.values[:min_len]

    # 计算RMSE（保持原逻辑，基于样本内对齐）
    metrics.is_rmse = calculate_rmse(actual_aligned, forecast_aligned)

    # 创建周度预测和目标值Series - 确保使用DatetimeIndex
    actual_index = pd.to_datetime(actual.index[:min_len])
    pred_series = pd.Series(forecast_aligned, index=actual_index)
    actual_series = pd.Series(actual_aligned, index=actual_index)

    # 计算MAE（新：使用下月配对）
    try:
        metrics.is_mae = calculate_next_month_mae(pred_series, actual_series)
        logger.debug(f"[IS] MAE计算成功: {metrics.is_mae:.4f}")
    except Exception as e:
        logger.error(f"[IS] MAE计算失败: {e}")
        metrics.is_mae = np.inf

    # 计算Hit Rate（新：使用新定义）
    try:
        metrics.is_hit_rate = calculate_next_month_hit_rate(pred_series, actual_series)
        logger.debug(f"[IS] Hit Rate计算成功: {metrics.is_hit_rate:.2f}%")
    except Exception as e:
        logger.error(f"[IS] Hit Rate计算失败: {e}")
        metrics.is_hit_rate = np.nan

    logger.debug(
        f"[IS] RMSE={metrics.is_rmse:.4f}, "
        f"MAE={metrics.is_mae:.4f}, "
        f"HitRate={metrics.is_hit_rate:.2f}%"
    )


def _evaluate_out_of_sample(
    metrics: EvaluationMetrics,
    forecast: np.ndarray,
    actual: pd.Series
) -> None:
    """
    计算样本外评估指标

    Args:
        metrics: 要更新的EvaluationMetrics对象
        forecast: 预测值数组
        actual: 真实值Series
    """
    # 对齐长度
    min_len = min(len(forecast), len(actual))
    forecast_aligned = forecast[:min_len]
    actual_aligned = actual.values[:min_len]

    # 计算RMSE（保持原逻辑，基于样本外对齐）
    metrics.oos_rmse = calculate_rmse(actual_aligned, forecast_aligned)

    # 创建周度预测和目标值Series - 确保使用DatetimeIndex
    actual_index = pd.to_datetime(actual.index[:min_len])
    pred_series = pd.Series(forecast_aligned, index=actual_index)
    actual_series = pd.Series(actual_aligned, index=actual_index)

    # 计算MAE（新：使用下月配对）
    try:
        metrics.oos_mae = calculate_next_month_mae(pred_series, actual_series)
        logger.debug(f"[OOS] MAE计算成功: {metrics.oos_mae:.4f}")
    except Exception as e:
        logger.error(f"[OOS] MAE计算失败: {e}")
        metrics.oos_mae = np.inf

    # 计算Hit Rate（新：使用新定义）
    try:
        metrics.oos_hit_rate = calculate_next_month_hit_rate(pred_series, actual_series)
        logger.debug(f"[OOS] Hit Rate计算成功: {metrics.oos_hit_rate:.2f}%")
    except Exception as e:
        logger.error(f"[OOS] Hit Rate计算失败: {e}")
        metrics.oos_hit_rate = np.nan

    logger.debug(
        f"[OOS] RMSE={metrics.oos_rmse:.4f}, "
        f"MAE={metrics.oos_mae:.4f}, "
        f"HitRate={metrics.oos_hit_rate:.2f}%"
    )


__all__ = ['train_dfm_with_forecast', 'evaluate_model_performance']
