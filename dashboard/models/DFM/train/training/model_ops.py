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
    calculate_aligned_rmse,
    calculate_aligned_mae,
    calculate_aligned_hit_rate
)

logger = get_logger(__name__)


# ==================== 训练功能 ====================

def train_dfm_with_forecast(
    predictor_data: pd.DataFrame,
    target_data: pd.Series,
    k_factors: int,
    training_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    observation_end: Optional[str] = None,
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
        training_start: 训练集开始日期（必填）
        train_end: 训练集结束日期（必填）
        validation_start: 验证集开始日期（必填）
        validation_end: 验证集结束日期（必填）
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
            training_start=training_start,
            train_end=train_end
        )

    except Exception as e:
        logger.error(f"[ModelOps] 模型训练失败: {e}")
        raise RuntimeError(f"DFM模型训练失败: {e}") from e

    # 3. 生成目标变量预测
    try:
        model_result = generate_target_forecast(
            model_result=model_result,
            target_data=target_data,
            training_start=training_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            observation_end=observation_end,
            progress_callback=progress_callback
        )

    except Exception as e:
        logger.error(f"[ModelOps] 预测生成失败: {e}")
        raise RuntimeError(f"目标预测生成失败: {e}") from e

    return model_result


# ==================== 评估功能 ====================

def evaluate_model_performance(
    model_result: DFMModelResult,
    target_data: pd.Series,
    train_end: str,
    validation_start: str,
    validation_end: str,
    observation_end: Optional[str] = None,
    alignment_mode: str = 'next_month'
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
        train_end: 训练集结束日期（必填）
        validation_start: 验证集开始日期（必填）
        validation_end: 验证集结束日期（必填）
        alignment_mode: 目标配对模式 ('current_month' 或 'next_month')

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
        val_data = target_data.loc[validation_start:validation_end]

        # 2. 样本内评估
        if model_result.forecast_is is not None and len(train_data) > 0:
            _evaluate_performance(
                metrics=metrics,
                forecast=model_result.forecast_is,
                actual=train_data,
                period_type="is",
                alignment_mode=alignment_mode
            )

        # 3. 样本外评估
        if model_result.forecast_oos is not None and len(val_data) > 0:
            _evaluate_performance(
                metrics=metrics,
                forecast=model_result.forecast_oos,
                actual=val_data,
                period_type="oos",
                alignment_mode=alignment_mode
            )

        # 4. 观察期评估
        if observation_end is not None and model_result.forecast_obs is not None:
            val_end_date = pd.to_datetime(validation_end)
            obs_end_date = pd.to_datetime(observation_end)

            if obs_end_date > val_end_date:
                obs_start_date = val_end_date + pd.DateOffset(weeks=1)
                obs_data = target_data.loc[obs_start_date:obs_end_date]

                if len(obs_data) > 0:
                    _evaluate_performance(
                        metrics=metrics,
                        forecast=model_result.forecast_obs,
                        actual=obs_data,
                        period_type="obs",
                        alignment_mode=alignment_mode
                    )
                    logger.info(f"观察期评估完成: {len(obs_data)} 个数据点")

    except Exception as e:
        logger.error(f"[ModelOps] 评估过程出错: {e}")
        import traceback
        traceback.print_exc()

    return metrics


def _evaluate_performance(
    metrics: EvaluationMetrics,
    forecast: np.ndarray,
    actual: pd.Series,
    period_type: str,
    alignment_mode: str = 'next_month'
) -> None:
    """
    计算评估指标（统一的样本内/样本外评估逻辑）

    根据alignment_mode选择配对评估方式：
    - next_month: m月nowcast与m+1月target配对（默认）
    - current_month: m月nowcast与m月target配对

    Args:
        metrics: 要更新的EvaluationMetrics对象
        forecast: 预测值数组
        actual: 真实值Series
        period_type: 时间段类型，'is'表示样本内，'oos'表示样本外
        alignment_mode: 配对模式 ('current_month' 或 'next_month')
    """
    min_len = min(len(forecast), len(actual))
    forecast_aligned = forecast[:min_len]

    actual_index = pd.to_datetime(actual.index[:min_len])
    pred_series = pd.Series(forecast_aligned, index=actual_index)
    actual_series = actual

    log_prefix = "IS" if period_type == "is" else "OOS"

    # 使用统一调度函数根据配对模式计算指标
    try:
        rmse = calculate_aligned_rmse(pred_series, actual_series, alignment_mode)
        logger.debug(f"[{log_prefix}] RMSE计算成功: {rmse:.4f}")
    except Exception as e:
        logger.error(f"[{log_prefix}] RMSE计算失败: {e}")
        rmse = np.inf

    try:
        mae = calculate_aligned_mae(pred_series, actual_series, alignment_mode)
        logger.debug(f"[{log_prefix}] MAE计算成功: {mae:.4f}")
    except Exception as e:
        logger.error(f"[{log_prefix}] MAE计算失败: {e}")
        mae = np.inf

    try:
        hit_rate = calculate_aligned_hit_rate(pred_series, actual_series, alignment_mode)
        logger.debug(f"[{log_prefix}] Hit Rate计算成功: {hit_rate:.2f}%")
    except Exception as e:
        logger.error(f"[{log_prefix}] Hit Rate计算失败: {e}")
        hit_rate = np.nan

    if period_type == "is":
        metrics.is_rmse = rmse
        metrics.is_mae = mae
        metrics.is_hit_rate = hit_rate
    elif period_type == "oos":
        metrics.oos_rmse = rmse
        metrics.oos_mae = mae
        metrics.oos_hit_rate = hit_rate
    elif period_type == "obs":
        metrics.obs_rmse = rmse
        metrics.obs_mae = mae
        metrics.obs_hit_rate = hit_rate

    logger.debug(
        f"[{log_prefix}] RMSE={rmse:.4f}, "
        f"MAE={mae:.4f}, "
        f"HitRate={hit_rate:.2f}%"
    )


__all__ = ['train_dfm_with_forecast', 'evaluate_model_performance']
