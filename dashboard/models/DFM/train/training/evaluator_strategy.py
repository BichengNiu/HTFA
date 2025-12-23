# -*- coding: utf-8 -*-
"""
DFM评估策略 - 函数式接口

提供简洁的函数式接口创建DFM评估器

重构说明:
- 将闭包函数改为模块级顶层函数,解决pickle序列化问题
- 通过参数显式传递config数据,而非闭包捕获
- 保持API兼容性
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Dict, Any
from dashboard.models.DFM.train.utils.logger import get_logger
from dashboard.models.DFM.train.training.model_ops import train_dfm_with_forecast, evaluate_model_performance

logger = get_logger(__name__)


# ========== 可序列化的顶层评估函数 ==========

def _evaluate_dfm_model(
    variables: List[str],
    target_variable: str,
    full_data: pd.DataFrame,
    k_factors: int,
    training_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    max_iterations: int,
    tolerance: float,
    **kwargs
) -> Tuple[float, float, None, None, float, float, bool, None, None]:
    """
    顶层DFM评估函数（可序列化）

    Args:
        variables: 变量列表（包含目标变量）
        target_variable: 目标变量名
        full_data: 完整数据DataFrame
        k_factors: 因子数
        training_start: 训练开始日期
        train_end: 训练结束日期
        validation_start: 验证开始日期
        validation_end: 验证结束日期
        max_iter: 最大迭代次数
        tolerance: 容差
        **kwargs: 其他参数

    Returns:
        9元组: (is_rmse, oos_rmse, _, _, is_win_rate, oos_win_rate,
               is_svd_error, _, _)
    """
    try:
        # 分离预测变量
        predictor_vars = [v for v in variables if v != target_variable]

        if len(predictor_vars) == 0:
            logger.warning("[Evaluator] 预测变量为空，返回无穷大RMSE")
            return (np.inf, np.inf, np.nan, np.nan, np.nan, np.nan, False, None, None)

        # 准备数据
        predictor_data = full_data[predictor_vars]
        target_data = full_data[target_variable]

        # 训练模型
        model_result = train_dfm_with_forecast(
            predictor_data=predictor_data,
            target_data=target_data,
            k_factors=k_factors,
            training_start=training_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            max_iter=max_iterations,
            max_lags=1,
            tolerance=tolerance,
            progress_callback=None
        )

        # 评估模型
        metrics = evaluate_model_performance(
            model_result=model_result,
            target_data=target_data,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end
        )

        return metrics.to_tuple()

    except Exception as e:
        logger.error(f"[Evaluator] 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return (np.inf, np.inf, None, None, -np.inf, -np.inf, True, None, None)


def _evaluate_variable_selection_model(
    variables: List[str],
    target_variable: str,
    full_data: pd.DataFrame,
    k_factors: int,
    training_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    max_iterations: int,
    tolerance: float,
    alignment_mode: str = 'next_month',
    **kwargs
) -> Tuple[float, float, float, float, float, float, bool, None, None]:
    """
    顶层变量选择评估函数（可序列化）

    Args:
        variables: 变量列表（包含目标变量）
        target_variable: 目标变量名
        full_data: 完整数据DataFrame
        k_factors: 因子数
        training_start: 训练开始日期
        train_end: 训练结束日期
        validation_start: 验证开始日期
        validation_end: 验证结束日期
        max_iter: 最大迭代次数
        tolerance: 容差
        alignment_mode: 目标配对模式 ('current_month' 或 'next_month')
        **kwargs: 其他参数

    Returns:
        9元组: (is_rmse, oos_rmse, None, None, is_win_rate, oos_win_rate, is_svd_error, None, None)
    """
    try:
        # 分离预测变量
        predictor_vars = [v for v in variables if v != target_variable]

        if len(predictor_vars) == 0:
            logger.warning("[VarSelectionEvaluator] 预测变量为空，返回无穷大RMSE")
            return (np.inf, np.inf, np.nan, np.nan, np.nan, np.nan, False, None, None)

        # 准备数据
        predictor_data = full_data[predictor_vars]
        target_data = full_data[target_variable]

        # 训练模型
        model_result = train_dfm_with_forecast(
            predictor_data=predictor_data,
            target_data=target_data,
            k_factors=k_factors,
            training_start=training_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            max_iter=max_iterations,
            max_lags=1,
            tolerance=tolerance,
            progress_callback=None
        )

        # 导入评估函数
        from dashboard.models.DFM.train.evaluation.metrics import (
            calculate_aligned_rmse,
            calculate_aligned_win_rate
        )

        # 训练期评估
        is_rmse = np.inf
        is_win_rate = np.nan
        if model_result.forecast_is is not None and len(model_result.forecast_is) > 0:
            train_data_len = len(model_result.forecast_is)
            train_index = pd.to_datetime(predictor_data.index[:train_data_len])
            train_nowcast = pd.Series(
                model_result.forecast_is,
                index=train_index
            )
            is_rmse = calculate_aligned_rmse(train_nowcast, target_data, alignment_mode)
            is_win_rate = calculate_aligned_win_rate(train_nowcast, target_data, alignment_mode)

        # 观察期评估
        oos_rmse = np.inf
        oos_win_rate = np.nan
        if model_result.forecast_oos is not None and len(model_result.forecast_oos) > 0:
            train_data_len = len(model_result.forecast_is) if model_result.forecast_is is not None else 0
            val_index = pd.to_datetime(predictor_data.index[train_data_len:train_data_len+len(model_result.forecast_oos)])

            if len(val_index) == len(model_result.forecast_oos):
                val_nowcast = pd.Series(
                    model_result.forecast_oos,
                    index=val_index
                )
                oos_rmse = calculate_aligned_rmse(val_nowcast, target_data, alignment_mode)
                oos_win_rate = calculate_aligned_win_rate(val_nowcast, target_data, alignment_mode)
            else:
                logger.warning(f"[VarSelectionEvaluator] 验证期索引长度不匹配: {len(val_index)} vs {len(model_result.forecast_oos)}")

        # 返回9元组（位置[4]和[5]为Win Rate）
        return (is_rmse, oos_rmse, np.nan, np.nan, is_win_rate, oos_win_rate, False, None, None)

    except Exception as e:
        logger.error(f"[VarSelectionEvaluator] 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return (np.inf, np.inf, np.nan, np.nan, np.nan, np.nan, True, None, None)


# ========== 兼容性包装器（返回可调用对象） ==========

def create_dfm_evaluator(config: 'TrainingConfig') -> Callable:
    """
    创建DFM评估器（函数式接口）

    返回一个可调用对象，用于BackwardSelector的评估回调。

    重构后：返回一个lambda包装器，调用可序列化的顶层函数。

    Args:
        config: 训练配置对象

    Returns:
        评估函数，签名为 (variables: List[str], **kwargs) -> Tuple[float, ...]

    Example:
        >>> evaluator = create_dfm_evaluator(config)
        >>> selector = BackwardSelector(evaluator_func=evaluator, ...)
        >>> metrics = evaluator(
        ...     variables=['target', 'var1', 'var2'],
        ...     full_data=data,
        ...     params={'k_factors': 3}
        ... )
    """
    def evaluate(variables: List[str], **kwargs) -> Tuple[float, float, None, None, float, float, bool, None, None]:
        """
        评估指定变量组合的DFM模型性能

        Args:
            variables: 变量列表（包含目标变量）
            **kwargs: 其他参数
                - full_data: 完整数据DataFrame
                - params: DFM参数字典（包含k_factors等）
                - max_iter: 最大迭代次数

        Returns:
            9元组: (is_rmse, oos_rmse, _, _, is_win_rate, oos_win_rate,
                   is_svd_error, _, _)
        """
        # 提取参数
        full_data = kwargs.get('full_data')
        k_factors = kwargs.get('params', {}).get('k_factors', 2)
        max_iterations = kwargs.get('max_iter', config.max_iterations)

        # 调用可序列化的顶层函数
        return _evaluate_dfm_model(
            variables=variables,
            target_variable=config.target_variable,
            full_data=full_data,
            k_factors=k_factors,
            training_start=config.training_start,
            train_end=config.train_end,
            validation_start=config.validation_start,
            validation_end=config.validation_end,
            max_iterations=max_iterations,
            tolerance=config.tolerance,
            **kwargs
        )

    return evaluate


def create_variable_selection_evaluator(config: 'TrainingConfig') -> Callable:
    """
    创建变量筛选专用评估器（函数式接口）

    与create_dfm_evaluator的区别：
    - 使用下月配对RMSE作为唯一评估指标
    - 不计算Hit Rate和MAE（用于最终评估）
    - 专门用于变量筛选阶段

    重构后：返回一个lambda包装器，调用可序列化的顶层函数。

    Returns:
        评估函数，签名为 (variables: List[str], **kwargs) -> Tuple[float, ...]
    """
    def evaluate(variables: List[str], **kwargs) -> Tuple[float, float, float, float, float, float, bool, None, None]:
        """
        评估指定变量组合的DFM模型性能（仅使用下月配对RMSE）

        Args:
            variables: 变量列表（包含目标变量）
            **kwargs: 其他参数

        Returns:
            9元组: (is_rmse, oos_rmse, None, None, np.nan, np.nan, is_svd_error, None, None)
                   只有RMSE有意义，其他字段为占位符
        """
        # 提取参数
        full_data = kwargs.get('full_data')
        k_factors = kwargs.get('params', {}).get('k_factors', 2)
        max_iterations = kwargs.get('max_iter', config.max_iterations)

        # 调用可序列化的顶层函数
        return _evaluate_variable_selection_model(
            variables=variables,
            target_variable=config.target_variable,
            full_data=full_data,
            k_factors=k_factors,
            training_start=config.training_start,
            train_end=config.train_end,
            validation_start=config.validation_start,
            validation_end=config.validation_end,
            max_iterations=max_iterations,
            tolerance=config.tolerance
        )

    return evaluate


# ========== 获取可序列化配置的辅助函数 ==========

def extract_serializable_config(config: 'TrainingConfig') -> Dict[str, Any]:
    """
    从TrainingConfig提取可序列化的配置字典

    用于并行评估时传递配置参数

    Args:
        config: 训练配置对象

    Returns:
        可序列化的配置字典
    """
    return {
        'target_variable': config.target_variable,
        'training_start': config.training_start,
        'train_end': config.train_end,
        'validation_start': config.validation_start,
        'validation_end': config.validation_end,
        'max_iterations': config.max_iterations,
        'tolerance': config.tolerance,
        'alignment_mode': config.target_alignment_mode
    }


__all__ = [
    'create_dfm_evaluator',
    'create_variable_selection_evaluator',
    '_evaluate_dfm_model',
    '_evaluate_variable_selection_model',
    'extract_serializable_config'
]
