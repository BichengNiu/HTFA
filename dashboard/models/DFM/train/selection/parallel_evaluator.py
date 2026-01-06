# -*- coding: utf-8 -*-
"""
并行变量评估器

提供变量选择过程中的并行评估功能

重构说明:
- 移除不可序列化的参数(evaluator_func, progress_callback)
- 直接调用可序列化的顶层评估函数
- 使用可序列化的配置字典代替闭包
- DRY重构：合并removal/addition函数为通用函数
"""

import logging
from typing import List, Tuple, Dict, Callable, Optional, Any, Literal

import pandas as pd

logger = logging.getLogger(__name__)

# 操作类型
OperationType = Literal['remove', 'add']


def _build_temp_predictors(
    var: str,
    current_predictors: List[str],
    operation: OperationType
) -> List[str]:
    """根据操作类型构建临时预测变量列表"""
    if operation == 'remove':
        return [v for v in current_predictors if v != var]
    else:  # add
        return current_predictors + [var]


def _get_operation_verb(operation: OperationType, past_tense: bool = False) -> str:
    """获取操作动词"""
    if operation == 'remove':
        return '移除' if not past_tense else '移除'
    else:
        return '添加' if not past_tense else '添加'


def evaluate_single_variable_change(
    var: str,
    current_predictors: List[str],
    target_variable: str,
    full_data: pd.DataFrame,
    k_factors: int,
    evaluator_config: Dict[str, Any],
    operation: OperationType
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    评估单个变量变更后的模型性能（可序列化版本）

    此函数设计为可被并行执行的独立任务，所有参数都可序列化

    Args:
        var: 待变更的变量名
        current_predictors: 当前预测变量列表
        target_variable: 目标变量名
        full_data: 完整数据DataFrame（可序列化）
        k_factors: 因子数
        evaluator_config: 评估器配置字典（可序列化）
            - training_start: 训练开始日期
            - train_end: 训练结束日期
            - validation_start: 验证开始日期
            - validation_end: 验证结束日期
            - max_iterations: 最大迭代次数
            - tolerance: 容差
            - alignment_mode: 目标配对模式 ('current_month' 或 'next_month')
        operation: 操作类型 ('remove' 或 'add')

    Returns:
        (变量名, 评估结果字典) 或 (变量名, None) 如果评估失败
    """
    try:
        # 构建变更后的变量列表
        temp_predictors = _build_temp_predictors(var, current_predictors, operation)
        if not temp_predictors:
            return (var, None)

        temp_variables = [target_variable] + temp_predictors

        # 检查因子数约束
        if k_factors >= len(temp_variables):
            logger.debug(
                f"跳过'{var}': k_factors({k_factors}) >= "
                f"变量数({len(temp_variables)})"
            )
            return (var, None)

        # 导入可序列化的顶层评估函数
        from dashboard.models.DFM.train.training.evaluator_strategy import _evaluate_variable_selection_model

        # 调用可序列化的评估函数
        result_tuple = _evaluate_variable_selection_model(
            variables=temp_variables,
            target_variable=target_variable,
            full_data=full_data,
            k_factors=k_factors,
            training_start=evaluator_config['training_start'],
            train_end=evaluator_config['train_end'],
            validation_start=evaluator_config['validation_start'],
            validation_end=evaluator_config['validation_end'],
            max_iterations=evaluator_config['max_iterations'],
            tolerance=evaluator_config['tolerance'],
            alignment_mode=evaluator_config.get('alignment_mode', 'next_month')
        )

        if len(result_tuple) != 9:
            logger.warning(f"评估返回了{len(result_tuple)}个值，跳过'{var}'")
            return (var, None)

        is_rmse, oos_rmse, _, _, is_win_rate, oos_win_rate, is_svd_error, _, _ = result_tuple

        # 返回评估结果
        result = {
            'var': var,
            'is_rmse': is_rmse,
            'oos_rmse': oos_rmse,
            'is_win_rate': is_win_rate,
            'oos_win_rate': oos_win_rate,
            'is_svd_error': is_svd_error
        }

        return (var, result)

    except Exception as e:
        verb = _get_operation_verb(operation)
        logger.exception(f"评估{verb}'{var}'时出错: {e}")
        return (var, None)


def parallel_evaluate_changes(
    current_predictors: List[str],
    candidate_vars: List[str],
    target_variable: str,
    full_data: pd.DataFrame,
    k_factors: int,
    evaluator_config: Dict[str, Any],
    operation: OperationType,
    n_jobs: int = -1,
    backend: str = 'loky',
    verbose: int = 0,
    progress_callback: Optional[Callable[[str], None]] = None
) -> List[Dict[str, Any]]:
    """
    并行评估所有候选变量的变更效果（可序列化版本）

    Args:
        current_predictors: 当前预测变量列表
        candidate_vars: 候选变量列表
        target_variable: 目标变量名
        full_data: 完整数据DataFrame（可序列化）
        k_factors: 因子数
        evaluator_config: 评估器配置字典（可序列化）
        operation: 操作类型 ('remove' 或 'add')
        n_jobs: 并行任务数（-1=所有核心）
        backend: 并行后端（'loky', 'multiprocessing', 'threading'）
        verbose: 是否显示进度
        progress_callback: 进度回调函数（仅在主进程使用，不传递给子进程）

    Returns:
        评估结果列表（仅包含成功的评估）
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        logger.error("未安装joblib库，无法使用并行评估，请安装: pip install joblib")
        raise

    verb = _get_operation_verb(operation)
    if progress_callback:
        progress_callback(
            f"  并行评估{verb} {len(candidate_vars)} 个候选变量 "
            f"(使用 {n_jobs if n_jobs > 0 else 'max-1'} 个核心, backend={backend})..."
        )

    # 并行评估所有变量
    results_with_none = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        prefer='processes',
        batch_size='auto',
        pre_dispatch='2*n_jobs'
    )(
        delayed(evaluate_single_variable_change)(
            var,
            current_predictors,
            target_variable,
            full_data,
            k_factors,
            evaluator_config,
            operation
        )
        for var in candidate_vars
    )

    # 过滤掉失败的评估（None结果）
    candidate_results = []
    for idx, (var, result) in enumerate(results_with_none, 1):
        if result is None:
            continue

        # 打印每个候选的结果
        if progress_callback:
            progress_callback(
                f"  [{idx}/{len(candidate_vars)}] 完成评估: '{var}' - "
                f"训练期RMSE: {result['is_rmse']:.4f}, "
                f"验证期RMSE: {result['oos_rmse']:.4f}"
            )

        candidate_results.append(result)

    return candidate_results


def serial_evaluate_changes(
    current_predictors: List[str],
    candidate_vars: List[str],
    target_variable: str,
    full_data: pd.DataFrame,
    k_factors: int,
    evaluator_config: Dict[str, Any],
    operation: OperationType,
    progress_callback: Optional[Callable[[str], None]] = None
) -> List[Dict[str, Any]]:
    """
    串行评估所有候选变量的变更效果（可序列化版本）

    Args:
        current_predictors: 当前预测变量列表
        candidate_vars: 候选变量列表
        target_variable: 目标变量名
        full_data: 完整数据DataFrame（可序列化）
        k_factors: 因子数
        evaluator_config: 评估器配置字典（可序列化）
        operation: 操作类型 ('remove' 或 'add')
        progress_callback: 进度回调函数

    Returns:
        评估结果列表（仅包含成功的评估）
    """
    candidate_results = []
    verb = _get_operation_verb(operation)

    for idx, var in enumerate(candidate_vars, 1):
        # 打印正在尝试的变量
        if progress_callback:
            msg = f"  [{idx}/{len(candidate_vars)}] 尝试{verb}: '{var}'"
            progress_callback(msg)

        # 评估该变量
        _, result = evaluate_single_variable_change(
            var,
            current_predictors,
            target_variable,
            full_data,
            k_factors,
            evaluator_config,
            operation
        )

        if result is None:
            continue

        # 打印评估结果
        if progress_callback:
            progress_callback(
                f"    训练期RMSE: {result['is_rmse']:.4f}, "
                f"验证期RMSE: {result['oos_rmse']:.4f}"
            )

        candidate_results.append(result)

    return candidate_results


def evaluate_changes(
    current_predictors: List[str],
    candidate_vars: List[str],
    target_variable: str,
    full_data: pd.DataFrame,
    k_factors: int,
    evaluator_config: Dict[str, Any],
    operation: OperationType,
    use_parallel: bool = False,
    n_jobs: int = -1,
    backend: str = 'loky',
    verbose: int = 0,
    progress_callback: Optional[Callable[[str], None]] = None
) -> List[Dict[str, Any]]:
    """
    评估所有候选变量的变更效果（可序列化版本）

    根据use_parallel参数选择并行或串行评估

    Args:
        current_predictors: 当前预测变量列表
        candidate_vars: 候选变量列表
        target_variable: 目标变量名
        full_data: 完整数据DataFrame（可序列化）
        k_factors: 因子数
        evaluator_config: 评估器配置字典（可序列化）
        operation: 操作类型 ('remove' 或 'add')
        use_parallel: 是否使用并行
        n_jobs: 并行任务数
        backend: 并行后端
        verbose: 是否显示进度
        progress_callback: 进度回调函数

    Returns:
        评估结果列表
    """
    if not use_parallel:
        return serial_evaluate_changes(
            current_predictors,
            candidate_vars,
            target_variable,
            full_data,
            k_factors,
            evaluator_config,
            operation,
            progress_callback
        )

    return parallel_evaluate_changes(
        current_predictors,
        candidate_vars,
        target_variable,
        full_data,
        k_factors,
        evaluator_config,
        operation,
        n_jobs,
        backend,
        verbose,
        progress_callback
    )


__all__ = [
    'evaluate_single_variable_change',
    'parallel_evaluate_changes',
    'serial_evaluate_changes',
    'evaluate_changes',
]
