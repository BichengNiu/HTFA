# -*- coding: utf-8 -*-
"""
并行变量评估器

提供变量选择过程中的并行评估功能
"""

import logging
from typing import List, Tuple, Dict, Callable, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


def evaluate_single_variable_removal(
    var_to_remove: str,
    current_predictors: List[str],
    target_variable: str,
    evaluator_func: Callable,
    eval_params: Dict[str, Any],
    k_factors: int
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    评估移除单个变量后的模型性能

    此函数设计为可被并行执行的独立任务

    Args:
        var_to_remove: 待移除的变量名
        current_predictors: 当前预测变量列表
        target_variable: 目标变量名
        evaluator_func: 评估函数
        eval_params: 评估参数字典
        k_factors: 因子数

    Returns:
        (变量名, 评估结果字典) 或 (变量名, None) 如果评估失败
    """
    try:
        # 构建移除该变量后的变量列表
        temp_predictors = [v for v in current_predictors if v != var_to_remove]
        if not temp_predictors:
            return (var_to_remove, None)

        temp_variables = [target_variable] + temp_predictors

        # 检查因子数约束
        if k_factors >= len(temp_variables):
            logger.debug(
                f"跳过'{var_to_remove}': k_factors({k_factors}) >= "
                f"剩余变量数({len(temp_variables)})"
            )
            return (var_to_remove, None)

        # 评估移除后的性能
        result_tuple = evaluator_func(variables=temp_variables, **eval_params)

        if len(result_tuple) != 9:
            logger.warning(f"评估返回了{len(result_tuple)}个值，跳过'{var_to_remove}'")
            return (var_to_remove, None)

        is_rmse, oos_rmse, _, _, is_hit_rate, oos_hit_rate, is_svd_error, _, _ = result_tuple

        # 返回评估结果
        result = {
            'var': var_to_remove,
            'is_rmse': is_rmse,
            'oos_rmse': oos_rmse,
            'is_hit_rate': is_hit_rate,
            'oos_hit_rate': oos_hit_rate,
            'is_svd_error': is_svd_error
        }

        return (var_to_remove, result)

    except Exception as e:
        logger.error(f"评估移除'{var_to_remove}'时出错: {e}")
        return (var_to_remove, None)


def parallel_evaluate_removals(
    current_predictors: List[str],
    target_variable: str,
    evaluator_func: Callable,
    eval_params: Dict[str, Any],
    k_factors: int,
    n_jobs: int = -1,
    backend: str = 'loky',
    verbose: int = 0,
    progress_callback: Optional[Callable[[str], None]] = None
) -> List[Dict[str, Any]]:
    """
    并行评估所有候选变量的移除效果

    Args:
        current_predictors: 当前预测变量列表
        target_variable: 目标变量名
        evaluator_func: 评估函数
        eval_params: 评估参数字典
        k_factors: 因子数
        n_jobs: 并行任务数（-1=所有核心）
        backend: 并行后端（'loky', 'multiprocessing', 'threading'）
        verbose: 是否显示进度
        progress_callback: 进度回调函数

    Returns:
        评估结果列表（仅包含成功的评估）
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        logger.error("未安装joblib库，无法使用并行评估，请安装: pip install joblib")
        raise

    if progress_callback:
        progress_callback(
            f"  并行评估 {len(current_predictors)} 个候选变量 "
            f"(使用 {n_jobs if n_jobs > 0 else 'max-1'} 个核心, backend={backend})..."
        )

    # 并行评估所有变量
    # 关键参数说明：
    # - prefer='processes': 强制使用多进程（避免GIL限制）
    # - require='sharedmem': 使用共享内存减少数据复制开销
    # - batch_size='auto': 自动批处理大小
    # - pre_dispatch='2*n_jobs': 预分配任务数
    results_with_none = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        prefer='processes',  # 强制使用多进程
        require=None,  # 不强制共享内存（避免兼容性问题）
        batch_size='auto',
        pre_dispatch='2*n_jobs'
    )(
        delayed(evaluate_single_variable_removal)(
            var,
            current_predictors,
            target_variable,
            evaluator_func,
            eval_params,
            k_factors
        )
        for var in current_predictors
    )

    # 过滤掉失败的评估（None结果）
    candidate_results = []
    for idx, (var, result) in enumerate(results_with_none, 1):
        if result is None:
            continue

        # 打印每个候选的结果
        if progress_callback:
            progress_callback(
                f"  [{idx}/{len(current_predictors)}] 完成评估: '{var}' - "
                f"训练期RMSE: {result['is_rmse']:.4f}, "
                f"验证期RMSE: {result['oos_rmse']:.4f}"
            )

        candidate_results.append(result)

    return candidate_results


def serial_evaluate_removals(
    current_predictors: List[str],
    target_variable: str,
    evaluator_func: Callable,
    eval_params: Dict[str, Any],
    k_factors: int,
    progress_callback: Optional[Callable[[str], None]] = None
) -> List[Dict[str, Any]]:
    """
    串行评估所有候选变量的移除效果（回退方案）

    Args:
        current_predictors: 当前预测变量列表
        target_variable: 目标变量名
        evaluator_func: 评估函数
        eval_params: 评估参数字典
        k_factors: 因子数
        progress_callback: 进度回调函数

    Returns:
        评估结果列表（仅包含成功的评估）
    """
    candidate_results = []

    for idx, var in enumerate(current_predictors, 1):
        # 打印正在尝试的变量
        if progress_callback:
            msg = f"  [{idx}/{len(current_predictors)}] 尝试移除: '{var}'"
            progress_callback(msg)

        # 评估该变量
        _, result = evaluate_single_variable_removal(
            var,
            current_predictors,
            target_variable,
            evaluator_func,
            eval_params,
            k_factors
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


def evaluate_removals_with_fallback(
    current_predictors: List[str],
    target_variable: str,
    evaluator_func: Callable,
    eval_params: Dict[str, Any],
    k_factors: int,
    use_parallel: bool = False,
    n_jobs: int = -1,
    backend: str = 'loky',
    verbose: int = 0,
    progress_callback: Optional[Callable[[str], None]] = None
) -> List[Dict[str, Any]]:
    """
    评估所有候选变量的移除效果（自动降级）

    首先尝试并行评估，如果失败则自动降级到串行

    Args:
        current_predictors: 当前预测变量列表
        target_variable: 目标变量名
        evaluator_func: 评估函数
        eval_params: 评估参数字典
        k_factors: 因子数
        use_parallel: 是否使用并行
        n_jobs: 并行任务数
        backend: 并行后端
        verbose: 是否显示进度
        progress_callback: 进度回调函数

    Returns:
        评估结果列表
    """
    if not use_parallel:
        # 串行评估
        return serial_evaluate_removals(
            current_predictors,
            target_variable,
            evaluator_func,
            eval_params,
            k_factors,
            progress_callback
        )

    try:
        # 尝试并行评估
        return parallel_evaluate_removals(
            current_predictors,
            target_variable,
            evaluator_func,
            eval_params,
            k_factors,
            n_jobs,
            backend,
            verbose,
            progress_callback
        )
    except Exception as e:
        logger.error(f"并行评估失败: {e}，降级到串行模式")
        if progress_callback:
            progress_callback(f"  并行评估失败，自动切换到串行模式...")

        # 降级到串行评估
        return serial_evaluate_removals(
            current_predictors,
            target_variable,
            evaluator_func,
            eval_params,
            k_factors,
            progress_callback
        )


__all__ = [
    'evaluate_single_variable_removal',
    'parallel_evaluate_removals',
    'serial_evaluate_removals',
    'evaluate_removals_with_fallback'
]
