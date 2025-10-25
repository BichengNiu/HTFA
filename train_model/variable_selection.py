# -*- coding: utf-8 -*-
"""
包含 DFM 变量选择相关功能的模块。
"""
import pandas as pd
import numpy as np
# 移除并行处理：import concurrent.futures
from tqdm import tqdm
import unicodedata
import time # <-- 新增导入 time
import logging # <-- 新增导入 logging
import sys # <-- 新增导入 sys
import traceback # <-- 新增导入 traceback
from typing import List, Dict, Tuple, Callable, Optional, Any
import os

# 导入优化相关类
from dashboard.DFM.train_model.precomputed_dfm_context import PrecomputedDFMContext
from dashboard.DFM.train_model.optimized_dfm_evaluator import OptimizedDFMEvaluator

# DFM 变量选择模块 - 已改为串行执行模式

# 假设 evaluate_dfm_params 函数可以从某个地方导入
# from .dfm_core import evaluate_dfm_params 
# 或者它被作为参数传递进来

logger = logging.getLogger(__name__) # <-- 获取 logger


# 线程池版本不需要子进程初始化函数
# 线程共享同一个Python进程和内存空间

# 线程池版本不需要这个函数，直接使用parallel_worker.py中的evaluate_single_removal
# 该函数已被重构为支持直接参数传递而非元组解包


def perform_global_backward_selection(
    initial_variables: List[str],
    initial_params: Dict, # 包含固定 k=N 的参数
    # initial_score_tuple: Tuple, # 不再需要初始分数，函数内部会计算
    target_variable: str, # 需要明确目标变量
    all_data: pd.DataFrame,
    # 移除 var_type_map 参数
    validation_start: str,
    validation_end: str,
    target_freq: str,
    train_end_date: str,
    n_iter: int,
    target_mean_original: float,
    target_std_original: float,
    max_workers: int,
    evaluate_dfm_func: Callable,
    max_lags: int = 1,
    log_file: Optional[object] = None,
    use_optimization: bool = True  # 新增：是否使用优化
) -> Tuple[List[str], Dict, Tuple, int, int]:
    """
    执行全局后向变量剔除。

    从所有预测变量开始，每次迭代评估移除每个变量后的性能，
    移除性能提升最大的那个变量（基于 HR -> -RMSE 优化）。
    当没有单个变量的移除能提升性能时停止。

    Args:
        initial_variables: 包含目标变量和所有初始预测变量的列表。
        initial_params: 固定的 DFM 参数 (包含 k=N)。
        target_variable: 目标变量名称。
        all_data: 包含所有变量的完整 DataFrame。
        validation_start: 验证期开始日期。
        validation_end: 验证期结束日期。
        target_freq: 目标频率。
        train_end_date: 训练期结束日期。
        n_iter: DFM 迭代次数。
        target_mean_original: 原始目标变量均值 (用于反标准化)。
        target_std_original: 原始目标变量标准差 (用于反标准化)。
        max_workers: 并行计算的最大进程数。
        evaluate_dfm_func: 用于评估 DFM 参数的函数。
        max_lags: 最大滞后阶数。
        log_file: 可选的日志文件句柄。
        use_optimization: 是否使用预计算优化 (默认True)。

    Returns:
        Tuple 包含:
            - final_variables: 最终选择的变量列表 (包含目标变量)。
            - final_params: 最终选择的最佳参数 (与 initial_params 相同)。
            - final_score_tuple: 最终的最佳得分元组 (HR, -RMSE)。
            - total_evaluations: 此过程中执行的评估总次数。
            - svd_error_count: 此过程中遇到的 SVD 错误次数。
    """
    total_evaluations_global = 0
    svd_error_count_global = 0
    
    # === 优化初始化 ===
    precomputed_context = None
    optimized_evaluator = None
    optimization_successful = False
    
    if use_optimization:
        try:
            logger.info("初始化预计算上下文...")
            precomputed_context = PrecomputedDFMContext(
                full_data=all_data,
                initial_variables=initial_variables,
                target_variable=target_variable,
                params=initial_params,
                validation_start=validation_start,
                validation_end=validation_end,
                target_freq=target_freq,
                train_end_date=train_end_date,
                target_mean_original=target_mean_original,
                target_std_original=target_std_original,
                max_iter=n_iter,
                max_lags=max_lags
            )
            
            if precomputed_context.is_context_valid():
                logger.info("初始化优化评估器...")
                optimized_evaluator = OptimizedDFMEvaluator(
                    precomputed_context=precomputed_context,
                    use_cache=True
                )
                optimization_successful = True
                logger.info("优化初始化成功，将使用预计算上下文进行评估")
            else:
                logger.warning("预计算上下文无效，将使用原始评估方法")
                
        except Exception as e:
            logger.warning(f"优化初始化失败: {e}，将使用原始评估方法")
            optimization_successful = False

    # 1. 初始化当前最优变量集 (仅预测变量)
    current_best_predictors = sorted([v for v in initial_variables if v != target_variable])
    if not current_best_predictors:
        logger.error("全局后向筛选：初始预测变量列表为空，无法进行筛选。")
        return initial_variables, initial_params, (-np.inf, np.inf), 0, 0

    # 2. 计算初始基准性能
    logger.info("全局后向筛选：计算初始基准性能...")
    initial_vars_for_eval = [target_variable] + current_best_predictors
    try:
        result_tuple_base = evaluate_dfm_func(
             variables=initial_vars_for_eval,
             full_data=all_data,
             target_variable=target_variable,
             params=initial_params,
             validation_start=validation_start,
             validation_end=validation_end,
             target_freq=target_freq,
             train_end_date=train_end_date,
             target_mean_original=target_mean_original,
             target_std_original=target_std_original,
             max_iter=n_iter,
             max_lags=max_lags
        )
        total_evaluations_global += 1
        if len(result_tuple_base) != 9:
             logger.error(f"全局后向筛选：初始评估返回了 {len(result_tuple_base)} 个值 (预期 9)。无法计算基准分数。")
             return initial_variables, initial_params, (-np.inf, np.inf), total_evaluations_global, svd_error_count_global
        is_rmse_base, oos_rmse_base, _, _, is_hit_rate_base, oos_hit_rate_base, is_svd_error_base, _, _ = result_tuple_base
        if is_svd_error_base: svd_error_count_global += 1

        combined_rmse_base = np.inf
        finite_rmses_base = [r for r in [is_rmse_base, oos_rmse_base] if r is not None and np.isfinite(r)]
        if finite_rmses_base: combined_rmse_base = np.mean(finite_rmses_base)

        combined_hit_rate_base = -np.inf
        finite_hit_rates_base = [hr for hr in [is_hit_rate_base, oos_hit_rate_base] if hr is not None and np.isfinite(hr)]
        if finite_hit_rates_base: combined_hit_rate_base = np.mean(finite_hit_rates_base)

        if not (np.isfinite(combined_rmse_base) and np.isfinite(combined_hit_rate_base)):
            logger.error("全局后向筛选：初始基准评估未能计算有效分数。无法继续。")
            return initial_variables, initial_params, (-np.inf, np.inf), total_evaluations_global, svd_error_count_global

        current_best_score_tuple = (combined_hit_rate_base, -combined_rmse_base)
        logger.info(f"初始基准得分 (HR={current_best_score_tuple[0]:.2f}%, RMSE={-current_best_score_tuple[1]:.6f})，变量数: {len(current_best_predictors)}")
        if log_file:
            log_file.write(f"\n--- 全局后向筛选开始 ---\n")
            log_file.write(f"初始预测变量数: {len(current_best_predictors)}\n")
            log_file.write(f"初始基准得分 (HR, -RMSE): {current_best_score_tuple}\n")

    except Exception as e_base:
        logger.error(f"全局后向筛选：计算初始基准性能时出错: {e_base}")
        traceback.print_exc(file=sys.stderr) # 打印 traceback 到 stderr
        return initial_variables, initial_params, (-np.inf, np.inf), total_evaluations_global, svd_error_count_global

    # 3. 初始化进度条
    # 总迭代次数最多是初始预测变量数 - 1 (至少保留一个预测变量)
    max_possible_removals = len(current_best_predictors) - 1 if len(current_best_predictors) > 1 else 0
    pbar = tqdm(total=max_possible_removals, desc="全局变量后向剔除", unit="var")

    # 4. 迭代移除变量
    iteration = 0
    while True:
        iteration += 1
        logger.info(f"\n--- 全局后向筛选：第 {iteration} 轮 (当前变量数: {len(current_best_predictors)}) ---")
        if len(current_best_predictors) <= 1:
            logger.info("剩余预测变量数已达下限 (1)，停止筛选。")
            break

        best_candidate_score_this_iter = (-np.inf, np.inf) # (HR, -RMSE) -> 越前面越大越好，越后面越小越好
        best_removal_candidate_var_this_iter = None
        results_this_iteration_map = {} # 存储移除变量 -> 结果

        # 串行评估每个变量的移除效果
        valid_removals = []
        for var_to_remove in current_best_predictors:
            temp_predictors = [v for v in current_best_predictors if v != var_to_remove]
            if not temp_predictors: # 如果移除后没有预测变量了，跳过
                continue

            temp_variables_for_eval = [target_variable] + temp_predictors
            # 检查因子数是否仍然小于变量数 (N > K)
            if initial_params.get('k_factors', 1) >= len(temp_variables_for_eval):
                 # logger.debug(f"跳过移除 {var_to_remove}：移除后变量数 ({len(temp_variables_for_eval)}) <= 因子数 ({initial_params.get('k_factors')})")
                 continue
            
            valid_removals.append(var_to_remove)

        if not valid_removals:
             logger.info("本轮无可行的评估任务，筛选结束。")
             break

        # 使用 tqdm 显示串行评估进度
        eval_method = "优化" if optimization_successful else "原始"
        eval_pbar_desc = f"迭代 {iteration} {eval_method}评估"
        eval_pbar = tqdm(total=len(valid_removals), desc=eval_pbar_desc, unit="eval", leave=False)

        # 串行执行每个变量移除的评估
        evaluation_start_time = time.time()
        for var_to_remove in valid_removals:
            total_evaluations_global += 1
            eval_pbar.update(1)
            
            try:
                temp_predictors = [v for v in current_best_predictors if v != var_to_remove]
                temp_variables_for_eval = [target_variable] + temp_predictors
                
                # 根据优化状态选择评估方法
                if optimization_successful and optimized_evaluator is not None:
                    # 使用优化评估器
                    result_tuple = optimized_evaluator.evaluate_dfm_optimized(
                        variables=temp_variables_for_eval,
                        fallback_evaluate_func=evaluate_dfm_func
                    )
                else:
                    # 使用原始评估函数
                    result_tuple = evaluate_dfm_func(
                        variables=temp_variables_for_eval,
                        full_data=all_data,
                        target_variable=target_variable,
                        params=initial_params, # 固定参数 k=N
                        validation_start=validation_start,
                        validation_end=validation_end,
                        target_freq=target_freq,
                        train_end_date=train_end_date,
                        max_iter=n_iter,
                        target_mean_original=target_mean_original,
                        target_std_original=target_std_original,
                        max_lags=max_lags
                    )
                
                if len(result_tuple) != 9:
                    logger.warning(f"评估函数返回了 {len(result_tuple)} 个值 (预期 9)，跳过移除 {var_to_remove} 的结果。")
                    continue
                is_rmse, oos_rmse, _, _, is_hit_rate, oos_hit_rate, is_svd_error, _, _ = result_tuple
                if is_svd_error:
                    svd_error_count_global += 1

                combined_rmse_removal = np.inf
                finite_rmses = [r for r in [is_rmse, oos_rmse] if np.isfinite(r)]
                if finite_rmses: combined_rmse_removal = np.mean(finite_rmses)

                combined_hit_rate_removal = -np.inf
                finite_hit_rates = [hr for hr in [is_hit_rate, oos_hit_rate] if np.isfinite(hr)]
                if finite_hit_rates: combined_hit_rate_removal = np.mean(finite_hit_rates)

                if np.isfinite(combined_rmse_removal) and np.isfinite(combined_hit_rate_removal):
                    current_score_tuple_eval = (combined_hit_rate_removal, -combined_rmse_removal) # 使用临时变量名
                    results_this_iteration_map[var_to_remove] = current_score_tuple_eval

                    # 实时比较，找到本轮最佳移除候选
                    if current_score_tuple_eval > best_candidate_score_this_iter:
                        best_candidate_score_this_iter = current_score_tuple_eval
                        best_removal_candidate_var_this_iter = var_to_remove
                else:
                    # logger.debug(f"移除 {var_to_remove} 的结果无效 (RMSE={combined_rmse_removal}, HR={combined_hit_rate_removal})")
                    pass # 不记录无效结果

            except Exception as exc:
                logger.error(f"处理移除 {var_to_remove} 的评估结果时出错: {exc}")
                traceback.print_exc(file=sys.stderr) # 添加 traceback 打印
                
        eval_pbar.close() # 关闭评估进度条
        
        # 记录本轮评估性能
        evaluation_time = time.time() - evaluation_start_time
        logger.info(f"本轮评估完成：{len(valid_removals)} 个变量，耗时 {evaluation_time:.3f}秒 ({'优化' if optimization_successful else '原始'}方法)")
            
        # 这个分支已被移除，使用上面的fallback机制

        # 5. 检查本轮结果，决定是否移除变量
        if best_removal_candidate_var_this_iter is not None:
             # 比较本轮找到的最佳分数与全局当前最佳分数
             # logger.debug(f"比较本轮最佳移除 ({best_removal_candidate_var_this_iter}) 得分 {best_candidate_score_this_iter} 与当前最佳得分 {current_best_score_tuple}")
             if best_candidate_score_this_iter > current_best_score_tuple:
                 # 找到了更好的解，执行移除
                 removed_var = best_removal_candidate_var_this_iter
                 old_score_str = f"(HR={current_best_score_tuple[0]:.2f}%, RMSE={-current_best_score_tuple[1]:.6f})"
                 new_score_str = f"(HR={best_candidate_score_this_iter[0]:.2f}%, RMSE={-best_candidate_score_this_iter[1]:.6f})"

                 logger.info(f"接受移除: '{removed_var}'，性能提升: {old_score_str} -> {new_score_str}")
                 if log_file:
                     log_file.write(f"Iter {iteration}: 移除 '{removed_var}'，得分 {old_score_str} -> {new_score_str}\n")

                 # 更新最优变量集和分数
                 current_best_predictors.remove(removed_var)
                 current_best_score_tuple = best_candidate_score_this_iter
                 pbar.update(1) # 更新总进度条

                 # 继续下一轮迭代
                 continue
             else:
                 logger.info("本轮最佳移除候选未优于当前最佳得分，筛选稳定。")
                 if log_file:
                     log_file.write(f"Iter {iteration}: 未找到更优移除，筛选结束。\n")
                 break # 跳出 while 循环
        else:
             logger.info("本轮未找到任何有效的移除候选，筛选结束。")
             if log_file:
                 log_file.write(f"Iter {iteration}: 未找到有效移除候选，筛选结束。\n")
             break # 跳出 while 循环

    pbar.close() # 关闭总进度条

    # 6. 打印优化统计信息
    if optimization_successful and optimized_evaluator is not None:
        logger.info("\n=== 优化统计信息 ===")
        optimized_evaluator.print_statistics()
    
    # 7. 返回最终结果
    final_variables = sorted([target_variable] + current_best_predictors)
    logger.info(f"\n全局后向筛选完成。最终选择 {len(current_best_predictors)} 个预测变量。")
    logger.info(f"最终得分 (HR, -RMSE): {current_best_score_tuple}")
    
    if log_file:
        log_file.write(f"\n--- 全局后向筛选结束 ---\n")
        log_file.write(f"最终预测变量数: {len(current_best_predictors)}\n")
        log_file.write(f"最终得分 (HR, -RMSE): {current_best_score_tuple}\n")
        # log_file.write(f"最终变量列表: {final_variables}\n") # 可能太长
        
        # 记录优化统计到日志文件
        if optimization_successful and optimized_evaluator is not None:
            stats = optimized_evaluator.get_statistics()
            log_file.write(f"\n--- 优化统计 ---\n")
            log_file.write(f"优化成功率: {stats['optimization_rate']:.1%}\n")
            log_file.write(f"总节省时间: {stats['total_time_saved']:.3f}秒\n")
            log_file.write(f"平均评估时间: {stats['avg_evaluation_time']:.3f}秒\n")

    return final_variables, initial_params, current_best_score_tuple, total_evaluations_global, svd_error_count_global

