# -*- coding: utf-8 -*-
"""
优化的DFM评估器模块
使用预计算上下文来加速DFM模型评估过程
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional, Callable, Any
from dashboard.DFM.train_model.precomputed_dfm_context import PrecomputedDFMContext
from dashboard.DFM.train_model.dfm_core import _fit_dfm_model, _calculate_nowcast, _calculate_metrics, _handle_step_failure
from dashboard.DFM.train_model.evaluation_cache import DFMEvaluationCache

logger = logging.getLogger(__name__)


class OptimizedDFMEvaluator:
    """
    优化的DFM评估器
    
    使用预计算上下文来避免重复的数据预处理步骤，
    显著提升变量选择过程中的评估性能。
    
    特性：
    1. 利用PrecomputedDFMContext避免重复数据处理
    2. 集成缓存机制减少重复计算
    3. 性能监控和统计
    4. 自动回退到原始评估方法
    """
    
    def __init__(
        self,
        precomputed_context: PrecomputedDFMContext,
        cache_instance: Optional[DFMEvaluationCache] = None,
        use_cache: bool = True
    ):
        """
        初始化优化评估器
        
        Args:
            precomputed_context: 预计算上下文实例
            cache_instance: 缓存实例（可选）
            use_cache: 是否使用缓存
        """
        self.context = precomputed_context
        self.use_cache = use_cache
        self.cache_instance = cache_instance
        
        # 如果没有提供缓存实例且启用缓存，创建一个
        if self.use_cache and self.cache_instance is None:
            from dashboard.DFM.train_model.evaluation_cache import get_global_cache
            self.cache_instance = get_global_cache()
        
        # 性能统计
        self.stats = {
            'total_evaluations': 0,
            'optimized_evaluations': 0,
            'fallback_evaluations': 0,
            'cache_hits': 0,
            'total_time_saved': 0.0,
            'total_evaluation_time': 0.0,
            'creation_time': time.time()
        }
        
        logger.info(f"OptimizedDFMEvaluator初始化完成，使用缓存: {self.use_cache}")
    
    def evaluate_dfm_optimized(
        self,
        variables: List[str],
        fallback_evaluate_func: Callable
    ) -> Tuple[float, float, float, float, float, float, bool, pd.DataFrame, pd.DataFrame]:
        """
        优化的DFM评估函数
        
        Args:
            variables: 变量列表
            fallback_evaluate_func: 回退函数，用于在优化失败时调用
            
        Returns:
            与原始evaluate_dfm_params相同的返回值格式
        """
        start_time = time.time()
        self.stats['total_evaluations'] += 1
        
        # 检查预计算上下文是否有效
        if not self.context.is_context_valid():
            logger.warning("预计算上下文无效，回退到原始评估方法")
            self.stats['fallback_evaluations'] += 1
            return self._evaluate_with_fallback(variables, fallback_evaluate_func, start_time)
        
        try:
            # 尝试使用缓存
            cache_key = None
            if self.use_cache and self.cache_instance is not None:
                cache_key = self.cache_instance.generate_cache_key(
                    variables=variables,
                    params=self.context.params,
                    target_variable=self.context.target_variable,
                    validation_start=self.context.validation_start,
                    validation_end=self.context.validation_end,
                    train_end_date=self.context.train_end_date,
                    target_freq=self.context.target_freq,
                    max_iter=self.context.max_iter,
                    target_mean_original=self.context.target_mean_original,
                    target_std_original=self.context.target_std_original,
                    max_lags=self.context.max_lags
                )
                
                cached_result = self.cache_instance.get(cache_key, compute_time_estimate=2.149)
                if cached_result is not None:
                    self.stats['cache_hits'] += 1
                    elapsed = time.time() - start_time
                    self.stats['total_evaluation_time'] += elapsed
                    logger.debug(f"优化评估缓存命中: 变量数={len(variables)}, 耗时={elapsed:.3f}秒")
                    return cached_result
            
            # 使用预计算上下文获取预处理数据
            success, error_msg, prepared_data, prepared_variables = (
                self.context.get_prepared_data_for_variables(variables)
            )
            
            if not success:
                logger.warning(f"预计算上下文数据获取失败: {error_msg}，回退到原始方法")
                self.stats['fallback_evaluations'] += 1
                return self._evaluate_with_fallback(variables, fallback_evaluate_func, start_time)
            
            # 执行DFM模型拟合（跳过重复的数据预处理步骤）
            k_factors = self.context.k_factors
            
            # 拟合DFM模型
            success, error_msg, dfm_results, is_svd_error = _fit_dfm_model(
                prepared_data, k_factors, self.context.max_iter, 
                self.context.max_lags, self.context.train_end_date
            )
            
            if not success:
                result = _handle_step_failure("DFM拟合", error_msg, prepared_variables, k_factors, is_svd_error)
                self._store_result_and_update_stats(cache_key, result, start_time, optimized=True)
                return result
            
            # 计算nowcast
            success, error_msg, nowcast_series_orig, lambda_df = _calculate_nowcast(
                dfm_results, self.context.target_variable, prepared_variables, k_factors,
                self.context.target_mean_original, self.context.target_std_original
            )
            
            if not success:
                result = _handle_step_failure("Nowcast计算", error_msg, prepared_variables, k_factors, is_svd_error, lambda_df)
                self._store_result_and_update_stats(cache_key, result, start_time, optimized=True)
                return result
            
            # 计算评估指标
            success, error_msg, metrics_dict, aligned_df_monthly = _calculate_metrics(
                nowcast_series_orig, self.context.full_data, self.context.target_variable,
                self.context.validation_start, self.context.validation_end, self.context.train_end_date
            )
            
            if not success:
                result = _handle_step_failure("指标计算", error_msg, prepared_variables, k_factors, is_svd_error, lambda_df)
                self._store_result_and_update_stats(cache_key, result, start_time, optimized=True)
                return result
            
            # 构建成功结果
            result = (
                metrics_dict.get('is_rmse', np.nan), metrics_dict.get('oos_rmse', np.nan),
                metrics_dict.get('is_mae', np.nan), metrics_dict.get('oos_mae', np.nan),
                metrics_dict.get('is_hit_rate', np.nan), metrics_dict.get('oos_hit_rate', np.nan),
                is_svd_error, lambda_df, aligned_df_monthly
            )
            
            self._store_result_and_update_stats(cache_key, result, start_time, optimized=True)
            return result
            
        except Exception as e:
            logger.error(f"优化评估过程中发生错误: {e}，回退到原始方法")
            self.stats['fallback_evaluations'] += 1
            return self._evaluate_with_fallback(variables, fallback_evaluate_func, start_time)
    
    def _evaluate_with_fallback(
        self, 
        variables: List[str], 
        fallback_evaluate_func: Callable,
        start_time: float
    ) -> Tuple:
        """使用回退函数进行评估"""
        try:
            result = fallback_evaluate_func(
                variables=variables,
                full_data=self.context.full_data,
                target_variable=self.context.target_variable,
                params=self.context.params,
                validation_start=self.context.validation_start,
                validation_end=self.context.validation_end,
                target_freq=self.context.target_freq,
                train_end_date=self.context.train_end_date,
                target_mean_original=self.context.target_mean_original,
                target_std_original=self.context.target_std_original,
                max_iter=self.context.max_iter,
                max_lags=self.context.max_lags,
                use_cache=self.use_cache,
                cache_instance=self.cache_instance
            )
            
            elapsed = time.time() - start_time
            self.stats['total_evaluation_time'] += elapsed
            logger.debug(f"回退评估完成: 变量数={len(variables)}, 耗时={elapsed:.3f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"回退评估也失败了: {e}")
            elapsed = time.time() - start_time
            self.stats['total_evaluation_time'] += elapsed
            return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False, None, None)
    
    def _store_result_and_update_stats(
        self, 
        cache_key: Optional[str], 
        result: Tuple, 
        start_time: float,
        optimized: bool = True
    ):
        """存储结果到缓存并更新统计信息"""
        elapsed = time.time() - start_time
        self.stats['total_evaluation_time'] += elapsed
        
        if optimized:
            self.stats['optimized_evaluations'] += 1
            # 估算时间节省（基于预计算上下文的节省）
            context_stats = self.context.get_statistics()
            avg_time_saved = context_stats.get('avg_time_saved_per_access', 0)
            self.stats['total_time_saved'] += avg_time_saved
        
        # 存储到缓存
        if (self.use_cache and cache_key is not None and 
            self.cache_instance is not None):
            self.cache_instance.put(cache_key, result)
        
        logger.debug(f"{'优化' if optimized else '回退'}评估完成: 耗时={elapsed:.3f}秒")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.stats.copy()
        stats['uptime_seconds'] = time.time() - stats['creation_time']
        stats['optimization_rate'] = (
            self.stats['optimized_evaluations'] / max(1, self.stats['total_evaluations'])
        )
        stats['cache_hit_rate'] = (
            self.stats['cache_hits'] / max(1, self.stats['total_evaluations'])
        )
        stats['avg_evaluation_time'] = (
            self.stats['total_evaluation_time'] / max(1, self.stats['total_evaluations'])
        )
        
        # 添加上下文统计
        if self.context is not None:
            stats['context_stats'] = self.context.get_statistics()
        
        return stats
    
    def print_statistics(self):
        """打印性能统计信息"""
        stats = self.get_statistics()
        print("\n=== 优化DFM评估器统计 ===")
        print(f"总评估次数: {stats['total_evaluations']}")
        print(f"优化评估次数: {stats['optimized_evaluations']}")
        print(f"回退评估次数: {stats['fallback_evaluations']}")
        print(f"缓存命中次数: {stats['cache_hits']}")
        print(f"优化成功率: {stats['optimization_rate']:.1%}")
        print(f"缓存命中率: {stats['cache_hit_rate']:.1%}")
        print(f"总节省时间: {stats['total_time_saved']:.3f}秒")
        print(f"平均评估时间: {stats['avg_evaluation_time']:.3f}秒")
        print(f"运行时间: {stats['uptime_seconds']:.1f}秒")
        print("=" * 25)
        
        # 打印上下文统计
        if self.context is not None:
            self.context.print_statistics()