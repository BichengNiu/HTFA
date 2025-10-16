# -*- coding: utf-8 -*-
"""
DFM优化系统综合性能基准测试模块

本模块提供全面的性能基准测试，对比优化前后的DFM变量选择系统：
1. 基线性能测量（无优化）
2. 优化性能测量（带预计算和缓存）
3. 详细的时间、内存和计算成本分析
4. 边界条件测试
5. 综合性能报告生成
"""

import time
import os
import sys
import pandas as pd
import numpy as np
import psutil
import gc
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import tracemalloc
import warnings

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 禁用相关警告以保持输出清洁
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    execution_time: float = 0.0
    memory_peak_mb: float = 0.0
    memory_delta_mb: float = 0.0
    evaluations_count: int = 0
    cache_hit_count: int = 0
    optimization_success_rate: float = 0.0
    avg_evaluation_time: float = 0.0
    total_time_saved: float = 0.0
    svd_error_count: int = 0
    final_variables_count: int = 0
    final_score_hr: float = 0.0
    final_score_rmse: float = 0.0
    additional_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class BenchmarkResult:
    """基准测试结果数据类"""
    test_name: str
    baseline_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    improvement_ratios: Dict[str, float] = field(default_factory=dict)
    test_parameters: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str = ""


class MemoryProfiler:
    """内存性能分析器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = 0.0
        self.peak_memory = 0.0
        self.tracemalloc_started = False
    
    def start_profiling(self):
        """开始内存性能分析"""
        gc.collect()  # 强制垃圾收集
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
        # 启动tracemalloc进行详细内存追踪
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.tracemalloc_started = True
    
    def update_peak_memory(self):
        """更新峰值内存使用"""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)
    
    def stop_profiling(self) -> Tuple[float, float]:
        """
        停止内存性能分析
        
        Returns:
            Tuple[float, float]: (peak_memory_mb, memory_delta_mb)
        """
        self.update_peak_memory()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        memory_delta = end_memory - self.start_memory
        
        if self.tracemalloc_started:
            tracemalloc.stop()
            self.tracemalloc_started = False
        
        gc.collect()  # 清理内存
        
        return self.peak_memory, memory_delta


@contextmanager
def memory_profiler():
    """内存性能分析上下文管理器"""
    profiler = MemoryProfiler()
    profiler.start_profiling()
    try:
        yield profiler
    finally:
        peak_memory, memory_delta = profiler.stop_profiling()
        profiler.peak_memory = peak_memory
        profiler.memory_delta = memory_delta


class DFMBenchmarkSuite:
    """DFM性能基准测试套件"""
    
    def __init__(self, test_data_generator: Optional[Callable] = None):
        """
        初始化基准测试套件
        
        Args:
            test_data_generator: 自定义测试数据生成器函数
        """
        self.test_data_generator = test_data_generator or self._default_data_generator
        self.results: List[BenchmarkResult] = []
        
        # 导入DFM相关模块
        self._import_dfm_modules()

    def _import_dfm_modules(self):
        """导入DFM相关模块"""
        from dashboard.DFM.train_model.variable_selection import perform_global_backward_selection
        from dashboard.DFM.train_model.dfm_core import evaluate_dfm_params
        from dashboard.DFM.train_model.precomputed_dfm_context import PrecomputedDFMContext
        from dashboard.DFM.train_model.optimized_dfm_evaluator import OptimizedDFMEvaluator

        self.perform_global_backward_selection = perform_global_backward_selection
        self.evaluate_dfm_params = evaluate_dfm_params
        self.PrecomputedDFMContext = PrecomputedDFMContext
        self.OptimizedDFMEvaluator = OptimizedDFMEvaluator

        logger.info("DFM模块导入成功")
    
    @staticmethod
    def _default_data_generator(
        n_vars: int = 15, 
        n_periods: int = 60, 
        start_date: str = '2018-01-01',
        freq: str = 'M',
        seed: int = 42
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        默认测试数据生成器
        
        Args:
            n_vars: 变量数量（包括目标变量）
            n_periods: 时间序列长度
            start_date: 开始日期
            freq: 数据频率
            seed: 随机种子
        
        Returns:
            Tuple[pd.DataFrame, Dict]: (测试数据, 参数配置)
        """
        np.random.seed(seed)
        
        # 创建时间索引
        dates = pd.date_range(start_date, periods=n_periods, freq=freq)
        
        # 创建测试数据
        data = pd.DataFrame(index=dates)
        
        # 目标变量 - 带趋势和周期性
        trend = np.linspace(0, 2, n_periods)
        seasonal = 0.5 * np.sin(2 * np.pi * np.arange(n_periods) / 12)
        noise = np.random.randn(n_periods) * 0.3
        data['target'] = trend + seasonal + noise
        
        # 预测变量 - 与目标变量有不同程度的相关性
        for i in range(1, n_vars):
            # 创建不同相关性强度的预测变量
            correlation_strength = 0.8 - (i-1) * 0.05  # 递减相关性
            lag = np.random.randint(0, 3)  # 随机滞后
            
            base_signal = np.roll(data['target'], lag) * correlation_strength
            var_noise = np.random.randn(n_periods) * 0.4
            data[f'var{i}'] = base_signal + var_noise
        
        # 添加一些缺失值来模拟真实数据
        missing_rate = 0.02  # 2%缺失率
        for col in data.columns:
            missing_indices = np.random.choice(
                data.index, 
                size=int(len(data) * missing_rate), 
                replace=False
            )
            data.loc[missing_indices, col] = np.nan
        
        # 参数配置
        params = {
            'target_variable': 'target',
            'variables': data.columns.tolist(),
            'params': {'k_factors': min(3, n_vars-2)},  # 确保因子数合理
            'validation_start': dates[int(n_periods * 0.7)].strftime('%Y-%m-%d'),
            'validation_end': dates[-1].strftime('%Y-%m-%d'),
            'target_freq': 'M',
            'train_end_date': dates[int(n_periods * 0.6)].strftime('%Y-%m-%d'),
            'max_iter': 20,
            'max_lags': 1,
            'target_mean_original': 0.0,
            'target_std_original': 1.0,
            'max_workers': 1  # 串行执行以保持结果一致性
        }
        
        return data, params
    
    def _run_variable_selection_baseline(
        self, 
        data: pd.DataFrame, 
        params: Dict[str, Any]
    ) -> PerformanceMetrics:
        """运行基线变量选择（无优化）"""
        logger.info("开始基线性能测试...")
        
        metrics = PerformanceMetrics()
        
        with memory_profiler() as mem_prof:
            start_time = time.time()
            
            try:
                result = self.perform_global_backward_selection(
                    initial_variables=params['variables'],
                    initial_params=params['params'],
                    target_variable=params['target_variable'],
                    all_data=data,
                    validation_start=params['validation_start'],
                    validation_end=params['validation_end'],
                    target_freq=params['target_freq'],
                    train_end_date=params['train_end_date'],
                    n_iter=params['max_iter'],
                    target_mean_original=params['target_mean_original'],
                    target_std_original=params['target_std_original'],
                    max_workers=params['max_workers'],
                    evaluate_dfm_func=self.evaluate_dfm_params,
                    max_lags=params['max_lags'],
                    use_optimization=False  # 关键：禁用优化
                )
                
                final_vars, final_params, final_score, eval_count, svd_errors = result
                
                metrics.execution_time = time.time() - start_time
                metrics.evaluations_count = eval_count
                metrics.svd_error_count = svd_errors
                metrics.final_variables_count = len([v for v in final_vars if v != params['target_variable']])
                metrics.final_score_hr = final_score[0] if len(final_score) > 0 else 0.0
                metrics.final_score_rmse = -final_score[1] if len(final_score) > 1 else np.inf
                
                logger.info(f"基线测试完成: {metrics.execution_time:.2f}秒, {metrics.evaluations_count}次评估")
                
            except Exception as e:
                logger.error(f"基线测试失败: {e}")
                metrics.execution_time = time.time() - start_time
                raise
        
        metrics.memory_peak_mb = mem_prof.peak_memory
        metrics.memory_delta_mb = mem_prof.memory_delta
        metrics.avg_evaluation_time = metrics.execution_time / max(1, metrics.evaluations_count)
        
        return metrics
    
    def _run_variable_selection_optimized(
        self, 
        data: pd.DataFrame, 
        params: Dict[str, Any]
    ) -> PerformanceMetrics:
        """运行优化版变量选择"""
        logger.info("开始优化性能测试...")
        
        metrics = PerformanceMetrics()
        
        with memory_profiler() as mem_prof:
            start_time = time.time()
            
            try:
                result = self.perform_global_backward_selection(
                    initial_variables=params['variables'],
                    initial_params=params['params'],
                    target_variable=params['target_variable'],
                    all_data=data,
                    validation_start=params['validation_start'],
                    validation_end=params['validation_end'],
                    target_freq=params['target_freq'],
                    train_end_date=params['train_end_date'],
                    n_iter=params['max_iter'],
                    target_mean_original=params['target_mean_original'],
                    target_std_original=params['target_std_original'],
                    max_workers=params['max_workers'],
                    evaluate_dfm_func=self.evaluate_dfm_params,
                    max_lags=params['max_lags'],
                    use_optimization=True  # 关键：启用优化
                )
                
                final_vars, final_params, final_score, eval_count, svd_errors = result
                
                metrics.execution_time = time.time() - start_time
                metrics.evaluations_count = eval_count
                metrics.svd_error_count = svd_errors
                metrics.final_variables_count = len([v for v in final_vars if v != params['target_variable']])
                metrics.final_score_hr = final_score[0] if len(final_score) > 0 else 0.0
                metrics.final_score_rmse = -final_score[1] if len(final_score) > 1 else np.inf
                
                logger.info(f"优化测试完成: {metrics.execution_time:.2f}秒, {metrics.evaluations_count}次评估")
                
            except Exception as e:
                logger.error(f"优化测试失败: {e}")
                metrics.execution_time = time.time() - start_time
                raise
        
        metrics.memory_peak_mb = mem_prof.peak_memory
        metrics.memory_delta_mb = mem_prof.memory_delta
        metrics.avg_evaluation_time = metrics.execution_time / max(1, metrics.evaluations_count)
        
        return metrics
    
    def _calculate_improvement_ratios(
        self, 
        baseline: PerformanceMetrics, 
        optimized: PerformanceMetrics
    ) -> Dict[str, float]:
        """计算性能改进比率"""
        ratios = {}
        
        # 时间改进（加速比）
        if baseline.execution_time > 0:
            ratios['time_speedup'] = baseline.execution_time / optimized.execution_time
            ratios['time_reduction_pct'] = (1 - optimized.execution_time / baseline.execution_time) * 100
        
        # 内存改进
        if baseline.memory_peak_mb > 0:
            ratios['memory_efficiency'] = baseline.memory_peak_mb / optimized.memory_peak_mb
            ratios['memory_reduction_pct'] = (1 - optimized.memory_peak_mb / baseline.memory_peak_mb) * 100
        
        # 评估效率改进
        if baseline.avg_evaluation_time > 0:
            ratios['evaluation_speedup'] = baseline.avg_evaluation_time / optimized.avg_evaluation_time
        
        # 准确性比较（确保优化没有损失精度）
        ratios['hr_accuracy_ratio'] = optimized.final_score_hr / max(baseline.final_score_hr, 1e-6)
        ratios['rmse_accuracy_ratio'] = baseline.final_score_rmse / max(optimized.final_score_rmse, 1e-6)
        
        return ratios
    
    def run_standard_benchmark(self) -> BenchmarkResult:
        """运行标准基准测试"""
        logger.info("=== 开始标准基准测试 ===")
        
        # 生成测试数据
        data, params = self.test_data_generator(n_vars=12, n_periods=48)
        
        try:
            # 运行基线测试
            baseline_metrics = self._run_variable_selection_baseline(data, params)
            
            # 运行优化测试
            optimized_metrics = self._run_variable_selection_optimized(data, params)
            
            # 计算改进比率
            improvement_ratios = self._calculate_improvement_ratios(baseline_metrics, optimized_metrics)
            
            result = BenchmarkResult(
                test_name="标准基准测试",
                baseline_metrics=baseline_metrics,
                optimized_metrics=optimized_metrics,
                improvement_ratios=improvement_ratios,
                test_parameters={
                    'n_variables': len(params['variables']),
                    'n_periods': len(data),
                    'k_factors': params['params']['k_factors']
                },
                success=True
            )
            
            self.results.append(result)
            logger.info("标准基准测试完成")
            
            return result
            
        except Exception as e:
            error_msg = f"标准基准测试失败: {e}"
            logger.error(error_msg)
            
            result = BenchmarkResult(
                test_name="标准基准测试",
                baseline_metrics=PerformanceMetrics(),
                optimized_metrics=PerformanceMetrics(),
                success=False,
                error_message=error_msg
            )
            
            self.results.append(result)
            return result
    
    def run_scalability_benchmark(self) -> List[BenchmarkResult]:
        """运行可扩展性基准测试（不同数据规模）"""
        logger.info("=== 开始可扩展性基准测试 ===")
        
        scalability_results = []
        test_configs = [
            {'n_vars': 8, 'n_periods': 36, 'name': '小规模'},
            {'n_vars': 12, 'n_periods': 48, 'name': '中等规模'},
            {'n_vars': 16, 'n_periods': 60, 'name': '大规模'},
            {'n_vars': 20, 'n_periods': 72, 'name': '特大规模'},
        ]
        
        for config in test_configs:
            logger.info(f"测试{config['name']}数据集...")
            
            try:
                # 生成不同规模的测试数据
                data, params = self.test_data_generator(
                    n_vars=config['n_vars'], 
                    n_periods=config['n_periods']
                )
                
                # 运行基线和优化测试
                baseline_metrics = self._run_variable_selection_baseline(data, params)
                optimized_metrics = self._run_variable_selection_optimized(data, params)
                
                # 计算改进比率
                improvement_ratios = self._calculate_improvement_ratios(baseline_metrics, optimized_metrics)
                
                result = BenchmarkResult(
                    test_name=f"可扩展性测试-{config['name']}",
                    baseline_metrics=baseline_metrics,
                    optimized_metrics=optimized_metrics,
                    improvement_ratios=improvement_ratios,
                    test_parameters={
                        'n_variables': config['n_vars'],
                        'n_periods': config['n_periods'],
                        'k_factors': params['params']['k_factors'],
                        'scale': config['name']
                    },
                    success=True
                )
                
                scalability_results.append(result)
                self.results.append(result)
                
                logger.info(f"{config['name']}测试完成，加速比: {improvement_ratios.get('time_speedup', 0):.2f}x")
                
            except Exception as e:
                error_msg = f"可扩展性测试({config['name']})失败: {e}"
                logger.error(error_msg)
                
                result = BenchmarkResult(
                    test_name=f"可扩展性测试-{config['name']}",
                    baseline_metrics=PerformanceMetrics(),
                    optimized_metrics=PerformanceMetrics(),
                    success=False,
                    error_message=error_msg,
                    test_parameters=config
                )
                
                scalability_results.append(result)
                self.results.append(result)
        
        logger.info("可扩展性基准测试完成")
        return scalability_results
    
    def run_edge_case_benchmark(self) -> List[BenchmarkResult]:
        """运行边界条件基准测试"""
        logger.info("=== 开始边界条件基准测试 ===")
        
        edge_results = []
        edge_configs = [
            {'n_vars': 5, 'n_periods': 24, 'name': '最小可行规模', 'k_factors': 1},
            {'n_vars': 25, 'n_periods': 84, 'name': '高维度数据', 'k_factors': 4},
            {'n_vars': 10, 'n_periods': 120, 'name': '长时间序列', 'k_factors': 2},
        ]
        
        for config in edge_configs:
            logger.info(f"测试{config['name']}...")
            
            try:
                # 生成边界条件测试数据
                data, params = self.test_data_generator(
                    n_vars=config['n_vars'], 
                    n_periods=config['n_periods']
                )
                
                # 调整因子数
                params['params']['k_factors'] = config['k_factors']
                
                # 运行基线和优化测试
                baseline_metrics = self._run_variable_selection_baseline(data, params)
                optimized_metrics = self._run_variable_selection_optimized(data, params)
                
                # 计算改进比率
                improvement_ratios = self._calculate_improvement_ratios(baseline_metrics, optimized_metrics)
                
                result = BenchmarkResult(
                    test_name=f"边界条件测试-{config['name']}",
                    baseline_metrics=baseline_metrics,
                    optimized_metrics=optimized_metrics,
                    improvement_ratios=improvement_ratios,
                    test_parameters={
                        'n_variables': config['n_vars'],
                        'n_periods': config['n_periods'],
                        'k_factors': config['k_factors'],
                        'edge_case': config['name']
                    },
                    success=True
                )
                
                edge_results.append(result)
                self.results.append(result)
                
                logger.info(f"{config['name']}测试完成")
                
            except Exception as e:
                error_msg = f"边界条件测试({config['name']})失败: {e}"
                logger.error(error_msg)
                
                result = BenchmarkResult(
                    test_name=f"边界条件测试-{config['name']}",
                    baseline_metrics=PerformanceMetrics(),
                    optimized_metrics=PerformanceMetrics(),
                    success=False,
                    error_message=error_msg,
                    test_parameters=config
                )
                
                edge_results.append(result)
                self.results.append(result)
        
        logger.info("边界条件基准测试完成")
        return edge_results
    
    def run_full_benchmark_suite(self) -> List[BenchmarkResult]:
        """运行完整的基准测试套件"""
        logger.info("========== 开始完整DFM性能基准测试 ==========")
        
        all_results = []
        
        # 1. 标准基准测试
        standard_result = self.run_standard_benchmark()
        all_results.append(standard_result)
        
        # 2. 可扩展性基准测试
        scalability_results = self.run_scalability_benchmark()
        all_results.extend(scalability_results)
        
        # 3. 边界条件基准测试
        edge_results = self.run_edge_case_benchmark()
        all_results.extend(edge_results)
        
        logger.info("完整基准测试套件执行完成")
        return all_results
    
    def generate_performance_report(self) -> str:
        """生成综合性能报告"""
        if not self.results:
            return "没有基准测试结果可用于生成报告。"
        
        report_lines = [
            "=" * 80,
            "DFM优化系统性能基准测试报告",
            "=" * 80,
            f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"总测试数: {len(self.results)}",
            f"成功测试数: {sum(1 for r in self.results if r.success)}",
            ""
        ]
        
        # 成功测试的总体统计
        successful_results = [r for r in self.results if r.success]
        if successful_results:
            avg_time_speedup = np.mean([r.improvement_ratios.get('time_speedup', 1) for r in successful_results])
            avg_memory_reduction = np.mean([r.improvement_ratios.get('memory_reduction_pct', 0) for r in successful_results])
            
            report_lines.extend([
                "总体性能改进摘要:",
                "-" * 40,
                f"平均时间加速比: {avg_time_speedup:.2f}x",
                f"平均内存节省: {avg_memory_reduction:.1f}%",
                ""
            ])
        
        # 详细测试结果
        report_lines.append("详细测试结果:")
        report_lines.append("-" * 40)
        
        for result in self.results:
            report_lines.append(f"\n测试名称: {result.test_name}")
            
            if result.success:
                baseline = result.baseline_metrics
                optimized = result.optimized_metrics
                ratios = result.improvement_ratios
                
                report_lines.extend([
                    f"  参数: {result.test_parameters}",
                    f"  基线性能:",
                    f"    执行时间: {baseline.execution_time:.3f}秒",
                    f"    内存峰值: {baseline.memory_peak_mb:.1f}MB",
                    f"    评估次数: {baseline.evaluations_count}",
                    f"    平均评估时间: {baseline.avg_evaluation_time:.3f}秒",
                    f"  优化性能:",
                    f"    执行时间: {optimized.execution_time:.3f}秒",
                    f"    内存峰值: {optimized.memory_peak_mb:.1f}MB", 
                    f"    评估次数: {optimized.evaluations_count}",
                    f"    平均评估时间: {optimized.avg_evaluation_time:.3f}秒",
                    f"  性能改进:",
                    f"    时间加速比: {ratios.get('time_speedup', 0):.2f}x",
                    f"    时间节省: {ratios.get('time_reduction_pct', 0):.1f}%",
                    f"    内存效率: {ratios.get('memory_efficiency', 0):.2f}x",
                    f"    内存节省: {ratios.get('memory_reduction_pct', 0):.1f}%",
                    f"    评估加速: {ratios.get('evaluation_speedup', 0):.2f}x"
                ])
            else:
                report_lines.extend([
                    f"  状态: 失败",
                    f"  错误信息: {result.error_message}"
                ])
        
        # 结论和建议
        if successful_results:
            report_lines.extend([
                "",
                "=" * 50,
                "结论和建议:",
                "=" * 50
            ])
            
            max_speedup_result = max(successful_results, key=lambda r: r.improvement_ratios.get('time_speedup', 1))
            min_speedup_result = min(successful_results, key=lambda r: r.improvement_ratios.get('time_speedup', 1))
            
            report_lines.extend([
                f"1. 优化效果最佳的场景: {max_speedup_result.test_name}",
                f"   时间加速比: {max_speedup_result.improvement_ratios.get('time_speedup', 1):.2f}x",
                f"",
                f"2. 优化效果最小的场景: {min_speedup_result.test_name}", 
                f"   时间加速比: {min_speedup_result.improvement_ratios.get('time_speedup', 1):.2f}x",
                f"",
                "3. 推荐使用场景:",
                "   - 中大规模数据集（变量数 > 10）",
                "   - 需要频繁变量选择的情况",
                "   - 对响应时间有较高要求的应用",
                "",
                "4. 优化机制工作良好，建议在生产环境中启用",
                "   use_optimization=True 参数来获得最佳性能。"
            ])
        
        report_lines.extend([
            "",
            "=" * 80,
            "报告结束",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def save_report_to_file(self, filename: str = None):
        """保存性能报告到文件"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"dfm_performance_benchmark_{timestamp}.txt"
        
        report_content = self.generate_performance_report()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"性能报告已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存报告失败: {e}")


def run_comprehensive_benchmark():
    """运行综合性能基准测试"""
    print("开始DFM优化系统综合性能基准测试...")
    
    # 创建基准测试套件
    benchmark_suite = DFMBenchmarkSuite()
    
    try:
        # 运行完整基准测试
        results = benchmark_suite.run_full_benchmark_suite()
        
        # 生成和显示报告
        report = benchmark_suite.generate_performance_report()
        print("\n" + report)
        
        # 保存报告到文件
        benchmark_suite.save_report_to_file()
        
        return benchmark_suite, results
        
    except Exception as e:
        logger.error(f"基准测试过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    # 运行综合基准测试
    suite, results = run_comprehensive_benchmark()