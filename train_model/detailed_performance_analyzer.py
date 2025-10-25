# -*- coding: utf-8 -*-
"""
详细性能分析器

专注于DFM优化系统各个组件的深度性能分析：
1. 组件级别的时间分解
2. 内存使用模式分析
3. 缓存效果统计
4. 预计算上下文效率分析
5. 逐步性能跟踪
"""

import time
import gc
import psutil
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import queue

logger = logging.getLogger(__name__)


@dataclass
class ComponentMetrics:
    """组件性能指标"""
    component_name: str
    execution_time: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_percent: float = 0.0
    calls_count: int = 0
    success_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    additional_stats: Dict[str, Any] = field(default_factory=dict)


class PerformanceTimer:
    """高精度性能计时器"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0.0
        self.splits = {}
    
    def start(self):
        """开始计时"""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self):
        """停止计时"""
        if self.start_time is None:
            raise ValueError("计时器尚未启动")
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        return self.elapsed_time
    
    def split(self, label: str):
        """记录分段时间"""
        if self.start_time is None:
            raise ValueError("计时器尚未启动")
        current_time = time.perf_counter()
        self.splits[label] = current_time - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.memory_samples = queue.Queue()
        
    def start_monitoring(self):
        """开始内存监控"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """停止内存监控并返回统计信息"""
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # 收集内存样本
        samples = []
        while not self.memory_samples.empty():
            try:
                samples.append(self.memory_samples.get_nowait())
            except queue.Empty:
                break
        
        if not samples:
            return {}
        
        return {
            'peak_memory_mb': max(samples),
            'average_memory_mb': np.mean(samples),
            'memory_std_mb': np.std(samples),
            'min_memory_mb': min(samples),
            'sample_count': len(samples)
        }
    
    def _monitor_loop(self):
        """内存监控循环"""
        while self.monitoring:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.memory_samples.put(memory_mb)
                time.sleep(self.interval)
            except Exception as e:
                logger.warning(f"内存监控出错: {e}")
                break


class ComponentProfiler:
    """组件性能分析器"""
    
    def __init__(self):
        self.metrics = {}
        self.call_stack = []
        self.memory_monitor = MemoryMonitor()
        
    def start_profiling_component(self, component_name: str):
        """开始分析组件"""
        if component_name not in self.metrics:
            self.metrics[component_name] = ComponentMetrics(component_name=component_name)
        
        metrics = self.metrics[component_name]
        metrics.calls_count += 1
        
        # 记录开始状态
        gc.collect()  # 强制垃圾收集以获得更准确的内存测量
        metrics.memory_before_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 开始内存监控
        self.memory_monitor.start_monitoring()
        
        # 记录到调用栈
        profile_entry = {
            'component': component_name,
            'start_time': time.perf_counter(),
            'start_memory': metrics.memory_before_mb
        }
        self.call_stack.append(profile_entry)
        
        return profile_entry
    
    def stop_profiling_component(self, component_name: str, success: bool = True):
        """停止分析组件"""
        if component_name not in self.metrics:
            logger.warning(f"尝试停止未开始的组件分析: {component_name}")
            return
        
        # 从调用栈中找到对应的入口
        profile_entry = None
        for i in range(len(self.call_stack) - 1, -1, -1):
            if self.call_stack[i]['component'] == component_name:
                profile_entry = self.call_stack.pop(i)
                break
        
        if profile_entry is None:
            logger.warning(f"在调用栈中未找到组件: {component_name}")
            return
        
        metrics = self.metrics[component_name]
        
        # 记录执行时间
        end_time = time.perf_counter()
        execution_time = end_time - profile_entry['start_time']
        metrics.execution_time += execution_time
        
        # 记录结束状态
        gc.collect()
        metrics.memory_after_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 停止内存监控并获取峰值
        memory_stats = self.memory_monitor.stop_monitoring()
        if memory_stats:
            metrics.memory_peak_mb = max(metrics.memory_peak_mb, memory_stats.get('peak_memory_mb', 0))
        
        # 记录成功状态
        if success:
            metrics.success_count += 1
        
        logger.debug(f"组件 {component_name} 分析完成: {execution_time:.3f}秒")
    
    @contextmanager
    def profile_component(self, component_name: str):
        """组件性能分析上下文管理器"""
        self.start_profiling_component(component_name)
        success = False
        try:
            yield self.metrics[component_name]
            success = True
        except Exception as e:
            logger.error(f"组件 {component_name} 执行出错: {e}")
            raise
        finally:
            self.stop_profiling_component(component_name, success)
    
    def get_component_metrics(self, component_name: str) -> Optional[ComponentMetrics]:
        """获取组件指标"""
        return self.metrics.get(component_name)
    
    def get_all_metrics(self) -> Dict[str, ComponentMetrics]:
        """获取所有组件指标"""
        return self.metrics.copy()
    
    def reset(self):
        """重置所有指标"""
        self.metrics.clear()
        self.call_stack.clear()


class DetailedDFMAnalyzer:
    """详细DFM性能分析器"""
    
    def __init__(self):
        self.profiler = ComponentProfiler()
        self.optimization_stats = {}
        self.baseline_stats = {}
        
        # 导入DFM模块
        self._import_dfm_modules()
    
    def _import_dfm_modules(self):
        """导入DFM模块"""
        from dashboard.DFM.train_model.variable_selection import perform_global_backward_selection
        from dashboard.DFM.train_model.dfm_core import evaluate_dfm_params
        from dashboard.DFM.train_model.precomputed_dfm_context import PrecomputedDFMContext
        from dashboard.DFM.train_model.optimized_dfm_evaluator import OptimizedDFMEvaluator

        self.perform_global_backward_selection = perform_global_backward_selection
        self.evaluate_dfm_params = evaluate_dfm_params
        self.PrecomputedDFMContext = PrecomputedDFMContext
        self.OptimizedDFMEvaluator = OptimizedDFMEvaluator
    
    def analyze_baseline_performance(
        self, 
        data: pd.DataFrame, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分析基线性能（无优化）"""
        logger.info("开始基线性能详细分析...")
        
        self.profiler.reset()
        baseline_results = {}
        
        with self.profiler.profile_component("baseline_variable_selection"):
            # 分析各个步骤
            with self.profiler.profile_component("baseline_initialization"):
                initial_variables = params['variables']
                target_variable = params['target_variable']
            
            with self.profiler.profile_component("baseline_evaluation_loop"):
                try:
                    result = self.perform_global_backward_selection(
                        initial_variables=initial_variables,
                        initial_params=params['params'],
                        target_variable=target_variable,
                        all_data=data,
                        validation_start=params['validation_start'],
                        validation_end=params['validation_end'],
                        target_freq=params['target_freq'],
                        train_end_date=params['train_end_date'],
                        n_iter=params['max_iter'],
                        target_mean_original=params['target_mean_original'],
                        target_std_original=params['target_std_original'],
                        max_workers=params['max_workers'],
                        evaluate_dfm_func=self._wrapped_evaluate_dfm_baseline,
                        max_lags=params['max_lags'],
                        use_optimization=False
                    )
                    
                    final_vars, final_params, final_score, eval_count, svd_errors = result
                    baseline_results.update({
                        'final_variables_count': len([v for v in final_vars if v != target_variable]),
                        'evaluations_count': eval_count,
                        'svd_error_count': svd_errors,
                        'final_score': final_score
                    })
                    
                except Exception as e:
                    logger.error(f"基线分析失败: {e}")
                    baseline_results['error'] = str(e)
        
        # 收集组件指标
        baseline_results['component_metrics'] = self.profiler.get_all_metrics()
        self.baseline_stats = baseline_results
        
        logger.info("基线性能分析完成")
        return baseline_results
    
    def analyze_optimized_performance(
        self, 
        data: pd.DataFrame, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分析优化性能"""
        logger.info("开始优化性能详细分析...")
        
        self.profiler.reset()
        optimized_results = {}
        
        with self.profiler.profile_component("optimized_variable_selection"):
            # 分析预计算上下文创建
            with self.profiler.profile_component("precomputed_context_creation"):
                try:
                    context = self.PrecomputedDFMContext(
                        full_data=data,
                        initial_variables=params['variables'],
                        target_variable=params['target_variable'],
                        params=params['params'],
                        validation_start=params['validation_start'],
                        validation_end=params['validation_end'],
                        target_freq=params['target_freq'],
                        train_end_date=params['train_end_date'],
                        target_mean_original=params['target_mean_original'],
                        target_std_original=params['target_std_original'],
                        max_iter=params['max_iter'],
                        max_lags=params['max_lags']
                    )
                    
                    context_metrics = self.profiler.get_component_metrics("precomputed_context_creation")
                    if context_metrics:
                        context_stats = context.get_statistics()
                        context_metrics.additional_stats.update(context_stats)
                    
                except Exception as e:
                    logger.error(f"预计算上下文创建失败: {e}")
                    optimized_results['context_error'] = str(e)
                    return optimized_results
            
            # 分析优化评估器创建
            with self.profiler.profile_component("optimized_evaluator_creation"):
                try:
                    evaluator = self.OptimizedDFMEvaluator(
                        precomputed_context=context,
                        use_cache=True
                    )
                except Exception as e:
                    logger.error(f"优化评估器创建失败: {e}")
                    optimized_results['evaluator_error'] = str(e)
                    return optimized_results
            
            # 分析优化评估循环
            with self.profiler.profile_component("optimized_evaluation_loop"):
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
                        evaluate_dfm_func=self._wrapped_evaluate_dfm_optimized,
                        max_lags=params['max_lags'],
                        use_optimization=True
                    )
                    
                    final_vars, final_params, final_score, eval_count, svd_errors = result
                    optimized_results.update({
                        'final_variables_count': len([v for v in final_vars if v != target_variable]),
                        'evaluations_count': eval_count,
                        'svd_error_count': svd_errors,
                        'final_score': final_score
                    })
                    
                    # 获取优化统计
                    evaluator_stats = evaluator.get_statistics()
                    optimized_results['evaluator_stats'] = evaluator_stats
                    
                except Exception as e:
                    logger.error(f"优化评估失败: {e}")
                    optimized_results['evaluation_error'] = str(e)
        
        # 收集组件指标
        optimized_results['component_metrics'] = self.profiler.get_all_metrics()
        self.optimization_stats = optimized_results
        
        logger.info("优化性能分析完成")
        return optimized_results
    
    def _wrapped_evaluate_dfm_baseline(self, **kwargs):
        """包装的基线评估函数，用于性能分析"""
        with self.profiler.profile_component("dfm_evaluation_baseline"):
            with self.profiler.profile_component("data_preparation"):
                # 数据准备步骤在evaluate_dfm_params中
                pass
            
            with self.profiler.profile_component("dfm_model_fitting"):
                # 模型拟合步骤在evaluate_dfm_params中
                pass
            
            with self.profiler.profile_component("metrics_calculation"):
                # 指标计算步骤在evaluate_dfm_params中
                result = self.evaluate_dfm_params(**kwargs)
        
        return result
    
    def _wrapped_evaluate_dfm_optimized(self, **kwargs):
        """包装的优化评估函数，用于性能分析"""
        with self.profiler.profile_component("dfm_evaluation_optimized"):
            # 使用优化的评估方法
            result = self.evaluate_dfm_params(**kwargs)
        
        return result
    
    def compare_performance(self) -> Dict[str, Any]:
        """比较基线和优化性能"""
        if not self.baseline_stats or not self.optimization_stats:
            logger.warning("缺少基线或优化统计数据，无法进行比较")
            return {}
        
        comparison = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'baseline_components': {},
            'optimized_components': {},
            'component_improvements': {},
            'overall_improvement': {}
        }
        
        # 比较组件性能
        baseline_components = self.baseline_stats.get('component_metrics', {})
        optimized_components = self.optimization_stats.get('component_metrics', {})
        
        for comp_name, baseline_metrics in baseline_components.items():
            comparison['baseline_components'][comp_name] = {
                'execution_time': baseline_metrics.execution_time,
                'memory_peak_mb': baseline_metrics.memory_peak_mb,
                'calls_count': baseline_metrics.calls_count,
                'success_rate': baseline_metrics.success_count / max(1, baseline_metrics.calls_count)
            }
        
        for comp_name, optimized_metrics in optimized_components.items():
            comparison['optimized_components'][comp_name] = {
                'execution_time': optimized_metrics.execution_time,
                'memory_peak_mb': optimized_metrics.memory_peak_mb,
                'calls_count': optimized_metrics.calls_count,
                'success_rate': optimized_metrics.success_count / max(1, optimized_metrics.calls_count)
            }
        
        # 计算改进比率
        for comp_name in baseline_components:
            if comp_name in optimized_components:
                baseline_time = baseline_components[comp_name].execution_time
                optimized_time = optimized_components[comp_name].execution_time
                
                if baseline_time > 0 and optimized_time > 0:
                    speedup = baseline_time / optimized_time
                    time_saved = baseline_time - optimized_time
                    
                    comparison['component_improvements'][comp_name] = {
                        'speedup_ratio': speedup,
                        'time_saved_seconds': time_saved,
                        'time_reduction_percent': (time_saved / baseline_time) * 100
                    }
        
        # 总体改进
        baseline_total = sum(m.execution_time for m in baseline_components.values())
        optimized_total = sum(m.execution_time for m in optimized_components.values())
        
        if baseline_total > 0 and optimized_total > 0:
            overall_speedup = baseline_total / optimized_total
            overall_time_saved = baseline_total - optimized_total
            
            comparison['overall_improvement'] = {
                'total_speedup': overall_speedup,
                'total_time_saved': overall_time_saved,
                'total_reduction_percent': (overall_time_saved / baseline_total) * 100
            }
        
        return comparison
    
    def generate_detailed_report(self) -> str:
        """生成详细分析报告"""
        comparison = self.compare_performance()
        
        if not comparison:
            return "无法生成详细报告：缺少性能比较数据"
        
        report_lines = [
            "=" * 80,
            "DFM优化系统详细性能分析报告",
            "=" * 80,
            f"分析时间: {comparison.get('timestamp', 'Unknown')}",
            ""
        ]
        
        # 总体改进摘要
        overall = comparison.get('overall_improvement', {})
        if overall:
            report_lines.extend([
                "总体性能改进:",
                "-" * 30,
                f"总体加速比: {overall.get('total_speedup', 0):.2f}x",
                f"总体时间节省: {overall.get('total_time_saved', 0):.3f}秒",
                f"总体时间减少: {overall.get('total_reduction_percent', 0):.1f}%",
                ""
            ])
        
        # 组件级别分析
        report_lines.extend([
            "组件级性能分析:",
            "-" * 30
        ])
        
        component_improvements = comparison.get('component_improvements', {})
        baseline_components = comparison.get('baseline_components', {})
        optimized_components = comparison.get('optimized_components', {})
        
        for comp_name, improvement in component_improvements.items():
            baseline_comp = baseline_components.get(comp_name, {})
            optimized_comp = optimized_components.get(comp_name, {})
            
            report_lines.extend([
                f"\n组件: {comp_name}",
                f"  基线执行时间: {baseline_comp.get('execution_time', 0):.3f}秒",
                f"  优化执行时间: {optimized_comp.get('execution_time', 0):.3f}秒",
                f"  性能改进:",
                f"    加速比: {improvement.get('speedup_ratio', 0):.2f}x", 
                f"    时间节省: {improvement.get('time_saved_seconds', 0):.3f}秒",
                f"    减少百分比: {improvement.get('time_reduction_percent', 0):.1f}%"
            ])
        
        # 优化统计信息
        if 'evaluator_stats' in self.optimization_stats:
            evaluator_stats = self.optimization_stats['evaluator_stats']
            report_lines.extend([
                "",
                "优化器统计信息:",
                "-" * 30,
                f"总评估次数: {evaluator_stats.get('total_evaluations', 0)}",
                f"优化评估次数: {evaluator_stats.get('optimized_evaluations', 0)}",
                f"回退评估次数: {evaluator_stats.get('fallback_evaluations', 0)}",
                f"缓存命中次数: {evaluator_stats.get('cache_hits', 0)}",
                f"优化成功率: {evaluator_stats.get('optimization_rate', 0):.1%}",
                f"缓存命中率: {evaluator_stats.get('cache_hit_rate', 0):.1%}",
                f"平均评估时间: {evaluator_stats.get('avg_evaluation_time', 0):.3f}秒"
            ])
        
        # 结论
        report_lines.extend([
            "",
            "=" * 50,
            "详细分析结论:",
            "=" * 50,
        ])
        
        if overall.get('total_speedup', 0) > 1:
            report_lines.extend([
                "优化系统工作良好，显著提升了性能",
                f"整体加速比达到 {overall.get('total_speedup', 0):.2f}x",
                "建议在生产环境中启用优化",
            ])
        else:
            report_lines.extend([
                "优化效果不明显，可能需要调整优化策略",
                "建议检查数据规模和参数配置",
            ])
        
        # 最佳性能组件
        best_improvement = max(
            component_improvements.items(),
            key=lambda x: x[1].get('speedup_ratio', 0),
            default=(None, {})
        )
        
        if best_improvement[0]:
            report_lines.extend([
                "",
                f"最大性能提升组件: {best_improvement[0]}",
                f"加速比: {best_improvement[1].get('speedup_ratio', 0):.2f}x"
            ])
        
        report_lines.extend([
            "",
            "=" * 80,
            "详细报告结束", 
            "=" * 80
        ])
        
        return "\n".join(report_lines)


def run_detailed_analysis():
    """运行详细性能分析"""
    logger.info("开始详细DFM性能分析...")
    
    # 创建分析器
    analyzer = DetailedDFMAnalyzer()
    
    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', periods=48, freq='M')
    data = pd.DataFrame(index=dates)
    
    # 目标变量
    data['target'] = np.random.randn(48).cumsum() * 0.1
    
    # 预测变量
    for i in range(1, 11):
        noise = np.random.randn(48) * 0.2
        data[f'var{i}'] = 0.3 * data['target'] + noise
    
    # 参数配置
    params = {
        'target_variable': 'target',
        'variables': data.columns.tolist(),
        'params': {'k_factors': 2},
        'validation_start': '2020-01-01',
        'validation_end': '2021-12-31',
        'target_freq': 'M',
        'train_end_date': '2019-12-31',
        'max_iter': 15,
        'max_lags': 1,
        'target_mean_original': 0.0,
        'target_std_original': 1.0,
        'max_workers': 1
    }
    
    try:
        # 分析基线性能
        baseline_results = analyzer.analyze_baseline_performance(data, params)
        
        # 分析优化性能
        optimized_results = analyzer.analyze_optimized_performance(data, params)
        
        # 生成和显示详细报告
        detailed_report = analyzer.generate_detailed_report()
        print("\n" + detailed_report)
        
        return analyzer, baseline_results, optimized_results
        
    except Exception as e:
        logger.error(f"详细分析过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    run_detailed_analysis()