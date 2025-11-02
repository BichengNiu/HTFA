# -*- coding: utf-8 -*-
"""
UI性能监控器
集成统一状态管理的性能监控机制，监控UI组件性能
"""

import time
import psutil
import threading
import os
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    component_id: str
    operation: str
    duration: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime
    metadata: Dict[str, Any] = None


class UIPerformanceMonitor:
    """
    UI性能监控器
    集成统一状态管理的性能监控机制
    """
    
    def __init__(self):
        """初始化UI性能监控器"""
        # 性能指标存储
        self.metrics = []
        self.max_metrics = 1000  # 最大存储指标数量
        
        # 性能阈值配置
        self.thresholds = {
            'render_time': 3.0,      # 渲染时间阈值（秒）
            'memory_usage': 100.0,   # 内存使用阈值（MB）
            'cpu_usage': 80.0,       # CPU使用阈值（%）
            'file_size': 50.0        # 文件大小阈值（MB）
        }
        
        # 性能统计
        self.stats = {
            'total_operations': 0,
            'slow_operations': 0,
            'memory_warnings': 0,
            'cpu_warnings': 0,
            'last_cleanup': datetime.now()
        }
        
        # 监控状态
        self.monitoring_enabled = True
        self.auto_cleanup_enabled = True
        self.cleanup_interval = timedelta(hours=1)
        
        # 启动后台清理线程
        if self.auto_cleanup_enabled:
            self._start_cleanup_thread()
        
        logger.info("UI性能监控器初始化完成")
    
    @contextmanager
    def monitor_operation(self, component_id: str, operation: str, **metadata):
        """
        监控操作性能的上下文管理器
        
        Args:
            component_id: 组件ID
            operation: 操作名称
            **metadata: 额外元数据
        """
        if not self.monitoring_enabled:
            yield
            return
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            cpu_avg = (start_cpu + end_cpu) / 2
            
            # 创建性能指标
            metric = PerformanceMetric(
                component_id=component_id,
                operation=operation,
                duration=duration,
                memory_usage=memory_delta,
                cpu_usage=cpu_avg,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
            # 记录指标
            self._record_metric(metric)
    
    def monitor_component_render(self, component_id: str):
        """
        监控组件渲染性能的装饰器
        
        Args:
            component_id: 组件ID
            
        Returns:
            装饰器函数
        """
        def decorator(render_func: Callable):
            def wrapper(*args, **kwargs):
                with self.monitor_operation(component_id, "render"):
                    return render_func(*args, **kwargs)
            return wrapper
        return decorator
    
    def monitor_file_operation(self, component_id: str, file_path: str, operation: str):
        """
        监控文件操作性能

        Args:
            component_id: 组件ID
            file_path: 文件路径
            operation: 操作类型

        Returns:
            上下文管理器
        """
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

        return self.monitor_operation(
            component_id,
            operation,
            file_path=file_path,
            file_size_mb=file_size
        )
    
    def _record_metric(self, metric: PerformanceMetric):
        """记录性能指标"""
        try:
            # 添加到本地存储
            self.metrics.append(metric)
            
            # 限制存储数量
            if len(self.metrics) > self.max_metrics:
                self.metrics = self.metrics[-self.max_metrics:]
            
            # 更新统计
            self._update_stats(metric)
            
            # 检查性能阈值
            self._check_thresholds(metric)

        except Exception as e:
            logger.error(f"记录性能指标失败: {e}")
    
    def _update_stats(self, metric: PerformanceMetric):
        """更新性能统计"""
        try:
            self.stats['total_operations'] += 1
            
            # 检查慢操作
            if metric.duration > self.thresholds['render_time']:
                self.stats['slow_operations'] += 1
            
            # 检查内存警告
            if metric.memory_usage > self.thresholds['memory_usage']:
                self.stats['memory_warnings'] += 1
            
            # 检查CPU警告
            if metric.cpu_usage > self.thresholds['cpu_usage']:
                self.stats['cpu_warnings'] += 1
                
        except Exception as e:
            logger.error(f"更新性能统计失败: {e}")
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """检查性能阈值"""
        try:
            warnings = []
            
            # 检查渲染时间
            if metric.duration > self.thresholds['render_time']:
                warnings.append(f"渲染时间过长: {metric.duration:.2f}s")
            
            # 检查内存使用
            if metric.memory_usage > self.thresholds['memory_usage']:
                warnings.append(f"内存使用过高: {metric.memory_usage:.2f}MB")
            
            # 检查CPU使用
            if metric.cpu_usage > self.thresholds['cpu_usage']:
                warnings.append(f"CPU使用过高: {metric.cpu_usage:.1f}%")
            
            # 检查文件大小
            if metric.metadata and metric.metadata.get('file_size_mb', 0) > self.thresholds['file_size']:
                warnings.append(f"文件过大: {metric.metadata['file_size_mb']:.2f}MB")
            
            # 记录警告
            if warnings:
                logger.warning(
                    f"性能警告 - 组件: {metric.component_id}, 操作: {metric.operation}, "
                    f"警告: {'; '.join(warnings)}"
                )
                
        except Exception as e:
            logger.error(f"检查性能阈值失败: {e}")
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """获取当前CPU使用率（%）"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def _start_cleanup_thread(self):
        """启动后台清理线程"""
        def cleanup_worker():
            while self.auto_cleanup_enabled:
                try:
                    time.sleep(300)  # 每5分钟检查一次
                    
                    now = datetime.now()
                    if now - self.stats['last_cleanup'] > self.cleanup_interval:
                        self._cleanup_old_metrics()
                        self.stats['last_cleanup'] = now
                        
                except Exception as e:
                    logger.error(f"性能监控清理线程错误: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("性能监控清理线程已启动")
    
    def _cleanup_old_metrics(self):
        """清理旧的性能指标"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            old_count = len(self.metrics)
            
            self.metrics = [
                metric for metric in self.metrics 
                if metric.timestamp > cutoff_time
            ]
            
            cleaned_count = old_count - len(self.metrics)
            if cleaned_count > 0:
                logger.info(f"清理了 {cleaned_count} 个旧的性能指标")
                
        except Exception as e:
            logger.error(f"清理旧性能指标失败: {e}")
    
    def get_performance_summary(self, component_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取性能摘要
        
        Args:
            component_id: 可选的组件ID过滤
            
        Returns:
            Dict[str, Any]: 性能摘要
        """
        try:
            # 过滤指标
            metrics = self.metrics
            if component_id:
                metrics = [m for m in metrics if m.component_id == component_id]
            
            if not metrics:
                return {'message': '没有性能数据'}
            
            # 计算统计信息
            durations = [m.duration for m in metrics]
            memory_usages = [m.memory_usage for m in metrics]
            cpu_usages = [m.cpu_usage for m in metrics]
            
            return {
                'total_operations': len(metrics),
                'avg_duration': sum(durations) / len(durations),
                'max_duration': max(durations),
                'min_duration': min(durations),
                'avg_memory_usage': sum(memory_usages) / len(memory_usages),
                'max_memory_usage': max(memory_usages),
                'avg_cpu_usage': sum(cpu_usages) / len(cpu_usages),
                'max_cpu_usage': max(cpu_usages),
                'slow_operations': len([d for d in durations if d > self.thresholds['render_time']]),
                'time_range': {
                    'start': min(m.timestamp for m in metrics).isoformat(),
                    'end': max(m.timestamp for m in metrics).isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"获取性能摘要失败: {e}")
            return {'error': str(e)}
    
    def get_slow_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最慢的操作
        
        Args:
            limit: 返回数量限制
            
        Returns:
            List[Dict[str, Any]]: 慢操作列表
        """
        try:
            sorted_metrics = sorted(self.metrics, key=lambda m: m.duration, reverse=True)
            
            return [
                {
                    'component_id': m.component_id,
                    'operation': m.operation,
                    'duration': m.duration,
                    'memory_usage': m.memory_usage,
                    'cpu_usage': m.cpu_usage,
                    'timestamp': m.timestamp.isoformat(),
                    'metadata': m.metadata
                }
                for m in sorted_metrics[:limit]
            ]
            
        except Exception as e:
            logger.error(f"获取慢操作失败: {e}")
            return []
    
    def set_threshold(self, metric_name: str, value: float):
        """
        设置性能阈值
        
        Args:
            metric_name: 指标名称
            value: 阈值
        """
        if metric_name in self.thresholds:
            self.thresholds[metric_name] = value
            logger.info(f"性能阈值已更新: {metric_name} = {value}")
        else:
            logger.warning(f"未知的性能指标: {metric_name}")
    
    def enable_monitoring(self):
        """启用性能监控"""
        self.monitoring_enabled = True
        logger.info("性能监控已启用")
    
    def disable_monitoring(self):
        """禁用性能监控"""
        self.monitoring_enabled = False
        logger.info("性能监控已禁用")
    
    def clear_metrics(self):
        """清除所有性能指标"""
        self.metrics.clear()
        self.stats = {
            'total_operations': 0,
            'slow_operations': 0,
            'memory_warnings': 0,
            'cpu_warnings': 0,
            'last_cleanup': datetime.now()
        }
        logger.info("性能指标已清除")


# 全局UI性能监控器实例
_ui_performance_monitor = None


def get_ui_performance_monitor() -> UIPerformanceMonitor:
    """
    获取UI性能监控器实例（单例模式）
    
    Returns:
        UIPerformanceMonitor: UI性能监控器实例
    """
    global _ui_performance_monitor
    if _ui_performance_monitor is None:
        _ui_performance_monitor = UIPerformanceMonitor()
    return _ui_performance_monitor


def reset_ui_performance_monitor():
    """重置UI性能监控器实例"""
    global _ui_performance_monitor
    _ui_performance_monitor = None
