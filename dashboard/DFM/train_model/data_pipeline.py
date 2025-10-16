# -*- coding: utf-8 -*-
"""
DFM训练模块数据处理流水线

提供高性能、可扩展的数据处理流水线，支持缓存、错误处理和进度监控
"""

import os
import json
import pickle
import hashlib
import logging
import traceback
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

from dashboard.DFM.train_model.interfaces import IDataPipeline


logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """流水线步骤数据类"""
    name: str
    processor: Any
    kwargs: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    cache_enabled: bool = True
    description: str = ""
    
    def __post_init__(self):
        if not self.description:
            self.description = f"Processing step: {self.name}"


class DataPipelineError(Exception):
    """数据流水线异常"""
    pass


class CacheManager:
    """缓存管理器
    
    管理流水线中间结果的缓存，支持基于哈希的缓存失效
    """
    
    def __init__(self, cache_dir: Union[str, Path] = None):
        """初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录，默认为临时目录
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / ".pipeline_cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存元数据文件
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """加载缓存元数据"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            self.metadata = {}
    
    def _save_metadata(self):
        """保存缓存元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, data: pd.DataFrame, step_name: str, kwargs: Dict[str, Any]) -> str:
        """生成缓存键
        
        Args:
            data: 输入数据
            step_name: 步骤名称
            kwargs: 步骤参数
            
        Returns:
            str: 缓存键
        """
        # 创建数据哈希
        data_hash = hashlib.md5(pd.util.hash_pandas_object(data).values.tobytes()).hexdigest()
        
        # 创建参数哈希
        kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
        kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()
        
        # 组合哈希
        combined = f"{step_name}_{data_hash}_{kwargs_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, data: pd.DataFrame, step_name: str, kwargs: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """获取缓存数据
        
        Args:
            data: 输入数据
            step_name: 步骤名称
            kwargs: 步骤参数
            
        Returns:
            Optional[pd.DataFrame]: 缓存的数据，如果不存在则返回None
        """
        cache_key = self._get_cache_key(data, step_name, kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            # 检查缓存是否过期（可选功能）
            metadata = self.metadata.get(cache_key, {})
            
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            logger.debug(f"Cache hit for step '{step_name}', key: {cache_key[:8]}...")
            return cached_data
            
        except Exception as e:
            logger.warning(f"Failed to load cached data for step '{step_name}': {e}")
            # 删除损坏的缓存文件
            try:
                cache_file.unlink()
                if cache_key in self.metadata:
                    del self.metadata[cache_key]
                    self._save_metadata()
            except Exception:
                pass
            return None
    
    def set(self, data: pd.DataFrame, result: pd.DataFrame, step_name: str, kwargs: Dict[str, Any]):
        """设置缓存数据
        
        Args:
            data: 输入数据
            result: 输出数据
            step_name: 步骤名称
            kwargs: 步骤参数
        """
        cache_key = self._get_cache_key(data, step_name, kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 更新元数据
            self.metadata[cache_key] = {
                'step_name': step_name,
                'created_at': datetime.now().isoformat(),
                'data_shape': data.shape,
                'result_shape': result.shape
            }
            self._save_metadata()
            
            logger.debug(f"Cached result for step '{step_name}', key: {cache_key[:8]}...")
            
        except Exception as e:
            logger.warning(f"Failed to cache result for step '{step_name}': {e}")
    
    def clear(self, pattern: Optional[str] = None):
        """清除缓存
        
        Args:
            pattern: 匹配模式，如果为None则清除所有缓存
        """
        try:
            if pattern is None:
                # 清除所有缓存
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                self.metadata.clear()
                logger.info("Cleared all cache")
            else:
                # 清除匹配模式的缓存
                keys_to_remove = []
                for key, metadata in self.metadata.items():
                    if pattern in metadata.get('step_name', ''):
                        cache_file = self.cache_dir / f"{key}.pkl"
                        if cache_file.exists():
                            cache_file.unlink()
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.metadata[key]
                
                logger.info(f"Cleared {len(keys_to_remove)} cache entries matching '{pattern}'")
            
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息
        
        Returns:
            Dict: 缓存统计信息
        """
        total_files = len(list(self.cache_dir.glob("*.pkl")))
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        
        return {
            'cache_dir': str(self.cache_dir),
            'total_files': total_files,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'metadata_entries': len(self.metadata)
        }


class DataPipeline(IDataPipeline):
    """数据处理流水线实现
    
    提供链式处理、缓存、错误处理和进度监控等功能
    """
    
    def __init__(self, 
                 name: str = "DataPipeline",
                 cache_enabled: bool = True,
                 cache_dir: Optional[Union[str, Path]] = None,
                 progress_enabled: bool = True):
        """初始化数据流水线
        
        Args:
            name: 流水线名称
            cache_enabled: 是否启用缓存
            cache_dir: 缓存目录
            progress_enabled: 是否显示进度条
        """
        self.name = name
        self.steps: List[PipelineStep] = []
        self.cache_enabled = cache_enabled
        self.progress_enabled = progress_enabled
        
        if cache_enabled:
            self.cache_manager = CacheManager(cache_dir)
        else:
            self.cache_manager = None
        
        # 执行历史
        self.execution_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized DataPipeline '{name}' with cache={'enabled' if cache_enabled else 'disabled'}")
    
    def add_step(self, 
                 name: str, 
                 processor: Any, 
                 cache_enabled: Optional[bool] = None,
                 description: str = "",
                 **kwargs) -> None:
        """添加处理步骤
        
        Args:
            name: 步骤名称
            processor: 处理器实例或函数
            cache_enabled: 是否对此步骤启用缓存
            description: 步骤描述
            **kwargs: 额外参数
        """
        if cache_enabled is None:
            cache_enabled = self.cache_enabled
        
        step = PipelineStep(
            name=name,
            processor=processor,
            kwargs=kwargs,
            cache_enabled=cache_enabled,
            description=description or f"Processing step: {name}"
        )
        
        self.steps.append(step)
        logger.debug(f"Added step '{name}' to pipeline '{self.name}'")
    
    def remove_step(self, name: str) -> None:
        """移除处理步骤
        
        Args:
            name: 步骤名称
        """
        original_count = len(self.steps)
        self.steps = [step for step in self.steps if step.name != name]
        
        if len(self.steps) < original_count:
            logger.debug(f"Removed step '{name}' from pipeline '{self.name}'")
        else:
            logger.warning(f"Step '{name}' not found in pipeline '{self.name}'")
    
    def get_steps(self) -> List[Tuple[str, Any]]:
        """获取所有步骤
        
        Returns:
            List: 步骤列表
        """
        return [(step.name, step.processor) for step in self.steps]
    
    def enable_step(self, name: str, enabled: bool = True):
        """启用/禁用步骤
        
        Args:
            name: 步骤名称
            enabled: 是否启用
        """
        for step in self.steps:
            if step.name == name:
                step.enabled = enabled
                logger.debug(f"Step '{name}' {'enabled' if enabled else 'disabled'}")
                return
        
        logger.warning(f"Step '{name}' not found in pipeline '{self.name}'")
    
    def _execute_step(self, step: PipelineStep, data: pd.DataFrame) -> pd.DataFrame:
        """执行单个步骤
        
        Args:
            step: 流水线步骤
            data: 输入数据
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        if not step.enabled:
            logger.debug(f"Skipping disabled step '{step.name}'")
            return data
        
        # 尝试从缓存获取结果
        if step.cache_enabled and self.cache_manager:
            cached_result = self.cache_manager.get(data, step.name, step.kwargs)
            if cached_result is not None:
                return cached_result
        
        # 执行处理
        try:
            if callable(step.processor):
                # 如果是函数
                if hasattr(step.processor, '__self__'):
                    # 如果是方法，传递kwargs
                    result = step.processor(data, **step.kwargs)
                else:
                    # 如果是普通函数
                    result = step.processor(data, **step.kwargs)
            elif hasattr(step.processor, 'process'):
                # 如果有process方法
                result = step.processor.process(data, **step.kwargs)
            else:
                raise DataPipelineError(f"Processor for step '{step.name}' is not callable and has no 'process' method")
            
            # 验证结果
            if not isinstance(result, pd.DataFrame):
                raise DataPipelineError(f"Step '{step.name}' returned non-DataFrame result: {type(result)}")
            
            # 缓存结果
            if step.cache_enabled and self.cache_manager:
                self.cache_manager.set(data, result, step.name, step.kwargs)
            
            return result
            
        except Exception as e:
            error_msg = f"Step '{step.name}' failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Step '{step.name}' traceback: {traceback.format_exc()}")
            raise DataPipelineError(error_msg) from e
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """执行流水线
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        if not isinstance(data, pd.DataFrame):
            raise DataPipelineError(f"Input data must be DataFrame, got {type(data)}")
        
        if data.empty:
            logger.warning("Input data is empty")
            return data
        
        # 记录执行开始
        execution_start = datetime.now()
        enabled_steps = [step for step in self.steps if step.enabled]
        
        logger.info(f"Executing pipeline '{self.name}' with {len(enabled_steps)} steps on data shape {data.shape}")
        
        # 初始化进度条
        progress_bar = None
        if self.progress_enabled and enabled_steps:
            progress_bar = tqdm(
                total=len(enabled_steps),
                desc=f"Pipeline '{self.name}'",
                unit="step"
            )
        
        current_data = data.copy()
        step_results = {}
        
        try:
            for i, step in enumerate(enabled_steps):
                step_start = datetime.now()
                
                # 更新进度条描述
                if progress_bar:
                    progress_bar.set_description(f"Pipeline '{self.name}': {step.name}")
                
                # 执行步骤
                try:
                    current_data = self._execute_step(step, current_data)
                    step_duration = (datetime.now() - step_start).total_seconds()
                    
                    step_results[step.name] = {
                        'success': True,
                        'duration': step_duration,
                        'input_shape': data.shape if i == 0 else step_results[enabled_steps[i-1].name]['output_shape'],
                        'output_shape': current_data.shape,
                        'description': step.description
                    }
                    
                    logger.debug(f"Step '{step.name}' completed in {step_duration:.2f}s, output shape: {current_data.shape}")
                    
                except Exception as e:
                    step_results[step.name] = {
                        'success': False,
                        'error': str(e),
                        'duration': (datetime.now() - step_start).total_seconds()
                    }
                    raise
                
                # 更新进度条
                if progress_bar:
                    progress_bar.update(1)
        
        finally:
            if progress_bar:
                progress_bar.close()
        
        # 记录执行历史
        execution_duration = (datetime.now() - execution_start).total_seconds()
        execution_record = {
            'timestamp': execution_start.isoformat(),
            'duration': execution_duration,
            'input_shape': data.shape,
            'output_shape': current_data.shape,
            'steps_executed': len(enabled_steps),
            'steps_results': step_results,
            'success': True
        }
        
        self.execution_history.append(execution_record)
        
        # 保持历史记录长度
        if len(self.execution_history) > 10:
            self.execution_history = self.execution_history[-10:]
        
        logger.info(f"Pipeline '{self.name}' completed in {execution_duration:.2f}s, output shape: {current_data.shape}")
        
        return current_data
    
    def rollback(self, checkpoint_data: pd.DataFrame, step_name: str) -> pd.DataFrame:
        """回滚到指定步骤之前的状态
        
        Args:
            checkpoint_data: 检查点数据
            step_name: 要回滚到的步骤名称
            
        Returns:
            pd.DataFrame: 回滚后的数据
        """
        logger.info(f"Rolling back to before step '{step_name}'")
        
        # 找到目标步骤的索引
        step_index = None
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                step_index = i
                break
        
        if step_index is None:
            raise DataPipelineError(f"Step '{step_name}' not found for rollback")
        
        # 执行回滚前的步骤
        current_data = checkpoint_data.copy()
        for step in self.steps[:step_index]:
            if step.enabled:
                current_data = self._execute_step(step, current_data)
        
        return current_data
    
    def clear_cache(self) -> None:
        """清除缓存"""
        if self.cache_manager:
            self.cache_manager.clear()
            logger.info(f"Cleared cache for pipeline '{self.name}'")
        else:
            logger.warning(f"Cache not enabled for pipeline '{self.name}'")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计信息
        
        Returns:
            Dict: 执行统计信息
        """
        if not self.execution_history:
            return {'total_executions': 0}
        
        durations = [record['duration'] for record in self.execution_history]
        
        stats = {
            'total_executions': len(self.execution_history),
            'avg_duration': np.mean(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'last_execution': self.execution_history[-1]['timestamp'],
            'cache_info': self.cache_manager.get_cache_info() if self.cache_manager else None
        }
        
        return stats
    
    def save_pipeline(self, file_path: Union[str, Path]) -> None:
        """保存流水线配置
        
        Args:
            file_path: 保存路径
        """
        config = {
            'name': self.name,
            'cache_enabled': self.cache_enabled,
            'progress_enabled': self.progress_enabled,
            'steps': [
                {
                    'name': step.name,
                    'kwargs': step.kwargs,
                    'enabled': step.enabled,
                    'cache_enabled': step.cache_enabled,
                    'description': step.description,
                    # 注意：processor不会被序列化，需要在加载时重新设置
                }
                for step in self.steps
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved pipeline configuration to {file_path}")
    
    def load_pipeline(self, file_path: Union[str, Path]) -> None:
        """加载流水线配置
        
        Args:
            file_path: 配置文件路径
            
        Note:
            此方法只加载配置，不加载processor实例，需要手动重新添加处理器
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.name = config.get('name', self.name)
        self.cache_enabled = config.get('cache_enabled', self.cache_enabled)
        self.progress_enabled = config.get('progress_enabled', self.progress_enabled)
        
        # 清空现有步骤
        self.steps.clear()
        
        logger.info(f"Loaded pipeline configuration from {file_path}")
        logger.warning("Processors need to be re-added manually after loading configuration")


# 工厂函数和预定义流水线

def create_dfm_data_pipeline(cache_dir: Optional[Union[str, Path]] = None) -> DataPipeline:
    """创建DFM数据处理流水线
    
    Args:
        cache_dir: 缓存目录
        
    Returns:
        DataPipeline: 配置好的数据流水线
    """
    pipeline = DataPipeline(
        name="DFM_DataPipeline",
        cache_enabled=True,
        cache_dir=cache_dir,
        progress_enabled=True
    )
    
    # 注意：具体的处理器需要在使用时添加
    logger.info("Created DFM data pipeline template")
    
    return pipeline