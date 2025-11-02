# -*- coding: utf-8 -*-
"""
UI异步处理器
提供异步处理能力，避免UI阻塞，提升用户体验
"""

import asyncio
import threading
import time
from typing import Callable, Any, Optional, Dict, List
# 移除并行处理：from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """处理任务数据类"""
    task_id: str
    description: str
    progress: float = 0.0
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class UIAsyncProcessor:
    """
    UI异步处理器
    提供异步处理能力，避免UI阻塞
    """
    
    def __init__(self, max_workers: int = 4):
        """
        初始化异步处理器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        # 移除并行处理：self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.executor = None  # 已改为同步模式，不再使用线程池
        self.tasks = {}  # 任务存储
        self.task_counter = 0
        
        # 进度回调存储
        self.progress_callbacks = {}
        
        logger.info(f"UI处理器初始化完成，已改为同步模式（原 max_workers: {max_workers}）")
    
    def submit_task(self, func: Callable, *args, 
                   description: str = "处理中...", 
                   progress_callback: Optional[Callable] = None,
                   **kwargs) -> str:
        """
        提交异步任务
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            description: 任务描述
            progress_callback: 进度回调函数
            **kwargs: 函数关键字参数
            
        Returns:
            str: 任务ID
        """
        try:
            # 生成任务ID
            self.task_counter += 1
            task_id = f"task_{self.task_counter}_{int(time.time())}"
            
            # 创建任务对象
            task = ProcessingTask(
                task_id=task_id,
                description=description,
                status="pending"
            )
            
            self.tasks[task_id] = task
            
            # 存储进度回调
            if progress_callback:
                self.progress_callbacks[task_id] = progress_callback
            
            # 包装函数以支持进度更新
            def wrapped_func():
                try:
                    task.status = "running"
                    task.start_time = time.time()
                    
                    # 创建进度更新函数
                    def update_progress(progress: float, message: str = ""):
                        task.progress = min(100.0, max(0.0, progress))
                        if message:
                            task.description = message
                        
                        # 调用进度回调
                        if task_id in self.progress_callbacks:
                            try:
                                self.progress_callbacks[task_id](task.progress, task.description)
                            except Exception as e:
                                logger.warning(f"进度回调执行失败: {e}")
                    
                    # 如果函数支持进度回调，传入update_progress
                    import inspect
                    sig = inspect.signature(func)
                    if 'progress_callback' in sig.parameters:
                        kwargs['progress_callback'] = update_progress
                    
                    # 执行函数
                    result = func(*args, **kwargs)
                    
                    # 任务完成
                    task.status = "completed"
                    task.progress = 100.0
                    task.result = result
                    task.end_time = time.time()
                    
                    return result
                    
                except Exception as e:
                    task.status = "failed"
                    task.error = e
                    task.end_time = time.time()
                    logger.error(f"异步任务执行失败: {task_id}, 错误: {e}")
                    raise
            
            # 改为同步直接执行
            logger.info(f"同步任务开始执行: {task_id}, 描述: {description}")
            wrapped_func()  # 直接同步执行
            logger.info(f"同步任务执行完成: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"提交异步任务失败: {e}")
            raise
    
    def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[ProcessingTask]: 任务对象
        """
        return self.tasks.get(task_id)
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        等待任务完成
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
            
        Returns:
            Any: 任务结果
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"任务不存在: {task_id}")
        
        start_time = time.time()
        while task.status in ["pending", "running"]:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"任务超时: {task_id}")
            
            time.sleep(0.1)
        
        if task.status == "failed":
            raise task.error
        
        return task.result
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否成功取消
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status in ["pending", "running"]:
            task.status = "cancelled"
            # 清理进度回调
            if task_id in self.progress_callbacks:
                del self.progress_callbacks[task_id]
            
            logger.info(f"任务已取消: {task_id}")
            return True
        
        return False
    
    def get_active_tasks(self) -> List[ProcessingTask]:
        """
        获取活跃任务列表
        
        Returns:
            List[ProcessingTask]: 活跃任务列表
        """
        return [
            task for task in self.tasks.values()
            if task.status in ["pending", "running"]
        ]
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """
        清理已完成的任务
        
        Args:
            max_age_hours: 最大保留时间（小时）
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - (max_age_hours * 3600)
            
            tasks_to_remove = []
            for task_id, task in self.tasks.items():
                if (task.status in ["completed", "failed", "cancelled"] and 
                    task.end_time and task.end_time < cutoff_time):
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
                if task_id in self.progress_callbacks:
                    del self.progress_callbacks[task_id]
            
            if tasks_to_remove:
                logger.info(f"清理了 {len(tasks_to_remove)} 个已完成的任务")
                
        except Exception as e:
            logger.error(f"清理已完成任务失败: {e}")
    
    def shutdown(self, wait: bool = True):
        """
        关闭异步处理器
        
        Args:
            wait: 是否等待所有任务完成
        """
        try:
            # 同步模式无需关闭线程池
            logger.info("同步处理器已关闭（无需实际关闭操作）")
        except Exception as e:
            logger.error(f"关闭处理器失败: {e}")


class StreamlitAsyncHelper:
    """
    Streamlit异步处理助手
    专门为Streamlit应用提供异步处理支持
    """
    
    def __init__(self, processor: UIAsyncProcessor):
        """
        初始化Streamlit异步助手
        
        Args:
            processor: UI异步处理器
        """
        self.processor = processor
    
    def process_with_progress_bar(self, st_obj, func: Callable, *args,
                                 description: str = "处理中...",
                                 **kwargs) -> Any:
        """
        带进度条的异步处理
        
        Args:
            st_obj: Streamlit对象
            func: 要执行的函数
            *args: 函数参数
            description: 任务描述
            **kwargs: 函数关键字参数
            
        Returns:
            Any: 处理结果
        """
        try:
            # 创建进度条
            progress_bar = st_obj.progress(0)
            status_text = st_obj.empty()
            
            # 进度回调函数
            def progress_callback(progress: float, message: str):
                progress_bar.progress(int(progress))
                status_text.text(f"{message} ({progress:.1f}%)")
            
            # 提交任务
            task_id = self.processor.submit_task(
                func, *args,
                description=description,
                progress_callback=progress_callback,
                **kwargs
            )
            
            # 等待任务完成
            try:
                result = self.processor.wait_for_task(task_id)
                
                # 完成状态
                progress_bar.progress(100)
                status_text.text("处理完成！")
                
                # 短暂显示完成状态后清理
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                return result
                
            except Exception as e:
                # 错误状态
                progress_bar.empty()
                status_text.empty()
                st_obj.error(f"处理失败: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Streamlit异步处理失败: {e}")
            raise
    
    def process_file_async(self, st_obj, file_processor: Callable, 
                          file_data: Any, description: str = "处理文件中...") -> Any:
        """
        异步文件处理
        
        Args:
            st_obj: Streamlit对象
            file_processor: 文件处理函数
            file_data: 文件数据
            description: 处理描述
            
        Returns:
            Any: 处理结果
        """
        return self.process_with_progress_bar(
            st_obj, file_processor, file_data,
            description=description
        )
    
    def process_data_async(self, st_obj, data_processor: Callable,
                          data: Any, description: str = "处理数据中...") -> Any:
        """
        异步数据处理
        
        Args:
            st_obj: Streamlit对象
            data_processor: 数据处理函数
            data: 数据
            description: 处理描述
            
        Returns:
            Any: 处理结果
        """
        return self.process_with_progress_bar(
            st_obj, data_processor, data,
            description=description
        )


# 全局异步处理器实例
_ui_async_processor = None
_streamlit_async_helper = None


def get_ui_async_processor() -> UIAsyncProcessor:
    """
    获取UI异步处理器实例（单例模式）
    
    Returns:
        UIAsyncProcessor: UI异步处理器实例
    """
    global _ui_async_processor
    if _ui_async_processor is None:
        _ui_async_processor = UIAsyncProcessor()
    return _ui_async_processor


def get_streamlit_async_helper() -> StreamlitAsyncHelper:
    """
    获取Streamlit异步助手实例（单例模式）
    
    Returns:
        StreamlitAsyncHelper: Streamlit异步助手实例
    """
    global _streamlit_async_helper
    if _streamlit_async_helper is None:
        processor = get_ui_async_processor()
        _streamlit_async_helper = StreamlitAsyncHelper(processor)
    return _streamlit_async_helper


def reset_async_processors():
    """重置异步处理器实例"""
    global _ui_async_processor, _streamlit_async_helper
    
    if _ui_async_processor:
        _ui_async_processor.shutdown()
    
    _ui_async_processor = None
    _streamlit_async_helper = None
