# -*- coding: utf-8 -*-
"""
UI通用辅助函数
消除UI模块中的重复代码，提供统一的工具函数
"""

import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import time
from datetime import datetime
from functools import wraps
import importlib
import math
from collections import Counter

logger = logging.getLogger(__name__)


class UICommonHelpers:
    """UI通用辅助函数类"""
    
    @staticmethod
    def safe_import(module_path: str, class_name: str = None, default=None):
        """
        安全导入模块或类

        Args:
            module_path: 模块路径
            class_name: 类名（可选）
            default: 导入失败时的默认值

        Returns:
            导入的模块或类，失败时返回default
        """
        module = importlib.import_module(module_path)

        if class_name:
            return getattr(module, class_name, default)
        else:
            return module
    
    @staticmethod
    def create_component_key(component_id: str, key_suffix: str) -> str:
        """
        创建组件专用的状态键
        
        Args:
            component_id: 组件ID
            key_suffix: 键后缀
            
        Returns:
            str: 完整的状态键
        """
        return f"{component_id}_{key_suffix}"
    
    @staticmethod
    def render_loading_state(st_obj, message: str = "加载中..."):
        """
        渲染加载状态
        
        Args:
            st_obj: Streamlit对象
            message: 加载消息
        """
        with st_obj.spinner(message):
            time.sleep(0.1)  # 确保spinner显示
    
    @staticmethod
    def render_error_state(st_obj, error_message: str, suggestion: str = None):
        """
        渲染错误状态
        
        Args:
            st_obj: Streamlit对象
            error_message: 错误消息
            suggestion: 建议（可选）
        """
        st_obj.error(f"{error_message}")
        if suggestion:
            st_obj.info(f"建议: {suggestion}")
    
    @staticmethod
    def render_success_state(st_obj, success_message: str):
        """
        渲染成功状态
        
        Args:
            st_obj: Streamlit对象
            success_message: 成功消息
        """
        st_obj.success(f"{success_message}")
    
    @staticmethod
    def render_warning_state(st_obj, warning_message: str):
        """
        渲染警告状态
        
        Args:
            st_obj: Streamlit对象
            warning_message: 警告消息
        """
        st_obj.warning(f"{warning_message}")
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, min_rows: int = 1, 
                          required_columns: List[str] = None) -> Tuple[bool, str]:
        """
        验证DataFrame
        
        Args:
            df: 要验证的DataFrame
            min_rows: 最小行数
            required_columns: 必需的列名列表
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误消息)
        """
        if df is None:
            return False, "数据为空"
        
        if df.empty:
            return False, "数据表为空"
        
        if len(df) < min_rows:
            return False, f"数据行数不足，需要至少 {min_rows} 行，当前 {len(df)} 行"
        
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"缺少必需的列: {', '.join(missing_columns)}"
        
        return True, ""
    
    @staticmethod
    def handle_duplicate_columns(df: pd.DataFrame, keep: str = 'first') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        处理重复列
        
        Args:
            df: 输入DataFrame
            keep: 保留策略 ('first', 'last', False)
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (处理后的DataFrame, 处理信息)
        """
        duplicate_mask = df.columns.duplicated(keep=False)
        
        if not duplicate_mask.any():
            return df, {'has_duplicates': False, 'duplicate_count': 0}
        
        # 统计重复列
        column_counts = Counter(df.columns)
        duplicated_names = {name: count for name, count in column_counts.items() if count > 1}
        
        # 处理重复列
        if keep == False:
            # 删除所有重复列
            df_cleaned = df.loc[:, ~duplicate_mask]
        else:
            # 保留指定的重复列
            df_cleaned = df.loc[:, ~df.columns.duplicated(keep=keep)]
        
        return df_cleaned, {
            'has_duplicates': True,
            'duplicate_count': duplicate_mask.sum(),
            'duplicated_names': duplicated_names,
            'removed_columns': duplicate_mask.sum()
        }
    
    @staticmethod
    def handle_duplicate_rows(df: pd.DataFrame, subset: List[str] = None, 
                             keep: str = 'first') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        处理重复行
        
        Args:
            df: 输入DataFrame
            subset: 用于判断重复的列子集
            keep: 保留策略 ('first', 'last', False)
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (处理后的DataFrame, 处理信息)
        """
        duplicate_count = df.duplicated(subset=subset).sum()
        
        if duplicate_count == 0:
            return df, {'has_duplicates': False, 'duplicate_count': 0}
        
        # 删除重复行
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        
        return df_cleaned, {
            'has_duplicates': True,
            'duplicate_count': duplicate_count,
            'removed_rows': duplicate_count
        }
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        格式化文件大小
        
        Args:
            size_bytes: 文件大小（字节）
            
        Returns:
            str: 格式化的文件大小
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        格式化时间间隔
        
        Args:
            seconds: 秒数
            
        Returns:
            str: 格式化的时间间隔
        """
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}min"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    @staticmethod
    def create_download_button(st_obj, data: Union[pd.DataFrame, str, bytes], 
                              filename: str, label: str = "下载", 
                              mime_type: str = None) -> bool:
        """
        创建下载按钮
        
        Args:
            st_obj: Streamlit对象
            data: 要下载的数据
            filename: 文件名
            label: 按钮标签
            mime_type: MIME类型
            
        Returns:
            bool: 是否点击了下载按钮
        """
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrame转CSV，使用utf-8-sig编码避免中文乱码
                csv_string = data.to_csv(index=False, encoding='utf-8-sig')
                csv_data = csv_string.encode('utf-8-sig')
                mime_type = mime_type or 'text/csv'
                return st_obj.download_button(
                    label=label,
                    data=csv_data,
                    file_name=filename,
                    mime=mime_type
                )
            elif isinstance(data, str):
                # 字符串数据
                mime_type = mime_type or 'text/plain'
                return st_obj.download_button(
                    label=label,
                    data=data.encode('utf-8'),
                    file_name=filename,
                    mime=mime_type
                )
            elif isinstance(data, bytes):
                # 字节数据
                mime_type = mime_type or 'application/octet-stream'
                return st_obj.download_button(
                    label=label,
                    data=data,
                    file_name=filename,
                    mime=mime_type
                )
            else:
                logger.error(f"不支持的数据类型: {type(data)}")
                return False
                
        except Exception as e:
            logger.error(f"创建下载按钮失败: {e}")
            return False
    
    @staticmethod
    def render_dataframe_info(st_obj, df: pd.DataFrame, title: str = "数据信息"):
        """
        渲染DataFrame信息
        
        Args:
            st_obj: Streamlit对象
            df: DataFrame
            title: 标题
        """
        with st_obj.expander(title):
            col1, col2, col3 = st_obj.columns(3)
            
            with col1:
                st_obj.metric("行数", len(df))
            
            with col2:
                st_obj.metric("列数", len(df.columns))
            
            with col3:
                memory_usage = df.memory_usage(deep=True).sum()
                st_obj.metric("内存使用", UICommonHelpers.format_file_size(memory_usage))
            
            # 数据类型信息
            st_obj.subheader("数据类型")
            dtype_info = df.dtypes.value_counts()
            st_obj.write(dtype_info)
            
            # 缺失值信息
            missing_info = df.isnull().sum()
            if missing_info.sum() > 0:
                st_obj.subheader("缺失值")
                missing_df = pd.DataFrame({
                    '列名': missing_info.index,
                    '缺失数量': missing_info.values,
                    '缺失比例': (missing_info.values / len(df) * 100).round(2)
                })
                missing_df = missing_df[missing_df['缺失数量'] > 0]
                st_obj.dataframe(missing_df)


def timing_decorator(operation_name: str = "操作"):
    """
    计时装饰器
    
    Args:
        operation_name: 操作名称
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"{operation_name}完成，耗时: {UICommonHelpers.format_duration(duration)}")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{operation_name}失败，耗时: {UICommonHelpers.format_duration(duration)}, 错误: {e}")
                raise
        return wrapper
    return decorator


def safe_execute(func: Callable, default_return=None, error_message: str = "操作失败"):
    """
    安全执行函数
    
    Args:
        func: 要执行的函数
        default_return: 默认返回值
        error_message: 错误消息
        
    Returns:
        函数执行结果或默认值
    """
    try:
        return func()
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        return default_return


# 全局辅助函数实例
ui_helpers = UICommonHelpers()
