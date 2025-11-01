# -*- coding: utf-8 -*-
"""
DFM UI组件基类

提供DFM组件的基础接口和通用功能
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import sys
import os
import logging

# 导入基础UI组件
from dashboard.ui.components.base import UIComponent
from dashboard.core import get_global_dfm_manager

logger = logging.getLogger(__name__)


class DFMServiceManager:
    """DFM服务管理器，负责与后端服务的交互"""

    def __init__(self):
        self._services = {}

    def get_data_prep_service(self):
        """获取数据预处理服务"""
        if 'data_prep' not in self._services:
            # 这里将来会连接到实际的后端服务
            self._services['data_prep'] = MockDataPrepService()
        return self._services['data_prep']
    
    def get_training_service(self):
        """获取模型训练服务"""
        if 'training' not in self._services:
            self._services['training'] = MockTrainingService()
        return self._services['training']
    
    def get_analysis_service(self):
        """获取模型分析服务"""
        if 'analysis' not in self._services:
            self._services['analysis'] = MockAnalysisService()
        return self._services['analysis']


class MockDataPrepService:
    """数据预处理服务的模拟实现"""
    
    def upload_file(self, file_data: bytes) -> Dict:
        return {'status': 'success', 'message': 'File uploaded successfully'}
    
    def configure_parameters(self, params: Dict) -> Dict:
        return {'status': 'success', 'message': 'Parameters configured'}
    
    def process_data(self, config: Dict) -> Dict:
        return {'status': 'success', 'message': 'Data processed'}
    
    def get_processing_status(self) -> Dict:
        return {'status': 'idle', 'progress': 0}


class MockTrainingService:
    """模型训练服务的模拟实现"""
    
    def configure_training(self, config: Dict) -> Dict:
        return {'status': 'success', 'message': 'Training configured'}
    
    def start_training(self, config: Dict) -> Dict:
        return {'status': 'success', 'message': 'Training started'}
    
    def get_training_status(self) -> Dict:
        return {'status': 'idle', 'progress': 0}
    
    def get_training_results(self) -> Dict:
        return {'status': 'no_results', 'results': None}


class MockAnalysisService:
    """模型分析服务的模拟实现"""
    
    def load_model_results(self, model_path: str) -> Dict:
        return {'status': 'success', 'message': 'Model loaded'}
    
    def generate_charts(self, chart_type: str, params: Dict) -> Dict:
        return {'status': 'success', 'chart_data': {}}
    
    def export_results(self, format: str) -> bytes:
        return b'mock_export_data'


class DFMComponent(UIComponent):
    """DFM组件基类"""
    
    def __init__(self, service_manager: Optional[DFMServiceManager] = None):
        super().__init__()
        self.service_manager = service_manager or DFMServiceManager()
        self._state_key_prefix = f"dfm_{self.get_component_id().lower()}"
    
    @abstractmethod
    def validate_input(self, data: Dict) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据字典
            
        Returns:
            bool: 验证是否通过
        """
        pass
    
    @abstractmethod
    def handle_service_error(self, error: Exception) -> None:
        """
        处理服务错误
        
        Args:
            error: 异常对象
        """
        pass
    
    def get_state_key_prefix(self) -> str:
        """
        获取状态键前缀
        
        Returns:
            str: 状态键前缀
        """
        return self._state_key_prefix
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        获取组件状态

        Args:
            key: 状态键
            default: 默认值

        Returns:
            Any: 状态值
        """
        try:
            # 直接使用st.session_state
            full_key = f"{self._module_name}.{key}"
            return st.session_state.get(full_key, default)
        except Exception as e:
            logger.error(f"获取状态失败: {e}")
            return default

    def set_state(self, key: str, value: Any) -> None:
        """
        设置组件状态

        Args:
            key: 状态键
            value: 状态值
        """
        try:
            # 直接使用st.session_state
            full_key = f"{self._module_name}.{key}"
            st.session_state[full_key] = value
        except Exception as e:
            logger.error(f"设置状态失败: {e}")
            raise RuntimeError(f"状态设置失败: {key} - {str(e)}")
    
    def clear_state(self, key: Optional[str] = None) -> None:
        """
        清除组件状态

        Args:
            key: 状态键，如果为None则清除所有相关状态
        """
        try:
            # 直接使用st.session_state
            if key is None:
                # 清除所有相关状态
                prefix = f"{self._module_name}."
                keys_to_delete = [k for k in st.session_state.keys() if k.startswith(prefix)]
                for k in keys_to_delete:
                    del st.session_state[k]
                logger.info(f"清除{self._module_name}模块所有状态")
            else:
                # 清除特定状态
                full_key = f"{self._module_name}.{key}"
                if full_key in st.session_state:
                    del st.session_state[full_key]
        except Exception as e:
            logger.error(f"清除状态失败: {e}")
            raise RuntimeError(f"状态清除失败: {key} - {str(e)}")
    
    def call_service_safely(self, service_func: Callable, *args, **kwargs) -> Optional[Dict]:
        """
        安全地调用服务函数
        
        Args:
            service_func: 服务函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Optional[Dict]: 服务返回结果，出错时返回None
        """
        try:
            return service_func(*args, **kwargs)
        except Exception as e:
            self.handle_service_error(e)
            return None
    
    def render_error_message(self, st_obj, message: str, details: str = "") -> None:
        """
        渲染错误消息
        
        Args:
            st_obj: Streamlit对象
            message: 错误消息
            details: 错误详情
        """
        st_obj.error(f"[ERROR] {message}")
        if details:
            with st_obj.expander("错误详情"):
                st_obj.text(details)
    
    def render_success_message(self, st_obj, message: str) -> None:
        """
        渲染成功消息
        
        Args:
            st_obj: Streamlit对象
            message: 成功消息
        """
        st_obj.success(f"[SUCCESS] {message}")
    
    def render_warning_message(self, st_obj, message: str) -> None:
        """
        渲染警告消息
        
        Args:
            st_obj: Streamlit对象
            message: 警告消息
        """
        st_obj.warning(f"[WARNING] {message}")
    
    def render_info_message(self, st_obj, message: str) -> None:
        """
        渲染信息消息
        
        Args:
            st_obj: Streamlit对象
            message: 信息消息
        """
        st_obj.info(f"[INFO] {message}")


__all__ = [
    'DFMComponent',
    'DFMServiceManager',
    'MockDataPrepService',
    'MockTrainingService', 
    'MockAnalysisService'
]
