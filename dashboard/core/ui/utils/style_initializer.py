# -*- coding: utf-8 -*-
"""
UI样式初始化器
负责UI模块的样式初始化和管理
从dashboard.core.app_initializer迁移而来
"""

import streamlit as st
import time
import logging
from typing import Dict, Any

from dashboard.core.ui.utils.style_loader import inject_cached_styles
from dashboard.core.ui.components.registry import get_component_registry
from dashboard.core.ui.utils.style_manager import get_style_manager

logger = logging.getLogger(__name__)

class UIInitializer:
    """UI初始化器"""
    
    def __init__(self):
        self.initialization_time = 0
        self.initialized_components = set()
    
    def load_styles(self):
        """加载应用样式"""
        start_time = time.time()
        
        try:
            inject_cached_styles()
            logger.info("UI样式加载成功")
        except Exception as e:
            logger.warning(f"Failed to load styles: {e}")
            # 回退到基本样式
            self._load_default_styles()
        
        setup_time = time.time() - start_time
        logger.debug(f"Styles loaded in {setup_time:.3f}s")
        self.initialized_components.add('styles')
    
    def _handle_style_loading_failure(self, error: Exception):
        """处理样式加载失败"""
        logger.error(f"样式加载失败: {error}")
        # 不再加载fallback样式，而是记录错误并继续
        # 应用程序应该能够在没有自定义样式的情况下正常运行
    
    def initialize_ui_components(self):
        """初始化UI组件"""
        start_time = time.time()

        # 初始化组件注册表
        registry = get_component_registry()
        stats = registry.get_registry_stats()
        logger.info(f"UI组件注册表初始化完成，共注册 {stats['total_components']} 个组件")

        setup_time = time.time() - start_time
        logger.debug(f"UI components initialized in {setup_time:.3f}s")
        self.initialized_components.add('components')
    
    def initialize_style_manager(self):
        """初始化样式管理器"""
        start_time = time.time()

        style_manager = get_style_manager()
        logger.info("样式管理器初始化完成")

        setup_time = time.time() - start_time
        logger.debug(f"Style manager initialized in {setup_time:.3f}s")
        self.initialized_components.add('style_manager')
    
    def full_initialize(self) -> Dict[str, Any]:
        """完整初始化UI模块"""
        total_start_time = time.time()
        
        # 按顺序执行UI初始化步骤
        initialization_steps = [
            ('styles', self.load_styles),
            ('style_manager', self.initialize_style_manager),
            ('components', self.initialize_ui_components)
        ]
        
        step_times = {}
        
        for step_name, step_func in initialization_steps:
            if step_name not in self.initialized_components:
                step_start = time.time()
                try:
                    step_func()
                    step_times[step_name] = time.time() - step_start
                except Exception as e:
                    logger.error(f"Failed to initialize UI {step_name}: {e}")
                    step_times[step_name] = -1
        
        self.initialization_time = time.time() - total_start_time
        
        return {
            'total_time': self.initialization_time,
            'step_times': step_times,
            'initialized_components': list(self.initialized_components)
        }
    
    def get_initialization_stats(self) -> Dict[str, Any]:
        """获取初始化统计信息"""
        return {
            'initialization_time': self.initialization_time,
            'initialized_components': list(self.initialized_components),
            'component_count': len(self.initialized_components)
        }
    
    def is_complete(self) -> bool:
        """检查UI初始化是否完成"""
        required_components = {'styles', 'style_manager', 'components'}
        return required_components.issubset(self.initialized_components)

# 全局UI初始化器实例
_ui_initializer = None

def get_ui_initializer() -> UIInitializer:
    """获取全局UI初始化器实例"""
    global _ui_initializer
    if _ui_initializer is None:
        _ui_initializer = UIInitializer()
    return _ui_initializer

def initialize_ui() -> Dict[str, Any]:
    """UI初始化函数"""
    initializer = get_ui_initializer()
    return initializer.full_initialize()

# 便捷函数
def load_ui_styles():
    """加载UI样式的便捷函数"""
    initializer = get_ui_initializer()
    initializer.load_styles()

def is_ui_initialized() -> bool:
    """检查UI是否已初始化"""
    initializer = get_ui_initializer()
    return initializer.is_complete()

logger.info("[UIInitializer] UI初始化器模块加载完成")
