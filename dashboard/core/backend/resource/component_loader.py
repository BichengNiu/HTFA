# -*- coding: utf-8 -*-
"""
组件懒加载器 - 第二阶段性能优化
实现按需加载UI组件，减少初始加载时间
"""

import streamlit as st
import time
import threading
from typing import Dict, Any, Callable, Optional, Set
from functools import wraps
import importlib
import sys
import os

class ComponentLoader:
    """智能组件懒加载器"""

    def __init__(self):
        self.loaded_components: Dict[str, Any] = {}
        self.loading_status: Dict[str, str] = {}
        self.load_times: Dict[str, float] = {}
        self._lock = threading.RLock()

        # 使用UI模块的组件注册表（UI移动后将更新路径）
        # TODO: 更新为 dashboard.core.ui.components.registry
        from dashboard.core.ui.components.registry import get_component_registry
        self.component_registry = get_component_registry()

    def load_component(self, component_name: str, force_reload: bool = False) -> Optional[Any]:
        """
        懒加载指定组件

        Args:
            component_name: 组件名称
            force_reload: 是否强制重新加载

        Returns:
            加载的组件模块，失败时返回None
        """
        # 检查是否已加载
        if not force_reload and component_name in self.loaded_components:
            return self.loaded_components[component_name]

        # 检查是否正在加载
        if self.loading_status.get(component_name) == 'loading':
            return None

        # 开始加载
        self.loading_status[component_name] = 'loading'
        start_time = time.time()

        # 获取模块路径
        if self.component_registry:
            module_path = self.component_registry.get_component_path(component_name)
        else:
            raise RuntimeError(f"组件注册表不可用，无法加载组件: {component_name}")

        if not module_path:
            raise ImportError(f"未知的组件: {component_name}")

        # 检查模块是否已在sys.modules中
        if module_path in sys.modules:
            module = sys.modules[module_path]
        else:
            # 动态导入模块
            module = importlib.import_module(module_path)

        # 缓存组件
        self.loaded_components[component_name] = module
        self.loading_status[component_name] = 'loaded'

        # 记录加载时间
        load_time = (time.time() - start_time) * 1000
        self.load_times[component_name] = load_time

        return module

    def _load_dependencies(self, component_name: str):
        """加载组件依赖"""
        if self.component_registry:
            dependencies = self.component_registry.get_component_dependencies(component_name)
        else:
            raise RuntimeError(f"组件注册表不可用，无法获取依赖关系: {component_name}")

        for dep in dependencies:
            if dep not in self.loaded_components and self.loading_status.get(dep) != 'loading':
                self.load_component(dep)

    def preload_critical_components(self):
        """预加载关键组件"""
        if self.component_registry:
            critical_components = list(self.component_registry.get_critical_components())
        else:
            critical_components = []

        for component in critical_components:
            self.load_component(component)

    def get_loading_stats(self) -> Dict[str, Any]:
        """获取加载统计信息"""
        if self.component_registry:
            total_components = self.component_registry.get_registry_stats()['total_components']
        else:
            total_components = 0

        return {
            'loaded_count': len(self.loaded_components),
            'total_components': total_components,
            'load_times': self.load_times.copy(),
            'loading_status': self.loading_status.copy(),
            'average_load_time': sum(self.load_times.values()) / len(self.load_times) if self.load_times else 0
        }

    def is_component_loaded(self, component_name: str) -> bool:
        """检查组件是否已加载"""
        return component_name in self.loaded_components

    def get_component(self, component_name: str) -> Optional[Any]:
        """获取已加载的组件（不触发加载）"""
        return self.loaded_components.get(component_name)

# 全局组件加载器实例
@st.cache_resource
def get_component_loader():
    """获取全局组件加载器实例"""
    return ComponentLoader()

def lazy_load_component(component_name: str):
    """装饰器：懒加载组件"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            loader = get_component_loader()
            component = loader.load_component(component_name)
            if component is None:
                st.error(f"组件 {component_name} 加载失败")
                return None
            return func(component, *args, **kwargs)
        return wrapper
    return decorator

def preload_components_async():
    """异步预加载组件"""
    def _preload():
        loader = get_component_loader()
        loader.preload_critical_components()

    # 在后台线程中预加载
    thread = threading.Thread(target=_preload, daemon=True)
    thread.start()
    return thread
