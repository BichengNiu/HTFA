# -*- coding: utf-8 -*-
"""
统一资源加载器 - 合并lazy_loader和component_loader
提供统一的懒加载功能，消除代码重复
"""

import streamlit as st
import time
import threading
import importlib
import sys
import logging
from typing import Dict, Any, Optional, Callable
logger = logging.getLogger(__name__)


class ResourceLoader:
    """统一资源加载器 - 整合模块和组件的懒加载功能"""

    def __init__(self):
        self.loaded_resources: Dict[str, Any] = {}
        self.loading_status: Dict[str, str] = {}
        self.load_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._loading_resources: set = set()

        self._setup_config()
        self._setup_component_registry()

    def _setup_config(self):
        """从配置系统加载配置"""
        from dashboard.core.config import get_core_config
        self.config = get_core_config()

    def _setup_component_registry(self):
        """设置组件注册表（如果存在）"""
        from dashboard.ui.components.registry import get_component_registry
        self.component_registry = get_component_registry()

    def load_resource(self, resource_name: str,
                     is_component: bool = False,
                     force_reload: bool = False) -> Optional[Any]:
        """
        加载资源（模块或组件）

        Args:
            resource_name: 资源名称
            is_component: 是否为组件（默认False，即模块）
            force_reload: 是否强制重新加载

        Returns:
            加载的资源，失败返回None
        """
        # 快速路径：已加载
        if not force_reload and resource_name in self.loaded_resources:
            return self.loaded_resources[resource_name]

        # 线程安全检查
        with self._lock:
            if not force_reload and resource_name in self.loaded_resources:
                return self.loaded_resources[resource_name]

            if resource_name in self._loading_resources:
                logger.warning(f"资源 {resource_name} 正在加载中，避免循环依赖")
                return None

            self.loading_status[resource_name] = 'loading'
            self._loading_resources.add(resource_name)

        # 开始加载
        start_time = time.time()

        try:
            # 获取模块路径
            module_path = self._get_resource_path(resource_name, is_component)

            if not module_path:
                raise ImportError(f"未知的资源: {resource_name}")

            # 动态导入
            if module_path in sys.modules and not force_reload:
                module = sys.modules[module_path]
            else:
                if force_reload and module_path in sys.modules:
                    del sys.modules[module_path]
                module = importlib.import_module(module_path)

            # 缓存结果
            with self._lock:
                self.loaded_resources[resource_name] = module
                self.loading_status[resource_name] = 'loaded'
                self.load_times[resource_name] = (time.time() - start_time) * 1000
                self._loading_resources.discard(resource_name)

            return module

        except Exception as e:
            with self._lock:
                self.loading_status[resource_name] = 'error'
                self._loading_resources.discard(resource_name)
            logger.error(f"加载资源 {resource_name} 失败: {e}")
            return None

    def _get_resource_path(self, resource_name: str,
                          is_component: bool) -> Optional[str]:
        """获取资源路径"""
        if is_component and self.component_registry:
            return self.component_registry.get_component_path(resource_name)
        return self.config.get_module_path(resource_name)

    def preload_critical_resources(self):
        """预加载关键资源"""
        critical_resources = ['data_loader', 'preview_main']

        for resource in critical_resources:
            try:
                self.load_resource(resource)
            except Exception as e:
                logger.warning(f"预加载资源 {resource} 失败: {e}")

    def get_loading_stats(self) -> Dict[str, Any]:
        """获取加载统计信息"""
        with self._lock:
            return {
                'loaded_count': len(self.loaded_resources),
                'total_configured': len(self.config.get_module_paths()),
                'load_times': self.load_times.copy(),
                'loading_status': self.loading_status.copy(),
                'average_load_time': (
                    sum(self.load_times.values()) / len(self.load_times)
                    if self.load_times else 0
                )
            }

    def is_loaded(self, resource_name: str) -> bool:
        """检查资源是否已加载"""
        return resource_name in self.loaded_resources

    def get_resource(self, resource_name: str) -> Optional[Any]:
        """获取已加载的资源（不触发加载）"""
        return self.loaded_resources.get(resource_name)

    def clear_cache(self):
        """清理资源缓存"""
        with self._lock:
            self.loaded_resources.clear()
            self.load_times.clear()
            self.loading_status.clear()


# 全局资源加载器实例
@st.cache_resource
def get_resource_loader() -> ResourceLoader:
    """获取全局资源加载器实例"""
    return ResourceLoader()


# 向后兼容的别名函数
def get_lazy_loader() -> ResourceLoader:
    """向后兼容：获取懒加载器"""
    return get_resource_loader()


def get_component_loader() -> ResourceLoader:
    """向后兼容：获取组件加载器"""
    return get_resource_loader()
