# -*- coding: utf-8 -*-
"""
Dashboard核心模块

提供应用初始化、资源加载、状态管理、导航管理等核心功能
"""

# 导航管理
from dashboard.core.navigation_manager import (
    NavigationManager,
    get_navigation_manager,
    reset_navigation_manager
)

# 基础设施（装饰器和基类）
from dashboard.core.base import (
    safe_operation,
    timed_operation,
    thread_safe,
    validate_required_attributes
)

# 资源加载器（统一的懒加载）
from dashboard.core.resource_loader import (
    ResourceLoader,
    get_resource_loader
)

# 独立的加载器模块
from dashboard.core.lazy_loader import get_lazy_loader
from dashboard.core.component_loader import get_component_loader

# 配置管理
from dashboard.core.config import CoreConfig, get_core_config

# 配置缓存
from dashboard.core.config_cache import ConfigCache, get_config_cache

# 应用初始化器
from dashboard.core.app_initializer import (
    AppInitializer,
    EnvironmentInitializer,
    StreamlitInitializer,
    CacheCleanupManager,
    get_app_initializer,
    initialize_app
)

# DFM兼容层
from dashboard.core.dfm_compat import (
    DFMStateManager,
    get_global_dfm_manager
)

__version__ = "4.0.0"
__all__ = [
    # 导航管理
    'NavigationManager',
    'get_navigation_manager',
    'reset_navigation_manager',

    # 基础设施
    'safe_operation',
    'timed_operation',
    'thread_safe',
    'validate_required_attributes',

    # 资源加载
    'ResourceLoader',
    'get_resource_loader',
    'get_lazy_loader',
    'get_component_loader',

    # 配置管理
    'CoreConfig',
    'get_core_config',

    # 配置缓存
    'ConfigCache',
    'get_config_cache',

    # 应用初始化
    'AppInitializer',
    'EnvironmentInitializer',
    'StreamlitInitializer',
    'CacheCleanupManager',
    'get_app_initializer',
    'initialize_app',

    # DFM兼容层
    'DFMStateManager',
    'get_global_dfm_manager',
]
