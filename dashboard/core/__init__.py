# -*- coding: utf-8 -*-
"""
Core基础框架模块
提供应用基础服务和UI框架，遵循垂直切分架构

架构说明：
- backend/: 后端基础服务（配置、导航、资源加载、初始化等）
- ui/: 前端UI框架（组件、工具、样式等）
"""

# ========== 后端服务导出 ==========

# 配置管理
from dashboard.core.backend.config import (
    CoreConfig,
    get_core_config,
    EnvironmentConfig,
    ResourcePathsConfig,
    NavigationConfig,
    ConfigCache,
    get_config_cache
)

# 导航管理
from dashboard.core.backend.navigation import (
    NavigationManager,
    NavigationStateKeys,
    get_navigation_manager,
    reset_navigation_manager
)

# 资源加载
from dashboard.core.backend.resource import (
    ResourceLoader,
    get_resource_loader,
    LazyModuleLoader,
    get_lazy_loader,
    get_cached_lazy_loader,
    ComponentLoader,
    get_component_loader,
    lazy_load_component,
    preload_components_async
)

# 应用初始化
from dashboard.core.backend.initialization import (
    AppInitializer,
    EnvironmentInitializer,
    StreamlitInitializer,
    CacheCleanupManager,
    get_app_initializer,
    initialize_app
)

# 工具装饰器
from dashboard.core.backend.utils import (
    safe_operation,
    timed_operation,
    thread_safe,
    validate_required_attributes,
    ThreadSafeSingleton
)

# ========== UI框架导出 ==========

# UI组件
from dashboard.core.ui.components import (
    UIComponent,
    BaseWelcomePage,
    WelcomeComponent,
    NavigationComponent,
    LayoutComponent,
    ModuleCard,
    FeatureCard,
    ComponentRegistry,
    get_component_registry
)

# 数据输入组件
from dashboard.core.ui.components.data_input import (
    DataInputComponent,
    UnifiedDataUploadComponent,
    DataUploadSidebar,
    DataValidationComponent,
    DataStagingComponent,
    DataPreviewComponent
)

# UI工具
from dashboard.core.ui.utils import (
    StyleManager,
    get_style_manager,
    StyleLoader,
    inject_cached_styles,
    load_cached_styles,
    get_style_loader,
    UIInitializer,
    initialize_ui,
    load_ui_styles,
    is_ui_initialized,
    get_ui_initializer,
    TabStateDetector
)

# UI常量
from dashboard.core.ui.constants import UIConstants, NavigationLevel

__version__ = "5.0.0"

__all__ = [
    # ===== Backend Services =====
    # Config
    'CoreConfig', 'get_core_config', 'EnvironmentConfig', 'ResourcePathsConfig',
    'NavigationConfig', 'ConfigCache', 'get_config_cache',

    # Navigation
    'NavigationManager', 'NavigationStateKeys', 'get_navigation_manager',
    'reset_navigation_manager',

    # Resource Loading
    'ResourceLoader', 'get_resource_loader', 'LazyModuleLoader', 'get_lazy_loader',
    'get_cached_lazy_loader', 'ComponentLoader', 'get_component_loader',
    'lazy_load_component', 'preload_components_async',

    # Initialization
    'AppInitializer', 'EnvironmentInitializer', 'StreamlitInitializer',
    'CacheCleanupManager', 'get_app_initializer', 'initialize_app',

    # Utils
    'safe_operation', 'timed_operation', 'thread_safe',
    'validate_required_attributes', 'ThreadSafeSingleton',

    # ===== UI Framework =====
    # Components
    'UIComponent', 'BaseWelcomePage', 'WelcomeComponent', 'NavigationComponent',
    'LayoutComponent', 'ModuleCard', 'FeatureCard', 'ComponentRegistry',
    'get_component_registry',

    # Data Input Components
    'DataInputComponent', 'UnifiedDataUploadComponent', 'DataUploadSidebar',
    'DataValidationComponent', 'DataStagingComponent', 'DataPreviewComponent',

    # UI Utils
    'StyleManager', 'get_style_manager', 'StyleLoader', 'inject_cached_styles',
    'load_cached_styles', 'get_style_loader', 'UIInitializer', 'initialize_ui',
    'load_ui_styles', 'is_ui_initialized', 'get_ui_initializer', 'TabStateDetector',

    # UI Constants
    'UIConstants', 'NavigationLevel'
]
