# -*- coding: utf-8 -*-
"""
核心配置管理器
提供统一的配置访问接口，消除硬编码配置
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass, field


# ==================== 配置数据类 ====================

@dataclass
class EnvironmentConfig:
    """环境配置"""
    vars: Dict[str, str] = field(default_factory=lambda: {
        'STREAMLIT_LOGGER_LEVEL': 'CRITICAL',
        'STREAMLIT_CLIENT_TOOLBAR_MODE': 'minimal',
        'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
        'STREAMLIT_CLIENT_SHOW_ERROR_DETAILS': 'false',
        'PYTHONWARNINGS': 'ignore',
        'STREAMLIT_SILENT_IMPORTS': 'true',
        'STREAMLIT_SUPPRESS_WARNINGS': 'true'
    })


@dataclass
class ResourcePathsConfig:
    """资源路径配置"""
    module_paths: Dict[str, str] = field(default_factory=lambda: {
        # 预览模块
        'data_loader': 'dashboard.preview.data_loader',
        'preview_main': 'dashboard.preview.main',  # 重构后的主模块

        # DFM模块
        'dfm_ui': 'dashboard.DFM.model_analysis.dfm_ui',
        'dfm_data_prep': 'dashboard.DFM.data_prep.data_prep_ui',
        'dfm_train_model': 'dashboard.DFM.train_model.train_model_ui',
        'news_analysis': 'dashboard.DFM.news_analysis.news_analysis_front_end',

        # 工具模块
        'stationarity_analysis': 'dashboard.ui.components.analysis.timeseries.stationarity',
        'dtw_analysis': 'dashboard.ui.components.analysis.timeseries.dtw',
        'correlation_analysis': 'dashboard.ui.components.analysis.timeseries.correlation',
        'lead_lag_analysis': 'dashboard.ui.components.analysis.timeseries.lead_lag',

        # UI组件
        'data_upload_sidebar': 'dashboard.ui.components.sidebar',
        'parameter_sidebar': 'dashboard.ui.components.sidebar'
    })


@dataclass
class NavigationConfig:
    """导航配置"""
    cache_keys: List[str] = field(default_factory=lambda: [
        'ui.button_state_cache',
        'ui.navigation_cache',
        'ui.module_selector_cache'
    ])


# ==================== 配置管理器 ====================

class CoreConfig:
    """
    核心配置管理器

    统一管理所有核心模块的配置，消除硬编码
    """

    def __init__(self):
        self.environment = EnvironmentConfig()
        self.resource_paths = ResourcePathsConfig()
        self.navigation = NavigationConfig()

    def get_env_vars(self) -> Dict[str, str]:
        """获取环境变量配置"""
        return self.environment.vars.copy()

    def get_module_paths(self) -> Dict[str, str]:
        """获取模块路径配置"""
        return self.resource_paths.module_paths.copy()

    def get_module_path(self, module_name: str) -> str:
        """
        获取指定模块的路径

        Args:
            module_name: 模块名称

        Returns:
            模块路径，如果不存在则返回None
        """
        return self.resource_paths.module_paths.get(module_name)

    def add_module_path(self, module_name: str, module_path: str):
        """
        添加模块路径（支持运行时扩展）

        Args:
            module_name: 模块名称
            module_path: 模块路径
        """
        self.resource_paths.module_paths[module_name] = module_path

    def get_cache_keys(self) -> List[str]:
        """获取缓存键列表"""
        return self.navigation.cache_keys.copy()

    def add_cache_key(self, cache_key: str):
        """
        添加缓存键

        Args:
            cache_key: 缓存键
        """
        if cache_key not in self.navigation.cache_keys:
            self.navigation.cache_keys.append(cache_key)


# ==================== 全局配置实例 ====================

_core_config = None


def get_core_config() -> CoreConfig:
    """
    获取核心配置实例（单例模式）

    Returns:
        CoreConfig: 核心配置实例
    """
    global _core_config
    if _core_config is None:
        _core_config = CoreConfig()
    return _core_config
