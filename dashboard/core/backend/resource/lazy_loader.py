# -*- coding: utf-8 -*-
"""
懒加载模块管理器
优化模块导入性能，减少启动时间
"""

import importlib
import sys
from typing import Dict, Any, Optional, Callable
import time
import logging

logger = logging.getLogger(__name__)

class LazyModuleLoader:
    """懒加载模块管理器"""

    def __init__(self):
        self.loaded_modules: Dict[str, Any] = {}
        self.module_configs: Dict[str, Dict] = {}
        self.load_times: Dict[str, float] = {}
        self._setup_module_configs()

    def _setup_module_configs(self):
        """设置模块配置"""
        self.module_configs = {
            # 预览模块
            'data_loader': {
                'module_path': 'dashboard.preview.data_loader',
                'functions': ['load_and_process_data']
            },
            'growth_calculator': {
                'module_path': 'dashboard.preview.growth_calculator',
                'functions': ['calculate_monthly_growth_summary']
            },
            'preview_main': {
                'module_path': 'dashboard.preview.main',
                'functions': ['display_industrial_tabs']
            },
            'weekly_data_tab': {
                'module_path': 'dashboard.preview.weekly_data_tab',
                'functions': ['display_weekly_tab']
            },
            'monthly_data_tab': {
                'module_path': 'dashboard.preview.monthly_data_tab',
                'functions': ['display_monthly_tab']
            },
            'daily_data_tab': {
                'module_path': 'dashboard.preview.daily_data_tab',
                'functions': ['display_daily_tab']
            },

            # DFM模块
            'dfm_ui': {
                'module_path': 'dashboard.models.DFM.results.dfm_ui',
                'functions': ['render_dfm_tab']
            },
            'dfm_data_prep': {
                'module_path': 'dashboard.models.DFM.prep.data_prep_ui',
                'functions': ['render_dfm_data_prep_tab']
            },
            'dfm_train_model': {
                'module_path': 'dashboard.DFM.train_model.train_model_ui',
                'functions': ['render_dfm_train_model_tab']
            },
            'news_analysis': {
                'module_path': 'dashboard.models.DFM.decomp.news_analysis_front_end',
                'functions': ['render_news_analysis_tab']
            },

            # 工具模块（已迁移到explore/ui目录）
            'stationarity_analysis': {
                'module_path': 'dashboard.explore.ui.stationarity',
                'functions': ['StationarityAnalysisComponent']
            },
            'dtw_analysis': {
                'module_path': 'dashboard.explore.ui.dtw',
                'functions': ['DTWAnalysisComponent']
            },
            'correlation_analysis': {
                'module_path': 'dashboard.explore.ui.correlation',
                'functions': ['CorrelationAnalysisComponent']
            },
            'lead_lag_analysis': {
                'module_path': 'dashboard.explore.ui.lead_lag',
                'functions': ['LeadLagAnalysisComponent']
            },
            # 新UI组件架构（UI移动后将更新路径）
            # TODO: 更新为 dashboard.core.ui.components.sidebar
            'data_upload_sidebar': {
                'module_path': 'dashboard.ui.components.sidebar',
                'functions': ['DataUploadSidebar']
            },
            'parameter_sidebar': {
                'module_path': 'dashboard.ui.components.sidebar',
                'functions': ['ParameterSidebar']
            }
        }

    def load_module(self, module_name: str) -> Optional[Any]:
        """懒加载指定模块"""
        # 检查是否已加载
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]

        if module_name not in self.module_configs:
            logger.warning(f"Module {module_name} not configured for lazy loading")
            return None

        config = self.module_configs[module_name]
        module_path = config['module_path']

        # 检查模块是否已经在加载中
        if hasattr(self, '_loading_modules'):
            if module_name in self._loading_modules:
                logger.warning(f"模块 {module_name} 正在加载中，避免循环导入")
                return None
        else:
            self._loading_modules = set()

        # 设置加载标记
        self._loading_modules.add(module_name)

        start_time = time.time()
        module = importlib.import_module(module_path)
        load_time = time.time() - start_time

        self.loaded_modules[module_name] = module
        self.load_times[module_name] = load_time

        # 清除加载标记
        if hasattr(self, '_loading_modules') and module_name in self._loading_modules:
            self._loading_modules.remove(module_name)

        return module

    def get_function(self, module_name: str, function_name: str) -> Optional[Callable]:
        """获取模块中的指定函数"""
        module = self.load_module(module_name)
        if module is None:
            return None

        try:
            func = getattr(module, function_name)
            return func
        except AttributeError:
            logger.error(f"Function {function_name} not found in module {module_name}")
            return None

    def preload_critical_modules(self):
        """预加载关键模块"""
        critical_modules = ['data_loader', 'preview_main']

        for module_name in critical_modules:
            self.load_module(module_name)

    def get_load_stats(self) -> Dict[str, float]:
        """获取加载统计信息"""
        return self.load_times.copy()

    def clear_cache(self):
        """清理模块缓存"""
        self.loaded_modules.clear()
        self.load_times.clear()

# 全局懒加载器实例
_lazy_loader = None

def get_lazy_loader() -> LazyModuleLoader:
    """获取全局懒加载器实例"""
    global _lazy_loader
    if _lazy_loader is None:
        _lazy_loader = LazyModuleLoader()
    return _lazy_loader

def get_cached_lazy_loader() -> LazyModuleLoader:
    """获取懒加载器实例（与get_lazy_loader等价）"""
    return get_lazy_loader()
