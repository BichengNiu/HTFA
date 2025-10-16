# -*- coding: utf-8 -*-
"""
统一UI组件库
提供标准化的UI组件和页面模板，包含静态资源管理
"""

__version__ = "1.0.0"
__author__ = "HFTA Development Team"

# 导入核心组件
from dashboard.ui.components.welcome import WelcomeComponent
from dashboard.ui.components.navigation import NavigationComponent
from dashboard.ui.components.layout import LayoutComponent
from dashboard.ui.components.cards import ModuleCard, FeatureCard
from dashboard.ui.utils.tab_detector import TabStateDetector
from dashboard.ui.utils.style_manager import StyleManager, get_style_manager
from dashboard.ui.constants import UIConstants, NavigationLevel

# 导入迁移的样式功能
from dashboard.ui.utils.style_loader import (
    StyleLoader,
    inject_cached_styles,
    load_cached_styles,
    get_style_loader
)

# 导入UI初始化功能
from dashboard.ui.utils.style_initializer import (
    UIInitializer,
    initialize_ui,
    load_ui_styles,
    is_ui_initialized,
    get_ui_initializer
)

# 导入组件注册功能
from dashboard.ui.components.registry import (
    ComponentRegistry,
    get_component_registry,
    register_ui_component,
    get_ui_component_path,
    get_ui_component_dependencies
)

# 导入分析相关组件
from dashboard.ui.components.analysis import (
    IndustrialFileUploadComponent,
    IndustrialTimeRangeSelectorComponent,
    IndustrialGroupDetailsComponent,
    IndustrialWelcomeComponent,
    TimeSeriesChartComponent,
    EnterpriseIndicatorsChartComponent
)

# 导入新的UI组件
from dashboard.ui.components.data_input import (
    DataInputComponent,
    UnifiedDataUploadComponent,
    DataUploadSidebar,
    DataValidationComponent,
    DataStagingComponent,
    DataPreviewComponent
)

from dashboard.ui.components.analysis.timeseries import (
    TimeSeriesAnalysisComponent,
    StationarityAnalysisComponent,
    CorrelationAnalysisComponent,
    LeadLagAnalysisComponent,
    UnifiedCorrelationAnalysisComponent,
    DTWAnalysisComponent,
    # WinRateAnalysisComponent
)


# 静态资源路径
import os
from pathlib import Path
STATIC_DIR = Path(__file__).parent / "static"

__all__ = [
    'WelcomeComponent',
    'NavigationComponent',
    'LayoutComponent',
    'ModuleCard',
    'FeatureCard',
    'TabStateDetector',
    'StyleManager',
    'get_style_manager',
    'UIConstants',
    'NavigationLevel',
    'STATIC_DIR',
    # 迁移的样式功能
    'StyleLoader',
    'inject_cached_styles',
    'load_cached_styles',
    'get_style_loader',
    # UI初始化功能
    'UIInitializer',
    'initialize_ui',
    'load_ui_styles',
    'is_ui_initialized',
    'get_ui_initializer',
    # 组件注册功能
    'ComponentRegistry',
    'get_component_registry',
    'register_ui_component',
    'get_ui_component_path',
    'get_ui_component_dependencies',
    # 分析组件
    'IndustrialFileUploadComponent',
    'IndustrialTimeRangeSelectorComponent',
    'IndustrialGroupDetailsComponent',
    'IndustrialWelcomeComponent',
    'TimeSeriesChartComponent',
    'EnterpriseIndicatorsChartComponent',
    # 数据输入组件
    'DataInputComponent',
    'UnifiedDataUploadComponent',
    'DataUploadSidebar',
    'DataValidationComponent',
    'DataStagingComponent',
    'DataPreviewComponent',
    # 时间序列分析组件
    'TimeSeriesAnalysisComponent',
    'StationarityAnalysisComponent',
    'CorrelationAnalysisComponent',
    'LeadLagAnalysisComponent',
    'DTWAnalysisComponent',
    # 'WinRateAnalysisComponent'
]
