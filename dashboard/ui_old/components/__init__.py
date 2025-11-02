# -*- coding: utf-8 -*-
"""
UI基础组件模块
"""

# 延迟导入避免循环导入
from dashboard.ui.components.base import UIComponent, BaseWelcomePage
from dashboard.ui.components.welcome import WelcomeComponent
from dashboard.ui.components.navigation import NavigationComponent
from dashboard.ui.components.layout import LayoutComponent
from dashboard.ui.components.cards import ModuleCard, FeatureCard
# sidebar组件使用延迟导入
# from dashboard.ui.components.sidebar import SidebarComponent

# 导入分析相关组件 - 暂时注释掉未实现的组件
# from dashboard.ui.components.analysis import (
#     IndustrialFileUploadComponent,
#     IndustrialTimeRangeSelectorComponent,
#     IndustrialGroupDetailsComponent,
#     IndustrialWelcomeComponent,
#     TimeSeriesChartComponent,
#     EnterpriseIndicatorsChartComponent
# )

# 导入已实现的组件 - 使用组件工厂解决导入问题
from dashboard.ui.components.analysis import TimeSeriesChartComponent, EnterpriseIndicatorsChartComponent

# 导入数据输入组件
from dashboard.ui.components.data_input import (
    DataInputComponent,
    UnifiedDataUploadComponent,
    DataUploadSidebar,
    DataValidationComponent,
    DataStagingComponent,
    DataPreviewComponent
)

# 导入时间序列分析组件
from dashboard.ui.components.analysis.timeseries import (
    TimeSeriesAnalysisComponent,
    StationarityAnalysisComponent,
    CorrelationAnalysisComponent,
    LeadLagAnalysisComponent,
    UnifiedCorrelationAnalysisComponent,
    DTWAnalysisComponent,
    # WinRateAnalysisComponent
)

# 导入组件注册功能
from dashboard.ui.components.registry import (
    ComponentRegistry,
    get_component_registry,
    register_ui_component,
    get_ui_component_path,
    get_ui_component_dependencies
)

__all__ = [
    'UIComponent',
    'BaseWelcomePage',
    'WelcomeComponent',
    'NavigationComponent',
    'LayoutComponent',
    'ModuleCard',
    'FeatureCard',
    # 'SidebarComponent',  # 使用延迟导入
    # 组件注册功能
    'ComponentRegistry',
    'get_component_registry',
    'register_ui_component',
    'get_ui_component_path',
    'get_ui_component_dependencies',
    # 分析组件 - 现在可以安全导出
    'TimeSeriesChartComponent',
    'EnterpriseIndicatorsChartComponent',
]

__all__.extend([
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
])
