# -*- coding: utf-8 -*-
"""
分析相关UI组件模块
提供工业分析等专业分析功能的UI组件
"""

# 使用组件工厂解决导入冲突问题
from dashboard.ui.components.analysis.component_factory import (
    get_component_factory,
    get_analysis_component,
    get_industrial_file_upload_component,
    get_industrial_time_range_selector_component,
    get_industrial_group_details_component,
    get_industrial_welcome_component,
    get_time_series_chart_component,
    get_enterprise_indicators_chart_component
)

# 导入已实现的基础组件 - 从industrial子目录导入
from dashboard.ui.components.analysis.industrial.base_analysis_component import BaseAnalysisComponent

# 通过工厂获取组件类
_factory = get_component_factory()
IndustrialFileUploadComponent = _factory.get_component('IndustrialFileUploadComponent')
IndustrialTimeRangeSelectorComponent = _factory.get_component('IndustrialTimeRangeSelectorComponent')
IndustrialGroupDetailsComponent = _factory.get_component('IndustrialGroupDetailsComponent')
IndustrialWelcomeComponent = _factory.get_component('IndustrialWelcomeComponent')

# 直接导入图表组件（无冲突）
from dashboard.ui.components.analysis.charts import (
    TimeSeriesChartComponent,
    EnterpriseIndicatorsChartComponent
)

from dashboard.ui.components.analysis.visualization import VisualizationComponent

__all__ = [
    'IndustrialFileUploadComponent',
    'IndustrialTimeRangeSelectorComponent',
    'IndustrialGroupDetailsComponent',
    'IndustrialWelcomeComponent',
    'BaseAnalysisComponent',
    'TimeSeriesChartComponent',
    'EnterpriseIndicatorsChartComponent',
    'VisualizationComponent'
]
