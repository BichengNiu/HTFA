"""
工业分析UI组件模块

该模块包含工业分析的所有UI组件：
- common: 通用组件（文件上传、时间选择器、欢迎页面、分组详情）
- shared: 共享组件（基础类、图表、表格）
"""

from dashboard.analysis.industrial.ui.common import (
    IndustrialFileUploadComponent,
    IndustrialTimeRangeSelectorComponent,
    IndustrialGroupDetailsComponent,
    IndustrialWelcomeComponent
)

from dashboard.analysis.industrial.ui.shared import (
    BaseAnalysisComponent,
    ChartComponent,
    TimeSeriesChartComponent,
    WeeklyChartComponent,
    TableComponent,
    SummaryTableComponent,
    DataTableComponent
)

__all__ = [
    'IndustrialFileUploadComponent',
    'IndustrialTimeRangeSelectorComponent',
    'IndustrialGroupDetailsComponent',
    'IndustrialWelcomeComponent',
    'BaseAnalysisComponent',
    'ChartComponent',
    'TimeSeriesChartComponent',
    'WeeklyChartComponent',
    'TableComponent',
    'SummaryTableComponent',
    'DataTableComponent'
]
