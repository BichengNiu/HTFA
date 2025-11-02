"""
工业分析组件模块
"""

from dashboard.ui.components.analysis.industrial.base_analysis_component import BaseAnalysisComponent
from dashboard.ui.components.analysis.industrial.chart_components import (
    ChartComponent,
    TimeSeriesChartComponent,
    WeeklyChartComponent
)
from dashboard.ui.components.analysis.industrial.table_components import (
    TableComponent,
    SummaryTableComponent,
    DataTableComponent
)

__all__ = [
    'BaseAnalysisComponent',
    'ChartComponent',
    'TimeSeriesChartComponent',
    'WeeklyChartComponent',
    'TableComponent',
    'SummaryTableComponent',
    'DataTableComponent'
]
