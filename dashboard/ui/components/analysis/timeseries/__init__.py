# -*- coding: utf-8 -*-
"""
时间序列分析组件模块
提供时间序列数据分析的各种组件
"""

from dashboard.ui.components.analysis.timeseries.base import TimeSeriesAnalysisComponent
from dashboard.ui.components.analysis.timeseries.stationarity import StationarityAnalysisComponent
from dashboard.ui.components.analysis.timeseries.correlation import CorrelationAnalysisComponent
from dashboard.ui.components.analysis.timeseries.lead_lag_analysis import LeadLagAnalysisComponent
from dashboard.ui.components.analysis.timeseries.unified_correlation import UnifiedCorrelationAnalysisComponent
from dashboard.ui.components.analysis.timeseries.dtw import DTWAnalysisComponent
# from dashboard.ui.components.analysis.timeseries.win_rate import WinRateAnalysisComponent

__all__ = [
    'TimeSeriesAnalysisComponent',
    'StationarityAnalysisComponent',
    'CorrelationAnalysisComponent',
    'LeadLagAnalysisComponent',
    'UnifiedCorrelationAnalysisComponent',
    'DTWAnalysisComponent',
    # 'WinRateAnalysisComponent'
]
