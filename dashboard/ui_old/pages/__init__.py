# -*- coding: utf-8 -*-
"""
页面级组件模块
"""

from dashboard.ui.pages.main_modules.data_preview import DataPreviewWelcomePage
from dashboard.ui.pages.main_modules.monitoring_analysis import MonitoringAnalysisWelcomePage
from dashboard.ui.pages.main_modules.model_analysis import ModelAnalysisWelcomePage
from dashboard.ui.pages.sub_modules.data_exploration import DataExplorationWelcomePage

__all__ = [
    'DataPreviewWelcomePage',
    'MonitoringAnalysisWelcomePage',
    'ModelAnalysisWelcomePage',
    'DataExplorationWelcomePage'
]
