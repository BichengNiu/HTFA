# -*- coding: utf-8 -*-
"""
DFM页面组件模块

提供DFM相关的页面级UI组件
"""

from dashboard.ui.pages.dfm.data_prep_page import render_dfm_data_prep_page
from dashboard.ui.pages.dfm.model_training_page import render_dfm_model_training_page
from dashboard.ui.pages.dfm.model_analysis_page import render_dfm_model_analysis_page
from dashboard.ui.pages.dfm.news_analysis_page import render_dfm_news_analysis_page

__all__ = [
    'render_dfm_data_prep_page',
    'render_dfm_model_training_page',
    'render_dfm_model_analysis_page',
    'render_dfm_news_analysis_page'
]

__version__ = '1.0.0'
