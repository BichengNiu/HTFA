# -*- coding: utf-8 -*-
"""
UI工具函数模块
"""

from dashboard.ui.utils.tab_detector import TabStateDetector
from dashboard.ui.utils.style_manager import StyleManager
from dashboard.ui.utils.validators import UIValidator
from dashboard.ui.utils.style_loader import StyleLoader, inject_cached_styles, load_cached_styles, get_style_loader
from dashboard.ui.utils.style_initializer import UIInitializer, initialize_ui, load_ui_styles, is_ui_initialized, get_ui_initializer

__all__ = [
    'TabStateDetector',
    'StyleManager',
    'UIValidator',
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
    'get_ui_initializer'
]
