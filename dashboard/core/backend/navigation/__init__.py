# -*- coding: utf-8 -*-
"""
导航管理模块
"""

from dashboard.core.backend.navigation.manager import (
    NavigationManager,
    NavigationStateKeys,
    get_navigation_manager,
    reset_navigation_manager
)

__all__ = [
    'NavigationManager',
    'NavigationStateKeys',
    'get_navigation_manager',
    'reset_navigation_manager'
]
