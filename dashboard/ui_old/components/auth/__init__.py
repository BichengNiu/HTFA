# -*- coding: utf-8 -*-
"""
认证相关UI组件
提供登录、用户管理等界面组件
"""

from dashboard.ui.components.auth.login_page import LoginPage
from dashboard.ui.components.auth.auth_middleware import AuthMiddleware
from dashboard.ui.components.auth.register_page import RegisterPage

__all__ = [
    'LoginPage',
    'AuthMiddleware',
    'RegisterPage'
]