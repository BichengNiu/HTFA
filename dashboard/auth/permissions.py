# -*- coding: utf-8 -*-
"""
权限管理器
提供基于用户直接权限的访问控制功能
"""

from typing import List, Dict, Optional, Set
import logging

from dashboard.auth.database import AuthDatabase
from dashboard.auth.models import User


# 权限模块映射
PERMISSION_MODULE_MAP = {
    "数据预览": ["data_preview"],
    "监测分析": ["monitoring_analysis"],
    "模型分析": ["model_analysis"],
    "数据探索": ["data_exploration"],
    "用户管理": ["user_management"]
}


class PermissionManager:
    """权限管理器 - 基于用户直接权限体系"""
    
    def __init__(self, db_path: str = None):
        """初始化权限管理器"""
        self.db = AuthDatabase(db_path)
        self.logger = logging.getLogger(__name__)
    
    def has_permission(self, user: User, permission: str) -> bool:
        """
        检查用户是否具有指定权限

        Args:
            user: 用户对象
            permission: 权限名称

        Returns:
            是否具有权限
        """
        if not user or not user.is_active:
            return False

        return permission in user.permissions
    
    def has_module_access(self, user: User, module_name: str) -> bool:
        """
        检查用户是否可以访问指定模块

        Args:
            user: 用户对象
            module_name: 模块名称

        Returns:
            是否可以访问
        """
        required_permissions = PERMISSION_MODULE_MAP.get(module_name, [])

        if not required_permissions:
            return True

        return any(self.has_permission(user, p) for p in required_permissions)
    
    def get_accessible_modules(self, user: User) -> List[str]:
        """
        获取用户可访问的模块列表

        Args:
            user: 用户对象

        Returns:
            可访问的模块名称列表
        """
        return [m for m in PERMISSION_MODULE_MAP.keys() if self.has_module_access(user, m)]

    def is_admin(self, user: User) -> bool:
        """
        检查用户是否为管理员

        Args:
            user: 用户对象

        Returns:
            是否为管理员
        """
        return self.has_permission(user, "user_management")

    def filter_accessible_modules(self, user: User, modules: Dict) -> Dict:
        """
        过滤用户可访问的模块配置

        Args:
            user: 用户对象
            modules: 原始模块配置

        Returns:
            过滤后的模块配置
        """
        return {
            main_module: sub_modules
            for main_module, sub_modules in modules.items()
            if self.has_module_access(user, main_module)
        }
