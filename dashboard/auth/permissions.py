# -*- coding: utf-8 -*-
"""
权限管理器
提供基于用户直接权限的访问控制功能
"""

from typing import List, Dict, Optional, Set
import logging

from dashboard.auth.database import AuthDatabase
from dashboard.auth.models import User, PERMISSION_MODULE_MAP


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
        try:
            # 检查用户是否激活
            if not user.is_active:
                return False
            
            # 直接检查用户的权限列表
            return permission in user.permissions
            
        except Exception as e:
            self.logger.error(f"检查权限时发生错误: {e}")
            return False
    
    def has_module_access(self, user: User, module_name: str) -> bool:
        """
        检查用户是否可以访问指定模块

        Args:
            user: 用户对象
            module_name: 模块名称

        Returns:
            是否可以访问
        """
        try:
            # 获取模块所需权限
            required_permissions = PERMISSION_MODULE_MAP.get(module_name, [])

            # 如果模块不需要权限，默认允许访问
            if not required_permissions:
                return True

            # 检查用户是否具有任一所需权限（使用any()简化）
            return any(self.has_permission(user, p) for p in required_permissions)

        except Exception as e:
            self.logger.error(f"检查模块访问权限时发生错误: {e}")
            return False
    
    def get_accessible_modules(self, user: User) -> List[str]:
        """
        获取用户可访问的模块列表

        Args:
            user: 用户对象

        Returns:
            可访问的模块名称列表
        """
        try:
            return [m for m in PERMISSION_MODULE_MAP.keys() if self.has_module_access(user, m)]
        except Exception as e:
            self.logger.error(f"获取可访问模块时发生错误: {e}")
            return []

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
        try:
            filtered_modules = {}
            
            for main_module, sub_modules in modules.items():
                if self.has_module_access(user, main_module):
                    filtered_modules[main_module] = sub_modules
            
            return filtered_modules
            
        except Exception as e:
            self.logger.error(f"过滤模块配置时发生错误: {e}")
            return {}
