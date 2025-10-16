# -*- coding: utf-8 -*-
"""
认证中间件
提供认证检查、会话管理等中间件功能
"""

import streamlit as st
from typing import Optional, Dict, Any, Callable
import logging
from datetime import datetime

# 导入认证相关模块
from dashboard.auth.authentication import AuthManager
from dashboard.auth.permissions import PermissionManager
from dashboard.auth.models import User, UserSession
from dashboard.core import get_unified_manager

# 导入持久化存储工具
from dashboard.ui.utils.persistent_storage import auth_storage_manager


class AuthMiddleware:
    """认证中间件"""
    
    def __init__(self):
        """初始化认证中间件"""
        self.auth_manager = AuthManager()
        self.permission_manager = PermissionManager()
        self.logger = logging.getLogger(__name__)
        
        # 获取统一状态管理器
        try:
            self.state_manager = get_unified_manager()
        except:
            self.state_manager = None
    
    def check_authentication(self) -> tuple[bool, Optional[User]]:
        """
        检查用户认证状态（带持久化存储恢复）
        
        Returns:
            (是否已认证, 用户对象)
        """
        try:
            # [新增] 首先尝试从持久化存储恢复认证状态
            self._try_restore_from_persistent_storage()
            
            # 从session state获取会话ID
            session_id = st.session_state.get('user_session_id')
            if not session_id:
                # [新增] 如果session state中没有，最后尝试从持久化存储直接获取
                session_id = self._get_session_id_from_persistent_storage()
                if session_id:
                    # 恢复到session state
                    st.session_state['user_session_id'] = session_id
                    self.logger.info(f"[AUTH_FIX] 从持久化存储恢复会话ID: {session_id[:8]}...")
                else:
                    self.logger.debug("[AUTH_FIX] 未找到有效的会话ID")
                    return False, None
            
            # 验证会话
            is_valid, user = self.auth_manager.validate_session(session_id)
            
            if is_valid and user:
                # 更新用户信息到session state
                st.session_state['current_user'] = user
                st.session_state['last_activity'] = datetime.now()
                
                # [新增] 如果从持久化存储恢复成功，更新持久化存储的活动时间
                self._update_persistent_storage_activity()
                
                # 同步到统一状态管理器
                try:
                    if self.state_manager:
                        self.state_manager.set_state('auth.current_user', user)
                        self.state_manager.set_state('auth.user_session_id', session_id)
                        
                        # 设置用户可访问模块
                        accessible_modules = self.permission_manager.get_accessible_modules(user)
                        self.state_manager.set_state('auth.user_accessible_modules', set(accessible_modules))
                except Exception as e:
                    self.logger.warning(f"统一状态管理器同步失败，但不影响认证: {e}")
                
                return True, user
            else:
                # 清除无效会话
                self._clear_user_session()
                return False, None
                
        except Exception as e:
            self.logger.error(f"检查认证状态失败: {e}")
            return False, None
    
    def require_authentication(self, show_login=True) -> Optional[User]:
        """
        要求用户认证，如果未认证则显示登录页面
        
        Args:
            show_login: 是否显示登录页面
            
        Returns:
            用户对象或None
        """
        is_authenticated, user = self.check_authentication()
        
        if is_authenticated and user:
            return user
        
        if show_login:
            # 显示登录页面
            from dashboard.ui.components.auth.login_page import render_login_page

            login_result = render_login_page()
            if login_result:
                success, login_data = login_result
                if success:
                    # 登录成功，保存会话信息
                    user = login_data['user']
                    session = login_data['session']
                    remember_me = login_data.get('remember_me', False)
                    
                    st.session_state['user_session_id'] = session.session_id
                    st.session_state['current_user'] = user
                    st.session_state['last_activity'] = datetime.now()
                    st.session_state['login_remember_flag'] = remember_me
                    
                    # [新增] 保存到持久化存储
                    try:
                        self._save_auth_state_to_persistent_storage(session.session_id, user, remember_me)
                    except Exception as e:
                        self.logger.warning(f"[AUTH_FIX] 保存到持久化存储失败，但不影响登录: {e}")
                    
                    # 同步到统一状态管理器
                    state_manager = get_unified_manager()
                    if state_manager:
                        state_manager.set_state('auth.current_user', user)
                        state_manager.set_state('auth.user_session_id', session.session_id)

                        # 设置用户可访问模块
                        accessible_modules = self.permission_manager.get_accessible_modules(user)
                        state_manager.set_state('auth.user_accessible_modules', set(accessible_modules))
                    
                    # 刷新页面以进入主应用
                    st.rerun()
            
            # 如果登录页面正在显示，停止后续页面渲染
            st.stop()
        
        return None
    
    def check_permission(self, user: User, module_name: str) -> bool:
        """
        检查用户是否有访问指定模块的权限
        
        Args:
            user: 用户对象
            module_name: 模块名称
            
        Returns:
            是否有权限
        """
        try:
            return self.permission_manager.has_module_access(user, module_name)
        except Exception as e:
            self.logger.error(f"检查权限失败: {e}")
            return False
    
    def require_permission(self, user: User, module_name: str) -> bool:
        """
        要求用户具有指定模块的访问权限
        
        Args:
            user: 用户对象
            module_name: 模块名称
            
        Returns:
            是否有权限
        """
        if not self.check_permission(user, module_name):
            st.error(f"权限不足：您无权访问「{module_name}」模块")
            st.info("如需访问权限，请联系系统管理员")
            st.stop()
            return False
        
        return True
    
    def filter_accessible_modules(self, user: User, module_config: Dict) -> Dict:
        """
        过滤用户可访问的模块配置
        
        Args:
            user: 用户对象
            module_config: 模块配置字典
            
        Returns:
            过滤后的模块配置
        """
        try:
            return self.permission_manager.filter_accessible_modules(user, module_config)
        except Exception as e:
            self.logger.error(f"过滤模块配置失败: {e}")
            return module_config
    
    def logout(self) -> bool:
        """
        用户登出
        
        Returns:
            是否登出成功
        """
        try:
            # 获取会话ID
            session_id = st.session_state.get('user_session_id')
            
            # 从服务器端删除会话
            if session_id:
                self.auth_manager.logout(session_id)
            
            # 清除客户端会话信息
            self._clear_user_session()
            
            self.logger.info("用户登出成功")
            return True
            
        except Exception as e:
            self.logger.error(f"登出失败: {e}")
            return False
    
    def _clear_user_session(self):
        """清除用户会话信息（包括持久化存储）"""
        # 清除session state中的用户信息
        session_keys = [
            'user_session_id',
            'current_user', 
            'last_activity',
            'login_remember_flag'
        ]
        
        for key in session_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        # [新增] 清除持久化存储
        try:
            auth_storage_manager.clear_auth_state()
            self.logger.debug("[AUTH_FIX] 已清除持久化存储中的认证状态")
        except Exception as e:
            self.logger.warning(f"[AUTH_FIX] 清除持久化存储失败: {e}")
        
        # 如果有统一状态管理器，也清除相关状态
        try:
            if self.state_manager:
                self.state_manager.set_state('auth.current_user', None)
                self.state_manager.set_state('auth.user_session_id', None)
                self.state_manager.set_state('auth.user_accessible_modules', set())
        except Exception as e:
            self.logger.warning(f"清除统一状态管理器状态失败: {e}")
    
    def _try_restore_from_persistent_storage(self) -> bool:
        """
        尝试从持久化存储恢复认证状态
        
        Returns:
            是否恢复成功
        """
        try:
            # 如果session state中已有认证信息，跳过恢复
            if st.session_state.get('user_session_id') and st.session_state.get('current_user'):
                return True
            
            # 尝试从持久化存储恢复
            restored_auth = auth_storage_manager.restore_auth_state_to_session()
            if restored_auth:
                self.logger.info(f"[AUTH_FIX] 认证状态恢复成功: 用户 {restored_auth['user_info'].get('username', 'unknown')} "
                               f"(来源: {restored_auth.get('restored_from', 'unknown')})")
                return True
            else:
                self.logger.debug("[AUTH_FIX] 未找到可恢复的认证状态")
                return False
                
        except Exception as e:
            self.logger.error(f"[AUTH_FIX] 从持久化存储恢复认证状态失败: {e}")
            return False
    
    def _get_session_id_from_persistent_storage(self) -> Optional[str]:
        """
        直接从持久化存储获取会话ID
        
        Returns:
            会话ID或None
        """
        try:
            # 尝试从文件存储获取会话ID
            if hasattr(auth_storage_manager, 'file_storage'):
                for storage_type in ["local", "session"]:
                    session_id = auth_storage_manager.file_storage.get_item(
                        auth_storage_manager.AUTH_KEYS['session_id'], 
                        storage_type=storage_type
                    )
                    if session_id:
                        self.logger.debug(f"[AUTH_FIX] 从持久化存储({storage_type})获取到会话ID")
                        return session_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"[AUTH_FIX] 从持久化存储获取会话ID失败: {e}")
            return None
    
    def _update_persistent_storage_activity(self):
        """更新持久化存储中的活动时间"""
        try:
            # 获取当前的记住登录设置
            remember_me = st.session_state.get('login_remember_flag', False)
            storage_type = "local" if remember_me else "session"
            
            # 更新活动时间
            current_time = int(datetime.now().timestamp() * 1000)
            if hasattr(auth_storage_manager, 'file_storage'):
                auth_storage_manager.file_storage.set_item(
                    auth_storage_manager.AUTH_KEYS['last_activity'],
                    current_time,
                    storage_type=storage_type
                )
            
            self.logger.debug(f"[AUTH_FIX] 已更新持久化存储活动时间({storage_type})")
            
        except Exception as e:
            self.logger.warning(f"[AUTH_FIX] 更新持久化存储活动时间失败: {e}")
    
    def _save_auth_state_to_persistent_storage(self, session_id: str, user: User, remember_me: bool = False) -> bool:
        """
        保存认证状态到持久化存储
        
        Args:
            session_id: 会话ID
            user: 用户对象
            remember_me: 是否记住登录
            
        Returns:
            是否保存成功
        """
        try:
            # 将用户对象转换为字典
            user_info = {
                'id': getattr(user, 'id', 0),
                'username': getattr(user, 'username', ''),
                'email': getattr(user, 'email', ''),
                'is_active': getattr(user, 'is_active', True),
                'permissions': getattr(user, 'permissions', [])
            }
            
            # 保存到持久化存储
            success = auth_storage_manager.save_auth_state(session_id, user_info, remember_me)
            if success:
                self.logger.info(f"[AUTH_FIX] 认证状态已保存到持久化存储: 用户 {user.username}, 记住登录: {remember_me}")
            else:
                self.logger.warning(f"[AUTH_FIX] 保存认证状态到持久化存储失败")
            
            return success
            
        except Exception as e:
            self.logger.error(f"[AUTH_FIX] 保存认证状态到持久化存储异常: {e}")
            return False
    
    def get_current_user(self) -> Optional[User]:
        """
        获取当前登录用户
        
        Returns:
            用户对象或None
        """
        return st.session_state.get('current_user')
    
    def is_authenticated(self) -> bool:
        """
        检查是否已认证
        
        Returns:
            是否已认证
        """
        is_auth, _ = self.check_authentication()
        return is_auth
    
    def render_user_info(self):
        """渲染用户信息组件（用于侧边栏等）"""
        user = self.get_current_user()
        if user:
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 用户信息")
                st.write(f"**用户名：** {user.username}")
                
                # 显示用户权限信息（转换为中文模块名称）
                if user.permissions:
                    from dashboard.auth.models import PERMISSION_MODULE_MAP
                    # 将权限代码转换为中文模块名称
                    accessible_modules = []
                    for module_name, required_perms in PERMISSION_MODULE_MAP.items():
                        if any(perm in user.permissions for perm in required_perms):
                            accessible_modules.append(module_name)
                    
                    if accessible_modules:
                        st.write(f"**权限：** {'、'.join(accessible_modules)}")
                    else:
                        st.write("**权限：** 无匹配模块")
                else:
                    st.write("**权限：** 无")
                
                # 登出按钮
                if st.button("退出登录", key="logout_button"):
                    if self.logout():
                        st.success("已成功退出登录")
                        st.rerun()
                    else:
                        st.error("退出登录失败")


def require_auth(show_login: bool = True):
    """
    认证装饰器函数
    
    Args:
        show_login: 是否显示登录页面
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            auth_middleware = AuthMiddleware()
            user = auth_middleware.require_authentication(show_login)
            if user:
                # 将用户对象传递给函数
                return func(current_user=user, *args, **kwargs)
            return None
        return wrapper
    return decorator


def require_permission(module_name: str):
    """
    权限检查装饰器
    
    Args:
        module_name: 需要的模块权限
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            auth_middleware = AuthMiddleware()
            user = auth_middleware.require_authentication()
            if user and auth_middleware.require_permission(user, module_name):
                return func(current_user=user, *args, **kwargs)
            return None
        return wrapper
    return decorator


# 全局中间件实例
_auth_middleware = None

def get_auth_middleware() -> AuthMiddleware:
    """获取全局认证中间件实例"""
    global _auth_middleware
    if _auth_middleware is None:
        _auth_middleware = AuthMiddleware()
    return _auth_middleware
