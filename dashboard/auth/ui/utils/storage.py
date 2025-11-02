# -*- coding: utf-8 -*-
"""
认证相关的持久化存储管理
从 dashboard.ui.utils.persistent_storage 迁移的认证存储功能
"""

import streamlit as st
import streamlit.components.v1 as components
import logging
import time
from typing import Any, Optional, Dict

# 导入基础存储类
from dashboard.core.ui.utils.persistent_storage import (
    FilePersistentStorage,
    BrowserPersistentStorage
)


class AuthStorageManager:
    """认证相关的持久化存储管理器"""

    def __init__(self):
        """初始化认证存储管理器"""
        self.file_storage = FilePersistentStorage()
        self.browser_storage = BrowserPersistentStorage()
        self.logger = logging.getLogger(__name__)

        # 认证相关的存储键名
        self.AUTH_KEYS = {
            'session_id': 'htfa_user_session_id',
            'user_info': 'htfa_current_user',
            'last_activity': 'htfa_last_activity',
            'remember_me': 'htfa_remember_me',
            'login_timestamp': 'htfa_login_timestamp'
        }

        # 添加调试标识
        self._debug_prefix = "[AUTH_STORAGE]"

    def save_auth_state(self, session_id: str, user_info: Dict[str, Any],
                       remember_me: bool = False) -> bool:
        """
        保存认证状态到持久化存储

        Args:
            session_id: 会话ID
            user_info: 用户信息字典
            remember_me: 是否记住登录

        Returns:
            是否保存成功
        """
        try:
            current_time = int(time.time() * 1000)

            # 选择存储类型：记住登录使用local，否则使用session
            storage_type = "local" if remember_me else "session"

            # 使用文件存储（更可靠）
            success_count = 0

            # 保存各项认证信息到文件存储
            storage_items = {
                self.AUTH_KEYS['session_id']: session_id,
                self.AUTH_KEYS['user_info']: user_info,
                self.AUTH_KEYS['last_activity']: current_time,
                self.AUTH_KEYS['remember_me']: remember_me,
                self.AUTH_KEYS['login_timestamp']: current_time
            }

            for key, value in storage_items.items():
                if self.file_storage.set_item(key, value, storage_type):
                    success_count += 1
                else:
                    self.logger.warning(f"{self._debug_prefix} 保存到文件存储失败: {key}")

            # 同时保存到session_state作为运行时缓存
            st.session_state['user_session_id'] = session_id
            st.session_state['current_user'] = user_info
            st.session_state['last_activity'] = current_time
            st.session_state['login_remember_flag'] = remember_me
            st.session_state[self.AUTH_KEYS['login_timestamp']] = current_time

            # 生成浏览器存储JavaScript（异步执行，不阻塞主流程）
            for key, value in storage_items.items():
                js_code = self.browser_storage.set_item_async(key, value, storage_type)
                components.html(js_code, height=0)

            self.logger.debug(f"{self._debug_prefix} 浏览器存储脚本已生成")

            success_rate = success_count / len(storage_items)
            self.logger.info(f"{self._debug_prefix} 认证状态保存完成 - 类型: {storage_type}, "
                           f"用户: {user_info.get('username', 'unknown')}, "
                           f"成功率: {success_rate:.1%} ({success_count}/{len(storage_items)})")

            return success_count >= 3  # 至少保存成功3项认为成功

        except Exception as e:
            self.logger.error(f"{self._debug_prefix} 保存认证状态失败: {e}")
            return False

    def restore_auth_state_to_session(self) -> Optional[Dict[str, Any]]:
        """
        从持久化存储恢复认证状态到session_state

        Returns:
            恢复的认证信息字典，失败则返回None
        """
        try:
            # 检查session_state中是否已有有效认证信息
            if st.session_state.get('user_session_id') and st.session_state.get('current_user'):
                self.logger.debug(f"{self._debug_prefix} Session state中已有认证信息，跳过恢复")
                return {
                    'session_id': st.session_state['user_session_id'],
                    'user_info': st.session_state['current_user'],
                    'restored_from': 'session_state'
                }

            # 首先尝试从文件存储恢复（更可靠）
            restored_auth = self._restore_from_file_storage()
            if restored_auth:
                self.logger.info(f"{self._debug_prefix} 从文件存储恢复认证状态成功: 用户 {restored_auth['user_info'].get('username', 'unknown')}")

                # 恢复到session_state
                self._apply_restored_auth_to_session(restored_auth)
                return restored_auth

            # 文件存储恢复失败，尝试从浏览器存储恢复（异步，用于后续访问）
            self._trigger_browser_storage_restore()

            self.logger.debug(f"{self._debug_prefix} 未找到可恢复的认证状态")
            return None

        except Exception as e:
            self.logger.error(f"{self._debug_prefix} 恢复认证状态失败: {e}")
            return None

    def _restore_from_file_storage(self) -> Optional[Dict[str, Any]]:
        """从文件存储恢复认证状态"""
        try:
            # 尝试从两种存储类型恢复（优先local，后session）
            for storage_type in ["local", "session"]:
                # 获取会话ID和用户信息
                session_id = self.file_storage.get_item(self.AUTH_KEYS['session_id'], storage_type=storage_type)
                user_info = self.file_storage.get_item(self.AUTH_KEYS['user_info'], storage_type=storage_type)

                if session_id and user_info:
                    # 检查会话是否过期
                    login_timestamp = self.file_storage.get_item(self.AUTH_KEYS['login_timestamp'],
                                                               default=int(time.time() * 1000),
                                                               storage_type=storage_type)
                    remember_me = self.file_storage.get_item(self.AUTH_KEYS['remember_me'],
                                                           default=False,
                                                           storage_type=storage_type)

                    # 计算过期时间
                    max_age = 7 * 24 * 60 * 60 * 1000 if remember_me else 24 * 60 * 60 * 1000  # 7天或1天
                    current_time = int(time.time() * 1000)

                    if current_time - login_timestamp < max_age:
                        # 会话未过期
                        return {
                            'session_id': session_id,
                            'user_info': user_info,
                            'last_activity': self.file_storage.get_item(self.AUTH_KEYS['last_activity'],
                                                                       default=current_time,
                                                                       storage_type=storage_type),
                            'remember_me': remember_me,
                            'login_timestamp': login_timestamp,
                            'storage_type': storage_type,
                            'restored_from': 'file_storage'
                        }
                    else:
                        # 会话已过期，清除存储
                        self.logger.info(f"{self._debug_prefix} 会话已过期，清除 {storage_type} 存储")
                        self._clear_storage_type(storage_type)

            return None

        except Exception as e:
            self.logger.error(f"{self._debug_prefix} 从文件存储恢复失败: {e}")
            return None

    def _apply_restored_auth_to_session(self, auth_data: Dict[str, Any]):
        """将恢复的认证数据应用到session_state"""
        try:
            st.session_state['user_session_id'] = auth_data['session_id']
            st.session_state['current_user'] = auth_data['user_info']
            st.session_state['last_activity'] = auth_data.get('last_activity', int(time.time() * 1000))
            st.session_state['login_remember_flag'] = auth_data.get('remember_me', False)

            self.logger.debug(f"{self._debug_prefix} 认证状态已应用到session_state")
        except Exception as e:
            self.logger.error(f"{self._debug_prefix} 应用认证状态到session_state失败: {e}")

    def _trigger_browser_storage_restore(self):
        """触发浏览器存储恢复（生成JavaScript）"""
        # 生成恢复所有认证项的JavaScript
        for key in self.AUTH_KEYS.values():
            # 尝试从localStorage和sessionStorage获取
            for storage_type in ["local", "session"]:
                js_code = self.browser_storage.get_item_async(key, storage_type)
                components.html(js_code, height=0)

        self.logger.debug(f"{self._debug_prefix} 浏览器存储恢复脚本已生成")

    def clear_auth_state(self) -> bool:
        """
        清除所有认证状态

        Returns:
            是否清除成功
        """
        try:
            success_count = 0

            # 清除文件存储
            for storage_type in ["local", "session"]:
                cleared_count = self._clear_storage_type(storage_type)
                success_count += cleared_count

            # 清除session_state
            session_keys = [
                'user_session_id', 'current_user', 'last_activity',
                'login_remember_flag', self.AUTH_KEYS['login_timestamp']
            ]
            for key in session_keys:
                if key in st.session_state:
                    del st.session_state[key]
                    success_count += 1

            # 生成浏览器存储清除JavaScript
            for key in self.AUTH_KEYS.values():
                for storage_type in ["local", "session"]:
                    js_code = self.browser_storage.remove_item_async(key, storage_type)
                    components.html(js_code, height=0)

            self.logger.debug(f"{self._debug_prefix} 浏览器存储清除脚本已生成")

            self.logger.info(f"{self._debug_prefix} 认证状态已清除，操作数量: {success_count}")
            return success_count > 0

        except Exception as e:
            self.logger.error(f"{self._debug_prefix} 清除认证状态失败: {e}")
            return False

    def _clear_storage_type(self, storage_type: str) -> int:
        """清除指定类型的存储"""
        cleared_count = 0
        for key in self.AUTH_KEYS.values():
            if self.file_storage.remove_item(key, storage_type):
                cleared_count += 1
        return cleared_count

    def get_debug_info(self) -> Dict[str, Any]:
        """获取调试信息"""
        debug_info = {
            'session_state_keys': [key for key in st.session_state.keys() if 'user' in key.lower() or 'auth' in key.lower()],
            'auth_keys': self.AUTH_KEYS,
            'storage_dir': str(self.file_storage.storage_dir),
            'file_count': len(list(self.file_storage.storage_dir.glob("*.json"))),
        }

        # 检查文件存储状态
        debug_info['file_storage_status'] = {}
        for storage_type in ["local", "session"]:
            for key_name, key in self.AUTH_KEYS.items():
                file_path = self.file_storage._get_file_path(key, storage_type)
                debug_info['file_storage_status'][f"{storage_type}_{key_name}"] = file_path.exists()

        return debug_info


# 工厂函数
def get_auth_storage_manager() -> AuthStorageManager:
    """获取当前session的认证存储管理器实例"""
    if '_auth_storage_manager' not in st.session_state:
        st.session_state['_auth_storage_manager'] = AuthStorageManager()
    return st.session_state['_auth_storage_manager']
