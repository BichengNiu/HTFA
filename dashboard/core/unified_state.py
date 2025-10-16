# -*- coding: utf-8 -*-
"""
核心统一状态管理器
提供核心的状态存储和管理功能
"""

import threading
import logging
from typing import Dict, Any, Optional, List

from dashboard.core.base import safe_operation, ThreadSafeSingleton


class UnifiedStateManager(ThreadSafeSingleton):
    """核心统一状态管理器 - 简化版"""

    def __init__(self):
        """初始化核心状态管理器"""
        if hasattr(self, '_initialized'):
            return

        super().__init__()

        import streamlit as st

        # 使用Streamlit的session_state来持久化状态
        if '_unified_states' not in st.session_state:
            st.session_state._unified_states = {}
        if '_unified_initialized_keys' not in st.session_state:
            st.session_state._unified_initialized_keys = set()

        self._states = st.session_state._unified_states
        self._initialized_keys = st.session_state._unified_initialized_keys

        # 线程安全
        self._lock = threading.RLock()

        # 日志
        self.logger = logging.getLogger(__name__)

        self.logger.info("Core UnifiedStateManager initialized")
        self._initialized = True

    @safe_operation(default_return=None)
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        获取状态值

        Args:
            key: 状态键（支持 "namespace.key" 格式）
            default: 默认值

        Returns:
            状态值或默认值
        """
        with self._lock:
            return self._states.get(key, default)

    @safe_operation(default_return=False)
    def set_state(self, key: str, value: Any, is_initialization: bool = False) -> bool:
        """
        设置状态值

        Args:
            key: 状态键（支持 "namespace.key" 格式）
            value: 状态值
            is_initialization: 是否为初始化设置

        Returns:
            是否设置成功
        """
        with self._lock:
            self._states[key] = value

            if is_initialization:
                self._initialized_keys.add(key)

            return True

    @safe_operation(default_return=False)
    def delete_state(self, key: str) -> bool:
        """
        删除状态

        Args:
            key: 状态键

        Returns:
            是否删除成功
        """
        with self._lock:
            if key in self._states:
                del self._states[key]
                self._initialized_keys.discard(key)
                return True
            return False

    @safe_operation(default_return=False)
    def clear_state(self, key: Optional[str] = None) -> bool:
        """
        清空状态

        Args:
            key: 状态键，None表示清空所有

        Returns:
            是否清空成功
        """
        if key is None:
            return self.clear_all_states()
        else:
            return self.delete_state(key)

    @safe_operation(default_return=False)
    def clear_all_states(self) -> bool:
        """
        清空所有状态

        Returns:
            是否清空成功
        """
        with self._lock:
            state_count = len(self._states)
            self._states.clear()
            self._initialized_keys.clear()
            self.logger.info(f"All states cleared ({state_count} states)")
            return True

    def has_state(self, key: str) -> bool:
        """检查状态是否存在"""
        with self._lock:
            return key in self._states

    def get_all_keys(self) -> List[str]:
        """获取所有状态键"""
        with self._lock:
            return list(self._states.keys())

    def get_state_count(self) -> int:
        """获取状态数量"""
        with self._lock:
            return len(self._states)

    def is_initialized(self, key: str) -> bool:
        """检查状态是否已初始化"""
        with self._lock:
            return key in self._initialized_keys

    def get_namespace_keys(self, namespace: str) -> List[str]:
        """
        获取指定命名空间的所有键

        Args:
            namespace: 命名空间

        Returns:
            键列表（不包含命名空间前缀）
        """
        with self._lock:
            prefix = f"{namespace}."
            keys = [k.replace(prefix, '', 1) for k in self._states.keys() if k.startswith(prefix)]
            return keys

    @safe_operation(default_return=False)
    def clear_namespace(self, namespace: str) -> bool:
        """
        清除指定命名空间的所有状态

        Args:
            namespace: 命名空间

        Returns:
            是否清除成功
        """
        with self._lock:
            prefix = f"{namespace}."
            keys_to_delete = [k for k in self._states.keys() if k.startswith(prefix)]

            for key in keys_to_delete:
                del self._states[key]
                self._initialized_keys.discard(key)

            self.logger.info(f"Cleared {len(keys_to_delete)} states in namespace '{namespace}'")
            return True

    # === 兼容性方法：DFM模块 ===

    def get_dfm_state(self, module_name: str, key: str, default: Any = None) -> Any:
        """
        获取DFM模块状态（兼容方法）

        Args:
            module_name: 模块名称
            key: 状态键
            default: 默认值

        Returns:
            状态值
        """
        full_key = f"dfm.{module_name}.{key}"
        return self.get_state(full_key, default)

    def set_dfm_state(self, module_name: str, key: str, value: Any) -> bool:
        """
        设置DFM模块状态（兼容方法）

        Args:
            module_name: 模块名称
            key: 状态键
            value: 状态值

        Returns:
            是否设置成功
        """
        full_key = f"dfm.{module_name}.{key}"
        return self.set_state(full_key, value)

    def clear_dfm_state(self, module_name: str, key: str) -> bool:
        """
        清除DFM模块状态（兼容方法）

        Args:
            module_name: 模块名称
            key: 状态键

        Returns:
            是否清除成功
        """
        full_key = f"dfm.{module_name}.{key}"
        return self.delete_state(full_key)

    # === 命名空间便捷方法 ===

    def get_namespaced(self, namespace: str, key: str, default: Any = None) -> Any:
        """
        获取命名空间状态（便捷方法）

        Args:
            namespace: 命名空间
            key: 状态键
            default: 默认值

        Returns:
            状态值
        """
        full_key = f"{namespace}.{key}"
        return self.get_state(full_key, default)

    def set_namespaced(self, namespace: str, key: str, value: Any) -> bool:
        """
        设置命名空间状态（便捷方法）

        Args:
            namespace: 命名空间
            key: 状态键
            value: 状态值

        Returns:
            是否设置成功
        """
        full_key = f"{namespace}.{key}"
        return self.set_state(full_key, value)

    def delete_namespaced(self, namespace: str, key: str) -> bool:
        """
        删除命名空间状态（便捷方法）

        Args:
            namespace: 命名空间
            key: 状态键

        Returns:
            是否删除成功
        """
        full_key = f"{namespace}.{key}"
        return self.delete_state(full_key)

    # === 兼容性方法：Tools模块 ===

    def get_tools_state(self, module_name: str, key: str, default: Any = None) -> Any:
        """
        获取Tools模块状态（兼容方法）

        Args:
            module_name: 模块名称
            key: 状态键
            default: 默认值

        Returns:
            状态值
        """
        full_key = f"tools.{module_name}.{key}"
        return self.get_state(full_key, default)

    def set_tools_state(self, module_name: str, key: str, value: Any) -> bool:
        """
        设置Tools模块状态（兼容方法）

        Args:
            module_name: 模块名称
            key: 状态键
            value: 状态值

        Returns:
            是否设置成功
        """
        full_key = f"tools.{module_name}.{key}"
        return self.set_state(full_key, value)

    def clear_tools_state(self, module_name: str, key: str) -> bool:
        """
        清除Tools模块状态（兼容方法）

        Args:
            module_name: 模块名称
            key: 状态键

        Returns:
            是否清除成功
        """
        full_key = f"tools.{module_name}.{key}"
        return self.delete_state(full_key)


def get_unified_manager():
    """
    获取统一状态管理器实例（单例模式）

    Returns:
        UnifiedStateManager: 统一状态管理器实例
    """
    import os

    # 在多进程环境中禁用状态管理系统
    if os.getenv('DISABLE_STATE_MANAGEMENT', 'false').lower() == 'true':
        return None

    try:
        return UnifiedStateManager.get_instance()
    except Exception as e:
        print(f"[CRITICAL ERROR] 统一状态管理器初始化失败: {e}")
        raise RuntimeError(f"统一状态管理器初始化失败: {e}")
