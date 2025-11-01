# -*- coding: utf-8 -*-
"""
DFM状态管理兼容层

提供向后兼容的DFM状态管理接口，内部使用st.session_state实现
"""

import streamlit as st
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DFMStateManager:
    """DFM状态管理器 - 基于st.session_state的兼容实现"""

    def get_dfm_state(self, module: str, key: str, default: Any = None) -> Any:
        """
        获取DFM状态值

        Args:
            module: 模块名称（如 'data_prep', 'train_model', 'model_analysis'）
            key: 状态键
            default: 默认值

        Returns:
            状态值或默认值
        """
        # 使用点分命名空间约定
        full_key = f"{module}.{key}"
        return st.session_state.get(full_key, default)

    def set_dfm_state(self, module: str, key: str, value: Any) -> bool:
        """
        设置DFM状态值

        Args:
            module: 模块名称
            key: 状态键
            value: 状态值

        Returns:
            是否设置成功
        """
        try:
            # 使用点分命名空间约定
            full_key = f"{module}.{key}"
            st.session_state[full_key] = value
            return True
        except Exception as e:
            logger.error(f"设置DFM状态失败: {module}.{key} - {e}")
            return False

    def delete_dfm_state(self, module: str, key: str) -> bool:
        """
        删除DFM状态

        Args:
            module: 模块名称
            key: 状态键

        Returns:
            是否删除成功
        """
        try:
            full_key = f"{module}.{key}"
            if full_key in st.session_state:
                del st.session_state[full_key]
            return True
        except Exception as e:
            logger.error(f"删除DFM状态失败: {module}.{key} - {e}")
            return False

    def clear_dfm_module_state(self, module: str) -> bool:
        """
        清除模块的所有状态

        Args:
            module: 模块名称

        Returns:
            是否清除成功
        """
        try:
            prefix = f"{module}."
            keys_to_delete = [key for key in st.session_state.keys() if key.startswith(prefix)]
            for key in keys_to_delete:
                del st.session_state[key]
            return True
        except Exception as e:
            logger.error(f"清除模块状态失败: {module} - {e}")
            return False


# 全局单例实例
_dfm_manager = None


def get_global_dfm_manager() -> DFMStateManager:
    """
    获取全局DFM状态管理器实例

    Returns:
        DFMStateManager实例
    """
    global _dfm_manager
    if _dfm_manager is None:
        _dfm_manager = DFMStateManager()
    return _dfm_manager
