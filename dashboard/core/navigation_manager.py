# -*- coding: utf-8 -*-
"""
导航管理器 - 简化版
统一导航状态管理，移除冗余功能
"""

import time
import streamlit as st
from typing import Optional, Dict, Any
from dashboard.core.base import safe_operation


class NavigationStateKeys:
    """导航状态键常量"""
    MAIN_MODULE = 'main_module'
    SUB_MODULE = 'sub_module'
    PREVIOUS_MAIN = 'previous_main'
    PREVIOUS_SUB = 'previous_sub'
    TRANSITIONING = 'transitioning'
    LAST_NAVIGATION_TIME = 'last_navigation_time'


class NavigationManager:
    """导航管理器 - 简化版，直接使用session_state"""

    def __init__(self):
        """初始化导航管理器"""
        self._state_prefix = 'navigation'

        # 从配置系统加载配置
        from dashboard.core.config import get_core_config
        self.config = get_core_config()

        self._initialize_state()

    def _get_state_key(self, key: str) -> str:
        """生成完整的状态键"""
        return f'{self._state_prefix}.{key}'

    def _initialize_state(self):
        """初始化导航状态"""
        defaults = {
            NavigationStateKeys.MAIN_MODULE: None,
            NavigationStateKeys.SUB_MODULE: None,
            NavigationStateKeys.PREVIOUS_MAIN: None,
            NavigationStateKeys.PREVIOUS_SUB: None,
            NavigationStateKeys.TRANSITIONING: False,
            NavigationStateKeys.LAST_NAVIGATION_TIME: 0
        }

        for key, value in defaults.items():
            state_key = self._get_state_key(key)
            if state_key not in st.session_state:
                st.session_state[state_key] = value

    # ========== 内部通用方法 ==========

    def _set_module_state(self,
                         current_key: str,
                         previous_key: str,
                         module_name: Optional[str],
                         clear_sub_module: bool = False) -> bool:
        """
        通用模块状态设置方法

        Args:
            current_key: 当前模块键名
            previous_key: 历史模块键名
            module_name: 模块名称
            clear_sub_module: 是否清空子模块

        Returns:
            是否设置成功
        """
        # 获取当前值
        current = st.session_state.get(self._get_state_key(current_key))

        if current == module_name:
            return True

        # 保存之前的状态
        st.session_state[self._get_state_key(previous_key)] = current

        # 设置新状态
        st.session_state[self._get_state_key(current_key)] = module_name

        # 主模块切换时清空子模块
        if clear_sub_module:
            st.session_state[self._get_state_key(NavigationStateKeys.SUB_MODULE)] = None

        # 更新导航时间和清除缓存
        st.session_state[self._get_state_key(NavigationStateKeys.LAST_NAVIGATION_TIME)] = time.time()
        self._clear_cache()

        return True

    # ========== 主模块管理 ==========

    def get_current_main_module(self) -> Optional[str]:
        """获取当前主模块"""
        return st.session_state.get(self._get_state_key(NavigationStateKeys.MAIN_MODULE))

    @safe_operation(default_return=False)
    def set_current_main_module(self, module_name: Optional[str]) -> bool:
        """
        设置当前主模块

        Args:
            module_name: 模块名称

        Returns:
            是否设置成功
        """
        return self._set_module_state(
            NavigationStateKeys.MAIN_MODULE,
            NavigationStateKeys.PREVIOUS_MAIN,
            module_name,
            clear_sub_module=True
        )

    # ========== 子模块管理 ==========

    def get_current_sub_module(self) -> Optional[str]:
        """获取当前子模块"""
        return st.session_state.get(self._get_state_key(NavigationStateKeys.SUB_MODULE))

    @safe_operation(default_return=False)
    def set_current_sub_module(self, sub_module_name: Optional[str]) -> bool:
        """
        设置当前子模块

        Args:
            sub_module_name: 子模块名称

        Returns:
            是否设置成功
        """
        return self._set_module_state(
            NavigationStateKeys.SUB_MODULE,
            NavigationStateKeys.PREVIOUS_SUB,
            sub_module_name,
            clear_sub_module=False
        )

    # ========== 历史状态查询 ==========

    def _get_previous_module(self, previous_key: str) -> Optional[str]:
        """
        通用历史模块获取方法

        Args:
            previous_key: 历史模块键名

        Returns:
            历史模块名称
        """
        return st.session_state.get(self._get_state_key(previous_key))

    def get_previous_main_module(self) -> Optional[str]:
        """获取之前的主模块"""
        return self._get_previous_module(NavigationStateKeys.PREVIOUS_MAIN)

    def get_previous_sub_module(self) -> Optional[str]:
        """获取之前的子模块"""
        return self._get_previous_module(NavigationStateKeys.PREVIOUS_SUB)

    def get_last_navigation_time(self) -> float:
        """获取最后导航时间"""
        return st.session_state.get(
            self._get_state_key(NavigationStateKeys.LAST_NAVIGATION_TIME),
            0
        )

    # ========== 转换状态管理 ==========

    def is_transitioning(self) -> bool:
        """检查是否正在转换"""
        return st.session_state.get(
            self._get_state_key(NavigationStateKeys.TRANSITIONING),
            False
        )

    def set_transitioning(self, transitioning: bool) -> bool:
        """设置转换状态"""
        st.session_state[self._get_state_key(NavigationStateKeys.TRANSITIONING)] = transitioning
        return True

    # ========== 状态重置 ==========

    @safe_operation(default_return=False)
    def reset_navigation(self):
        """重置导航状态"""
        st.session_state[self._get_state_key(NavigationStateKeys.MAIN_MODULE)] = None
        st.session_state[self._get_state_key(NavigationStateKeys.SUB_MODULE)] = None
        st.session_state[self._get_state_key(NavigationStateKeys.PREVIOUS_MAIN)] = None
        st.session_state[self._get_state_key(NavigationStateKeys.PREVIOUS_SUB)] = None
        st.session_state[self._get_state_key(NavigationStateKeys.TRANSITIONING)] = False
        return True

    def _clear_cache(self):
        """清除导航相关缓存"""
        cache_keys = self.config.get_cache_keys()
        for key in cache_keys:
            if key in st.session_state:
                del st.session_state[key]

    # ========== 状态信息 ==========

    def get_state_info(self) -> Dict[str, Any]:
        """获取当前导航状态信息"""
        return {
            'main_module': self.get_current_main_module(),
            'sub_module': self.get_current_sub_module(),
            'previous_main': self.get_previous_main_module(),
            'previous_sub': self.get_previous_sub_module(),
            'transitioning': self.is_transitioning(),
            'last_navigation_time': self.get_last_navigation_time()
        }


# 全局单例获取函数
def get_navigation_manager():
    """
    获取导航管理器实例

    Returns:
        NavigationManager: 导航管理器实例
    """
    # 每个session创建独立实例
    if '_navigation_manager' not in st.session_state:
        st.session_state['_navigation_manager'] = NavigationManager()
    return st.session_state['_navigation_manager']


def reset_navigation_manager():
    """重置导航管理器（用于测试）"""
    if '_navigation_manager' in st.session_state:
        del st.session_state['_navigation_manager']
