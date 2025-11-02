# -*- coding: utf-8 -*-
"""
标签页状态检测工具
重构版本：直接使用st.session_state
"""

import streamlit as st
import time
from typing import Optional, Dict, Any, List
from dashboard.core.ui.constants import UIConstants, NavigationLevel

class TabStateDetector:
    """标签页状态检测器"""

    def __init__(self):
        self._cache = {}
        self._cache_timeout = 5.0  # 缓存5秒，提升性能

    def detect_active_tab(self, sub_module: str) -> Optional[str]:
        """
        检测当前激活的标签页

        Args:
            sub_module: 子模块名称（如"数据探索"）

        Returns:
            激活的标签页名称，如果没有激活的标签页则返回None
        """
        cache_key = f"active_tab_{sub_module}"
        current_time = time.time()

        # 检查缓存
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if current_time - cached_time < self._cache_timeout:
                return cached_result

        # 执行检测
        result = self._detect_tab_state(sub_module)

        # 更新缓存
        self._cache[cache_key] = (current_time, result)

        return result

    def _detect_tab_state(self, sub_module: str) -> Optional[str]:
        """内部标签页状态检测逻辑"""

        # 方法1: 检查明确设置的活跃标签页状态
        # 根据子模块选择对应的状态键
        if sub_module == "数据探索":
            active_tab_key = UIConstants.STATE_KEYS["data_exploration"]["active_tab"]
        else:
            active_tab_key = UIConstants.STATE_KEYS["data_exploration"]["active_tab"]

        # 检查活跃标签页
        active_tab = st.session_state.get(active_tab_key, None)
        if active_tab:
            print(f"[TabDetector] 检测到活跃标签页: {active_tab}")
            return active_tab

        # 备用方法: 检查标志和时间戳
        try:
            return self._detect_by_flags_and_timestamps(sub_module)
        except Exception as e:
            print(f"[TabDetector] 标志检测失败: {e}")
            return None


    def _detect_by_flags_and_timestamps(self, sub_module: str = None) -> Optional[str]:
        """通过标志和时间戳检测活跃标签页"""
        # 根据子模块选择对应的状态键
        flags = UIConstants.STATE_KEYS["data_exploration"]["tab_flags"]
        timestamps = UIConstants.STATE_KEYS["data_exploration"]["timestamps"]

        # 检查哪个标签页的标志为True
        active_tabs = []
        for tab_name, flag_key in flags.items():
            if st.session_state.get(flag_key, False):
                timestamp = st.session_state.get(timestamps[tab_name], 0)
                active_tabs.append((tab_name, timestamp))

        if active_tabs:
            # 返回时间戳最新的标签页
            active_tabs.sort(key=lambda x: x[1], reverse=True)
            latest_tab = active_tabs[0][0]
            print(f"[TabDetector] 从标志检测到最新活跃标签页: {latest_tab}")
            return latest_tab

        return None

    def has_active_tab(self, sub_module: str) -> bool:
        """
        检查是否有激活的标签页

        Args:
            sub_module: 子模块名称

        Returns:
            是否有激活的标签页
        """
        return self.detect_active_tab(sub_module) is not None

    def get_navigation_level(self, main_module: str, sub_module: Optional[str]) -> NavigationLevel:
        """
        获取当前导航层级

        Args:
            main_module: 主模块名称
            sub_module: 子模块名称

        Returns:
            当前导航层级
        """
        if not sub_module:
            # 如果没有选择子模块，清除所有标签页状态
            self._clear_all_tab_states()
            return NavigationLevel.MAIN_MODULE

        if sub_module and not self.has_active_tab(sub_module):
            return NavigationLevel.SUB_MODULE

        return NavigationLevel.FUNCTION

    def _clear_all_tab_states(self):
        """清除所有标签页状态"""
        # 清除数据探索的活跃标签页状态
        exploration_active_tab = UIConstants.STATE_KEYS["data_exploration"]["active_tab"]
        tab_key = f'ui.tabs.{exploration_active_tab}'
        if tab_key in st.session_state:
            del st.session_state[tab_key]

        # 清除缓存
        self.clear_cache()

    def should_show_sidebar(self, main_module: str, sub_module: Optional[str]) -> bool:
        """
        判断是否应该显示侧边栏

        Args:
            main_module: 主模块名称
            sub_module: 子模块名称

        Returns:
            是否应该显示侧边栏
        """
        # 只有在第三层（具体功能层）才显示侧边栏
        if main_module == "数据探索":
            return True

        return False

    def clear_cache(self):
        """清空检测缓存"""
        self._cache.clear()

    def force_clear_all_states(self):
        """强制清除所有相关状态 - 用于导航重置"""
        print("[TabDetector] 强制清除所有状态")

        # 清除缓存
        self.clear_cache()

        # 清除状态中的相关状态
        self._clear_all_tab_states()

        print("[TabDetector] 强制清除完成")

    def _clear_session_state_tabs(self):
        """清除session_state中的标签页状态"""
        # 清除数据探索的活跃标签页状态
        exploration_active_tab = UIConstants.STATE_KEYS["data_exploration"]["active_tab"]
        st.session_state[exploration_active_tab] = None

        # 清除其他标签页状态标志
        all_keys = list(st.session_state.keys())
        tab_flags = [key for key in all_keys if 'currently_in_' in key and '_tab' in key]
        for flag in tab_flags:
            st.session_state[flag] = False

# 全局单例
_tab_detector = None

def get_tab_detector() -> TabStateDetector:
    """获取全局标签页检测器实例"""
    global _tab_detector
    if _tab_detector is None:
        _tab_detector = TabStateDetector()
    return _tab_detector
