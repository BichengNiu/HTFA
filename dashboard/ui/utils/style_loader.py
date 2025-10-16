# -*- coding: utf-8 -*-
"""
样式加载器
优化CSS加载性能，实现样式缓存
迁移自dashboard.core.style_loader
"""

import streamlit as st
import os
from pathlib import Path
import hashlib
import time
import logging

logger = logging.getLogger(__name__)

# 全局StyleLoader实例，避免重复创建
_global_style_loader = None

def get_style_loader():
    """获取全局StyleLoader实例"""
    global _global_style_loader
    if _global_style_loader is None:
        _global_style_loader = StyleLoader()
    return _global_style_loader

class StyleLoader:
    """样式加载器"""
    
    def __init__(self):
        # 调整路径指向ui/static目录
        self.static_dir = Path(__file__).parent.parent / "static"
        self.cache = {}
        self.file_hashes = {}
    
    def _get_file_hash(self, file_path: Path) -> str:
        """获取文件哈希值"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"Error getting file hash for {file_path}: {e}")
            return ""
    
    def _load_css_file(self, file_path: Path) -> str:
        """加载CSS文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading CSS file {file_path}: {e}")
            return ""
    
    def load_styles(self, css_file: str = "styles.css") -> str:
        """加载样式文件"""
        file_path = self.static_dir / css_file
        
        if not file_path.exists():
            logger.warning(f"CSS file not found: {file_path}")
            return ""
        
        # 检查文件是否已更改
        current_hash = self._get_file_hash(file_path)
        cache_key = str(file_path)
        
        if (cache_key in self.cache and 
            cache_key in self.file_hashes and 
            self.file_hashes[cache_key] == current_hash):
            # 使用缓存的内容
            return self.cache[cache_key]
        
        # 加载新内容
        content = self._load_css_file(file_path)
        
        # 更新缓存
        self.cache[cache_key] = content
        self.file_hashes[cache_key] = current_hash

        # 只在首次加载时打印日志，避免重复打印
        if cache_key not in self.cache:
            logger.info(f"Loaded CSS file: {css_file}")
        return content
    
    def _load_js_file(self, file_path: Path) -> str:
        """加载JavaScript文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading JS file {file_path}: {e}")
            return ""

    def inject_styles(self, css_file: str = "styles.css"):
        """注入样式和JavaScript到Streamlit应用"""
        css_content = self.load_styles(css_file)

        # 加载JavaScript文件
        js_file_path = self.static_dir / "button_state.js"
        js_content = ""
        if js_file_path.exists():
            js_content = self._load_js_file(js_file_path)
            logger.info(f"Loaded JavaScript file: {js_file_path}")

        if css_content:
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

            # 直接注入JavaScript
            if js_content:
                st.markdown(f"<script>{js_content}</script>", unsafe_allow_html=True)
                # 减少重复日志，改为debug级别
                logger.debug("JavaScript injected successfully")
    
    def clear_cache(self):
        """清理样式缓存"""
        self.cache.clear()
        self.file_hashes.clear()

# 全局样式加载器实例
_style_loader = None

def get_style_loader() -> StyleLoader:
    """获取全局样式加载器实例"""
    global _style_loader
    if _style_loader is None:
        _style_loader = StyleLoader()
    return _style_loader

def load_cached_styles(css_file: str = "styles.css") -> str:
    """加载样式文件 - 优化版本，使用智能缓存"""
    loader = get_style_loader()  # 使用全局实例
    return loader.load_styles(css_file)

def inject_cached_styles(css_file: str = "styles.css"):
    """注入样式到Streamlit应用 - 使用统一状态管理器"""
    # 直接注入CSS，不依赖统一状态管理器
    static_dir = Path(__file__).parent.parent / "static"
    css_path = static_dir / css_file
    
    if css_path.exists():
        try:
            with open(css_path, 'r', encoding='utf-8') as f:
                css_content = f.read()
            
            # 直接注入CSS
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            return
        except Exception:
            pass
    
    # 原始逻辑作为后备
    # 获取统一状态管理器（强制模式）
    from dashboard.core import get_unified_manager
    state_manager = get_unified_manager()
    if state_manager is None:
        return

    # 状态键定义
    css_cache_key = f"ui.css.cache_{css_file}"
    css_hash_key = f"ui.css.hash_{css_file}"
    css_time_key = f"ui.css.time_{css_file}"

    # 获取CSS文件路径和当前哈希 - 调整为ui/static路径
    static_dir = Path(__file__).parent.parent / "static"
    css_path = static_dir / css_file

    if not css_path.exists():
        logger.warning(f"CSS file not found: {css_path}")
        return

    try:
        # 获取文件修改时间和哈希
        file_mtime = os.path.getmtime(css_path)
        with open(css_path, 'rb') as f:
            current_hash = hashlib.md5(f.read()).hexdigest()

        logger.debug(f"CSS file {css_file} - hash: {current_hash[:8]}, mtime: {file_mtime}")

        # 使用统一状态管理器检查缓存
        if state_manager:
            cached_hash = state_manager.get_state(css_hash_key)
            cached_time = state_manager.get_state(css_time_key)
            cached_content = state_manager.get_state(css_cache_key)

            if cached_hash and cached_time and cached_content:
                logger.debug(f"Found cached CSS - hash: {cached_hash[:8]}, mtime: {cached_time}")

                # 如果哈希和修改时间都没变，使用缓存但仍然重新注入
                if cached_hash == current_hash and cached_time == file_mtime:
                    logger.info(f"Using cached CSS for {css_file} (hash: {current_hash[:8]})")
                    st.markdown(f"<style>{cached_content}</style>", unsafe_allow_html=True)
                    # 注入JavaScript
                    _inject_javascript()
                    return
                else:
                    logger.info(f"CSS cache invalid for {css_file} - reloading")
            else:
                logger.debug(f"No CSS cache found for {css_file} - first load")
        else:
            # 如果状态管理器不可用，直接加载文件
            cached_hash = None
            cached_time = None

        # 加载新CSS并缓存
        css_content = load_cached_styles(css_file)
        if css_content:
            # 使用统一状态管理器缓存
            if state_manager:
                state_manager.set_state(css_cache_key, css_content)
                state_manager.set_state(css_hash_key, current_hash)
                state_manager.set_state(css_time_key, file_mtime)

            # 添加缓存检查，避免重复打印
            if css_cache_key not in state_manager.get_all_keys():
                logger.info(f"Loaded and cached CSS file: {css_file}")
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

            # 注入JavaScript
            _inject_javascript()
        else:
            # 如果加载失败，尝试直接加载
            try:
                with open(css_path, 'r', encoding='utf-8') as f:
                    css_content = f.read()

                # 使用统一状态管理器缓存
                if state_manager:
                    state_manager.set_state(css_cache_key, css_content)
                    state_manager.set_state(css_hash_key, current_hash)
                    state_manager.set_state(css_time_key, file_mtime)

                logger.info(f"Directly loaded and cached CSS file: {css_file}")
                st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

                # 注入JavaScript
                _inject_javascript()
            except Exception as e:
                logger.error(f"Failed to load CSS file directly: {e}")

    except Exception as e:
        logger.error(f"Error in inject_cached_styles: {e}")
        # 回退到原始方法
        css_content = load_cached_styles(css_file)
        if css_content:
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            # 注入JavaScript
            _inject_javascript()


def _inject_javascript():
    """注入JavaScript代码"""
    try:
        # 获取JavaScript文件路径
        static_dir = Path(__file__).parent.parent / "static"
        js_path = static_dir / "button_state.js"

        if js_path.exists():
            with open(js_path, 'r', encoding='utf-8') as f:
                js_content = f.read()

            st.markdown(f"<script>{js_content}</script>", unsafe_allow_html=True)
            # 改为debug级别，减少重复日志
            logger.debug("JavaScript injected successfully")
        else:
            logger.warning(f"JavaScript file not found: {js_path}")
    except Exception as e:
        logger.error(f"Failed to inject JavaScript: {e}")
