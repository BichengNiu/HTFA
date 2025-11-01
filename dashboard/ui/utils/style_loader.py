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
    """注入样式到Streamlit应用"""
    loader = get_style_loader()
    loader.inject_styles(css_file)
