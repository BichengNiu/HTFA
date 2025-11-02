# -*- coding: utf-8 -*-
"""
配置缓存器 - 简化版
实现配置文件的智能缓存，移除双层缓存
"""

import streamlit as st
import os
import json
from typing import Any, Optional, Callable
from pathlib import Path


class ConfigCache:
    """智能配置缓存器 - 简化版"""

    def __init__(self, cache_ttl: int = 3600):
        """
        初始化配置缓存器

        Args:
            cache_ttl: 缓存生存时间（秒）
        """
        self.cache_ttl = cache_ttl
        self.stats = {
            'loads': 0,
            'errors': 0
        }

    @st.cache_data(ttl=3600)
    def load_config(_self, config_name: str, file_path: str = None,
                   loader_func: Callable = None, force_reload: bool = False) -> Optional[Any]:
        """
        加载配置（带缓存）

        Args:
            config_name: 配置名称
            file_path: 配置文件路径
            loader_func: 自定义加载函数
            force_reload: 是否强制重新加载

        Returns:
            配置数据
        """
        try:
            # 使用自定义加载函数或默认加载
            if loader_func:
                data = loader_func(file_path) if file_path else loader_func()
            elif file_path:
                data = _self._load_config_file(file_path)
            else:
                raise ValueError("必须提供file_path或loader_func")

            _self.stats['loads'] += 1
            return data

        except Exception as e:
            _self.stats['errors'] += 1
            return None

    def _load_config_file(self, file_path: str) -> Any:
        """默认配置文件加载器"""
        file_ext = Path(file_path).suffix.lower()

        with open(file_path, 'r', encoding='utf-8') as f:
            if file_ext == '.json':
                return json.load(f)
            elif file_ext in ['.yml', '.yaml']:
                import yaml
                return yaml.safe_load(f)
            else:
                return f.read()

    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        return {
            'total_loads': self.stats['loads'],
            'total_errors': self.stats['errors']
        }


@st.cache_resource
def get_config_cache():
    """获取全局配置缓存实例"""
    return ConfigCache()
