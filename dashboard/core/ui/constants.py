# -*- coding: utf-8 -*-
"""
UI组件库常量定义
"""

from enum import Enum
from typing import Dict, List, Any
import time
from functools import lru_cache

class NavigationLevel(Enum):
    """导航层级枚举"""
    MAIN_MODULE = "main_module"      # 第一层：主模块选择
    SUB_MODULE = "sub_module"        # 第二层：子模块选择  
    FUNCTION = "function"            # 第三层：具体功能

class UIConstants:
    """UI常量定义"""
    
    # 主模块配置
    MAIN_MODULES = {
        "数据预览": {
            "icon": "📊",
            "description": "查看和预览各类经济数据，提供工业数据的全面展示和分析",
            "sub_modules": None
        },
        "监测分析": {
            "icon": "[CHART]",
            "description": "对经济运行数据进行深度监测和分析，提供专业的分析报告",
            "sub_modules": ["工业"]
        },
        "模型分析": {
            "icon": "[MODEL]",
            "description": "使用先进的机器学习模型进行经济预测和分析",
            "sub_modules": ["DFM 模型"]
        },
        "数据探索": {
            "icon": "🔍",
            "description": "深入探索时间序列数据的统计特性和内在规律，包括平稳性分析和相关性分析",
            "sub_modules": []
        }
    }
    
    # 子模块配置
    SUB_MODULES = {}
    
    # HTML模板常量
    HTML_TEMPLATES = {
        # 子模块卡片模板
        "sub_module_card": """
        <div style="
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 3em; margin-bottom: 0.5rem;">{icon}</div>
            <h3 style="margin: 0.5rem 0; color: #333;">{title}</h3>
            <p style="color: #666; font-size: 0.9em; margin: 0;">{description}</p>
        </div>
        """,

        # 功能卡片模板
        "function_card": """
        <div style="
            border: 1px solid #f0f0f0;
            border-radius: 6px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            background: #fafafa;
            text-align: center;
            height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 2.5em; margin-bottom: 0.5rem;">{icon}</div>
            <h4 style="margin: 0.5rem 0; color: #333;">{title}</h4>
            <p style="color: #666; font-size: 0.85em; margin: 0;">{description}</p>
        </div>
        """,

        # 模块卡片模板（带悬停效果）
        "module_card": """
        <div style="
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            height: {height};
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.15)';"
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)';">
            <div style="font-size: 3em; margin-bottom: 0.5rem;">{icon}</div>
            <h3 style="margin: 0.5rem 0; color: #333; font-weight: 600;">{title}</h3>
            <p style="color: #666; font-size: 0.9em; margin: 0; line-height: 1.4;">{description}</p>
        </div>
        """
    }

    # 样式常量
    STYLES = {
        "welcome_container": """
            <div style="
                text-align: center;
                padding: 2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
                margin: 1rem 0;
                color: white;
            ">
        """,
        "module_card": """
            <div style="
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 1.5rem;
                margin: 1rem 0;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s ease;
            ">
        """,
        "feature_card": """
            <div style="
                border: 1px solid #f0f0f0;
                border-radius: 6px;
                padding: 1rem;
                margin: 0.5rem;
                background: #fafafa;
                cursor: pointer;
            ">
        """
    }
    
    # 标签页索引映射
    TAB_INDEX_MAPPING = {
        "数据探索": {
            0: "stationarity",      # 平稳性分析
            1: "time_lag_corr"      # 相关性分析（包含DTW和领先滞后）
        }
    }
    
    # 状态键名映射
    STATE_KEYS = {
        # 数据探索状态键
        "data_exploration": {
            "active_tab": "data_exploration_active_tab",
            "tab_flags": {
                "stationarity": "currently_in_stationarity_tab",
                "time_lag_corr": "currently_in_time_lag_corr_tab"
            },
            "timestamps": {
                "stationarity": "stationarity_tab_set_time",
                "time_lag_corr": "time_lag_tab_set_time"
            }
        },
        "active_tab": "data_exploration_active_tab",
        "tab_flags": {
            "stationarity": "currently_in_stationarity_tab",
            "time_lag_corr": "currently_in_time_lag_corr_tab",
            "dtw": "currently_in_dtw_tab",
            "lead_lag": "currently_in_lead_lag_tab"
        },
        "timestamps": {
            "stationarity": "stationarity_tab_set_time",
            "time_lag_corr": "time_lag_tab_set_time",
            "dtw": "dtw_tab_set_time",
            "lead_lag": "lead_lag_tab_set_time"
        }
    }

class TemplateManager:
    """HTML模板管理器，提供缓存和格式化功能"""

    _cache = {}
    _cache_timeout = 300  # 缓存5分钟

    @classmethod
    @lru_cache(maxsize=128)
    def get_template(cls, template_name: str) -> str:
        """获取HTML模板（带缓存）"""
        if template_name not in UIConstants.HTML_TEMPLATES:
            raise ValueError(f"模板 '{template_name}' 不存在")
        return UIConstants.HTML_TEMPLATES[template_name]

    @classmethod
    def render_template(cls, template_name: str, **kwargs) -> str:
        """渲染HTML模板"""
        template = cls.get_template(template_name)
        return template.format(**kwargs)

    @classmethod
    def clear_cache(cls):
        """清除模板缓存"""
        cls.get_template.cache_clear()
        cls._cache.clear()
