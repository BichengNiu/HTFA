"""
Industrial Analysis Module
工业分析模块
"""

# 导入统一的工业分析模块
from dashboard.analysis.industrial.industrial_analysis import render_industrial_analysis

# 从重构后的模块导入
from dashboard.analysis.industrial.macro_analysis import render_macro_operations_tab
from dashboard.analysis.industrial.enterprise_analysis import render_enterprise_operations_tab

__all__ = [
    'render_industrial_analysis',
    'render_macro_operations_tab',
    'render_enterprise_operations_tab'
]
