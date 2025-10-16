"""
Monitoring Analysis Module
监测分析模块
"""

# 使用修正的路径导入工业分析模块
from dashboard.analysis.industrial.macro_analysis import render_macro_operations_tab as render_industrial_macro_tab
from dashboard.analysis.industrial.enterprise_analysis import render_enterprise_operations_tab as render_industrial_enterprise_tab

__all__ = [
    'render_industrial_macro_tab',
    'render_industrial_enterprise_tab'
]
