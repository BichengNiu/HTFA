"""
文本处理工具模块（prep模块专用）

从共享工具库导入，保持向后兼容
"""

# 从共享工具库导入，消除模块间耦合
from dashboard.models.DFM.utils.text_utils import normalize_text, normalize_column_name

__all__ = [
    'normalize_text',
    'normalize_column_name'
]
