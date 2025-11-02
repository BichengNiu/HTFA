"""
工业分析通用UI组件
"""

from dashboard.analysis.industrial.ui.common.file_upload import IndustrialFileUploadComponent
from dashboard.analysis.industrial.ui.common.time_selector import IndustrialTimeRangeSelectorComponent
from dashboard.analysis.industrial.ui.common.group_detail import IndustrialGroupDetailsComponent
from dashboard.analysis.industrial.ui.common.welcome import IndustrialWelcomeComponent

__all__ = [
    'IndustrialFileUploadComponent',
    'IndustrialTimeRangeSelectorComponent',
    'IndustrialGroupDetailsComponent',
    'IndustrialWelcomeComponent'
]
