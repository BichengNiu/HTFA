# Dashboard Preview 模块初始化
# 导出主要的模块和函数

# 数据加载
from dashboard.preview.data_loader import load_and_process_data, normalize_string, LoadedIndustrialData

# 新的统一模块
from dashboard.preview.config import COLORS, UI_TEXT, UNIFIED_FREQUENCY_CONFIGS, PLOT_CONFIGS, SUMMARY_CONFIGS, FREQUENCY_ORDER
from dashboard.preview.calculators import calculate_summary
from dashboard.preview.plotting import plot_indicator
from dashboard.preview.components import create_filter_ui, display_summary_table
from dashboard.preview.tabs import display_time_series_tab, display_overview_tab
from dashboard.preview.frequency_utils import (
    get_indicator_frequencies,
    filter_indicators_by_frequency,
    iter_all_frequencies,
    create_empty_frequency_dict,
    get_all_frequency_names
)

# 主入口模块
from dashboard.preview.main import display_industrial_tabs

__all__ = [
    # 数据加载
    'load_and_process_data',
    'normalize_string',
    'LoadedIndustrialData',

    # 配置
    'COLORS',
    'UI_TEXT',
    'UNIFIED_FREQUENCY_CONFIGS',
    'PLOT_CONFIGS',
    'SUMMARY_CONFIGS',
    'FREQUENCY_ORDER',

    # 核心功能
    'calculate_summary',
    'plot_indicator',
    'create_filter_ui',
    'display_summary_table',
    'display_time_series_tab',
    'display_overview_tab',

    # 频率工具
    'get_indicator_frequencies',
    'filter_indicators_by_frequency',
    'iter_all_frequencies',
    'create_empty_frequency_dict',
    'get_all_frequency_names',

    # 主入口
    'display_industrial_tabs'
] 