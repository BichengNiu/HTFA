# -*- coding: utf-8 -*-
"""
Preview模块集中配置
所有颜色、样式、文本、计算参数都在这里
"""

# 颜色配置
COLORS = {
    'current_year': 'red',
    'previous_year': 'blue',
    'historical_mean': 'grey',
    'historical_range': 'rgba(211, 211, 211, 0.5)',
    'positive': '#ffcdd2',
    'negative': '#c8e6c9'
}

# UI文本配置
UI_TEXT = {
    'select_industry': '选择行业大类',
    'select_type': '选择指标类型',
    'all_option': '全部',
    'download_label': '下载数据摘要 (CSV)',
    'no_data_warning': '筛选条件 "{}" 没有匹配的指标。',
    'loading_message': '正在生成 {} 的图表...',
    'empty_data': '{}数据尚未加载或为空,请返回"数据概览"模块上传数据。'
}

# 基础频率配置（最小配置，避免冗余）
_BASE_FREQUENCY_CONFIGS = {
    'weekly': {
        'display_name': '周度',
        'summary_config': {
            'sort_column': '环比上周',
            'highlight_columns': ['环比上周'],
            'percentage_columns': ['环比上周'],
        }
    },
    'monthly': {
        'display_name': '月度',
        'summary_config': {
            'sort_column': '环比上月',
            'highlight_columns': ['环比上月', '同比上年'],
            'percentage_columns': ['环比上月', '同比上年'],
        }
    },
    'daily': {
        'display_name': '日度',
        'summary_config': {
            'sort_column': '环比昨日',
            'highlight_columns': ['环比昨日', '同比上年'],
            'percentage_columns': ['环比昨日', '环比上周', '环比上月', '同比上年'],
        }
    },
    'ten_day': {
        'display_name': '旬度',
        'summary_config': {
            'sort_column': '环比上旬',
            'highlight_columns': ['环比上旬'],
            'percentage_columns': ['环比上旬'],
        }
    },
    'yearly': {
        'display_name': '年度',
        'summary_config': {
            'sort_column': '同比上年',
            'highlight_columns': ['同比上年'],
            'percentage_columns': ['同比上年'],
        }
    }
}


def _build_frequency_config(english_name: str, base_config: dict) -> dict:
    """从基础配置构建完整的频率配置（自动生成冗余字段）

    Args:
        english_name: 英文频率名
        base_config: 基础配置

    Returns:
        完整配置字典
    """
    display_name = base_config['display_name']

    # 自动生成冗余字段
    full_config = {
        'english_name': english_name,
        'display_name': display_name,
        'key_prefix': english_name,
        'df_key': f'{english_name}_df',
        'industries_key': f'{english_name}_industries',
        'empty_message': f'{display_name}数据尚未加载或为空,请返回"数据概览"模块上传数据。',
        'summary_config': {
            **base_config['summary_config'],
            'download_prefix': f'{display_name}数据摘要'
        }
    }

    return full_config


# 统一频率配置（自动从基础配置生成）
UNIFIED_FREQUENCY_CONFIGS = {
    freq_name: _build_frequency_config(freq_name, base_cfg)
    for freq_name, base_cfg in _BASE_FREQUENCY_CONFIGS.items()
}

# 绘图配置
PLOT_CONFIGS = {
    'weekly': {
        'x_range': range(1, 54),
        'x_label_func': lambda x: f"W{x}",
        'x_tick_interval': 4,
        'historical_years': 5,
        'layout': {
            'margin': {'l': 50, 'r': 30, 't': 60, 'b': 100},
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.2,
                'xanchor': 'center',
                'x': 0.5
            }
        }
    },
    'monthly': {
        'x_range': range(1, 13),
        'x_label_func': lambda x: f"{x}月",
        'x_tick_interval': 1,
        'historical_years': 5,
        'layout': {
            'margin': {'l': 50, 'r': 30, 't': 60, 'b': 100},
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.2,
                'xanchor': 'center',
                'x': 0.5
            }
        }
    },
    'daily': {
        'x_range': range(1, 367),
        'x_label_func': lambda x: f"{x}日",
        'x_tick_vals': [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
        'x_tick_labels': ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月'],
        'historical_years': 5,
        'layout': {
            'margin': {'l': 50, 'r': 30, 't': 60, 'b': 100},
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.2,
                'xanchor': 'center',
                'x': 0.5
            }
        }
    },
    'ten_day': {
        'x_range': range(1, 37),
        'x_label_func': lambda x: f"第{x}旬",
        'x_tick_vals': list(range(2, 37, 3)),
        'x_tick_labels': [f"{i}月" for i in range(1, 13)],
        'historical_years': 5,
        'layout': {
            'margin': {'l': 50, 'r': 30, 't': 60, 'b': 100},
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.2,
                'xanchor': 'center',
                'x': 0.5
            }
        }
    },
    'yearly': {
        'layout': {
            'margin': {'l': 50, 'r': 30, 't': 60, 'b': 100}
        }
    }
}

# 摘要计算配置
SUMMARY_CONFIGS = {
    'weekly': {
        'indicator_name_column': '周度指标名称',
        'date_column': '最新日期',
        'column_order': [
            '周度指标名称', '最新日期', '最新值', '上周值',
            '环比上周'
        ]
    },
    'monthly': {
        'indicator_name_column': '月度指标名称',
        'date_column': '最新月份',
        'column_order': [
            '月度指标名称', '最新月份', '最新值', '上月值',
            '环比上月', '同比上年'
        ]
    },
    'daily': {
        'indicator_name_column': '日度指标名称',
        'date_column': '最新日期',
        'column_order': [
            '日度指标名称', '最新日期', '最新值',
            '昨日值', '环比昨日',
            '上周均值', '上月均值'
        ]
    },
    'ten_day': {
        'indicator_name_column': '旬度指标名称',
        'date_column': '最新日期',
        'column_order': [
            '旬度指标名称', '最新日期', '最新值',
            '上旬值', '环比上旬'
        ]
    },
    'yearly': {
        'indicator_name_column': '年度指标名称',
        'date_column': '最新年份',
        'column_order': [
            '年度指标名称', '最新年份', '最新值', '上年值', '同比上年',
            '两年前值', '三年前值'
        ]
    }
}

# 频率显示顺序
FREQUENCY_ORDER = ['日度', '周度', '旬度', '月度', '年度']

# 中文到英文频率映射（为frequency_utils提供快速查找）
CHINESE_TO_ENGLISH_FREQ = {
    '周度': 'weekly',
    '月度': 'monthly',
    '日度': 'daily',
    '旬度': 'ten_day',
    '年度': 'yearly'
}

# 英文到中文频率映射
ENGLISH_TO_CHINESE_FREQ = {v: k for k, v in CHINESE_TO_ENGLISH_FREQ.items()}
