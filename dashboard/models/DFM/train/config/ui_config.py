# -*- coding: utf-8 -*-
"""
UI配置管理
统一管理所有UI相关的配置和默认值
"""

from datetime import date
from typing import Dict
import pandas as pd


class UIConfig:
    """UI配置类 - 单一配置来源"""

    # 日期默认值
    DEFAULT_TRAINING_START = date(2020, 1, 1)
    DEFAULT_VALIDATION_START = date(2025, 7, 1)
    DEFAULT_OBSERVATION_START = date(2025, 12, 31)  # 观察期开始日期（原validation_end）

    # 估计方法选项
    ESTIMATION_METHODS = {
        'single_stage': '一次估计法 (推荐)',
        'two_stage': '二次估计法'
    }
    DEFAULT_ESTIMATION_METHOD = 'single_stage'

    # 目标对齐方式选项
    TARGET_ALIGNMENT_OPTIONS = {
        'next_month': "下月值 (m月预测m+1月)",
        'current_month': "本月值 (m月预测m月)"
    }
    DEFAULT_TARGET_ALIGNMENT = 'next_month'

    # 因子选择策略
    FACTOR_STRATEGIES = {
        'fixed_number': "固定因子数 (默认)",
        'cumulative_variance': "累积方差贡献",
        'kaiser': "Kaiser准则 (特征值>1)"
    }
    DEFAULT_FACTOR_STRATEGY = 'fixed_number'

    # 因子数配置
    DEFAULT_K_FACTORS = 4
    K_FACTORS_MIN = 1
    K_FACTORS_MAX = 15

    # 累积方差配置
    DEFAULT_CUM_VARIANCE = 0.8
    CUM_VARIANCE_MIN = 0.5
    CUM_VARIANCE_MAX = 0.99
    CUM_VARIANCE_STEP = 0.01

    # Kaiser准则配置
    DEFAULT_KAISER_THRESHOLD = 1.0
    KAISER_THRESHOLD_MIN = 0.5
    KAISER_THRESHOLD_MAX = 2.0
    KAISER_THRESHOLD_STEP = 0.1

    # 变量选择方法
    VARIABLE_SELECTION_METHODS = {
        'none': "无筛选 (使用全部已选变量)",
        'backward': "后向选择法 (逐步移除不重要变量)",
        'stepwise': "向前向后法 (逐步添加并检查冗余变量)"
    }
    DEFAULT_VAR_SELECTION = 'none'

    # 筛选策略选项
    SELECTION_CRITERIA = {
        'rmse': "RMSE",
        'win_rate': "胜率",
        'hybrid': "混合"
    }
    DEFAULT_SELECTION_CRITERION = 'hybrid'

    # 混合策略优先级选项
    HYBRID_PRIORITIES = {
        'win_rate_first': "胜率优先",
        'rmse_first': "RMSE优先"
    }
    DEFAULT_HYBRID_PRIORITY = 'win_rate_first'

    # 训练期权重配置 (2025-12新增)
    DEFAULT_TRAINING_WEIGHT = 50  # 默认50%训练期权重（百分比）
    TRAINING_WEIGHT_MIN = 0       # 0%=仅验证期
    TRAINING_WEIGHT_MAX = 100     # 100%=仅训练期
    TRAINING_WEIGHT_STEP = 10

    # EM算法配置
    EM_MAX_ITERATIONS = 100
    EM_MIN_ITERATIONS = 10
    EM_TOLERANCE = 1e-6

    # 因子AR阶数
    DEFAULT_FACTOR_AR_ORDER = 1
    FACTOR_AR_ORDER_MIN = 0
    FACTOR_AR_ORDER_MAX = 5

    # UI布局配置
    NUM_COLS_INDUSTRY = 3
    MAX_ITERATIONS_STEP = 10

    # 文件上传配置
    EXCEL_MAPPING_SHEET = '指标体系'
    INDICATOR_COLUMN_NAME = '指标名称'
    INDUSTRY_COLUMN_NAME = '行业'
    TYPE_COLUMN_NAME = '类型'

    @classmethod
    def get_date_defaults(cls) -> Dict[str, date]:
        """获取日期默认值字典"""
        # 计算验证期结束日期 = 观察期开始日期 - 1周
        validation_end_timestamp = pd.Timestamp(cls.DEFAULT_OBSERVATION_START) - pd.Timedelta(weeks=1)

        return {
            'training_start': cls.DEFAULT_TRAINING_START,
            'validation_start': cls.DEFAULT_VALIDATION_START,
            'validation_end': validation_end_timestamp.date()  # 观察期开始的前一周
        }

    @classmethod
    def get_safe_option_index(cls, options: Dict, value: str, default: str) -> int:
        """
        安全获取选项索引，如果值无效则返回默认值的索引

        Args:
            options: 选项字典
            value: 当前值
            default: 默认值

        Returns:
            有效的索引
        """
        keys = list(options.keys())
        if value in keys:
            return keys.index(value)
        return keys.index(default) if default in keys else 0
