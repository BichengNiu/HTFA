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

    # 因子选择策略
    FACTOR_STRATEGIES = {
        'information_criteria': "信息准则",
        'fixed_number': "固定因子数",
        'cumulative_variance': "累积方差贡献"
    }
    DEFAULT_FACTOR_STRATEGY = 'fixed_number'

    # 信息准则选项
    INFORMATION_CRITERIA = {
        'bic': "BIC",
        'aic': "AIC",
        'hqc': "HQC"
    }
    DEFAULT_IC = 'bic'
    IC_MAX_FACTORS = 10

    # 因子数配置
    DEFAULT_K_FACTORS = 4
    K_FACTORS_MIN = 1
    K_FACTORS_MAX = 20

    # 累积方差配置
    DEFAULT_CUM_VARIANCE = 0.85
    CUM_VARIANCE_MIN = 0.1
    CUM_VARIANCE_MAX = 0.99
    CUM_VARIANCE_STEP = 0.05

    # 变量选择方法
    VARIABLE_SELECTION_METHODS = {
        'none': "无筛选 (使用全部已选变量)",
        'global_backward': "全局后向剔除 (在已选变量中筛选)"
    }
    DEFAULT_VAR_SELECTION = 'global_backward'

    # EM算法配置
    EM_MAX_ITERATIONS = 100
    EM_MIN_ITERATIONS = 10
    EM_TOLERANCE = 1e-6
    EM_CONVERGENCE_CRITERIA = {
        'params': "参数变化",
        'likelihood': "似然函数"
    }
    DEFAULT_EM_CONVERGENCE = 'params'

    # 缺失值处理方法
    MISSING_VALUE_METHODS = {
        'interpolate': "线性插值",
        'forward_fill': "前向填充",
        'drop': "删除缺失"
    }
    DEFAULT_MISSING_METHOD = 'interpolate'

    # 因子AR阶数
    DEFAULT_FACTOR_AR_ORDER = 1
    FACTOR_AR_ORDER_MIN = 1
    FACTOR_AR_ORDER_MAX = 4

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
    def get_factor_config(cls, strategy: str) -> Dict:
        """根据策略获取因子配置"""
        if strategy == 'information_criteria':
            return {
                'strategy': strategy,
                'max_factors': cls.IC_MAX_FACTORS,
                'criterion': cls.DEFAULT_IC
            }
        elif strategy == 'fixed_number':
            return {
                'strategy': strategy,
                'k_factors': cls.DEFAULT_K_FACTORS
            }
        elif strategy == 'cumulative_variance':
            return {
                'strategy': strategy,
                'threshold': cls.DEFAULT_CUM_VARIANCE
            }
        else:
            raise ValueError(f"未知的因子选择策略: {strategy}")
