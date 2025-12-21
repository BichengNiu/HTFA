# -*- coding: utf-8 -*-
"""
配置文件 - DFM模块配置
"""


class UIDefaults:
    """UI相关的默认配置"""
    NUM_COLS_INDUSTRY = 3
    VARIABLE_SELECTION_OPTIONS = {
        'none': "无筛选 (使用全部已选变量)",
        'global_backward': "全局后向剔除 (在已选变量中筛选)"
    }
    FACTOR_SELECTION_STRATEGY_OPTIONS = {
        'information_criteria': "信息准则",
        'fixed_number': "固定因子数",
        'cumulative_variance': "累积方差贡献"
    }
    INFORMATION_CRITERION_OPTIONS = {
        'bic': "BIC",
        'aic': "AIC",
        'hqc': "HQC"
    }
    EM_CONVERGENCE_CRITERION_OPTIONS = {
        'params': "参数变化",
        'likelihood': "似然函数"
    }
    IC_MAX_FACTORS_DEFAULT = 10
    MAX_ITERATIONS_DEFAULT = 30
    MAX_ITERATIONS_MIN = 1
    MAX_ITERATIONS_STEP = 10
    CUM_VARIANCE_MIN = 0.1


class TrainDefaults:
    """训练相关的默认配置"""
    VARIABLE_SELECTION_METHOD = 'none'
    FACTOR_SELECTION_STRATEGY = 'information_criteria'
    EM_MAX_ITER = 100
    FIXED_NUMBER_OF_FACTORS = 3
    CUM_VARIANCE_THRESHOLD = 0.8
    FACTOR_AR_ORDER = 1
    INFORMATION_CRITERION = 'bic'
    EM_CONVERGENCE_CRITERION = 'params'
    EM_TOLERANCE = 1e-6
    IC_MAX_FACTORS = 10
    K_FACTORS_RANGE_MIN = 1
    TRAINING_YEARS_BACK = 5
    VALIDATION_END_YEAR = 2024
    VALIDATION_END_MONTH = 12
    VALIDATION_END_DAY = 31
