# -*- coding: utf-8 -*-
"""
配置文件 - DFM模块配置
"""
import os

# 全局配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOWCAST_MODEL_INPUT_DIR = 'models'
NOWCAST_MODEL_FILENAME = 'final_dfm_model.joblib'
NOWCAST_METADATA_FILENAME = 'final_model_metadata.pkl'
NOWCAST_EVOLUTION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'news_analysis_output')

# 目标变量和频率
TARGET_VARIABLE = '规模以上工业增加值:当月同比'
TARGET_FREQ = 'M'

class DataDefaults:
    """数据相关的默认配置"""
    TARGET_VARIABLE = '规模以上工业增加值:当月同比'
    TARGET_FREQ = 'M'
    DEFAULT_START_DATE = '2010-01-31'
    DEFAULT_END_DATE = '2025-07-03'

class NewsAnalysisDefaults:
    """新闻分析相关的默认配置"""
    DEFAULT_MODEL_FREQUENCY = 'MS'  # Month Start frequency
    DEFAULT_TARGET_MONTH = None
    DEFAULT_PLOT_START_DATE = None
    DEFAULT_PLOT_END_DATE = None

class ModelDefaults:
    """模型相关的默认配置"""
    DEFAULT_N_FACTORS = 3
    DEFAULT_MAX_ITER = 100
    DEFAULT_TOL = 1e-4

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
    MISSING_VALUE_OPTIONS = {
        'interpolate': "线性插值",
        'forward_fill': "前向填充",
        'drop': "删除缺失"
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
    MISSING_VALUE_METHOD = 'interpolate'
    IC_MAX_FACTORS = 10
    K_FACTORS_RANGE_MIN = 1
    TRAINING_YEARS_BACK = 5
    VALIDATION_END_YEAR = 2024
    VALIDATION_END_MONTH = 12
    VALIDATION_END_DAY = 31