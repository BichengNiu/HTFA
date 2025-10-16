# -*- coding: utf-8 -*-
"""
常量定义模块

定义explore模块中使用的所有常量，避免魔法数字
"""

# ==================== 数据验证相关常量 ====================

# ADF检验所需的最小样本数
# 原因：ADF检验需要足够的样本来估计自回归模型
MIN_SAMPLES_ADF = 5

# 相关性计算所需的最小样本数
# 原因：至少需要2个点才能计算相关系数
MIN_SAMPLES_CORRELATION = 2

# KL散度计算所需的最小样本数
# 原因：需要足够样本来构建有意义的概率分布
MIN_SAMPLES_KL_DIVERGENCE = 10

# 胜率计算所需的最小样本数
# 原因：需要至少2个点才能计算变化方向
MIN_SAMPLES_WIN_RATE = 2

# ==================== KL散度相关常量 ====================

# KL散度默认分箱数
# 原因：10个分箱通常能够很好地近似连续分布
DEFAULT_KL_BINS = 10

# KL散度平滑参数
# 原因：避免log(0)和除零错误
DEFAULT_KL_SMOOTHING_ALPHA = 1e-9

# KL散度每个分箱的最小点数
# 原因：确保每个分箱有足够的样本
MIN_POINTS_PER_BIN = 2

# ==================== 时间序列相关常量 ====================

# 时间间隔容差（天）
TIMEDELTA_TOLERANCE_DAYS = {
    'Daily': (0.8, 1.2),
    'Weekly': (6, 8),
    'Monthly': (25, 35),
    'Quarterly': (85, 100),
    'Annual': (350, 380)
}

# 频率映射：从业务名称到pandas频率代码
FREQUENCY_MAPPINGS = {
    'Daily': 'D',
    'Weekly': 'W-MON',
    'Monthly': 'ME',
    'Quarterly': 'QE',
    'Annual': 'YE'
}

# 频率优先级（用于自动选择对齐频率）
FREQUENCY_PRIORITY = {
    'Daily': 1,
    'Weekly': 2,
    'Monthly': 3,
    'Quarterly': 4,
    'Annual': 5
}

# ==================== 文本消息模板 ====================

# ERROR_MESSAGES: 统一的错误消息模板
# 用于所有explore子模块，确保错误消息的一致性（DRY原则）
ERROR_MESSAGES = {
    # 数据验证相关
    'empty_series': '输入序列为空',
    'insufficient_data': '有效数据点不足',
    'not_numeric': '序列非数值类型',
    'no_variance': '序列方差为零',
    'calculation_failed': '计算失败',

    # 序列验证相关
    'target_series_invalid': '目标序列无效',
    'ref_series_invalid': '参考序列无效',
    'candidate_not_found': '候选变量未找到',

    # 分析相关
    'correlation_calc_error': '相关性计算错误',
    'kl_calc_error': 'KL散度计算错误',
    'no_common_data': '对齐后无共同有效数据点',

    # 目标相关
    'no_target_change': '目标无变化',
}

# ==================== 默认参数 ====================

# 默认聚合方法
DEFAULT_AGG_METHOD = 'mean'

# 默认标准化方法
DEFAULT_STANDARDIZATION_METHOD = 'zscore'

# 默认最大滞后阶数
DEFAULT_MAX_LAGS = 12

# 默认DTW窗口大小
DEFAULT_DTW_WINDOW = 10
