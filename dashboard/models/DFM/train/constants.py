# -*- coding: utf-8 -*-
"""
DFM训练模块常量定义

集中管理所有magic number和配置常量
"""

# ==================== DFM模型常量 ====================

# 随机种子
DFM_RANDOM_SEED = 42

# 数据质量要求
MIN_DATA_POINTS_PER_VARIABLE = 10  # 每个变量最少需要的有效数据点数
MIN_FACTORS_FOR_REGRESSION = 2  # 回归至少需要的因子数

# ==================== AR(1)模型默认参数 ====================

# 单因子AR(1)默认系数
DEFAULT_AR1_COEFFICIENT = 0.95

# 单因子Q矩阵默认方差
DEFAULT_Q_VARIANCE = 0.1

# B矩阵默认缩放系数
DEFAULT_B_SCALE = 0.1

# ==================== 业务逻辑常量 ====================

# 季节性掩码月份（春节因素）
SEASONAL_MASK_MONTHS = [1, 2]

# ==================== 数值稳定性常量 ====================

# 正定性检查的最小特征值
MIN_EIGENVALUE_EPSILON = 1e-7

# 零标准差的替代值
ZERO_STD_REPLACEMENT = 1.0

# SVD fallback最小值
SVD_FALLBACK_MIN_VALUE = 1e-6

# R矩阵最小方差
R_MATRIX_MIN_VARIANCE = 1e-6
