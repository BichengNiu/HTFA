# -*- coding: utf-8 -*-
"""
训练模块常量定义

集中管理DFM训练模块使用的所有常量和配置参数
"""

# ==================== 随机种子 ====================
RANDOM_SEED = 42  # DFM模型训练的随机种子，确保结果可重现


# ==================== 数值稳定性阈值 ====================
MIN_EPSILON = 1e-6  # 最小epsilon值，用于正定性保证
NUMERICAL_EPSILON = 1e-7  # 数值计算精度阈值
JITTER_EPSILON = 1e-4  # 卡尔曼滤波的jitter值，避免数值奇异


# ==================== 数据验证 ====================
MIN_REQUIRED_DATA_POINTS = 10  # 变量需要的最小有效数据点数
MIN_REQUIRED_SAMPLES_FOR_REGRESSION = 5  # OLS回归所需的最小样本数


# ==================== EM算法默认参数 ====================
DEFAULT_MAX_ITERATIONS = 30  # EM算法默认最大迭代次数
DEFAULT_TOLERANCE = 1e-6  # EM算法默认收敛容差
DEFAULT_MAX_LAGS = 1  # 因子自回归默认最大滞后阶数


# ==================== 单因子初始化参数 ====================
# 用于k=1时的A矩阵和Q矩阵初始化
SINGLE_FACTOR_A_INIT = 0.95  # 单因子AR(1)系数初始值
SINGLE_FACTOR_Q_INIT = 0.1  # 单因子状态噪声方差初始值


# ==================== 矩阵初始化参数 ====================
B_MATRIX_INIT_SCALE = 0.1  # B矩阵（冲击矩阵）初始化比例因子
DEFAULT_MATRIX_INIT = 0.1  # 默认矩阵初始化值（Q、R等）


# ==================== 因子选择方法 ====================
FACTOR_SELECTION_METHODS = ['fixed', 'cumulative', 'elbow']  # 可用的因子选择方法
DEFAULT_PCA_THRESHOLD = 0.9  # PCA累积方差阈值默认值
DEFAULT_ELBOW_THRESHOLD = 0.1  # Elbow方法边际方差阈值默认值


# ==================== 变量选择方法 ====================
VARIABLE_SELECTION_METHODS = ['backward', 'forward', 'none']  # 可用的变量选择方法


# ==================== 文件导出 ====================
DEFAULT_MODEL_COMPRESSION_LEVEL = 3  # joblib模型文件压缩级别
PICKLE_PROTOCOL = 4  # pickle协议版本（最高协议以获得最佳性能）


__all__ = [
    # 随机种子
    'RANDOM_SEED',

    # 数值稳定性
    'MIN_EPSILON',
    'NUMERICAL_EPSILON',
    'JITTER_EPSILON',

    # 数据验证
    'MIN_REQUIRED_DATA_POINTS',
    'MIN_REQUIRED_SAMPLES_FOR_REGRESSION',

    # EM算法
    'DEFAULT_MAX_ITERATIONS',
    'DEFAULT_TOLERANCE',
    'DEFAULT_MAX_LAGS',

    # 初始化参数
    'SINGLE_FACTOR_A_INIT',
    'SINGLE_FACTOR_Q_INIT',
    'B_MATRIX_INIT_SCALE',
    'DEFAULT_MATRIX_INIT',

    # 因子选择
    'FACTOR_SELECTION_METHODS',
    'DEFAULT_PCA_THRESHOLD',
    'DEFAULT_ELBOW_THRESHOLD',

    # 变量选择
    'VARIABLE_SELECTION_METHODS',

    # 文件导出
    'DEFAULT_MODEL_COMPRESSION_LEVEL',
    'PICKLE_PROTOCOL',
]
