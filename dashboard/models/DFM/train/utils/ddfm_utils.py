# -*- coding: utf-8 -*-
"""
DDFM工具函数模块

整合深度动态因子模型所需的工具函数
"""

from typing import Tuple
import numpy as np

# 常量定义
_FACTOR_ORDER_ERROR = "仅支持AR(1)或AR(2)因子动态"


def mse_missing(y_actual, y_predicted):
    """
    处理缺失数据的MSE损失函数（TensorFlow版本）

    Args:
        y_actual: 实际值张量
        y_predicted: 预测值张量

    Returns:
        MSE损失值
    """
    import tensorflow as tf

    mask = tf.where(tf.math.is_nan(y_actual), tf.zeros_like(y_actual), tf.ones_like(y_actual))
    y_actual_ = tf.where(tf.math.is_nan(y_actual), tf.zeros_like(y_actual), y_actual)
    y_predicted_ = tf.multiply(y_predicted, mask)
    return tf.reduce_mean(tf.square(y_actual_ - y_predicted_))


def convergence_checker(y_prev: np.ndarray, y_now: np.ndarray, y_actual: np.ndarray) -> Tuple[float, float]:
    """
    检查收敛性

    Args:
        y_prev: 上一次迭代的预测值
        y_now: 当前迭代的预测值
        y_actual: 实际值

    Returns:
        (相对变化量, 当前损失)
    """
    from sklearn.metrics import mean_squared_error as mse

    loss_minus = mse(y_prev[~np.isnan(y_actual)], y_actual[~np.isnan(y_actual)])
    loss = mse(y_now[~np.isnan(y_actual)], y_actual[~np.isnan(y_actual)])

    # 防止除零错误
    if loss_minus == 0:
        raise ValueError("前一次迭代损失为0，无法计算相对变化量")

    return np.abs(loss - loss_minus) / loss_minus, loss


def convert_decoder_to_numpy(decoder, has_bias: bool, factor_order: int,
                             structure_decoder: tuple = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    将Keras解码器转换为numpy矩阵

    Args:
        decoder: Keras解码器模型
        has_bias: 是否有偏置项
        factor_order: 因子自回归阶数
        structure_decoder: 解码器结构，None表示单层

    Returns:
        (偏置, 观测矩阵H)
    """
    if structure_decoder is None:
        if has_bias:
            ws, bs = decoder.get_layer(index=-1).get_weights()
        else:
            ws = decoder.get_layer(index=-1).get_weights()[0]
            bs = np.zeros(ws.shape[1])

        # 构建观测方程矩阵
        if factor_order == 2:
            emission = np.hstack((
                ws.T,  # 权重项
                np.zeros((ws.shape[1], ws.shape[0])),  # 滞后因子的零矩阵
                np.identity(ws.shape[1])  # 特质项
            ))
        elif factor_order == 1:
            emission = np.hstack((
                ws.T,  # 权重项
                np.identity(ws.shape[1])  # 特质项
            ))
        else:
            raise NotImplementedError(_FACTOR_ORDER_ERROR)
    else:
        raise NotImplementedError("非线性解码器尚未实现")

    return bs, emission


def get_transition_params(f_t: np.ndarray, eps_t: np.ndarray, factor_order: int,
                          bool_no_miss: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算状态转移方程参数

    Args:
        f_t: 公共因子
        eps_t: 特质项
        factor_order: 因子自回归阶数
        bool_no_miss: 非缺失值标记数组

    Returns:
        (转移矩阵A, 噪声协方差Q, 初始均值mu_0, 初始协方差Sigma_0, 状态序列x_t)
    """
    if factor_order == 2:
        f_past = np.hstack((f_t[1:-1, :], f_t[:-2, :]))
        A_f = (np.linalg.pinv(f_past.T @ f_past) @ f_past.T @ f_t[2:, :]).T
    elif factor_order == 1:
        f_past = f_t[:-1, :]
        A_f = (np.linalg.pinv(f_past.T @ f_past) @ f_past.T @ f_t[1:, :]).T
    else:
        raise NotImplementedError(_FACTOR_ORDER_ERROR)

    # 获取特质项的AR系数
    A_eps, _, _ = get_idio(eps_t, bool_no_miss)

    # 构建伴随形式状态 x_t = [f_t, f_{t-1}, eps_t]
    if factor_order == 2:
        x_t = np.vstack((f_t[1:, :].T, f_t[:-1, :].T, eps_t[1:, :].T))
        A = np.vstack((
            np.hstack((A_f, np.zeros((A_f.shape[0], eps_t.shape[1])))),  # VAR因子
            np.hstack((np.identity(A_f.shape[0]), np.zeros((A_f.shape[0], A_f.shape[0] + eps_t.shape[1])))),
            np.hstack((np.zeros((eps_t.shape[1], A_f.shape[1])), A_eps))  # AR(1)特质项
        ))
    elif factor_order == 1:
        x_t = np.vstack((f_t.T, eps_t.T))
        A = np.vstack((
            np.hstack((A_f, np.zeros((A_f.shape[0], eps_t.shape[1])))),  # VAR因子
            np.hstack((np.zeros((eps_t.shape[1], A_f.shape[1])), A_eps))  # AR(1)特质项
        ))
    else:
        raise NotImplementedError(_FACTOR_ORDER_ERROR)

    # 误差项协方差矩阵
    w_t = x_t[:, 1:] - A @ x_t[:, :-1]
    W = np.diag(np.diag(np.cov(w_t)))

    # 设置初始状态的无条件矩
    mu_0 = np.mean(x_t, axis=1)
    Sigma_0 = np.cov(x_t)

    # 特质项与因子不相关，特质项之间协方差为对角阵
    Sigma_0[:A_f.shape[1], A_f.shape[1]:] = 0
    Sigma_0[A_f.shape[1]:, :A_f.shape[1]] = 0
    Sigma_0[A_f.shape[1]:, A_f.shape[1]:] = np.diag(np.diag(Sigma_0[A_f.shape[1]:, A_f.shape[1]:]))

    return A, W, mu_0, Sigma_0, x_t


def get_idio(eps: np.ndarray, idx_no_missings: np.ndarray, min_obs: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    估计特质项的AR(1)参数（支持混合频率数据）

    对于不同频率的变量，在其原始频率上估计AR(1)参数：
    - 周度变量：检查相邻行是否都非缺失
    - 月度变量：检查在原始频率上相邻的观测（即两个非空观测之间没有其他非空观测）

    Args:
        eps: 特质项序列 (T, n_variables)
        idx_no_missings: 非缺失值索引 (T, n_variables)
        min_obs: 最小观测数

    Returns:
        (AR系数矩阵, 均值, 标准差)
    """
    phi = np.zeros((eps.shape[1], eps.shape[1]))
    mu_eps = np.zeros(eps.shape[1])
    std_eps = np.zeros(eps.shape[1])

    for j in range(eps.shape[1]):
        # 找出该变量的非空观测位置
        non_miss_indices = np.where(idx_no_missings[:, j])[0]

        if len(non_miss_indices) < 2:
            # 观测太少，使用默认值
            mu_eps[j] = 0.0
            std_eps[j] = 1.0
            phi[j, j] = 0.0
            continue

        # 计算非空观测之间的间隔
        gaps = np.diff(non_miss_indices)
        median_gap = np.median(gaps)

        # 判断频率：间隔<=2视为周度，否则视为月度/其他低频
        is_weekly = median_gap <= 2

        if is_weekly:
            # 周度变量：检查相邻行都非空
            to_select = idx_no_missings[:, j]
            to_select = np.hstack((np.array([False]), to_select[:-1] * to_select[1:]))
            eps_t = eps[to_select, j]
            # 构造eps[t-1]序列
            to_select_prev = np.hstack((to_select[1:], np.array([False])))
            eps_t_1 = eps[to_select_prev, j]
        else:
            # 月度/低频变量：使用原始频率上的连续观测对
            eps_t = eps[non_miss_indices[1:], j]
            eps_t_1 = eps[non_miss_indices[:-1], j]

        n_pairs = len(eps_t)

        if n_pairs >= min_obs:
            mu_eps[j] = np.mean(eps_t)
            std_eps[j] = np.std(eps_t)

            if std_eps[j] > 1e-10:  # 数值稳定性检查
                cov1_eps = np.cov(eps_t, eps_t_1)[0][1]
                phi_raw = cov1_eps / (std_eps[j] ** 2)
                # 限制phi在平稳范围内
                phi[j, j] = np.clip(phi_raw, -0.99, 0.99)
            else:
                phi[j, j] = 0.0
        else:
            # 观测不足，使用默认值
            valid_eps = eps[idx_no_missings[:, j], j]
            if len(valid_eps) > 0:
                mu_eps[j] = np.mean(valid_eps)
                std_eps[j] = np.std(valid_eps) if len(valid_eps) > 1 else 1.0
            else:
                mu_eps[j] = 0.0
                std_eps[j] = 1.0
            phi[j, j] = 0.0

    return phi, mu_eps, std_eps
