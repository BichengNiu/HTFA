# -*- coding: utf-8 -*-
"""
卡尔曼滤波器模块

实现离散卡尔曼滤波和平滑算法
参考: dashboard/DFM/train_model/DiscreteKalmanFilter.py
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional
from dashboard.DFM.train_ref.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class KalmanFilterResult:
    """卡尔曼滤波结果"""
    x_filtered: np.ndarray      # 滤波状态估计
    P_filtered: np.ndarray      # 滤波协方差
    x_predicted: np.ndarray     # 预测状态估计
    P_predicted: np.ndarray     # 预测协方差
    loglikelihood: float        # 对数似然
    innovation: np.ndarray      # 新息序列


@dataclass
class KalmanSmootherResult:
    """卡尔曼平滑结果"""
    x_smoothed: np.ndarray      # 平滑状态估计
    P_smoothed: np.ndarray      # 平滑协方差
    P_lag_smoothed: np.ndarray  # 滞后协方差


class KalmanFilter:
    """卡尔曼滤波器

    状态空间模型:
        x_{t+1} = A * x_t + B * u_t + w_t,  w_t ~ N(0, Q)
        z_t = H * x_t + v_t,                v_t ~ N(0, R)

    Args:
        A: 状态转移矩阵
        B: 控制矩阵
        H: 观测矩阵
        Q: 过程噪声协方差
        R: 观测噪声协方差
        x0: 初始状态
        P0: 初始协方差
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray
    ):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x0 = x0
        self.P0 = P0

        self.n_states = A.shape[0]
        self.n_obs = H.shape[0]

    def filter(
        self,
        Z: np.ndarray,
        U: Optional[np.ndarray] = None
    ) -> KalmanFilterResult:
        """卡尔曼滤波（完全匹配train_model实现）

        Args:
            Z: 观测序列 (n_time, n_obs) - 匹配train_model格式
            U: 控制输入 (n_time, n_control) - 匹配train_model格式

        Returns:
            KalmanFilterResult: 滤波结果
        """
        import scipy.linalg

        # 输入检查：确保所有状态空间矩阵有效
        matrices_to_check = {
            'A': self.A, 'B': self.B, 'H': self.H,
            'Q': self.Q, 'R': self.R, 'x0': self.x0, 'P0': self.P0
        }
        for name, mat in matrices_to_check.items():
            if not np.all(np.isfinite(mat)):
                nan_count = np.sum(np.isnan(mat))
                inf_count = np.sum(np.isinf(mat))
                raise ValueError(
                    f"Kalman滤波器初始化失败：矩阵{name}包含{nan_count}个NaN和{inf_count}个Inf。"
                    f"形状: {mat.shape}"
                )

        n_time = Z.shape[0]

        if U is None:
            U = np.zeros((n_time, self.B.shape[1]))

        # 注意：为了匹配老代码，x和P的索引从0开始，但x[0]是初始状态
        x_filt = np.zeros((n_time, self.n_states))
        P_filt = [np.zeros((self.n_states, self.n_states)) for _ in range(n_time)]

        x_pred = np.zeros((n_time, self.n_states))
        P_pred = [np.zeros((self.n_states, self.n_states)) for _ in range(n_time)]

        innovation = np.zeros((n_time, self.n_obs))
        loglikelihood = 0.0

        # 初始化
        x_filt[0, :] = self.x0
        P_filt[0] = self.P0.copy()
        x_pred[0, :] = self.x0
        P_pred[0] = self.P0.copy()

        # 从t=1开始循环（匹配老代码从i=1开始）
        for t in range(1, n_time):
            # 获取有效观测的索引
            ix = np.where(~np.isnan(Z[t, :]))[0]

            # 先执行预测步（使用t-1时刻的滤波结果）
            x_pred[t, :] = self.A @ x_filt[t-1, :] + self.B @ U[t, :]
            P_pred_raw = self.A @ P_filt[t-1] @ self.A.T + self.Q
            # 添加jitter保证数值稳定性
            p_jitter = np.eye(self.n_states) * 1e-6
            P_pred[t] = P_pred_raw + p_jitter

            # DEBUG: 输出第1个时间步的预测步结果
            if t == 1:
                print(f"\n[KALMAN-DEBUG] 新代码 t={t} 预测步:")
                print(f"  x_filt[{t-1},:] = {x_filt[t-1, :]}")
                print(f"  U[{t},:] = {U[t, :]}")
                print(f"  x_pred[{t},:] = {x_pred[t, :]}")
                print(f"  P_filt[{t-1}] diag = {np.diag(P_filt[t-1])}")
                print(f"  P_pred_raw diag = {np.diag(P_pred_raw)}")
                print(f"  P_pred[{t}] diag = {np.diag(P_pred[t])}")

            if len(ix) == 0:
                # 没有观测，滤波结果等于预测结果
                x_filt[t, :] = x_pred[t, :]
                P_filt[t] = P_pred[t].copy()
                innovation[t, :] = np.nan
                continue

            # 提取有效观测和对应的H、R矩阵
            z_t = Z[t, ix]
            H_t = self.H[ix, :]
            R_t = self.R[np.ix_(ix, ix)]

            # 更新步
            innov_t = z_t - H_t @ x_pred[t, :]
            innovation[t, ix] = innov_t

            S_t = H_t @ P_pred[t] @ H_t.T + R_t
            # 添加jitter保证数值稳定性
            jitter = np.eye(S_t.shape[0]) * 1e-4

            # DEBUG: 输出第1个时间步的更新步中间结果
            if t == 1:
                print(f"\n[KALMAN-DEBUG] 新代码 t={t} 更新步:")
                print(f"  有效观测数: {len(ix)}")
                print(f"  Z[{t}, ix] = {z_t[:5]}")  # 只显示前5个
                print(f"  H_t @ x_pred = {(H_t @ x_pred[t, :])[:5]}")
                print(f"  innov_t[:5] = {innov_t[:5]}")
                print(f"  S_t diag[:5] = {np.diag(S_t)[:5]}")

            try:
                # 使用scipy.linalg.solve提高数值稳定性
                # K_t = P_pred[t] @ H_t.T @ inv(S_t + jitter)
                K_t = scipy.linalg.solve((S_t + jitter).T, (P_pred[t] @ H_t.T).T, assume_a='pos').T
            except np.linalg.LinAlgError:
                logger.warning(f"时间步{t}: 新息协方差矩阵奇异，使用伪逆")
                K_t = P_pred[t] @ H_t.T @ np.linalg.pinv(S_t + jitter)

            x_filt[t, :] = x_pred[t, :] + K_t @ innov_t
            P_filt[t] = (np.eye(self.n_states) - K_t @ H_t) @ P_pred[t]
            # 对称化协方差矩阵
            P_filt[t] = (P_filt[t] + P_filt[t].T) / 2.0

            # DEBUG: 输出第1个时间步的最终滤波结果
            if t == 1:
                print(f"\n[KALMAN-DEBUG] 新代码 t={t} 滤波结果:")
                print(f"  K_t[0,:] = {K_t[0, :]}")  # 只显示第一行
                print(f"  x_filt[{t},:] = {x_filt[t, :]}")
                print(f"  P_filt[{t}] diag = {np.diag(P_filt[t])}")

            # 计算对数似然（只用于有观测的时间步）
            try:
                sign, logdet = np.linalg.slogdet(S_t)
                if sign > 0:
                    loglikelihood += -0.5 * (
                        len(ix) * np.log(2 * np.pi)
                        + logdet
                        + innov_t.T @ np.linalg.solve(S_t, innov_t)
                    )
            except Exception as e:
                logger.warning(f"时间步{t}: 似然计算失败 - {e}")

        # 转换P_filt和P_pred为3D数组以保持接口一致
        P_filt_array = np.array([P_filt[t] for t in range(n_time)])
        P_pred_array = np.array([P_pred[t] for t in range(n_time)])
        P_filt_array = np.transpose(P_filt_array, (1, 2, 0))
        P_pred_array = np.transpose(P_pred_array, (1, 2, 0))

        return KalmanFilterResult(
            x_filtered=x_filt,
            P_filtered=P_filt_array,
            x_predicted=x_pred,
            P_predicted=P_pred_array,
            loglikelihood=loglikelihood,
            innovation=innovation
        )

    def smooth(
        self,
        filter_result: KalmanFilterResult
    ) -> KalmanSmootherResult:
        """卡尔曼平滑（RTS平滑器，完全匹配train_model实现）

        Args:
            filter_result: 滤波结果

        Returns:
            KalmanSmootherResult: 平滑结果
        """
        import scipy.linalg

        n_time = filter_result.x_filtered.shape[0]

        # filter_result已经是(n_time, n_states)格式,直接使用
        x_filt = filter_result.x_filtered  # (n_time, n_states)
        x_pred = filter_result.x_predicted  # (n_time, n_states)

        # P_filtered和P_predicted已经是(n_states, n_states, n_time)格式
        # 转换为list格式
        P_filt = [filter_result.P_filtered[:, :, t] for t in range(n_time)]
        P_pred = [filter_result.P_predicted[:, :, t] for t in range(n_time)]

        # 初始化平滑结果
        x_smooth = np.zeros((n_time, self.n_states))
        x_smooth[n_time-1, :] = x_filt[n_time-1, :]

        P_smooth = [np.zeros((self.n_states, self.n_states)) for _ in range(n_time)]
        P_smooth[n_time-1] = P_filt[n_time-1].copy()

        P_lag_smooth = [np.zeros((self.n_states, self.n_states)) for _ in range(n_time - 1)]

        # RTS平滑器 - 反向迭代
        for i in reversed(range(n_time - 1)):
            try:
                # 使用scipy.linalg.solve提高数值稳定性
                # J_i = P_filt[i] @ A.T @ inv(P_pred[i+1])
                J_i = scipy.linalg.solve(P_pred[i+1].T, (P_filt[i] @ self.A.T).T, assume_a='pos').T
            except scipy.linalg.LinAlgError:
                # 回退到伪逆
                J_i = P_filt[i] @ self.A.T @ np.linalg.pinv(P_pred[i+1])

            # 平滑状态
            delta_x = x_smooth[i+1, :] - x_pred[i+1, :]
            x_smooth[i, :] = x_filt[i, :] + J_i @ delta_x

            # 平滑协方差
            delta_P = P_smooth[i+1] - P_pred[i+1]
            P_smooth[i] = P_filt[i] + J_i @ delta_P @ J_i.T

            # 滞后协方差
            P_lag_smooth[i] = J_i @ P_smooth[i+1]

        # 转换回(n_states, n_time)格式
        x_smooth = x_smooth.T

        # 转换P_smooth为3D数组
        P_smooth_array = np.array([P_smooth[t] for t in range(n_time)])
        P_smooth_array = np.transpose(P_smooth_array, (1, 2, 0))

        # 转换P_lag_smooth为3D数组
        P_lag_smooth_array = np.array([P_lag_smooth[t] for t in range(n_time - 1)])
        P_lag_smooth_array = np.transpose(P_lag_smooth_array, (1, 2, 0))

        return KalmanSmootherResult(
            x_smoothed=x_smooth,
            P_smoothed=P_smooth_array,
            P_lag_smoothed=P_lag_smooth_array
        )


def kalman_filter(
    Z: np.ndarray,
    U: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray
) -> KalmanFilterResult:
    """卡尔曼滤波函数接口

    Args:
        Z: 观测序列
        U: 控制输入
        A: 状态转移矩阵
        B: 控制矩阵
        H: 观测矩阵
        Q: 过程噪声协方差
        R: 观测噪声协方差
        x0: 初始状态
        P0: 初始协方差

    Returns:
        KalmanFilterResult: 滤波结果
    """
    kf = KalmanFilter(A, B, H, Q, R, x0, P0)
    return kf.filter(Z, U)


def kalman_smoother(
    filter_result: KalmanFilterResult,
    A: np.ndarray,
    Q: np.ndarray
) -> KalmanSmootherResult:
    """卡尔曼平滑函数接口

    Args:
        filter_result: 滤波结果
        A: 状态转移矩阵
        Q: 过程噪声协方差

    Returns:
        KalmanSmootherResult: 平滑结果
    """
    n_states = A.shape[0]
    B = np.eye(n_states)
    H = np.eye(n_states)
    R = np.eye(n_states)
    x0 = np.zeros(n_states)
    P0 = np.eye(n_states)

    kf = KalmanFilter(A, B, H, Q, R, x0, P0)
    return kf.smooth(filter_result)
