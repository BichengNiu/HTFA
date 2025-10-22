# -*- coding: utf-8 -*-
"""
动态因子模型核心实现

实现基于EM算法的DFM估计
参考: dashboard/DFM/train_model/DynamicFactorModel.py
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
from sklearn.decomposition import PCA
from dashboard.DFM.train_ref.core.kalman import KalmanFilter, kalman_filter
from dashboard.DFM.train_ref.core.estimator import (
    estimate_loadings,
    estimate_transition_matrix,
    estimate_covariance_matrices,
    _ensure_positive_definite
)
from dashboard.DFM.train_ref.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class DFMResults:
    """DFM模型估计结果"""
    factors: pd.DataFrame          # 平滑因子估计
    loadings: np.ndarray           # 因子载荷矩阵
    transition_matrix: np.ndarray  # 状态转移矩阵
    process_noise_cov: np.ndarray  # 过程噪声协方差 Q
    measurement_noise_cov: np.ndarray  # 观测噪声协方差 R
    loglikelihood: float           # 对数似然
    n_iter: int                    # 实际迭代次数
    converged: bool                # 是否收敛


class DFMModel:
    """动态因子模型

    使用EM算法估计DFM模型参数

    状态空间表示:
        观测方程: Z_t = Lambda * F_t + eps_t
        状态方程: F_t = A * F_{t-1} + eta_t

    Args:
        n_factors: 因子数量
        max_lags: 因子自回归最大滞后阶数
        max_iter: EM算法最大迭代次数
        tolerance: 收敛容忍度
    """

    def __init__(
        self,
        n_factors: int,
        max_lags: int = 1,
        max_iter: int = 30,
        tolerance: float = 1e-6
    ):
        self.n_factors = n_factors
        self.max_lags = max_lags
        self.max_iter = max_iter
        self.tolerance = tolerance

        self.results_: Optional[DFMResults] = None

    def fit(
        self,
        data: pd.DataFrame,
        train_end: Optional[str] = None
    ) -> DFMResults:
        """拟合DFM模型

        Args:
            data: 观测数据 (时间 × 变量)
            train_end: 训练期结束日期

        Returns:
            DFMResults: 拟合结果
        """
        # 设置确定性随机种子（匹配老代码）
        DFM_SEED = 42
        np.random.seed(DFM_SEED)
        import random
        random.seed(DFM_SEED)
        logger.info(f"设置随机种子: {DFM_SEED}")

        logger.info(f"开始DFM拟合: 数据{data.shape}, 因子数={self.n_factors}, 滞后={self.max_lags}")

        Z_orig = data.copy()

        if train_end:
            Z_train = data.loc[:train_end]
            logger.info(f"使用训练期数据初始化: {Z_train.shape}")
        else:
            Z_train = data

        # 数据预处理：计算中心化和标准化数据（使用训练期参数）
        obs_centered, Z_standardized_full, means, stds = self._preprocess_data(Z_train, data)

        # 仅用训练期数据进行PCA
        if train_end:
            Z_for_pca = Z_standardized_full[:len(Z_train)]
            obs_centered_for_pca = obs_centered.iloc[:len(Z_train)]
        else:
            Z_for_pca = Z_standardized_full
            obs_centered_for_pca = obs_centered

        # PCA初始化：得到因子、载荷和V矩阵（用于R矩阵计算）
        initial_factors, initial_loadings, V = self._initialize_factors_pca(
            Z_for_pca, obs_centered_for_pca, means, stds
        )

        self.results_ = self._em_algorithm(
            obs_centered,  # 使用中心化数据进行EM算法
            initial_factors,
            initial_loadings,
            V,  # V矩阵用于R矩阵计算
            stds,
            data.index,
            Z_train.index
        )

        logger.info(f"DFM拟合完成: 迭代{self.results_.n_iter}次, "
                   f"收敛={self.results_.converged}, "
                   f"LogLik={self.results_.loglikelihood:.2f}")

        return self.results_

    def _preprocess_data(
        self,
        train_data: pd.DataFrame,
        full_data: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """数据预处理：中心化和标准化（匹配老代码实现）

        Args:
            train_data: 训练期数据（用于计算均值和标准差）
            full_data: 完整数据（需要处理的数据），如果为None则使用train_data

        Returns:
            Tuple: (中心化数据DataFrame, 标准化数据ndarray, 均值, 标准差)
        """
        if full_data is None:
            full_data = train_data

        # 使用训练期数据计算均值和标准差
        means = train_data.mean(skipna=True).values
        stds = train_data.std(skipna=True).values

        # 处理零标准差
        stds = np.where(stds > 0, stds, 1.0)

        # 中心化数据（匹配老代码obs_centered）
        obs_centered = full_data - means

        # 标准化数据并填充NaN为0（匹配老代码的z）
        Z_standardized = (obs_centered / stds).fillna(0).values

        logger.debug(f"数据预处理完成: centered={obs_centered.shape}, standardized={Z_standardized.shape}")

        return obs_centered, Z_standardized, means, stds

    def _initialize_factors_pca(
        self,
        Z_standardized: np.ndarray,
        obs_centered: pd.DataFrame,
        means: np.ndarray,
        stds: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """使用PCA初始化因子（完全匹配老代码的SVD实现）

        Args:
            Z_standardized: 标准化的观测数据 (n_time, n_obs) - 用于PCA
            obs_centered: 中心化的观测数据 (n_time, n_obs) - 用于计算载荷
            means: 均值向量
            stds: 标准差向量

        Returns:
            Tuple: (初始因子DataFrame, 初始载荷矩阵, V矩阵)
        """
        # 使用SVD分解（匹配老代码）
        U, s, Vh = np.linalg.svd(Z_standardized, full_matrices=False)

        # 初始因子 F0 = U_k * S_k （匹配老代码）
        factors_init = U[:, :self.n_factors] * s[:self.n_factors]

        factors_df = pd.DataFrame(
            factors_init,
            columns=[f'Factor{i+1}' for i in range(self.n_factors)]
        )

        # V矩阵（用于R矩阵计算）
        V = Vh.T  # (n_obs, n_obs)

        # 关键修改：使用中心化数据（而非标准化数据）计算载荷
        # 这与老代码完全一致：calculate_factor_loadings(obs_centered, factors_init_df)
        from dashboard.DFM.train_ref.core.estimator import estimate_loadings

        initial_loadings = estimate_loadings(
            obs_centered,  # 使用中心化数据（匹配老代码）
            factors_df
        )

        logger.info(f"PCA初始化完成: 因子形状={factors_init.shape}, 载荷形状={initial_loadings.shape}")

        return factors_df, initial_loadings, V

    def _em_algorithm(
        self,
        obs_centered: pd.DataFrame,
        initial_factors: pd.DataFrame,
        initial_loadings: np.ndarray,
        V: np.ndarray,
        stds: np.ndarray,
        index: pd.DatetimeIndex,
        train_index: pd.DatetimeIndex
    ) -> DFMResults:
        """EM算法估计DFM参数（匹配老代码）

        Args:
            obs_centered: 中心化观测数据（DataFrame）
            initial_factors: 初始因子估计
            initial_loadings: 初始载荷矩阵
            V: SVD分解得到的V矩阵（用于R矩阵计算）
            stds: 标准差向量（用于计算R矩阵）
            index: 完整数据时间索引
            train_index: 训练期时间索引

        Returns:
            DFMResults: 估计结果
        """
        n_time, n_obs = obs_centered.shape
        Z = obs_centered.values.T  # 转换为(n_obs, n_time)供Kalman滤波使用
        n_states = self.n_factors * self.max_lags

        factors_current = initial_factors.copy()

        # 使用初始载荷矩阵
        Lambda = initial_loadings.copy()

        # 估计A矩阵和Q矩阵
        if self.n_factors == 1:
            # 对于单因子情况，使用AR模型
            from statsmodels.tsa.api import AutoReg
            ar_model = AutoReg(factors_current.iloc[:, 0].dropna(), lags=self.max_lags, trend='n')
            ar_results = ar_model.fit()

            if self.max_lags == 1:
                A = np.array([[ar_results.params.iloc[0]]])  # (1, 1) 矩阵
            else:
                # 构造companion form
                n_states = self.max_lags
                A = np.zeros((n_states, n_states))
                A[0, :] = ar_results.params.iloc[:self.max_lags].values
                if self.max_lags > 1:
                    A[1:, :-1] = np.eye(self.max_lags - 1)

            # 使用AR残差计算Q矩阵
            Q = np.array([[ar_results.sigma2]])
            if self.max_lags > 1:
                Q_full = np.zeros((self.max_lags, self.max_lags))
                Q_full[0, 0] = ar_results.sigma2
                Q = Q_full
        else:
            # 对于多因子情况，使用VAR模型（完全匹配老代码）
            from statsmodels.tsa.api import VAR
            var_model = VAR(factors_current.dropna())
            var_results = var_model.fit(self.max_lags)

            # 使用VAR系数初始化A矩阵（完全匹配老代码line 322）
            if self.max_lags == 1:
                A = var_results.coefs[0]
            else:
                # VAR(p): 构造companion form矩阵
                n_factors_orig = factors_current.shape[1]
                A = np.zeros((n_factors_orig * self.max_lags, n_factors_orig * self.max_lags))
                # 填充VAR系数
                for lag in range(self.max_lags):
                    A[:n_factors_orig, lag*n_factors_orig:(lag+1)*n_factors_orig] = var_results.coefs[lag]
                # 构造companion form下半部分
                if self.max_lags > 1:
                    A[n_factors_orig:, :-n_factors_orig] = np.eye(n_factors_orig * (self.max_lags - 1))

            # 使用VAR残差计算初始Q矩阵（匹配老代码）
            Q = np.cov(var_results.resid, rowvar=False)
            Q = np.diag(np.maximum(np.diag(Q), 1e-6))

        # 计算R矩阵（匹配老代码：psi_diag * obs_std^2）
        # 只使用训练期数据计算R矩阵
        obs_centered_for_R = obs_centered.loc[train_index] if len(train_index) < len(index) else obs_centered
        R = self._compute_R_matrix(initial_factors.values, V, stds, obs_centered_for_R)

        loglik_prev = -np.inf
        converged = False

        # 初始化Kalman滤波的初始状态（匹配老代码）
        x0 = np.zeros(n_states)
        P0 = np.eye(n_states)

        # 生成外部shock矩阵U（匹配老代码默认行为：使用零矩阵）
        # 老代码默认 error='False'，即使用零冲击
        # 如果需要随机冲击，应在调用时明确指定
        U = np.zeros((n_states, n_time))

        # DEBUG: 打印第一次Kalman滤波前的参数
        print(f"\n[DEBUG] 新代码进入第1次Kalman滤波前的参数:")
        print(f"  obs_centered[:3, :3] =\n{obs_centered.iloc[:3, :3].values}")
        print(f"  Z.shape = {Z.shape}, dtype = {Z.dtype}")
        print(f"  U.shape = {U.shape}, dtype = {U.dtype}, 全为0: {np.allclose(U, 0)}")
        print(f"  Lambda.shape = {Lambda.shape}, dtype = {Lambda.dtype}")
        print(f"  Lambda[:3, :] =\n{Lambda[:3, :]}")
        print(f"  A.dtype = {A.dtype}, Q.dtype = {Q.dtype}, R.dtype = {R.dtype}")
        print(f"  A =\n{A}")
        print(f"  Q_diag = {np.diag(Q)}")
        print(f"  R_diag[:5] = {np.diag(R)[:5]}")
        print(f"  x0 = {x0}, dtype = {x0.dtype}")
        print(f"  P0_diag = {np.diag(P0)}, dtype = {P0.dtype}")
        print(f"  n_states = {n_states}, n_obs = {n_obs}, n_factors = {self.n_factors}")

        for iteration in range(self.max_iter):
            logger.debug(f"EM迭代 {iteration + 1}/{self.max_iter}")

            H = np.zeros((n_obs, n_states))
            H[:, :self.n_factors] = Lambda

            # B矩阵：匹配老代码line 393: B_current = np.eye(n_factors) * 0.1
            B = np.eye(n_states) * 0.1

            # DEBUG: 打印第一次迭代的H和B矩阵
            if iteration == 0:
                print(f"\n[DEBUG] 新代码第1次迭代构造H和B矩阵:")
                print(f"  H.shape = {H.shape}")
                print(f"  H[:3, :] =\n{H[:3, :]}")
                print(f"  H是否等于Lambda: {np.allclose(H, Lambda) if H.shape == Lambda.shape else 'shape不同'}")
                print(f"  B.shape = {B.shape}, dtype = {B.dtype}")
                print(f"  B =\n{B}")
                print(f"\n[DEBUG] 新代码调用Kalman滤波器前的数据校验:")
                print(f"  Z.sum() = {Z.sum():.15f}")
                print(f"  Z[:, 0] = {Z[:, 0]}")
                print(f"  Z[:, 1] = {Z[:, 1]}")
                print(f"  R对角线[:5]: {np.diag(R)[:5]}")

            kf = KalmanFilter(A, B, H, Q, R, x0, P0)
            filter_result = kf.filter(Z, U)  # Z已经是(n_obs, n_time)格式
            smoother_result = kf.smooth(filter_result)

            loglik_current = filter_result.loglikelihood

            if iteration > 0:
                loglik_diff = loglik_current - loglik_prev
                logger.debug(f"  LogLik: {loglik_current:.2f} (增量: {loglik_diff:.4f})")

                if abs(loglik_diff) < self.tolerance:
                    logger.info(f"EM算法收敛于迭代{iteration + 1}")
                    converged = True
                    break

            loglik_prev = loglik_current

            factors_smoothed = smoother_result.x_smoothed[:self.n_factors, :].T
            factors_df = pd.DataFrame(
                factors_smoothed,
                columns=[f'Factor{i+1}' for i in range(self.n_factors)]
            )

            # DEBUG: 打印第一次迭代的E步结果
            if iteration == 0:
                print(f"\n[DEBUG] 新代码第1次迭代E步平滑因子前3行:")
                print(factors_smoothed[:3])

            # 使用中心化数据（而非标准化数据）估计载荷
            Lambda = estimate_loadings(
                obs_centered,  # 使用中心化数据（匹配老代码）
                factors_df
            )

            A = estimate_transition_matrix(factors_smoothed, self.max_lags)

            # 估计协方差矩阵（使用中心化数据）
            Q, R = estimate_covariance_matrices(
                smoother_result,
                obs_centered,  # 使用中心化数据
                Lambda,
                self.n_factors,
                A  # 传入A矩阵用于Q矩阵计算
            )

            # 更新下一次迭代的初始状态（匹配老代码）
            x0 = smoother_result.x_smoothed[:, 0].copy()  # 第一个时间点的平滑状态
            P0 = smoother_result.P_smoothed[:, :, 0].copy()  # 第一个时间点的平滑协方差

            factors_current = factors_df

        factors_final = pd.DataFrame(
            factors_current.values,
            index=index,
            columns=[f'Factor{i+1}' for i in range(self.n_factors)]
        )

        return DFMResults(
            factors=factors_final,
            loadings=Lambda,
            transition_matrix=A,
            process_noise_cov=Q,
            measurement_noise_cov=R,
            loglikelihood=loglik_current,
            n_iter=iteration + 1,
            converged=converged
        )

    def _compute_R_matrix(
        self,
        factors: np.ndarray,
        V: np.ndarray,
        stds: np.ndarray,
        obs_centered: pd.DataFrame
    ) -> np.ndarray:
        """计算R矩阵（完全匹配老代码DynamicFactorModel.py line 357-382）

        Args:
            factors: 因子矩阵 (n_time, n_factors)
            V: SVD的V矩阵 (n_obs, n_obs)
            stds: 标准差向量 (n_obs,)
            obs_centered: 中心化观测数据 (n_time, n_obs)

        Returns:
            R矩阵 (n_obs, n_obs)
        """
        # 匹配老代码line 357-382的实现
        # 计算标准化数据的重构和残差
        z_standardized = (obs_centered / stds).fillna(0).values  # (n_time, n_obs)

        # 重构标准化数据：factors @ V[:, :n_factors].T
        reconstructed_z = factors @ V[:, :self.n_factors].T  # (n_time, n_obs)
        residuals_z = z_standardized - reconstructed_z

        # R矩阵：标准化残差的方差 * 原始标准差的平方
        psi_diag = np.nanvar(residuals_z, axis=0)  # 标准化残差的方差
        original_std_sq = stds ** 2  # 原始标准差的平方
        R_diag_current = psi_diag * original_std_sq  # 恢复到原始尺度
        R_diag_current = np.maximum(R_diag_current, 1e-6)  # 确保正定性

        return np.diag(R_diag_current)


def fit_dfm(
    data: pd.DataFrame,
    n_factors: int,
    max_lags: int = 1,
    max_iter: int = 30,
    train_end: Optional[str] = None
) -> DFMResults:
    """拟合DFM模型的函数接口

    Args:
        data: 观测数据
        n_factors: 因子数量
        max_lags: 最大滞后阶数
        max_iter: 最大迭代次数
        train_end: 训练期结束日期

    Returns:
        DFMResults: 拟合结果
    """
    model = DFMModel(
        n_factors=n_factors,
        max_lags=max_lags,
        max_iter=max_iter
    )

    return model.fit(data, train_end)
