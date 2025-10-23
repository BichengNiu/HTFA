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

        # 数据质量检查：过滤掉有效数据点不足的变量
        min_required_points = max(self.n_factors + 5, 20)  # 至少需要n_factors+5个数据点
        valid_counts = data.notna().sum()
        valid_vars = valid_counts[valid_counts >= min_required_points].index.tolist()

        if len(valid_vars) < len(data.columns):
            dropped_vars = set(data.columns) - set(valid_vars)
            logger.warning(f"过滤掉{len(dropped_vars)}个数据不足的变量（需要至少{min_required_points}个有效点）")
            logger.debug(f"过滤掉的变量: {list(dropped_vars)[:5]}...")  # 只显示前5个
            data = data[valid_vars]

        if len(valid_vars) < self.n_factors:
            raise ValueError(f"有效变量数（{len(valid_vars)}）少于因子数（{self.n_factors}），无法进行DFM估计")

        logger.info(f"数据质量检查完成: 保留{len(valid_vars)}个变量")

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

        # 检查Lambda是否有NaN行
        nan_rows = np.isnan(initial_loadings).any(axis=1)
        n_nan_rows = nan_rows.sum()

        if n_nan_rows > 0:
            logger.warning(f"载荷矩阵有{n_nan_rows}行包含NaN，将使用替代方法估计")

            # 对有NaN的行，使用SVD直接估计载荷
            # Lambda = V[:n_factors].T * sqrt(s[:n_factors])
            V_short = Vh[:self.n_factors, :].T  # (n_obs, n_factors)
            svd_loadings = V_short * np.sqrt(s[:self.n_factors])

            # 填充NaN行
            for i in np.where(nan_rows)[0]:
                logger.debug(f"变量{i}的载荷使用SVD估计")
                initial_loadings[i, :] = svd_loadings[i, :]

        # 最后检查：确保没有NaN或Inf
        if np.any(np.isnan(initial_loadings)) or np.any(np.isinf(initial_loadings)):
            raise ValueError(f"载荷矩阵仍包含NaN或Inf，无法继续。请检查输入数据质量。")

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
        Z = obs_centered.values  # (n_time, n_obs)格式,匹配train_model
        n_states = self.n_factors * self.max_lags

        factors_current = initial_factors.copy()

        # 使用初始载荷矩阵
        Lambda = initial_loadings.copy()

        # 初始化A矩阵和Q矩阵
        # 关键修复：k=1用固定值（匹配老代码），k>=2用VAR估计
        if self.n_factors == 1:
            # 单因子情况：使用固定初始值（匹配老代码line 354-355）
            # 不使用AutoReg估计，因为初始PCA因子可能不准确，导致算法发散
            if self.max_lags == 1:
                A = np.array([[0.95]])
                Q = np.array([[0.1]])
            else:
                # AR(p) companion form
                A = np.zeros((self.max_lags, self.max_lags))
                A[0, :] = 0.95 / self.max_lags
                if self.max_lags > 1:
                    A[1:, :-1] = np.eye(self.max_lags - 1)
                Q = np.zeros((self.max_lags, self.max_lags))
                Q[0, 0] = 0.1
        else:
            # 多因子情况：使用VAR模型估计（保持原逻辑，已验证成功）
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

        # 初始化B矩阵（匹配老代码line 393: B_current = np.eye(n_factors) * 0.1）
        B = np.eye(n_states) * 0.1

        # 生成外部shock矩阵U（完全匹配老代码默认行为）
        # 老代码有个Python陷阱：if error: 将字符串'False'当作True处理！
        # 所以即使error='False'，也会生成随机U矩阵
        # 为了保持一致性，这里也生成相同的随机U矩阵
        DFM_SEED = 42  # 匹配老代码的种子
        np.random.seed(DFM_SEED)
        # 注意：老代码U shape是(n_time, n_shocks)，这里n_shocks=n_factors=n_states（max_lags=1时）
        U = np.random.randn(n_time, n_states)  # (n_time, n_states)格式,匹配train_model

        for iteration in range(self.max_iter):
            logger.debug(f"EM迭代 {iteration + 1}/{self.max_iter}")

            H = np.zeros((n_obs, n_states))
            H[:, :self.n_factors] = Lambda

            # 注意：B矩阵在循环外初始化，并在M步后更新（匹配老代码）

            kf = KalmanFilter(A, B, H, Q, R, x0, P0)
            filter_result = kf.filter(Z, U)  # Z:(n_time, n_obs), U:(n_time, n_states) - 匹配train_model
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

            # 使用中心化数据（而非标准化数据）估计载荷
            Lambda_new = estimate_loadings(
                obs_centered,  # 使用中心化数据（匹配老代码）
                factors_df
            )

            # 处理Lambda中的NaN：使用上一次迭代的值
            nan_rows = np.isnan(Lambda_new).any(axis=1)
            if np.any(nan_rows):
                logger.warning(f"EM迭代{iteration}: Lambda有{nan_rows.sum()}行包含NaN，保留上一次迭代的值")
                Lambda_new[nan_rows, :] = Lambda[nan_rows, :]

            # 最终检查：如果仍有NaN（第一次迭代且初始化失败），抛出错误
            if np.any(np.isnan(Lambda_new)):
                raise ValueError(
                    f"EM迭代{iteration}: Lambda仍包含NaN，无法继续。"
                    f"可能原因：数据质量不足或变量有效数据点太少。"
                )

            Lambda = Lambda_new

            A = estimate_transition_matrix(factors_smoothed, self.max_lags)

            # 估计协方差矩阵和B矩阵（使用中心化数据，匹配老代码）
            B, Q, R = estimate_covariance_matrices(
                smoother_result,
                obs_centered,  # 使用中心化数据
                Lambda,
                self.n_factors,
                A,  # 传入A矩阵用于Q矩阵计算
                n_shocks=self.n_factors  # 传入n_shocks以计算B矩阵
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
