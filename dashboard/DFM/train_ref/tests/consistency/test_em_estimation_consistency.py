# -*- coding: utf-8 -*-
"""
Phase 4.1: EM参数估计函数一致性测试

测试train_model和train_ref中EM参数估计函数的一致性
严格验证每个参数估计函数的输出

验证策略: 极严格数值容差(rtol=1e-10, atol=1e-14)
前置条件: Phase 3 (卡尔曼滤波/平滑) 100%通过
"""

import numpy as np
import pandas as pd
import pytest
from dashboard.DFM.train_ref.tests.consistency.base import ConsistencyTestBase


# =============================================================================
# 导入两个版本的实现
# =============================================================================
# train_model版本
from dashboard.DFM.train_model.DiscreteKalmanFilter import (
    calculate_factor_loadings as calc_loadings_old,
    _calculate_prediction_matrix as calc_A_old,
    _calculate_shock_matrix as calc_shock_old,
    KalmanFilter as KF_old,
    FIS as Smoother_old
)

# train_ref版本
from dashboard.DFM.train_ref.core.estimator import (
    estimate_loadings as estimate_loadings_new,
    estimate_transition_matrix as estimate_A_new,
    estimate_covariance_matrices as estimate_QR_new,
    _ensure_positive_definite
)
from dashboard.DFM.train_ref.core.kalman import KalmanFilter as KF_new


class TestEMEstimationConsistency(ConsistencyTestBase):
    """EM参数估计一致性测试类

    测试内容:
    - 载荷矩阵估计一致性
    - 状态转移矩阵估计一致性
    - 过程噪声协方差Q估计一致性
    - 观测噪声协方差R估计一致性
    - 正定性保证函数一致性
    """

    @classmethod
    def setup_class(cls):
        """初始化测试数据（需要先运行卡尔曼平滑）"""
        # 加载small数据集
        cls.small_dataset = cls.load_simulated_dataset('small')

        # 提取数据
        cls.Z_df = cls.small_dataset['Z']  # DataFrame (n_time, n_obs)
        cls.n_time_small = cls.Z_df.shape[0]
        cls.n_obs_small = cls.Z_df.shape[1]
        cls.n_factors_small = cls.small_dataset['n_factors']

        # 状态空间维度
        cls.n_states = cls.n_factors_small
        cls.n_controls = cls.n_factors_small

        print(f"\n[SETUP] Small数据集维度:")
        print(f"  观测数量(n_obs): {cls.n_obs_small}")
        print(f"  时间长度(n_time): {cls.n_time_small}")
        print(f"  因子数量(n_factors): {cls.n_factors_small}")

        # 生成状态空间参数(使用固定种子)
        np.random.seed(42)

        # A: 状态转移矩阵
        A_raw = np.random.randn(cls.n_states, cls.n_states) * 0.3
        eigenvalues, eigenvectors = np.linalg.eig(A_raw)
        eigenvalues = eigenvalues * 0.8 / np.abs(eigenvalues)
        cls.A = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
        cls.A = np.real(cls.A)

        # B: 控制矩阵
        cls.B = np.random.randn(cls.n_states, cls.n_controls) * 0.1

        # H: 观测矩阵
        cls.H = np.random.randn(cls.n_obs_small, cls.n_states)

        # Q: 过程噪声协方差
        Q_raw = np.random.randn(cls.n_states, cls.n_states)
        cls.Q = Q_raw @ Q_raw.T * 0.01

        # R: 观测噪声协方差
        cls.R = np.eye(cls.n_obs_small) * 0.1

        # x0: 初始状态
        cls.x0 = np.random.randn(cls.n_states)

        # P0: 初始协方差
        P0_raw = np.random.randn(cls.n_states, cls.n_states)
        cls.P0 = P0_raw @ P0_raw.T

        # 控制输入(设为零)
        cls.U_df = pd.DataFrame(
            np.zeros((cls.n_time_small, cls.n_controls)),
            index=cls.Z_df.index,
            columns=[f'u{i+1}' for i in range(cls.n_controls)]
        )

        # 状态名称
        cls.state_names = [f'Factor{i+1}' for i in range(cls.n_states)]

        # 运行卡尔曼滤波+平滑获得平滑因子
        print(f"\n[SETUP] 运行卡尔曼滤波+平滑...")

        # train_model版本
        res_kf_old = KF_old(
            Z=cls.Z_df,
            U=cls.U_df,
            A=cls.A,
            B=cls.B,
            H=cls.H,
            state_names=cls.state_names,
            x0=cls.x0,
            P0=cls.P0,
            Q=cls.Q,
            R=cls.R
        )
        res_sm_old = Smoother_old(res_kf_old)

        cls.factors_old_df = res_sm_old.x_sm  # DataFrame (n_time, n_states)

        # train_ref版本
        Z_array = cls.Z_df.to_numpy()  # (n_time, n_obs) - 正确的维度
        U_array = cls.U_df.to_numpy()  # (n_time, n_controls) - 正确的维度

        kf_new = KF_new(
            A=cls.A,
            B=cls.B,
            H=cls.H,
            Q=cls.Q,
            R=cls.R,
            x0=cls.x0,
            P0=cls.P0
        )
        res_kf_new = kf_new.filter(Z=Z_array, U=U_array)
        res_sm_new = kf_new.smooth(res_kf_new)

        cls.factors_new_df = pd.DataFrame(
            res_sm_new.x_smoothed.T,  # 转为(n_time, n_states)
            index=cls.Z_df.index,
            columns=cls.state_names
        )

        cls.smoothed_result_new = res_sm_new

        print(f"  平滑完成")
        print(f"  factors_old shape: {cls.factors_old_df.shape}")
        print(f"  factors_new shape: {cls.factors_new_df.shape}")

        # 验证平滑因子一致性(前置条件)
        factors_diff = np.max(np.abs(
            cls.factors_old_df.to_numpy() - cls.factors_new_df.to_numpy()
        ))
        print(f"  平滑因子差异: {factors_diff:.6e}")
        assert factors_diff < 1e-10, "前置条件失败: 平滑因子不一致"

    def test_001_loadings_estimation_consistency(self):
        """测试001: 载荷矩阵估计一致性

        验证内容:
        - Lambda = estimate_loadings(Z, F)
        - 使用OLS回归: y_i ~ F (无截距)

        预期: Lambda矩阵完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试001: 载荷矩阵估计一致性")
        print("="*60)

        # train_model版本
        # 注意: calculate_factor_loadings使用sm.OLS
        print("\n[Step 1] 运行train_model载荷估计...")
        Lambda_old = calc_loadings_old(
            self.Z_df,
            self.factors_old_df
        )

        print(f"  Lambda_old shape: {Lambda_old.shape}")
        print(f"  Lambda_old[0, :]: {Lambda_old[0, :]}")

        # train_ref版本
        # 注意: estimate_loadings使用sklearn LinearRegression
        print("\n[Step 2] 运行train_ref载荷估计...")
        Lambda_new = estimate_loadings_new(
            self.Z_df,
            self.factors_new_df
        )

        print(f"  Lambda_new shape: {Lambda_new.shape}")
        print(f"  Lambda_new[0, :]: {Lambda_new[0, :]}")

        # 对比
        print("\n[Step 3] 对比载荷矩阵...")

        # 检查NaN
        nan_old = np.isnan(Lambda_old).sum()
        nan_new = np.isnan(Lambda_new).sum()
        print(f"  NaN count: old={nan_old}, new={nan_new}")

        # 只对比非NaN元素
        valid_mask = ~(np.isnan(Lambda_old) | np.isnan(Lambda_new))
        if not np.all(valid_mask):
            print(f"  警告: 存在NaN元素, valid={valid_mask.sum()}/{Lambda_old.size}")

        Lambda_old_valid = Lambda_old[valid_mask]
        Lambda_new_valid = Lambda_new[valid_mask]

        diff = np.abs(Lambda_old_valid - Lambda_new_valid)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"\n[差异统计]")
        print(f"  最大差异: {max_diff:.6e}")
        print(f"  平均差异: {mean_diff:.6e}")
        print(f"  标准差:   {np.std(diff):.6e}")

        # 使用极严格容差验证
        print(f"\n[验证] 使用极严格容差(rtol=1e-10, atol=1e-14)")
        self.assert_allclose_strict(
            Lambda_new_valid,
            Lambda_old_valid,
            name="载荷矩阵 Lambda (有效元素)"
        )

        print(f"\n[PASS] 载荷矩阵估计一致性验证通过!")

    def test_002_transition_matrix_estimation_consistency(self):
        """测试002: 状态转移矩阵估计一致性

        验证内容:
        - A = estimate_transition_matrix(F)
        - 公式: A = (F_t' F_{t-1})(F_{t-1}' F_{t-1})^-1

        预期: A矩阵完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试002: 状态转移矩阵估计一致性")
        print("="*60)

        # train_model版本
        print("\n[Step 1] 运行train_model转移矩阵估计...")
        A_old = calc_A_old(
            self.factors_old_df
        )

        print(f"  A_old shape: {A_old.shape}")
        print(f"  A_old:\n{A_old}")

        # 检查稳定性(特征值)
        eigenvalues_old = np.linalg.eigvals(A_old)
        max_eig_old = np.max(np.abs(eigenvalues_old))
        print(f"  max|eigenvalue|: {max_eig_old:.4f}")

        # train_ref版本
        print("\n[Step 2] 运行train_ref转移矩阵估计...")
        A_new = estimate_A_new(
            self.factors_new_df.to_numpy(),
            max_lags=1
        )

        print(f"  A_new shape: {A_new.shape}")
        print(f"  A_new:\n{A_new}")

        eigenvalues_new = np.linalg.eigvals(A_new)
        max_eig_new = np.max(np.abs(eigenvalues_new))
        print(f"  max|eigenvalue|: {max_eig_new:.4f}")

        # 对比
        print("\n[Step 3] 对比转移矩阵...")

        diff = np.abs(A_old - A_new)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"\n[差异统计]")
        print(f"  最大差异: {max_diff:.6e}")
        print(f"  平均差异: {mean_diff:.6e}")
        print(f"  Frobenius范数差异: {np.linalg.norm(A_old - A_new, 'fro'):.6e}")

        # 使用极严格容差验证
        print(f"\n[验证] 使用极严格容差(rtol=1e-10, atol=1e-14)")
        self.assert_allclose_strict(
            A_new,
            A_old,
            name="状态转移矩阵 A"
        )

        print(f"\n[PASS] 状态转移矩阵估计一致性验证通过!")

    def test_003_process_noise_covariance_estimation_consistency(self):
        """测试003: 过程噪声协方差Q估计一致性

        验证内容:
        - Q = estimate_covariance_matrices()[0]
        - 公式: Q = E[F_t F_t'] - A E[F_{t-1} F_{t-1}'] A'
        - 正定性保证

        预期: Q矩阵完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试003: 过程噪声协方差Q估计一致性")
        print("="*60)

        # 先估计A矩阵
        A_old = calc_A_old(self.factors_old_df)
        A_new = estimate_A_new(self.factors_new_df.to_numpy(), max_lags=1)

        # train_model版本
        print("\n[Step 1] 运行train_model Q矩阵估计...")
        B_old, Q_old = calc_shock_old(
            self.factors_old_df,
            A_old,
            n_shocks=self.n_factors_small
        )

        print(f"  Q_old shape: {Q_old.shape}")
        print(f"  Q_old:\n{Q_old}")

        # 检查正定性
        eigenvalues_Q_old = np.linalg.eigvals(Q_old)
        min_eig_Q_old = np.min(eigenvalues_Q_old)
        print(f"  min(eigenvalue): {min_eig_Q_old:.6e}")
        print(f"  是否正定: {min_eig_Q_old > 0}")

        # train_ref版本
        print("\n[Step 2] 运行train_ref Q矩阵估计...")

        # 注意: estimate_covariance_matrices需要载荷矩阵Lambda
        Lambda_new = estimate_loadings_new(self.Z_df, self.factors_new_df)

        B_new, Q_new, R_new = estimate_QR_new(
            self.smoothed_result_new,
            self.Z_df,
            Lambda_new,
            self.n_factors_small,
            A_new,
            n_shocks=self.n_factors_small
        )

        print(f"  Q_new shape: {Q_new.shape}")
        print(f"  Q_new:\n{Q_new}")

        eigenvalues_Q_new = np.linalg.eigvals(Q_new)
        min_eig_Q_new = np.min(eigenvalues_Q_new)
        print(f"  min(eigenvalue): {min_eig_Q_new:.6e}")
        print(f"  是否正定: {min_eig_Q_new > 0}")

        # 对比
        print("\n[Step 3] 对比Q矩阵...")

        diff = np.abs(Q_old - Q_new)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"\n[差异统计]")
        print(f"  最大差异: {max_diff:.6e}")
        print(f"  平均差异: {mean_diff:.6e}")
        print(f"  Frobenius范数差异: {np.linalg.norm(Q_old - Q_new, 'fro'):.6e}")

        # 使用极严格容差验证
        print(f"\n[验证] 使用极严格容差(rtol=1e-10, atol=1e-14)")
        self.assert_allclose_strict(
            Q_new,
            Q_old,
            name="过程噪声协方差 Q"
        )

        print(f"\n[PASS] 过程噪声协方差Q估计一致性验证通过!")

    def test_004_observation_noise_covariance_estimation_consistency(self):
        """测试004: 观测噪声协方差R估计一致性

        验证内容:
        - R = estimate_covariance_matrices()[1]
        - 公式: R = diag(Var(Z - Lambda * F))
        - 对角矩阵结构

        预期: R矩阵完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试004: 观测噪声协方差R估计一致性")
        print("="*60)

        # 需要先估计Lambda和A
        Lambda_old = calc_loadings_old(self.Z_df, self.factors_old_df)
        Lambda_new = estimate_loadings_new(self.Z_df, self.factors_new_df)
        A_new = estimate_A_new(self.factors_new_df.to_numpy(), max_lags=1)

        # train_model版本
        # 注意: EMstep函数会计算R，但我们需要单独提取逻辑
        print("\n[Step 1] 运行train_model R矩阵估计...")

        # 手动计算R (匹配EMstep的逻辑)
        f_np = self.factors_old_df.to_numpy()  # (n_time, n_factors)
        y_np = self.Z_df.to_numpy()  # (n_time, n_obs)

        predicted_y_old = (Lambda_old @ f_np.T).T  # (n_time, n_obs)
        residuals_old = y_np - predicted_y_old
        R_diag_old = np.nanvar(residuals_old, axis=0)
        R_diag_old = np.maximum(R_diag_old, 1e-7)
        R_old = np.diag(R_diag_old)

        print(f"  R_old shape: {R_old.shape}")
        print(f"  R_old diagonal[:5]: {np.diag(R_old)[:5]}")

        # 检查对角结构
        off_diag_old = R_old - np.diag(np.diag(R_old))
        max_off_diag_old = np.max(np.abs(off_diag_old))
        print(f"  最大非对角元素: {max_off_diag_old:.6e}")

        # train_ref版本
        print("\n[Step 2] 运行train_ref R矩阵估计...")

        B_new, Q_new, R_new = estimate_QR_new(
            self.smoothed_result_new,
            self.Z_df,
            Lambda_new,
            self.n_factors_small,
            A_new,
            n_shocks=self.n_factors_small
        )

        print(f"  R_new shape: {R_new.shape}")
        print(f"  R_new diagonal[:5]: {np.diag(R_new)[:5]}")

        off_diag_new = R_new - np.diag(np.diag(R_new))
        max_off_diag_new = np.max(np.abs(off_diag_new))
        print(f"  最大非对角元素: {max_off_diag_new:.6e}")

        # 对比
        print("\n[Step 3] 对比R矩阵...")

        # 只对比对角元素(R应该是对角矩阵)
        R_diag_old_vec = np.diag(R_old)
        R_diag_new_vec = np.diag(R_new)

        diff = np.abs(R_diag_old_vec - R_diag_new_vec)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"\n[差异统计](对角元素)")
        print(f"  最大差异: {max_diff:.6e}")
        print(f"  平均差异: {mean_diff:.6e}")
        print(f"  标准差:   {np.std(diff):.6e}")

        # 使用极严格容差验证
        print(f"\n[验证] 使用极严格容差(rtol=1e-10, atol=1e-14)")
        self.assert_allclose_strict(
            R_diag_new_vec,
            R_diag_old_vec,
            name="观测噪声协方差 R (对角元素)"
        )

        # 也验证完整矩阵
        self.assert_allclose_strict(
            R_new,
            R_old,
            name="观测噪声协方差 R (完整矩阵)"
        )

        print(f"\n[PASS] 观测噪声协方差R估计一致性验证通过!")

    def test_005_positive_definite_guarantee_consistency(self):
        """测试005: 正定性保证函数一致性

        验证内容:
        - ensure_positive_definite()
        - 特征值调整策略: max(λ, epsilon)

        预期: 调整后的矩阵完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试005: 正定性保证函数一致性")
        print("="*60)

        # 创建一个非正定矩阵
        print("\n[Step 1] 创建测试矩阵...")
        np.random.seed(123)
        n = 3
        M = np.random.randn(n, n)
        M = (M + M.T) / 2  # 对称化

        # 修改特征值使其非正定
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        eigenvalues[0] = -0.5  # 负特征值
        eigenvalues[1] = 0.0   # 零特征值
        M_non_pd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        print(f"  原始特征值: {eigenvalues}")
        print(f"  是否正定: {np.all(eigenvalues > 0)}")

        # train_model版本
        # 注意: _calculate_shock_matrix中的逻辑 (lines 115-121)
        print("\n[Step 2] 运行train_model正定性保证...")

        epsilon_old = 1e-7
        evals_old, evecs_old = np.linalg.eigh(M_non_pd)
        evals_corrected_old = np.maximum(evals_old, epsilon_old)
        M_pd_old = evecs_old @ np.diag(evals_corrected_old) @ evecs_old.T

        print(f"  调整后特征值: {evals_corrected_old}")
        print(f"  min(eigenvalue): {np.min(evals_corrected_old):.6e}")

        # train_ref版本
        print("\n[Step 3] 运行train_ref正定性保证...")

        M_pd_new = _ensure_positive_definite(M_non_pd, epsilon=1e-7)

        evals_new = np.linalg.eigvals(M_pd_new)
        print(f"  调整后特征值: {np.sort(evals_new)}")
        print(f"  min(eigenvalue): {np.min(evals_new):.6e}")

        # 对比
        print("\n[Step 4] 对比调整后的矩阵...")

        diff = np.abs(M_pd_old - M_pd_new)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"\n[差异统计]")
        print(f"  最大差异: {max_diff:.6e}")
        print(f"  平均差异: {mean_diff:.6e}")

        # 使用极严格容差验证
        print(f"\n[验证] 使用极严格容差(rtol=1e-10, atol=1e-14)")
        self.assert_allclose_strict(
            M_pd_new,
            M_pd_old,
            name="正定性保证后的矩阵"
        )

        print(f"\n[PASS] 正定性保证函数一致性验证通过!")


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '-s'])
