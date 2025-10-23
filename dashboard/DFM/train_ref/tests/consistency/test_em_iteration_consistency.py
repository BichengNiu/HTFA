# -*- coding: utf-8 -*-
"""
Phase 4.2: EM完整迭代一致性测试

测试train_model和train_ref中EM算法完整迭代的一致性
验证E步+M步的完整循环,参数演化轨迹,收敛判定逻辑

验证策略: 极严格数值容差(rtol=1e-10, atol=1e-14)
前置条件: Phase 4.1 (EM参数估计函数) 100%通过
"""

import numpy as np
import pandas as pd
import pytest
from dashboard.DFM.train_ref.tests.consistency.base import ConsistencyTestBase


# =============================================================================
# 导入两个版本的实现
# =============================================================================
# train_model版本
from dashboard.DFM.train_model.DynamicFactorModel import DFM_EMalgo as DFM_old
from dashboard.DFM.train_model.DiscreteKalmanFilter import (
    KalmanFilter as KF_old,
    FIS as Smoother_old,
    EMstep as EMstep_old
)

# train_ref版本
from dashboard.DFM.train_ref.core.factor_model import DFMModel as DFM_new
from dashboard.DFM.train_ref.core.kalman import KalmanFilter as KF_new
from dashboard.DFM.train_ref.core.estimator import (
    estimate_loadings,
    estimate_transition_matrix,
    estimate_covariance_matrices
)


class TestEMIterationConsistency(ConsistencyTestBase):
    """EM完整迭代一致性测试类

    测试内容:
    - 单次EM迭代(E步+M步)一致性
    - 多次EM迭代参数演化轨迹一致性
    - 收敛判定逻辑一致性
    - 不同初始化方法一致性
    """

    @classmethod
    def setup_class(cls):
        """初始化测试数据"""
        # 加载small数据集
        cls.small_dataset = cls.load_simulated_dataset('small')

        # 提取数据
        cls.Z_df = cls.small_dataset['Z']  # DataFrame (n_time, n_obs)
        cls.n_time_small = cls.Z_df.shape[0]
        cls.n_obs_small = cls.Z_df.shape[1]
        cls.n_factors_small = cls.small_dataset['n_factors']

        print(f"\n[SETUP] Small数据集维度:")
        print(f"  观测数量(n_obs): {cls.n_obs_small}")
        print(f"  时间长度(n_time): {cls.n_time_small}")
        print(f"  因子数量(n_factors): {cls.n_factors_small}")

        # 设置EM算法参数
        cls.max_iter = 5  # 测试用,限制迭代次数
        cls.tolerance = 1e-6  # 收敛容差

    def test_001_single_em_iteration_consistency(self):
        """测试001: 单次EM迭代(E步+M步)一致性

        验证内容:
        - 从相同的初始参数开始
        - E步: Kalman滤波+平滑的一致性
        - M步: 参数估计的一致性
        - 验证更新后的Lambda, A, Q, R完全一致

        预期: 单次迭代后的所有参数完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试001: 单次EM迭代(E步+M步)一致性")
        print("="*60)

        # 固定随机种子以确保初始化一致
        np.random.seed(42)

        print("\n[Step 1] 准备输入数据和初始参数...")

        # 使用PCA初始化获得相同的初始参数
        # 标准化数据
        means = self.Z_df.mean(skipna=True).values
        stds = self.Z_df.std(skipna=True).values
        stds = np.where(stds > 0, stds, 1.0)

        obs_centered = self.Z_df - means
        Z_standardized = (obs_centered / stds).fillna(0).values

        # PCA初始化
        U, s, Vh = np.linalg.svd(Z_standardized, full_matrices=False)
        factors_init = U[:, :self.n_factors_small] * s[:self.n_factors_small]
        factors_df = pd.DataFrame(
            factors_init,
            index=self.Z_df.index,
            columns=[f'Factor{i+1}' for i in range(self.n_factors_small)]
        )

        # 使用sklearn估计初始载荷(与train_ref一致)
        from sklearn.linear_model import LinearRegression
        Lambda_init = estimate_loadings(obs_centered, factors_df)

        # 使用VAR初始化A和Q矩阵
        from statsmodels.tsa.api import VAR
        var_model = VAR(factors_df.dropna())
        var_results = var_model.fit(1)
        A_init = var_results.coefs[0]
        Q_init = np.cov(var_results.resid, rowvar=False)
        Q_init = np.diag(np.maximum(np.diag(Q_init), 1e-6))

        # 初始化R矩阵
        V = Vh.T
        reconstructed_z = factors_init @ V[:, :self.n_factors_small].T
        residuals_z = Z_standardized - reconstructed_z
        psi_diag = np.nanvar(residuals_z, axis=0)
        R_init = np.diag(np.maximum(psi_diag * stds**2, 1e-6))

        # 初始状态
        x0_init = np.zeros(self.n_factors_small)
        P0_init = np.eye(self.n_factors_small)

        print(f"  Lambda_init shape: {Lambda_init.shape}")
        print(f"  A_init shape: {A_init.shape}")
        print(f"  Q_init shape: {Q_init.shape}")
        print(f"  R_init shape: {R_init.shape}")
        print(f"  x0_init: {x0_init}")
        print(f"  P0_init diagonal: {np.diag(P0_init)}")

        # =========================================================================
        # train_model版本: 执行一次E步+M步
        # =========================================================================
        print("\n[Step 2] 运行train_model单次EM迭代...")

        # 准备train_model的输入格式
        state_names = [f'Factor{i+1}' for i in range(self.n_factors_small)]
        Z_array = obs_centered.values.T  # (n_obs, n_time)
        U_array = np.zeros((self.n_factors_small, self.n_time_small))  # 零控制

        # 构造H矩阵
        H_old = Lambda_init.copy()
        B_old = np.eye(self.n_factors_small) * 0.1

        # E步: Kalman滤波+平滑
        kf_old = KF_old(
            Z=obs_centered,  # 使用中心化数据
            U=pd.DataFrame(
                U_array.T,
                index=self.Z_df.index,
                columns=state_names
            ),
            A=A_init,
            B=B_old,
            H=H_old,
            state_names=state_names,
            x0=x0_init,
            P0=P0_init,
            Q=Q_init,
            R=R_init
        )

        smoother_old = Smoother_old(kf_old)

        # M步: 参数估计
        em_old = EMstep_old(smoother_old, n_shocks=self.n_factors_small)

        # 提取更新后的参数
        Lambda_old_iter1 = np.array(em_old.Lambda)
        A_old_iter1 = np.array(em_old.A)
        Q_old_iter1 = np.array(em_old.Q)
        R_old_iter1 = np.array(em_old.R)
        x0_old_iter1 = np.array(em_old.x_sm.iloc[0])
        P0_old_iter1 = smoother_old.P_sm[0]

        print(f"  Lambda_old[迭代1] shape: {Lambda_old_iter1.shape}")
        print(f"  Lambda_old[0, :]: {Lambda_old_iter1[0, :]}")
        print(f"  A_old[迭代1]:\n{A_old_iter1}")
        print(f"  Q_old[迭代1] diagonal: {np.diag(Q_old_iter1)}")
        print(f"  R_old[迭代1] diagonal[:5]: {np.diag(R_old_iter1)[:5]}")

        # =========================================================================
        # train_ref版本: 执行一次E步+M步
        # =========================================================================
        print("\n[Step 3] 运行train_ref单次EM迭代...")

        # 准备train_ref的输入格式
        Z_new = obs_centered.values  # (n_time, n_obs)
        U_new = np.zeros((self.n_time_small, self.n_factors_small))

        # 构造H矩阵
        n_states = self.n_factors_small
        H_new = np.zeros((self.n_obs_small, n_states))
        H_new[:, :self.n_factors_small] = Lambda_init
        B_new = np.eye(n_states) * 0.1

        # E步: Kalman滤波+平滑
        kf_new = KF_new(
            A=A_init,
            B=B_new,
            H=H_new,
            Q=Q_init,
            R=R_init,
            x0=x0_init,
            P0=P0_init
        )

        filter_result = kf_new.filter(Z_new, U_new)
        smoother_result = kf_new.smooth(filter_result)

        # M步: 参数估计
        factors_smoothed = smoother_result.x_smoothed[:self.n_factors_small, :].T
        factors_df_new = pd.DataFrame(
            factors_smoothed,
            index=self.Z_df.index,
            columns=state_names
        )

        Lambda_new_iter1 = estimate_loadings(obs_centered, factors_df_new)
        A_new_iter1 = estimate_transition_matrix(factors_smoothed, max_lags=1)
        B_new_iter1, Q_new_iter1, R_new_iter1 = estimate_covariance_matrices(
            smoother_result,
            obs_centered,
            Lambda_new_iter1,
            self.n_factors_small,
            A_new_iter1,
            n_shocks=self.n_factors_small
        )

        x0_new_iter1 = smoother_result.x_smoothed[:, 0].copy()
        P0_new_iter1 = smoother_result.P_smoothed[:, :, 0].copy()

        print(f"  Lambda_new[迭代1] shape: {Lambda_new_iter1.shape}")
        print(f"  Lambda_new[0, :]: {Lambda_new_iter1[0, :]}")
        print(f"  A_new[迭代1]:\n{A_new_iter1}")
        print(f"  Q_new[迭代1] diagonal: {np.diag(Q_new_iter1)}")
        print(f"  R_new[迭代1] diagonal[:5]: {np.diag(R_new_iter1)[:5]}")

        # =========================================================================
        # 对比单次迭代结果
        # =========================================================================
        print("\n[Step 4] 对比单次EM迭代后的参数...")

        # 对比Lambda
        diff_Lambda = np.max(np.abs(Lambda_old_iter1 - Lambda_new_iter1))
        print(f"\n[Lambda差异]")
        print(f"  最大差异: {diff_Lambda:.6e}")
        self.assert_allclose_strict(
            Lambda_new_iter1,
            Lambda_old_iter1,
            name="载荷矩阵Lambda[迭代1]"
        )

        # 对比A
        diff_A = np.max(np.abs(A_old_iter1 - A_new_iter1))
        print(f"\n[A矩阵差异]")
        print(f"  最大差异: {diff_A:.6e}")
        self.assert_allclose_strict(
            A_new_iter1,
            A_old_iter1,
            name="状态转移矩阵A[迭代1]"
        )

        # 对比Q
        diff_Q = np.max(np.abs(Q_old_iter1 - Q_new_iter1))
        print(f"\n[Q矩阵差异]")
        print(f"  最大差异: {diff_Q:.6e}")
        self.assert_allclose_strict(
            Q_new_iter1,
            Q_old_iter1,
            name="过程噪声协方差Q[迭代1]"
        )

        # 对比R
        diff_R = np.max(np.abs(R_old_iter1 - R_new_iter1))
        print(f"\n[R矩阵差异]")
        print(f"  最大差异: {diff_R:.6e}")
        self.assert_allclose_strict(
            R_new_iter1,
            R_old_iter1,
            name="观测噪声协方差R[迭代1]"
        )

        # 对比初始状态
        diff_x0 = np.max(np.abs(x0_old_iter1 - x0_new_iter1))
        print(f"\n[初始状态x0差异]")
        print(f"  最大差异: {diff_x0:.6e}")
        self.assert_allclose_strict(
            x0_new_iter1,
            x0_old_iter1,
            name="更新初始状态x0[迭代1]"
        )

        # 对比初始协方差
        diff_P0 = np.max(np.abs(P0_old_iter1 - P0_new_iter1))
        print(f"\n[初始协方差P0差异]")
        print(f"  最大差异: {diff_P0:.6e}")
        self.assert_allclose_strict(
            P0_new_iter1,
            P0_old_iter1,
            name="更新初始协方差P0[迭代1]"
        )

        print(f"\n[PASS] 单次EM迭代一致性验证通过!")

    def test_002_multiple_em_iterations_consistency(self):
        """测试002: 多次EM迭代参数演化轨迹一致性

        验证内容:
        - 执行3次完整EM迭代
        - 逐次验证每次迭代后的参数一致性
        - 验证参数演化轨迹(收敛方向)一致

        预期: 所有迭代的参数完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试002: 多次EM迭代参数演化轨迹一致性")
        print("="*60)

        # 固定随机种子
        np.random.seed(42)

        n_test_iters = 3  # 测试3次迭代

        print(f"\n[Step 1] 使用DFM完整流程执行{n_test_iters}次迭代...")

        # =========================================================================
        # train_model版本: 完整DFM训练
        # =========================================================================
        print("\n[Step 2] train_model执行EM迭代...")

        # 调用DFM_EMalgo执行n_test_iters次迭代
        # 注意: DFM_EMalgo返回最终结果,我们需要修改它来返回中间结果
        # 这里我们直接调用底层循环

        results_old = self._run_em_iterations_old(
            self.Z_df,
            n_factors=self.n_factors_small,
            n_iters=n_test_iters
        )

        print(f"  完成{n_test_iters}次迭代")
        for i in range(n_test_iters):
            print(f"  迭代{i+1}: Lambda[0, :] = {results_old['Lambda'][i][0, :]}")

        # =========================================================================
        # train_ref版本: 完整DFM训练
        # =========================================================================
        print("\n[Step 3] train_ref执行EM迭代...")

        results_new = self._run_em_iterations_new(
            self.Z_df,
            n_factors=self.n_factors_small,
            n_iters=n_test_iters
        )

        print(f"  完成{n_test_iters}次迭代")
        for i in range(n_test_iters):
            print(f"  迭代{i+1}: Lambda[0, :] = {results_new['Lambda'][i][0, :]}")

        # =========================================================================
        # 逐次迭代对比
        # =========================================================================
        print("\n[Step 4] 逐次对比每次迭代的参数...")

        for i in range(n_test_iters):
            print(f"\n--- 迭代 {i+1} ---")

            Lambda_old = results_old['Lambda'][i]
            Lambda_new = results_new['Lambda'][i]
            diff_Lambda = np.max(np.abs(Lambda_old - Lambda_new))
            print(f"  Lambda最大差异: {diff_Lambda:.6e}")
            self.assert_allclose_strict(
                Lambda_new, Lambda_old,
                name=f"Lambda[迭代{i+1}]"
            )

            A_old = results_old['A'][i]
            A_new = results_new['A'][i]
            diff_A = np.max(np.abs(A_old - A_new))
            print(f"  A最大差异: {diff_A:.6e}")
            self.assert_allclose_strict(
                A_new, A_old,
                name=f"A[迭代{i+1}]"
            )

            Q_old = results_old['Q'][i]
            Q_new = results_new['Q'][i]
            diff_Q = np.max(np.abs(Q_old - Q_new))
            print(f"  Q最大差异: {diff_Q:.6e}")
            self.assert_allclose_strict(
                Q_new, Q_old,
                name=f"Q[迭代{i+1}]"
            )

            R_old = results_old['R'][i]
            R_new = results_new['R'][i]
            diff_R = np.max(np.abs(R_old - R_new))
            print(f"  R最大差异: {diff_R:.6e}")
            self.assert_allclose_strict(
                R_new, R_old,
                name=f"R[迭代{i+1}]"
            )

        print(f"\n[PASS] 多次EM迭代一致性验证通过!")

    def test_003_long_iteration_sequence_consistency(self):
        """测试003: 长迭代序列一致性

        验证内容:
        - 执行较长序列(10次)EM迭代
        - 验证最终收敛参数一致性
        - 验证迭代轨迹的稳定性

        注意: train_model没有收敛判定,只运行固定次数
              train_ref有收敛判定,但此测试中我们运行固定次数进行对比

        预期: 运行相同次数迭代后,最终参数完全一致
        """
        print("\n" + "="*60)
        print("测试003: 长迭代序列一致性")
        print("="*60)

        # 固定随机种子
        np.random.seed(42)

        # 设置较长的迭代次数
        n_iters_test = 10

        print(f"\n[Step 1] 设置测试参数: n_iters={n_iters_test}")

        # =========================================================================
        # train_model版本
        # =========================================================================
        print(f"\n[Step 2] train_model执行{n_iters_test}次EM迭代...")

        results_old = self._run_em_iterations_old(
            self.Z_df,
            n_factors=self.n_factors_small,
            n_iters=n_iters_test
        )

        print(f"  完成{n_iters_test}次迭代")
        print(f"  最终Lambda[0, :] = {results_old['Lambda'][-1][0, :]}")
        print(f"  最终A:\n{results_old['A'][-1]}")

        # =========================================================================
        # train_ref版本
        # =========================================================================
        print(f"\n[Step 3] train_ref执行{n_iters_test}次EM迭代...")

        results_new = self._run_em_iterations_new(
            self.Z_df,
            n_factors=self.n_factors_small,
            n_iters=n_iters_test
        )

        print(f"  完成{n_iters_test}次迭代")
        print(f"  最终Lambda[0, :] = {results_new['Lambda'][-1][0, :]}")
        print(f"  最终A:\n{results_new['A'][-1]}")

        # =========================================================================
        # 对比最终收敛结果
        # =========================================================================
        print("\n[Step 4] 对比最终收敛参数...")

        # 对比最终迭代的参数
        i = n_iters_test - 1
        print(f"\n[最终迭代 {i+1}]")

        Lambda_old_final = results_old['Lambda'][i]
        Lambda_new_final = results_new['Lambda'][i]
        diff_Lambda = np.max(np.abs(Lambda_old_final - Lambda_new_final))
        print(f"  Lambda最大差异: {diff_Lambda:.6e}")
        self.assert_allclose_strict(
            Lambda_new_final, Lambda_old_final,
            name=f"Lambda[最终迭代]"
        )

        A_old_final = results_old['A'][i]
        A_new_final = results_new['A'][i]
        diff_A = np.max(np.abs(A_old_final - A_new_final))
        print(f"  A最大差异: {diff_A:.6e}")
        self.assert_allclose_strict(
            A_new_final, A_old_final,
            name=f"A[最终迭代]"
        )

        Q_old_final = results_old['Q'][i]
        Q_new_final = results_new['Q'][i]
        diff_Q = np.max(np.abs(Q_old_final - Q_new_final))
        print(f"  Q最大差异: {diff_Q:.6e}")
        self.assert_allclose_strict(
            Q_new_final, Q_old_final,
            name=f"Q[最终迭代]"
        )

        R_old_final = results_old['R'][i]
        R_new_final = results_new['R'][i]
        diff_R = np.max(np.abs(R_old_final - R_new_final))
        print(f"  R最大差异: {diff_R:.6e}")
        self.assert_allclose_strict(
            R_new_final, R_old_final,
            name=f"R[最终迭代]"
        )

        # 验证迭代轨迹的一致性(抽查几个中间迭代)
        print("\n[迭代轨迹一致性验证]")
        check_iters = [0, n_iters_test//2, n_iters_test-1]
        for iter_idx in check_iters:
            Lambda_diff = np.max(np.abs(
                results_old['Lambda'][iter_idx] - results_new['Lambda'][iter_idx]
            ))
            print(f"  迭代{iter_idx+1}: Lambda差异={Lambda_diff:.6e}")
            assert Lambda_diff < 1e-10, f"迭代{iter_idx+1}差异过大"

        print(f"\n[PASS] 长迭代序列一致性验证通过!")

    def test_004_initialization_methods_consistency(self):
        """测试004: 不同初始化方法一致性

        验证内容:
        - PCA初始化一致性(已在前3个测试中验证)
        - 随机初始化后第一次EM迭代一致性
        - 用户指定初始化后第一次EM迭代一致性

        预期: 给定相同初始参数,第一次EM迭代结果一致
        """
        print("\n" + "="*60)
        print("测试004: 不同初始化方法一致性")
        print("="*60)

        # 固定随机种子
        np.random.seed(123)  # 使用不同种子

        print("\n[Step 1] 测试随机初始化...")

        # 生成随机初始参数
        Lambda_random = np.random.randn(self.n_obs_small, self.n_factors_small)
        A_random = np.random.randn(self.n_factors_small, self.n_factors_small) * 0.5
        Q_random = np.eye(self.n_factors_small) * 0.1
        R_random = np.eye(self.n_obs_small) * 0.1
        x0_random = np.random.randn(self.n_factors_small)
        P0_random = np.eye(self.n_factors_small)

        print(f"  Lambda_random[0, :]: {Lambda_random[0, :]}")
        print(f"  A_random:\n{A_random}")

        # 执行一次EM迭代(train_model)
        results_old = self._run_single_iteration_with_init_old(
            self.Z_df,
            Lambda_random, A_random, Q_random, R_random,
            x0_random, P0_random
        )

        # 执行一次EM迭代(train_ref)
        results_new = self._run_single_iteration_with_init_new(
            self.Z_df,
            Lambda_random, A_random, Q_random, R_random,
            x0_random, P0_random
        )

        print("\n[Step 2] 对比随机初始化后的第一次迭代结果...")

        # 对比Lambda
        diff_Lambda = np.max(np.abs(results_old['Lambda'] - results_new['Lambda']))
        print(f"  Lambda最大差异: {diff_Lambda:.6e}")
        self.assert_allclose_strict(
            results_new['Lambda'],
            results_old['Lambda'],
            name="Lambda[随机初始化]"
        )

        # 对比A
        diff_A = np.max(np.abs(results_old['A'] - results_new['A']))
        print(f"  A最大差异: {diff_A:.6e}")
        self.assert_allclose_strict(
            results_new['A'],
            results_old['A'],
            name="A[随机初始化]"
        )

        print("\n[Step 3] 测试另一组随机初始化...")

        # 使用另一组随机初始化(不同的种子)
        np.random.seed(456)
        Lambda_random2 = np.random.randn(self.n_obs_small, self.n_factors_small)
        A_random2 = np.random.randn(self.n_factors_small, self.n_factors_small) * 0.3
        Q_random2 = np.eye(self.n_factors_small) * 0.05
        R_random2 = np.eye(self.n_obs_small) * 0.05
        x0_random2 = np.random.randn(self.n_factors_small)
        P0_random2 = np.eye(self.n_factors_small)

        print(f"  Lambda_random2[0, :]: {Lambda_random2[0, :]}")
        print(f"  A_random2:\n{A_random2}")

        # 执行一次EM迭代(train_model)
        results_old2 = self._run_single_iteration_with_init_old(
            self.Z_df,
            Lambda_random2, A_random2, Q_random2, R_random2,
            x0_random2, P0_random2
        )

        # 执行一次EM迭代(train_ref)
        results_new2 = self._run_single_iteration_with_init_new(
            self.Z_df,
            Lambda_random2, A_random2, Q_random2, R_random2,
            x0_random2, P0_random2
        )

        print("\n[Step 4] 对比第二组随机初始化后的第一次迭代结果...")

        # 对比Lambda
        diff_Lambda2 = np.max(np.abs(
            results_old2['Lambda'] - results_new2['Lambda']
        ))
        print(f"  Lambda最大差异: {diff_Lambda2:.6e}")
        self.assert_allclose_strict(
            results_new2['Lambda'],
            results_old2['Lambda'],
            name="Lambda[第二组随机初始化]"
        )

        # 对比A
        diff_A2 = np.max(np.abs(results_old2['A'] - results_new2['A']))
        print(f"  A最大差异: {diff_A2:.6e}")
        self.assert_allclose_strict(
            results_new2['A'],
            results_old2['A'],
            name="A[第二组随机初始化]"
        )

        # 对比Q和R
        diff_Q2 = np.max(np.abs(results_old2['Q'] - results_new2['Q']))
        print(f"  Q最大差异: {diff_Q2:.6e}")
        self.assert_allclose_strict(
            results_new2['Q'],
            results_old2['Q'],
            name="Q[第二组随机初始化]"
        )

        diff_R2 = np.max(np.abs(results_old2['R'] - results_new2['R']))
        print(f"  R最大差异: {diff_R2:.6e}")
        self.assert_allclose_strict(
            results_new2['R'],
            results_old2['R'],
            name="R[第二组随机初始化]"
        )

        print(f"\n[PASS] 不同初始化方法一致性验证通过!")

    # =========================================================================
    # 辅助方法: train_model版本的EM迭代
    # =========================================================================

    def _run_em_iterations_old(self, Z_df, n_factors, n_iters):
        """运行train_model的EM迭代,返回中间结果"""
        # 初始化
        means = Z_df.mean(skipna=True).values
        stds = Z_df.std(skipna=True).values
        stds = np.where(stds > 0, stds, 1.0)
        obs_centered = Z_df - means
        Z_std = (obs_centered / stds).fillna(0).values

        # PCA初始化
        U, s, Vh = np.linalg.svd(Z_std, full_matrices=False)
        factors_init = U[:, :n_factors] * s[:n_factors]
        factors_df = pd.DataFrame(
            factors_init,
            index=Z_df.index,
            columns=[f'Factor{i+1}' for i in range(n_factors)]
        )

        # 初始载荷
        from dashboard.DFM.train_model.DiscreteKalmanFilter import calculate_factor_loadings
        Lambda = calculate_factor_loadings(obs_centered, factors_df)

        # VAR初始化A和Q
        from statsmodels.tsa.api import VAR
        var_model = VAR(factors_df.dropna())
        var_results = var_model.fit(1)
        A = var_results.coefs[0]
        Q = np.cov(var_results.resid, rowvar=False)
        Q = np.diag(np.maximum(np.diag(Q), 1e-6))

        # 初始化R
        V = Vh.T
        reconstructed = factors_init @ V[:, :n_factors].T
        residuals = Z_std - reconstructed
        R = np.diag(np.maximum(np.nanvar(residuals, axis=0) * stds**2, 1e-6))

        # 初始状态
        x0 = np.zeros(n_factors)
        P0 = np.eye(n_factors)

        state_names = [f'Factor{i+1}' for i in range(n_factors)]
        U_df = pd.DataFrame(
            np.zeros((len(Z_df), n_factors)),
            index=Z_df.index,
            columns=state_names
        )

        # 存储中间结果
        results = {
            'Lambda': [],
            'A': [],
            'Q': [],
            'R': []
        }

        # EM迭代
        for i in range(n_iters):
            B = np.eye(n_factors) * 0.1

            kf = KF_old(
                Z=obs_centered,
                U=U_df,
                A=A,
                B=B,
                H=Lambda,
                state_names=state_names,
                x0=x0,
                P0=P0,
                Q=Q,
                R=R
            )

            smoother = Smoother_old(kf)
            em = EMstep_old(smoother, n_shocks=n_factors)

            # 保存当前迭代结果
            results['Lambda'].append(np.array(em.Lambda))
            results['A'].append(np.array(em.A))
            results['Q'].append(np.array(em.Q))
            results['R'].append(np.array(em.R))

            # 更新参数
            Lambda = np.array(em.Lambda)
            A = np.array(em.A)
            Q = np.array(em.Q)
            R = np.array(em.R)
            x0 = np.array(em.x_sm.iloc[0])
            P0 = smoother.P_sm[0]

        return results

    def _run_em_iterations_new(self, Z_df, n_factors, n_iters):
        """运行train_ref的EM迭代,返回中间结果"""
        # 初始化
        means = Z_df.mean(skipna=True).values
        stds = Z_df.std(skipna=True).values
        stds = np.where(stds > 0, stds, 1.0)
        obs_centered = Z_df - means
        Z_std = (obs_centered / stds).fillna(0).values

        # PCA初始化
        U, s, Vh = np.linalg.svd(Z_std, full_matrices=False)
        factors_init = U[:, :n_factors] * s[:n_factors]
        factors_df = pd.DataFrame(
            factors_init,
            index=Z_df.index,
            columns=[f'Factor{i+1}' for i in range(n_factors)]
        )

        # 初始载荷
        Lambda = estimate_loadings(obs_centered, factors_df)

        # VAR初始化A和Q
        from statsmodels.tsa.api import VAR
        var_model = VAR(factors_df.dropna())
        var_results = var_model.fit(1)
        A = var_results.coefs[0]
        Q = np.cov(var_results.resid, rowvar=False)
        Q = np.diag(np.maximum(np.diag(Q), 1e-6))

        # 初始化R
        V = Vh.T
        reconstructed = factors_init @ V[:, :n_factors].T
        residuals = Z_std - reconstructed
        R = np.diag(np.maximum(np.nanvar(residuals, axis=0) * stds**2, 1e-6))

        # 初始状态
        x0 = np.zeros(n_factors)
        P0 = np.eye(n_factors)

        Z = obs_centered.values  # (n_time, n_obs)
        U = np.zeros((len(Z_df), n_factors))

        # 存储中间结果
        results = {
            'Lambda': [],
            'A': [],
            'Q': [],
            'R': []
        }

        # EM迭代
        for i in range(n_iters):
            n_time, n_obs = Z.shape
            H = np.zeros((n_obs, n_factors))
            H[:, :n_factors] = Lambda
            B = np.eye(n_factors) * 0.1

            kf = KF_new(A, B, H, Q, R, x0, P0)
            filter_result = kf.filter(Z, U)
            smoother_result = kf.smooth(filter_result)

            # M步
            factors_smoothed = smoother_result.x_smoothed[:n_factors, :].T
            factors_df = pd.DataFrame(
                factors_smoothed,
                index=Z_df.index,
                columns=[f'Factor{j+1}' for j in range(n_factors)]
            )

            Lambda = estimate_loadings(obs_centered, factors_df)
            A = estimate_transition_matrix(factors_smoothed, max_lags=1)
            B, Q, R = estimate_covariance_matrices(
                smoother_result,
                obs_centered,
                Lambda,
                n_factors,
                A,
                n_shocks=n_factors
            )

            # 保存当前迭代结果
            results['Lambda'].append(Lambda.copy())
            results['A'].append(A.copy())
            results['Q'].append(Q.copy())
            results['R'].append(R.copy())

            # 更新初始状态
            x0 = smoother_result.x_smoothed[:, 0].copy()
            P0 = smoother_result.P_smoothed[:, :, 0].copy()

        return results

    def _run_em_until_convergence_old(self, Z_df, n_factors, max_iter, tolerance):
        """运行train_model的EM直到收敛"""
        # 初始化(与_run_em_iterations_old相同)
        means = Z_df.mean(skipna=True).values
        stds = Z_df.std(skipna=True).values
        stds = np.where(stds > 0, stds, 1.0)
        obs_centered = Z_df - means
        Z_std = (obs_centered / stds).fillna(0).values

        U, s, Vh = np.linalg.svd(Z_std, full_matrices=False)
        factors_init = U[:, :n_factors] * s[:n_factors]
        factors_df = pd.DataFrame(
            factors_init,
            index=Z_df.index,
            columns=[f'Factor{i+1}' for i in range(n_factors)]
        )

        from dashboard.DFM.train_model.DiscreteKalmanFilter import calculate_factor_loadings
        Lambda = calculate_factor_loadings(obs_centered, factors_df)

        from statsmodels.tsa.api import VAR
        var_model = VAR(factors_df.dropna())
        var_results = var_model.fit(1)
        A = var_results.coefs[0]
        Q = np.cov(var_results.resid, rowvar=False)
        Q = np.diag(np.maximum(np.diag(Q), 1e-6))

        V = Vh.T
        reconstructed = factors_init @ V[:, :n_factors].T
        residuals = Z_std - reconstructed
        R = np.diag(np.maximum(np.nanvar(residuals, axis=0) * stds**2, 1e-6))

        x0 = np.zeros(n_factors)
        P0 = np.eye(n_factors)

        state_names = [f'Factor{i+1}' for i in range(n_factors)]
        U_df = pd.DataFrame(
            np.zeros((len(Z_df), n_factors)),
            index=Z_df.index,
            columns=state_names
        )

        loglik_list = []
        converged = False

        # EM迭代
        for i in range(max_iter):
            B = np.eye(n_factors) * 0.1

            kf = KF_old(
                Z=obs_centered,
                U=U_df,
                A=A,
                B=B,
                H=Lambda,
                state_names=state_names,
                x0=x0,
                P0=P0,
                Q=Q,
                R=R
            )

            smoother = Smoother_old(kf)

            # 计算对数似然
            loglik_current = kf.loglik
            loglik_list.append(loglik_current)

            # 检查收敛
            if i > 0:
                loglik_diff = loglik_current - loglik_list[i-1]
                if abs(loglik_diff) < tolerance:
                    converged = True
                    break

            # M步
            em = EMstep_old(smoother, n_shocks=n_factors)

            Lambda = np.array(em.Lambda)
            A = np.array(em.A)
            Q = np.array(em.Q)
            R = np.array(em.R)
            x0 = np.array(em.x_sm.iloc[0])
            P0 = smoother.P_sm[0]

        return {
            'n_iter': i + 1,
            'converged': converged,
            'loglik': loglik_list
        }

    def _run_em_until_convergence_new(self, Z_df, n_factors, max_iter, tolerance):
        """运行train_ref的EM直到收敛"""
        # 初始化
        means = Z_df.mean(skipna=True).values
        stds = Z_df.std(skipna=True).values
        stds = np.where(stds > 0, stds, 1.0)
        obs_centered = Z_df - means
        Z_std = (obs_centered / stds).fillna(0).values

        U, s, Vh = np.linalg.svd(Z_std, full_matrices=False)
        factors_init = U[:, :n_factors] * s[:n_factors]
        factors_df = pd.DataFrame(
            factors_init,
            index=Z_df.index,
            columns=[f'Factor{i+1}' for i in range(n_factors)]
        )

        Lambda = estimate_loadings(obs_centered, factors_df)

        from statsmodels.tsa.api import VAR
        var_model = VAR(factors_df.dropna())
        var_results = var_model.fit(1)
        A = var_results.coefs[0]
        Q = np.cov(var_results.resid, rowvar=False)
        Q = np.diag(np.maximum(np.diag(Q), 1e-6))

        V = Vh.T
        reconstructed = factors_init @ V[:, :n_factors].T
        residuals = Z_std - reconstructed
        R = np.diag(np.maximum(np.nanvar(residuals, axis=0) * stds**2, 1e-6))

        x0 = np.zeros(n_factors)
        P0 = np.eye(n_factors)

        Z = obs_centered.values
        U = np.zeros((len(Z_df), n_factors))

        loglik_list = []
        converged = False

        # EM迭代
        for i in range(max_iter):
            n_obs, n_time = Z.shape
            H = np.zeros((n_obs, n_factors))
            H[:, :n_factors] = Lambda
            B = np.eye(n_factors) * 0.1

            kf = KF_new(A, B, H, Q, R, x0, P0)
            filter_result = kf.filter(Z, U)
            smoother_result = kf.smooth(filter_result)

            # 计算对数似然
            loglik_current = filter_result.loglikelihood
            loglik_list.append(loglik_current)

            # 检查收敛
            if i > 0:
                loglik_diff = loglik_current - loglik_list[i-1]
                if abs(loglik_diff) < tolerance:
                    converged = True
                    break

            # M步
            factors_smoothed = smoother_result.x_smoothed[:n_factors, :].T
            factors_df = pd.DataFrame(
                factors_smoothed,
                index=Z_df.index,
                columns=[f'Factor{j+1}' for j in range(n_factors)]
            )

            Lambda = estimate_loadings(obs_centered, factors_df)
            A = estimate_transition_matrix(factors_smoothed, max_lags=1)
            B, Q, R = estimate_covariance_matrices(
                smoother_result,
                obs_centered,
                Lambda,
                n_factors,
                A,
                n_shocks=n_factors
            )

            x0 = smoother_result.x_smoothed[:, 0].copy()
            P0 = smoother_result.P_smoothed[:, :, 0].copy()

        return {
            'n_iter': i + 1,
            'converged': converged,
            'loglik': loglik_list
        }

    def _run_single_iteration_with_init_old(
        self, Z_df, Lambda, A, Q, R, x0, P0
    ):
        """使用指定初始参数运行train_model单次迭代"""
        n_factors = Lambda.shape[1]

        # 准备数据
        means = Z_df.mean(skipna=True).values
        obs_centered = Z_df - means

        state_names = [f'Factor{i+1}' for i in range(n_factors)]
        U_df = pd.DataFrame(
            np.zeros((len(Z_df), n_factors)),
            index=Z_df.index,
            columns=state_names
        )

        B = np.eye(n_factors) * 0.1

        # E步
        kf = KF_old(
            Z=obs_centered,
            U=U_df,
            A=A,
            B=B,
            H=Lambda,
            state_names=state_names,
            x0=x0,
            P0=P0,
            Q=Q,
            R=R
        )

        smoother = Smoother_old(kf)

        # M步
        em = EMstep_old(smoother, n_shocks=n_factors)

        return {
            'Lambda': np.array(em.Lambda),
            'A': np.array(em.A),
            'Q': np.array(em.Q),
            'R': np.array(em.R)
        }

    def _run_single_iteration_with_init_new(
        self, Z_df, Lambda, A, Q, R, x0, P0
    ):
        """使用指定初始参数运行train_ref单次迭代"""
        n_factors = Lambda.shape[1]

        # 准备数据
        means = Z_df.mean(skipna=True).values
        obs_centered = Z_df - means
        Z = obs_centered.values  # (n_time, n_obs)
        U = np.zeros((len(Z_df), n_factors))

        n_time, n_obs = Z.shape
        H = np.zeros((n_obs, n_factors))
        H[:, :n_factors] = Lambda
        B = np.eye(n_factors) * 0.1

        # E步
        kf = KF_new(A, B, H, Q, R, x0, P0)
        filter_result = kf.filter(Z, U)
        smoother_result = kf.smooth(filter_result)

        # M步
        factors_smoothed = smoother_result.x_smoothed[:n_factors, :].T
        factors_df = pd.DataFrame(
            factors_smoothed,
            index=Z_df.index,
            columns=[f'Factor{i+1}' for i in range(n_factors)]
        )

        Lambda_new = estimate_loadings(obs_centered, factors_df)
        A_new = estimate_transition_matrix(factors_smoothed, max_lags=1)
        B_new, Q_new, R_new = estimate_covariance_matrices(
            smoother_result,
            obs_centered,
            Lambda_new,
            n_factors,
            A_new,
            n_shocks=n_factors
        )

        return {
            'Lambda': Lambda_new,
            'A': A_new,
            'Q': Q_new,
            'R': R_new
        }


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '-s'])
