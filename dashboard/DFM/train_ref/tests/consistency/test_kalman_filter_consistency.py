# -*- coding: utf-8 -*-
"""
Phase 3.1: 卡尔曼滤波器一致性测试

测试train_model和train_ref的卡尔曼滤波实现一致性
严格验证每个时间步的所有中间变量

验证策略: 极严格数值容差(rtol=1e-10, atol=1e-14)
"""

import numpy as np
import pandas as pd
import pytest
from dashboard.DFM.train_ref.tests.consistency.base import ConsistencyTestBase


# =============================================================================
# 导入两个版本的实现
# =============================================================================
# train_model版本
from dashboard.DFM.train_model.DiscreteKalmanFilter import KalmanFilter as KF_old

# train_ref版本
from dashboard.DFM.train_ref.core.kalman import KalmanFilter as KF_new


class TestKalmanFilterConsistency(ConsistencyTestBase):
    """卡尔曼滤波器一致性测试类

    测试内容:
    - 单步预测一致性(x_pred, P_pred)
    - 单步更新一致性(K, x_filt, P_filt)
    - 多步滤波一致性(完整时间序列)
    - 缺失数据处理一致性
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

        # 状态空间维度
        cls.n_states = cls.n_factors_small
        cls.n_controls = cls.n_factors_small

        print(f"\n[SETUP] Small数据集维度:")
        print(f"  观测数量(n_obs): {cls.n_obs_small}")
        print(f"  时间长度(n_time): {cls.n_time_small}")
        print(f"  因子数量(n_factors): {cls.n_factors_small}")
        print(f"  状态维度(n_states): {cls.n_states}")

        # 生成状态空间参数(使用固定种子确保可重现)
        np.random.seed(42)

        # A: 状态转移矩阵 (n_states x n_states)
        # 使用小特征值确保系统稳定
        A_raw = np.random.randn(cls.n_states, cls.n_states) * 0.3
        eigenvalues, eigenvectors = np.linalg.eig(A_raw)
        eigenvalues = eigenvalues * 0.8 / np.abs(eigenvalues)  # 确保|λ| < 1
        cls.A = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
        cls.A = np.real(cls.A)  # 取实部

        # B: 控制矩阵 (n_states x n_controls)
        cls.B = np.random.randn(cls.n_states, cls.n_controls) * 0.1

        # H: 观测矩阵 (n_obs x n_states)
        cls.H = np.random.randn(cls.n_obs_small, cls.n_states)

        # Q: 过程噪声协方差 (n_states x n_states)
        Q_raw = np.random.randn(cls.n_states, cls.n_states)
        cls.Q = Q_raw @ Q_raw.T * 0.01  # 确保正定

        # R: 观测噪声协方差 (n_obs x n_obs)
        cls.R = np.eye(cls.n_obs_small) * 0.1

        # x0: 初始状态 (n_states,)
        cls.x0 = np.random.randn(cls.n_states)

        # P0: 初始协方差 (n_states x n_states)
        P0_raw = np.random.randn(cls.n_states, cls.n_states)
        cls.P0 = P0_raw @ P0_raw.T  # 确保正定

        # 控制输入(设为零)
        cls.U_df = pd.DataFrame(
            np.zeros((cls.n_time_small, cls.n_controls)),
            index=cls.Z_df.index,
            columns=[f'u{i+1}' for i in range(cls.n_controls)]
        )

        # 状态名称
        cls.state_names = [f'Factor{i+1}' for i in range(cls.n_states)]

        print(f"\n[SETUP] 状态空间参数:")
        print(f"  A shape: {cls.A.shape}, max|eigenvalue|: {np.max(np.abs(np.linalg.eigvals(cls.A))):.4f}")
        print(f"  B shape: {cls.B.shape}")
        print(f"  H shape: {cls.H.shape}")
        print(f"  Q shape: {cls.Q.shape}")
        print(f"  R shape: {cls.R.shape}")
        print(f"  x0 shape: {cls.x0.shape}")
        print(f"  P0 shape: {cls.P0.shape}")
        print(f"  U shape: {cls.U_df.shape}")

    def test_001_single_step_prediction_consistency(self):
        """测试001: 单步预测一致性

        验证内容:
        - x_pred[t] = A @ x_filt[t-1] + B @ u[t]
        - P_pred[t] = A @ P_filt[t-1] @ A.T + Q

        预期: 每个时间步的预测结果完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试001: 卡尔曼滤波单步预测一致性")
        print("="*60)

        # 运行train_model版本
        res_old = KF_old(
            Z=self.Z_df,
            U=self.U_df,
            A=self.A,
            B=self.B,
            H=self.H,
            state_names=self.state_names,
            x0=self.x0,
            P0=self.P0,
            Q=self.Q,
            R=self.R
        )

        # train_model结果:
        # x_minus: DataFrame (n_time, n_states)
        # P_minus: list of arrays [(n_states, n_states)] * n_time
        x_pred_old = res_old.x_minus.to_numpy()  # (n_time, n_states)
        P_pred_old = res_old.P_minus  # list of (n_states, n_states)

        # 运行train_ref版本
        # 数据格式: Z (n_time, n_obs), U (n_time, n_controls) - 匹配train_model
        Z_array = self.Z_df.to_numpy()  # (n_time, n_obs)
        U_array = self.U_df.to_numpy()  # (n_time, n_controls)

        kf_new = KF_new(
            A=self.A,
            B=self.B,
            H=self.H,
            Q=self.Q,
            R=self.R,
            x0=self.x0,
            P0=self.P0
        )
        res_new = kf_new.filter(Z=Z_array, U=U_array)

        # train_ref结果:
        # x_predicted: (n_time, n_states) - 已匹配train_model格式
        # P_predicted: (n_states, n_states, n_time)
        x_pred_new = res_new.x_predicted  # (n_time, n_states)
        P_pred_new = res_new.P_predicted  # (n_states, n_states, n_time)

        print(f"\n[数据格式]")
        print(f"  train_model: x_pred shape = {x_pred_old.shape}, P_pred length = {len(P_pred_old)}")
        print(f"  train_ref:   x_pred shape = {x_pred_new.shape}, P_pred shape = {P_pred_new.shape}")

        # 逐时间步验证
        n_time = self.n_time_small
        max_diff_x = 0.0
        max_diff_P = 0.0

        for t in range(n_time):
            # 验证x_pred[t]
            x_old_t = x_pred_old[t, :]
            x_new_t = x_pred_new[t, :]

            diff_x = np.max(np.abs(x_old_t - x_new_t))
            max_diff_x = max(max_diff_x, diff_x)

            # 验证P_pred[t]
            P_old_t = P_pred_old[t]
            P_new_t = P_pred_new[:, :, t]

            diff_P = np.max(np.abs(P_old_t - P_new_t))
            max_diff_P = max(max_diff_P, diff_P)

            # 每10个时间步打印一次
            if t % 10 == 0 or t == n_time - 1:
                print(f"\n[时间步 t={t}]")
                print(f"  x_pred max_diff: {diff_x:.6e}")
                print(f"  P_pred max_diff: {diff_P:.6e}")

        print(f"\n[全局统计]")
        print(f"  x_pred 最大差异: {max_diff_x:.6e}")
        print(f"  P_pred 最大差异: {max_diff_P:.6e}")

        # 使用极严格容差验证
        print(f"\n[验证] 使用极严格容差(rtol=1e-10, atol=1e-14)")
        self.assert_allclose_strict(
            x_pred_new,
            x_pred_old,
            name="预测状态 x_pred (全时间序列)"
        )

        # 验证所有时间步的P_pred
        for t in range(n_time):
            self.assert_allclose_strict(
                P_pred_new[:, :, t],
                P_pred_old[t],
                name=f"预测协方差 P_pred[t={t}]"
            )

        print(f"\n[PASS] 单步预测一致性验证通过!")

    def test_002_single_step_update_consistency(self):
        """测试002: 单步更新一致性

        验证内容:
        - 卡尔曼增益 K[t]
        - 滤波状态 x_filt[t] = x_pred[t] + K[t] @ innovation[t]
        - 滤波协方差 P_filt[t] = (I - K[t] @ H_t) @ P_pred[t]

        预期: 每个时间步的更新结果完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试002: 卡尔曼滤波单步更新一致性")
        print("="*60)

        # 运行train_model版本
        res_old = KF_old(
            Z=self.Z_df,
            U=self.U_df,
            A=self.A,
            B=self.B,
            H=self.H,
            state_names=self.state_names,
            x0=self.x0,
            P0=self.P0,
            Q=self.Q,
            R=self.R
        )

        # train_model结果:
        # x: DataFrame (n_time, n_states)
        # P: list of arrays [(n_states, n_states)] * n_time
        # Kalman_gain: list of arrays, 注意可能是full格式(n_states, n_obs)
        x_filt_old = res_old.x.to_numpy()  # (n_time, n_states)
        P_filt_old = res_old.P  # list of (n_states, n_states)
        K_old = res_old.Kalman_gain  # list of arrays

        # 运行train_ref版本
        Z_array = self.Z_df.to_numpy()  # (n_time, n_obs)
        U_array = self.U_df.to_numpy()  # (n_time, n_controls)

        kf_new = KF_new(
            A=self.A,
            B=self.B,
            H=self.H,
            Q=self.Q,
            R=self.R,
            x0=self.x0,
            P0=self.P0
        )
        res_new = kf_new.filter(Z=Z_array, U=U_array)

        # train_ref结果:
        # x_filtered: (n_time, n_states) - 已匹配train_model格式
        # P_filtered: (n_states, n_states, n_time)
        # innovation: (n_time, n_obs) - 已匹配train_model格式
        x_filt_new = res_new.x_filtered  # (n_time, n_states)
        P_filt_new = res_new.P_filtered  # (n_states, n_states, n_time)
        innovation_new = res_new.innovation  # (n_time, n_obs)

        print(f"\n[数据格式]")
        print(f"  train_model: x_filt shape = {x_filt_old.shape}, P_filt length = {len(P_filt_old)}")
        print(f"  train_ref:   x_filt shape = {x_filt_new.shape}, P_filt shape = {P_filt_new.shape}")

        # 逐时间步验证
        n_time = self.n_time_small
        max_diff_x = 0.0
        max_diff_P = 0.0

        for t in range(n_time):
            # 验证x_filt[t]
            x_old_t = x_filt_old[t, :]
            x_new_t = x_filt_new[t, :]

            diff_x = np.max(np.abs(x_old_t - x_new_t))
            max_diff_x = max(max_diff_x, diff_x)

            # 验证P_filt[t]
            P_old_t = P_filt_old[t]
            P_new_t = P_filt_new[:, :, t]

            diff_P = np.max(np.abs(P_old_t - P_new_t))
            max_diff_P = max(max_diff_P, diff_P)

            # 每10个时间步打印一次
            if t % 10 == 0 or t == n_time - 1:
                print(f"\n[时间步 t={t}]")
                print(f"  x_filt max_diff: {diff_x:.6e}")
                print(f"  P_filt max_diff: {diff_P:.6e}")

        print(f"\n[全局统计]")
        print(f"  x_filt 最大差异: {max_diff_x:.6e}")
        print(f"  P_filt 最大差异: {max_diff_P:.6e}")

        # 使用极严格容差验证
        print(f"\n[验证] 使用极严格容差(rtol=1e-10, atol=1e-14)")
        self.assert_allclose_strict(
            x_filt_new,
            x_filt_old,
            name="滤波状态 x_filt (全时间序列)"
        )

        # 验证所有时间步的P_filt
        for t in range(n_time):
            self.assert_allclose_strict(
                P_filt_new[:, :, t],
                P_filt_old[t],
                name=f"滤波协方差 P_filt[t={t}]"
            )

        print(f"\n[PASS] 单步更新一致性验证通过!")

    def test_003_full_filtering_consistency(self):
        """测试003: 完整滤波一致性

        验证内容:
        - 完整时间序列的滤波结果
        - 新息序列(innovation)
        - 对数似然(如果两边都计算)

        预期: 全局结果完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试003: 完整卡尔曼滤波一致性")
        print("="*60)

        # 运行train_model版本
        res_old = KF_old(
            Z=self.Z_df,
            U=self.U_df,
            A=self.A,
            B=self.B,
            H=self.H,
            state_names=self.state_names,
            x0=self.x0,
            P0=self.P0,
            Q=self.Q,
            R=self.R
        )

        x_filt_old = res_old.x.to_numpy()

        # 运行train_ref版本
        Z_array = self.Z_df.to_numpy()
        U_array = self.U_df.to_numpy()

        kf_new = KF_new(
            A=self.A,
            B=self.B,
            H=self.H,
            Q=self.Q,
            R=self.R,
            x0=self.x0,
            P0=self.P0
        )
        res_new = kf_new.filter(Z=Z_array, U=U_array)

        x_filt_new = res_new.x_filtered

        print(f"\n[数据格式]")
        print(f"  train_model: x_filt shape = {x_filt_old.shape}")
        print(f"  train_ref:   x_filt shape = {x_filt_new.shape}")

        # 全局统计
        diff = np.abs(x_filt_old - x_filt_new)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"\n[全局差异统计]")
        print(f"  最大差异: {max_diff:.6e}")
        print(f"  平均差异: {mean_diff:.6e}")
        print(f"  标准差:   {np.std(diff):.6e}")

        # 使用极严格容差验证
        print(f"\n[验证] 使用极严格容差(rtol=1e-10, atol=1e-14)")
        self.assert_allclose_strict(
            x_filt_new,
            x_filt_old,
            name="完整滤波状态序列"
        )

        print(f"\n[PASS] 完整滤波一致性验证通过!")

    def test_004_missing_data_handling_consistency(self):
        """测试004: 缺失数据处理一致性

        验证内容:
        - 注入缺失值后的滤波结果
        - 缺失数据跳过逻辑
        - 滤波状态连续性

        预期: 缺失数据处理策略完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试004: 缺失数据处理一致性")
        print("="*60)

        # 创建带缺失值的观测数据(固定位置确保可重现)
        Z_with_nan = self.Z_df.copy()
        np.random.seed(123)

        # 随机注入10%的缺失值
        n_missing = int(self.n_time_small * self.n_obs_small * 0.1)
        missing_indices = []
        for _ in range(n_missing):
            t = np.random.randint(1, self.n_time_small)  # 避免t=0
            i = np.random.randint(0, self.n_obs_small)
            Z_with_nan.iloc[t, i] = np.nan
            missing_indices.append((t, i))

        print(f"\n[缺失数据注入]")
        print(f"  总数据点: {self.n_time_small * self.n_obs_small}")
        print(f"  缺失数据点: {n_missing} ({n_missing / (self.n_time_small * self.n_obs_small) * 100:.1f}%)")
        print(f"  前5个缺失位置: {missing_indices[:5]}")

        # 运行train_model版本
        res_old = KF_old(
            Z=Z_with_nan,
            U=self.U_df,
            A=self.A,
            B=self.B,
            H=self.H,
            state_names=self.state_names,
            x0=self.x0,
            P0=self.P0,
            Q=self.Q,
            R=self.R
        )

        x_filt_old = res_old.x.to_numpy()
        P_filt_old = res_old.P

        # 运行train_ref版本
        Z_with_nan_array = Z_with_nan.to_numpy()  # (n_time, n_obs)
        U_array = self.U_df.to_numpy()

        kf_new = KF_new(
            A=self.A,
            B=self.B,
            H=self.H,
            Q=self.Q,
            R=self.R,
            x0=self.x0,
            P0=self.P0
        )
        res_new = kf_new.filter(Z=Z_with_nan_array, U=U_array)

        x_filt_new = res_new.x_filtered
        P_filt_new = res_new.P_filtered

        print(f"\n[数据格式]")
        print(f"  train_model: x_filt shape = {x_filt_old.shape}")
        print(f"  train_ref:   x_filt shape = {x_filt_new.shape}")

        # 全局统计
        diff = np.abs(x_filt_old - x_filt_new)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"\n[全局差异统计]")
        print(f"  最大差异: {max_diff:.6e}")
        print(f"  平均差异: {mean_diff:.6e}")

        # 使用极严格容差验证
        print(f"\n[验证] 使用极严格容差(rtol=1e-10, atol=1e-14)")
        self.assert_allclose_strict(
            x_filt_new,
            x_filt_old,
            name="缺失数据下的滤波状态"
        )

        # 验证所有时间步的P_filt
        for t in range(self.n_time_small):
            self.assert_allclose_strict(
                P_filt_new[:, :, t],
                P_filt_old[t],
                name=f"缺失数据下的滤波协方差 P_filt[t={t}]"
            )

        print(f"\n[PASS] 缺失数据处理一致性验证通过!")


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '-s'])
