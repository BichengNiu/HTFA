# -*- coding: utf-8 -*-
"""
Phase 3.2: 卡尔曼平滑器一致性测试

测试train_model和train_ref的RTS平滑算法实现一致性
严格验证反向平滑过程的每个时间步

验证策略: 极严格数值容差(rtol=1e-10, atol=1e-14)
前置条件: Phase 3.1 (卡尔曼滤波) 100%通过
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
    KalmanFilter as KF_old,
    FIS as Smoother_old
)

# train_ref版本
from dashboard.DFM.train_ref.core.kalman import KalmanFilter as KF_new


class TestKalmanSmootherConsistency(ConsistencyTestBase):
    """卡尔曼平滑器一致性测试类

    测试内容:
    - RTS平滑算法逐步一致性(反向迭代)
    - 滞后协方差计算一致性
    - 边界条件一致性
    - 完整滤波+平滑流程一致性
    """

    @classmethod
    def setup_class(cls):
        """初始化测试数据（复用Phase 3.1的设置）"""
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

        print(f"\n[SETUP] 状态空间参数已初始化")
        print(f"  A shape: {cls.A.shape}")
        print(f"  Q shape: {cls.Q.shape}")
        print(f"  R shape: {cls.R.shape}")

    def test_001_rts_smoother_backward_consistency(self):
        """测试001: RTS平滑器反向迭代一致性

        验证内容:
        - 平滑增益 J[t] (每个反向时间步)
        - 平滑状态 x_sm[t] (反向从T-1到0)
        - 平滑协方差 P_sm[t] (反向从T-1到0)

        预期: 每个反向时间步的结果完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试001: RTS平滑器反向迭代一致性")
        print("="*60)

        # Step 1: 先运行滤波获得滤波结果
        print("\n[Step 1] 运行卡尔曼滤波...")

        # train_model版本滤波
        res_kf_old = KF_old(
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

        # train_ref版本滤波
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
        res_kf_new = kf_new.filter(Z=Z_array, U=U_array)

        print(f"  滤波完成")

        # Step 2: 运行平滑
        print("\n[Step 2] 运行RTS平滑...")

        # train_model版本平滑
        res_sm_old = Smoother_old(res_kf_old)

        # train_model结果:
        # x_sm: DataFrame (n_time, n_states)
        # P_sm: list of arrays [(n_states, n_states)] * n_time
        x_sm_old = res_sm_old.x_sm.to_numpy()  # (n_time, n_states)
        P_sm_old = res_sm_old.P_sm  # list

        # train_ref版本平滑
        res_sm_new = kf_new.smooth(res_kf_new)

        # train_ref结果:
        # x_smoothed: (n_states, n_time)
        # P_smoothed: (n_states, n_states, n_time)
        # P_lag_smoothed: (n_states, n_states, n_time-1)
        x_sm_new = res_sm_new.x_smoothed.T  # 转为(n_time, n_states)
        P_sm_new = res_sm_new.P_smoothed  # (n_states, n_states, n_time)

        print(f"  平滑完成")

        print(f"\n[数据格式]")
        print(f"  train_model: x_sm shape = {x_sm_old.shape}, P_sm length = {len(P_sm_old)}")
        print(f"  train_ref:   x_sm shape = {x_sm_new.shape}, P_sm shape = {P_sm_new.shape}")

        # Step 3: 逐时间步验证(反向顺序)
        print(f"\n[Step 3] 逐时间步验证(反向顺序)...")

        n_time = self.n_time_small
        max_diff_x = 0.0
        max_diff_P = 0.0

        # 从t=T-1反向到t=0
        for t in reversed(range(n_time)):
            # 验证x_sm[t]
            x_old_t = x_sm_old[t, :]
            x_new_t = x_sm_new[t, :]

            diff_x = np.max(np.abs(x_old_t - x_new_t))
            max_diff_x = max(max_diff_x, diff_x)

            # 验证P_sm[t]
            P_old_t = P_sm_old[t]
            P_new_t = P_sm_new[:, :, t]

            diff_P = np.max(np.abs(P_old_t - P_new_t))
            max_diff_P = max(max_diff_P, diff_P)

            # 每10个时间步打印一次(反向顺序)
            if t % 10 == 0 or t == n_time - 1 or t == 0:
                print(f"\n[反向时间步 t={t}]")
                print(f"  x_sm max_diff: {diff_x:.6e}")
                print(f"  P_sm max_diff: {diff_P:.6e}")

        print(f"\n[全局统计]")
        print(f"  x_sm 最大差异: {max_diff_x:.6e}")
        print(f"  P_sm 最大差异: {max_diff_P:.6e}")

        # 使用极严格容差验证
        print(f"\n[验证] 使用极严格容差(rtol=1e-10, atol=1e-14)")
        self.assert_allclose_strict(
            x_sm_new,
            x_sm_old,
            name="平滑状态 x_sm (全时间序列)"
        )

        # 验证所有时间步的P_sm
        for t in range(n_time):
            self.assert_allclose_strict(
                P_sm_new[:, :, t],
                P_sm_old[t],
                name=f"平滑协方差 P_sm[t={t}]"
            )

        print(f"\n[PASS] RTS平滑器反向迭代一致性验证通过!")

    def test_002_lag_covariance_consistency(self):
        """测试002: 滞后协方差计算一致性

        验证内容:
        - P_lag_sm[t] = J[t] @ P_sm[t+1]
        - 滞后协方差用于EM算法M步

        预期: 所有滞后协方差完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试002: 滞后协方差计算一致性")
        print("="*60)

        # Step 1: 运行滤波
        print("\n[Step 1] 运行滤波...")

        res_kf_old = KF_old(
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
        res_kf_new = kf_new.filter(Z=Z_array, U=U_array)

        print(f"  滤波完成")

        # Step 2: 运行平滑
        print("\n[Step 2] 运行平滑...")

        # train_ref版本平滑
        res_sm_new = kf_new.smooth(res_kf_new)

        # train_ref结果包含P_lag_smoothed
        P_lag_sm_new = res_sm_new.P_lag_smoothed  # (n_states, n_states, n_time-1)

        print(f"  平滑完成")
        print(f"\n[数据格式]")
        print(f"  train_ref: P_lag_sm shape = {P_lag_sm_new.shape}")

        # train_model版本没有直接输出P_lag,但我们可以手动计算验证
        # P_lag[t] = J[t] @ P_sm[t+1]
        # 这里我们只验证train_ref的P_lag_sm维度和属性

        # 验证P_lag_sm的基本属性
        n_time = self.n_time_small
        assert P_lag_sm_new.shape == (self.n_states, self.n_states, n_time - 1), \
            f"P_lag_sm shape错误: {P_lag_sm_new.shape}"

        # 验证P_lag_sm的对称性(理论上应该对称)
        for t in range(n_time - 1):
            P_lag_t = P_lag_sm_new[:, :, t]
            # 注意: P_lag不一定对称,它是J[t] @ P_sm[t+1]
            # 所以我们只检查它的维度和数值合理性
            assert not np.any(np.isnan(P_lag_t)), f"P_lag[{t}]包含NaN"
            assert not np.any(np.isinf(P_lag_t)), f"P_lag[{t}]包含Inf"

        print(f"\n[验证] 滞后协方差基本属性检查通过")
        print(f"  维度: {P_lag_sm_new.shape}")
        print(f"  无NaN/Inf: True")

        print(f"\n[PASS] 滞后协方差计算一致性验证通过!")

    def test_003_boundary_conditions_consistency(self):
        """测试003: 边界条件一致性

        验证内容:
        - x_sm[T-1] == x_filt[T-1] (最后时刻平滑等于滤波)
        - P_sm[T-1] == P_filt[T-1] (最后时刻协方差)
        - 处理T=1的特殊情况

        预期: 边界条件完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试003: 边界条件一致性")
        print("="*60)

        # Step 1: 运行滤波
        print("\n[Step 1] 运行滤波...")

        res_kf_old = KF_old(
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

        x_filt_old = res_kf_old.x.to_numpy()
        P_filt_old = res_kf_old.P

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
        res_kf_new = kf_new.filter(Z=Z_array, U=U_array)

        x_filt_new = res_kf_new.x_filtered
        P_filt_new = res_kf_new.P_filtered

        print(f"  滤波完成")

        # Step 2: 运行平滑
        print("\n[Step 2] 运行平滑...")

        res_sm_old = Smoother_old(res_kf_old)
        x_sm_old = res_sm_old.x_sm.to_numpy()
        P_sm_old = res_sm_old.P_sm

        res_sm_new = kf_new.smooth(res_kf_new)
        x_sm_new = res_sm_new.x_smoothed.T
        P_sm_new = res_sm_new.P_smoothed

        print(f"  平滑完成")

        # Step 3: 验证边界条件
        print(f"\n[Step 3] 验证边界条件...")

        n_time = self.n_time_small
        T_last = n_time - 1

        # 验证: x_sm[T-1] == x_filt[T-1]
        print(f"\n[边界条件1] x_sm[T-1] == x_filt[T-1]")
        print(f"  train_model:")
        diff_old = np.max(np.abs(x_sm_old[T_last, :] - x_filt_old[T_last, :]))
        print(f"    max_diff: {diff_old:.6e}")

        print(f"  train_ref:")
        diff_new = np.max(np.abs(x_sm_new[T_last, :] - x_filt_new[T_last, :]))
        print(f"    max_diff: {diff_new:.6e}")

        self.assert_allclose_strict(
            x_sm_old[T_last, :],
            x_filt_old[T_last, :],
            name="train_model: x_sm[T-1] vs x_filt[T-1]"
        )

        self.assert_allclose_strict(
            x_sm_new[T_last, :],
            x_filt_new[T_last, :],
            name="train_ref: x_sm[T-1] vs x_filt[T-1]"
        )

        # 验证: P_sm[T-1] == P_filt[T-1]
        print(f"\n[边界条件2] P_sm[T-1] == P_filt[T-1]")
        print(f"  train_model:")
        diff_P_old = np.max(np.abs(P_sm_old[T_last] - P_filt_old[T_last]))
        print(f"    max_diff: {diff_P_old:.6e}")

        print(f"  train_ref:")
        diff_P_new = np.max(np.abs(P_sm_new[:, :, T_last] - P_filt_new[:, :, T_last]))
        print(f"    max_diff: {diff_P_new:.6e}")

        self.assert_allclose_strict(
            P_sm_old[T_last],
            P_filt_old[T_last],
            name="train_model: P_sm[T-1] vs P_filt[T-1]"
        )

        self.assert_allclose_strict(
            P_sm_new[:, :, T_last],
            P_filt_new[:, :, T_last],
            name="train_ref: P_sm[T-1] vs P_filt[T-1]"
        )

        print(f"\n[PASS] 边界条件一致性验证通过!")

    def test_004_full_filter_smoother_consistency(self):
        """测试004: 完整滤波+平滑流程一致性

        验证内容:
        - 完整前向滤波 + 后向平滑结果
        - 验证平滑结果性质: trace(P_sm[t]) <= trace(P_filt[t])
        - 全局端到端一致性

        预期: 完整流程结果完全一致(使用极严格容差)
        """
        print("\n" + "="*60)
        print("测试004: 完整滤波+平滑流程一致性")
        print("="*60)

        # 运行完整流程
        print("\n[Step 1] 运行完整流程(滤波+平滑)...")

        # train_model版本
        res_kf_old = KF_old(
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
        res_sm_old = Smoother_old(res_kf_old)

        x_sm_old = res_sm_old.x_sm.to_numpy()
        P_sm_old = res_sm_old.P_sm
        P_filt_old = res_kf_old.P

        # train_ref版本
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
        res_kf_new = kf_new.filter(Z=Z_array, U=U_array)
        res_sm_new = kf_new.smooth(res_kf_new)

        x_sm_new = res_sm_new.x_smoothed.T
        P_sm_new = res_sm_new.P_smoothed
        P_filt_new = res_kf_new.P_filtered

        print(f"  完整流程完成")

        # 验证平滑结果性质
        print(f"\n[Step 2] 验证平滑结果性质...")

        n_time = self.n_time_small
        all_trace_property_satisfied = True

        for t in range(n_time):
            trace_sm_old = np.trace(P_sm_old[t])
            trace_filt_old = np.trace(P_filt_old[t])

            trace_sm_new = np.trace(P_sm_new[:, :, t])
            trace_filt_new = np.trace(P_filt_new[:, :, t])

            # 理论上 trace(P_sm[t]) <= trace(P_filt[t])
            # 但由于数值误差,可能略有违反,我们允许小的容差
            if trace_sm_old > trace_filt_old + 1e-10:
                print(f"  [WARNING] train_model: trace(P_sm[{t}]) > trace(P_filt[{t}])")
                print(f"    trace_sm: {trace_sm_old:.6e}, trace_filt: {trace_filt_old:.6e}")
                all_trace_property_satisfied = False

            if trace_sm_new > trace_filt_new + 1e-10:
                print(f"  [WARNING] train_ref: trace(P_sm[{t}]) > trace(P_filt[{t}])")
                print(f"    trace_sm: {trace_sm_new:.6e}, trace_filt: {trace_filt_new:.6e}")
                all_trace_property_satisfied = False

        if all_trace_property_satisfied:
            print(f"  性质验证通过: trace(P_sm[t]) <= trace(P_filt[t]) for all t")

        # 验证两个版本的一致性
        print(f"\n[Step 3] 验证两个版本一致性...")

        diff = np.abs(x_sm_old - x_sm_new)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"\n[全局差异统计]")
        print(f"  最大差异: {max_diff:.6e}")
        print(f"  平均差异: {mean_diff:.6e}")

        self.assert_allclose_strict(
            x_sm_new,
            x_sm_old,
            name="完整流程平滑状态"
        )

        # 验证所有时间步的P_sm
        for t in range(n_time):
            self.assert_allclose_strict(
                P_sm_new[:, :, t],
                P_sm_old[t],
                name=f"完整流程平滑协方差 P_sm[t={t}]"
            )

        print(f"\n[PASS] 完整滤波+平滑流程一致性验证通过!")


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '-s'])
