# -*- coding: utf-8 -*-
"""
Phase 5: 核心端到端一致性测试

测试范围:
1. 给定相同数据和超参数,完整训练流程的一致性
2. 不同因子数配置的一致性 (k=1,2,3)
3. 不同迭代次数的一致性
4. 预测性能指标的一致性

测试策略:
- 使用模拟数据集,跳过变量选择逻辑
- 直接对比DFM训练核心流程
- 验证最终模型参数和预测结果

数值容差: rtol=1e-10, atol=1e-14
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# 导入测试基类
from dashboard.DFM.train_ref.tests.consistency.base import ConsistencyTestBase

# 导入train_ref组件
from dashboard.DFM.train_ref.core.factor_model import DFMModel

# 导入train_model组件
from dashboard.DFM.train_model.DynamicFactorModel import DFM_EMalgo


class TestEndToEndCore(ConsistencyTestBase):
    """核心端到端一致性测试类

    测试内容:
    - 完整DFM训练流程(PCA初始化 + EM迭代 + Kalman估计)
    - 不同超参数配置一致性
    - 预测性能指标一致性
    """

    def setup_method(self):
        """测试前准备"""
        # 加载标准数据集
        self.small_dataset = self.load_simulated_dataset('small')
        self.medium_dataset = self.load_simulated_dataset('medium')

        # 提取小数据集参数
        self.Z_df = self.small_dataset['Z']
        self.n_time, self.n_obs = self.Z_df.shape  # DataFrame格式: (n_time, n_obs)
        self.n_factors_small = self.small_dataset['n_factors']

        print(f"\n[Setup] 数据集加载完成:")
        print(f"  时间步数: {self.n_time}")
        print(f"  观测变量数: {self.n_obs}")
        print(f"  因子数: {self.n_factors_small}")

    def test_001_basic_end_to_end_consistency(self):
        """测试001: 基础端到端训练一致性

        验证内容:
        - 相同数据和超参数 → 相同模型参数
        - Lambda, A, Q, R矩阵一致性
        - 平滑因子一致性
        - 预测结果一致性

        预期结果: 所有差异 < 1e-10
        """
        print("\n" + "="*60)
        print("测试001: 基础端到端训练一致性")
        print("="*60)

        # 配置参数
        k_factors = self.n_factors_small
        max_iterations = 10
        tolerance = 1e-6
        max_lags = 1

        print(f"\n[配置参数]")
        print(f"  k_factors: {k_factors}")
        print(f"  max_iterations: {max_iterations}")
        print(f"  tolerance: {tolerance}")
        print(f"  max_lags: {max_lags}")

        # ===== Step 1: train_model 训练 =====
        print(f"\n[Step 1] 运行train_model完整训练...")

        # 准备数据
        Z_df_old = self.Z_df.copy()

        # 调用train_model的DFM_EMalgo
        try:
            result_old = DFM_EMalgo(
                observation=Z_df_old,
                n_factors=k_factors,
                n_shocks=k_factors,
                n_iter=max_iterations,
                max_lags=max_lags
            )

            # 提取结果 (DFMEMResultsWrapper对象,使用属性访问)
            Lambda_old = result_old.Lambda
            A_old = result_old.A
            Q_old = result_old.Q
            R_old = result_old.R
            x_smooth_old = result_old.x_sm.values.T  # DataFrame (n_time, n_factors) → ndarray (n_factors, n_time)

            print(f"  train_model训练完成")
            print(f"  Lambda shape: {Lambda_old.shape}")
            print(f"  A shape: {A_old.shape}")
            print(f"  Q shape: {Q_old.shape}")
            print(f"  R shape: {R_old.shape}")
            print(f"  x_smooth shape: {x_smooth_old.shape}")

        except Exception as e:
            pytest.fail(f"train_model训练失败: {e}")

        # ===== Step 2: train_ref 训练 =====
        print(f"\n[Step 2] 运行train_ref完整训练...")

        # 准备数据
        Z_df_new = self.Z_df.copy()

        # 创建DFMModel并训练
        try:
            model = DFMModel(
                n_factors=k_factors,
                max_lags=max_lags,
                max_iter=max_iterations,
                tolerance=tolerance
            )
            results = model.fit(Z_df_new)

            # 提取结果 (DFMResults对象)
            Lambda_new = results.loadings
            A_new = results.transition_matrix
            Q_new = results.process_noise_cov
            R_new = results.measurement_noise_cov
            x_smooth_new = results.factors.values.T  # factors是DataFrame,转为ndarray (n_factors, n_time)

            print(f"  train_ref训练完成")
            print(f"  Lambda shape: {Lambda_new.shape}")
            print(f"  A shape: {A_new.shape}")
            print(f"  Q shape: {Q_new.shape}")
            print(f"  R shape: {R_new.shape}")
            print(f"  x_smooth shape: {x_smooth_new.shape}")

        except Exception as e:
            pytest.fail(f"train_ref训练失败: {e}")

        # ===== Step 3: 对比模型参数 =====
        print(f"\n[Step 3] 对比模型参数...")

        # Lambda对比
        print(f"\n[Lambda对比]")
        Lambda_diff = np.abs(Lambda_old - Lambda_new)
        print(f"  最大差异: {np.max(Lambda_diff):.6e}")
        print(f"  平均差异: {np.mean(Lambda_diff):.6e}")
        print(f"  Lambda_old[0,:]: {Lambda_old[0,:]}")
        print(f"  Lambda_new[0,:]: {Lambda_new[0,:]}")

        # A对比
        print(f"\n[A对比]")
        A_diff = np.abs(A_old - A_new)
        print(f"  最大差异: {np.max(A_diff):.6e}")
        print(f"  平均差异: {np.mean(A_diff):.6e}")
        print(f"  A_old:\n{A_old}")
        print(f"  A_new:\n{A_new}")

        # Q对比
        print(f"\n[Q对比]")
        Q_diff = np.abs(Q_old - Q_new)
        print(f"  最大差异: {np.max(Q_diff):.6e}")
        print(f"  平均差异: {np.mean(Q_diff):.6e}")

        # R对比 (对角矩阵)
        print(f"\n[R对比]")
        R_diag_old = np.diag(R_old)
        R_diag_new = np.diag(R_new)
        R_diff = np.abs(R_diag_old - R_diag_new)
        print(f"  对角元素最大差异: {np.max(R_diff):.6e}")
        print(f"  对角元素平均差异: {np.mean(R_diff):.6e}")

        # 平滑因子对比
        print(f"\n[平滑因子对比]")
        # train_model: x_smooth shape = (n_factors + n_shocks, n_time)
        # train_ref: x_smooth shape = (n_factors + n_shocks, n_time)
        # 只对比前n_factors行 (因子部分)
        x_smooth_old_factors = x_smooth_old[:k_factors, :]
        x_smooth_new_factors = x_smooth_new[:k_factors, :]

        x_smooth_diff = np.abs(x_smooth_old_factors - x_smooth_new_factors)
        print(f"  最大差异: {np.max(x_smooth_diff):.6e}")
        print(f"  平均差异: {np.mean(x_smooth_diff):.6e}")

        # ===== Step 4: 验证 =====
        print(f"\n[Step 4] 使用严格容差验证...")

        # Lambda验证
        self.assert_allclose_strict(
            Lambda_old, Lambda_new,
            name="Lambda矩阵",
            rtol=1e-10, atol=1e-14
        )

        # A验证
        self.assert_allclose_strict(
            A_old, A_new,
            name="A矩阵",
            rtol=1e-10, atol=1e-14
        )

        # Q验证
        self.assert_allclose_strict(
            Q_old, Q_new,
            name="Q矩阵",
            rtol=1e-10, atol=1e-14
        )

        # R验证
        self.assert_allclose_strict(
            R_diag_old, R_diag_new,
            name="R矩阵对角元素",
            rtol=1e-10, atol=1e-14
        )

        # 平滑因子验证
        self.assert_allclose_strict(
            x_smooth_old_factors, x_smooth_new_factors,
            name="平滑因子",
            rtol=1e-10, atol=1e-14
        )

        print(f"\n[PASS] 基础端到端训练一致性验证通过!")


    def test_002_different_k_factors_consistency(self):
        """测试002: 不同因子数配置一致性

        验证内容:
        - k=1, k=2, k=3 的训练结果一致性
        - 每个k下的模型参数维度正确
        - 每个k下的数值一致性

        预期结果: 所有k值下,差异 < 1e-10
        """
        print("\n" + "="*60)
        print("测试002: 不同因子数配置一致性")
        print("="*60)

        k_factors_list = [1, 2, 3]
        max_iterations = 5  # 减少迭代次数加快测试

        for k in k_factors_list:
            print(f"\n{'='*60}")
            print(f"测试 k={k}")
            print(f"{'='*60}")

            # train_model训练
            print(f"\n[train_model] k={k}")
            try:
                result_old = DFM_EMalgo(
                    observation=self.Z_df.copy(),
                    n_factors=k,
                    n_shocks=k,
                    n_iter=max_iterations,
                    max_lags=1
                )
                Lambda_old = result_old.Lambda
                A_old = result_old.A
                print(f"  Lambda shape: {Lambda_old.shape}")
                print(f"  A shape: {A_old.shape}")
            except Exception as e:
                pytest.fail(f"train_model k={k} 失败: {e}")

            # train_ref训练
            print(f"\n[train_ref] k={k}")
            try:
                model = DFMModel(
                    n_factors=k,
                    max_lags=1,
                    max_iter=max_iterations,
                    tolerance=1e-6
                )
                results = model.fit(self.Z_df.copy())
                Lambda_new = results.loadings
                A_new = results.transition_matrix
                print(f"  Lambda shape: {Lambda_new.shape}")
                print(f"  A shape: {A_new.shape}")
            except Exception as e:
                pytest.fail(f"train_ref k={k} 失败: {e}")

            # 验证维度
            assert Lambda_old.shape == Lambda_new.shape, f"k={k}: Lambda维度不一致"
            assert A_old.shape == A_new.shape, f"k={k}: A维度不一致"
            assert Lambda_old.shape == (self.n_obs, k), f"k={k}: Lambda维度错误"
            assert A_old.shape == (k, k), f"k={k}: A维度错误"

            # 数值对比
            print(f"\n[对比] k={k}")
            Lambda_diff = np.max(np.abs(Lambda_old - Lambda_new))
            A_diff = np.max(np.abs(A_old - A_new))
            print(f"  Lambda最大差异: {Lambda_diff:.6e}")
            print(f"  A最大差异: {A_diff:.6e}")

            # 严格验证
            self.assert_allclose_strict(
                Lambda_old, Lambda_new,
                name=f"k={k} Lambda矩阵",
                rtol=1e-10, atol=1e-14
            )
            self.assert_allclose_strict(
                A_old, A_new,
                name=f"k={k} A矩阵",
                rtol=1e-10, atol=1e-14
            )

            print(f"[PASS] k={k} 一致性验证通过")

        print(f"\n[PASS] 不同因子数配置一致性验证通过!")


    def test_003_different_iterations_consistency(self):
        """测试003: 不同迭代次数配置一致性

        验证内容:
        - max_iter=5, 10, 15 的训练结果一致性
        - 迭代次数增加不影响收敛参数一致性

        预期结果: 所有迭代次数下,差异 < 1e-10
        """
        print("\n" + "="*60)
        print("测试003: 不同迭代次数配置一致性")
        print("="*60)

        iterations_list = [5, 10, 15]
        k_factors = 2

        for max_iter in iterations_list:
            print(f"\n{'='*60}")
            print(f"测试 max_iter={max_iter}")
            print(f"{'='*60}")

            # train_model训练
            print(f"\n[train_model] max_iter={max_iter}")
            result_old = DFM_EMalgo(
                observation=self.Z_df.copy(),
                n_factors=k_factors,
                n_shocks=k_factors,
                n_iter=max_iter,
                max_lags=1
            )
            Lambda_old = result_old.Lambda

            # train_ref训练
            print(f"\n[train_ref] max_iter={max_iter}")
            model = DFMModel(
                n_factors=k_factors,
                max_lags=1,
                max_iter=max_iter,
                tolerance=1e-6
            )
            results = model.fit(self.Z_df.copy())
            Lambda_new = results.loadings

            # 对比
            print(f"\n[对比] max_iter={max_iter}")
            Lambda_diff = np.max(np.abs(Lambda_old - Lambda_new))
            print(f"  Lambda最大差异: {Lambda_diff:.6e}")

            # 严格验证
            self.assert_allclose_strict(
                Lambda_old, Lambda_new,
                name=f"max_iter={max_iter} Lambda矩阵",
                rtol=1e-10, atol=1e-14
            )

            print(f"[PASS] max_iter={max_iter} 一致性验证通过")

        print(f"\n[PASS] 不同迭代次数配置一致性验证通过!")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, '-v', '-s'])
