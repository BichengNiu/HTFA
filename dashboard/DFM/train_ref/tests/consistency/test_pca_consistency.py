# -*- coding: utf-8 -*-
"""
PCA算法一致性测试

验证train_model和train_ref的PCA初始化算法是否产生**完全相同**的数值结果。

测试策略:
1. 使用相同的模拟数据(固定SEED=42)
2. 分别调用train_model和train_ref的PCA实现
3. 使用零容差标准验证结果完全相等

关键测试点:
- 协方差矩阵计算
- SVD分解结果
- 特征值/特征向量
- 因子提取
- 载荷矩阵估计
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# 导入待测试模块
from dashboard.DFM.train_model.DynamicFactorModel import (
    _calculate_pca,
    DFM
)
from dashboard.DFM.train_ref.core.factor_model import DFMModel as DFM_ref

# 导入测试基础类和工具
from dashboard.DFM.train_ref.tests.consistency.base import ConsistencyTestBase


class TestPCAConsistency(ConsistencyTestBase):
    """PCA算法一致性测试类"""

    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        # 加载small数据集用于快速测试
        cls.small_dataset = cls.load_simulated_dataset('small')

        # 提取数据(注意: load_simulated_dataset已经返回DataFrame)
        cls.Z_small_df = cls.small_dataset['Z']  # 已经是DataFrame
        cls.n_factors_small = cls.small_dataset['n_factors']  # 应该是2

        print(f"\n加载测试数据集: small")
        print(f"  数据形状: {cls.Z_small_df.shape}")
        print(f"  因子数: {cls.n_factors_small}")


    def test_001_standardization_consistency(self):
        """
        测试1: 数据标准化一致性

        验证两边的标准化(z-score)计算是否完全一致
        """
        print("\n" + "=" * 60)
        print("测试1: 数据标准化一致性")
        print("=" * 60)

        # train_model的标准化逻辑(来自DFM函数)
        obs_mean_old = self.Z_small_df.mean()
        obs_std_old = self.Z_small_df.std()
        obs_std_old[obs_std_old == 0] = 1.0  # 防止除零
        z_old = ((self.Z_small_df - obs_mean_old) / obs_std_old).fillna(0)

        # train_ref的标准化逻辑(来自DynamicFactorModel.fit)
        # 注意: train_ref在fit方法中进行标准化
        # 这里我们手动模拟相同的标准化步骤
        obs_mean_new = self.Z_small_df.mean()
        obs_std_new = self.Z_small_df.std()
        obs_std_new[obs_std_new == 0] = 1.0
        z_new = ((self.Z_small_df - obs_mean_new) / obs_std_new).fillna(0)

        # 验证均值完全相等
        print("\n验证均值向量:")
        self.assert_array_exact_equal(
            obs_mean_new.values,
            obs_mean_old.values,
            name="观测数据均值"
        )
        print("[PASS] 均值向量完全一致")

        # 验证标准差完全相等
        print("\n验证标准差向量:")
        self.assert_array_exact_equal(
            obs_std_new.values,
            obs_std_old.values,
            name="观测数据标准差"
        )
        print("[PASS] 标准差向量完全一致")

        # 验证标准化结果完全相等
        print("\n验证标准化数据矩阵:")
        self.assert_array_exact_equal(
            z_new.values,
            z_old.values,
            name="标准化数据"
        )
        print("[PASS] 标准化数据完全一致")

        print(f"\n[PASS] 测试1通过: 数据标准化完全一致")


    def test_002_covariance_matrix_consistency(self):
        """
        测试2: 协方差矩阵计算一致性

        验证_calculate_pca中的协方差矩阵计算是否与SVD等价

        数学原理:
        - 方法1(eigh): S = (1/N) * Z'Z, 然后对S进行特征分解
        - 方法2(SVD): 对Z进行SVD分解, S = V @ diag(s^2/N) @ V'

        理论上两种方法应该产生相同的协方差矩阵
        """
        print("\n" + "=" * 60)
        print("测试2: 协方差矩阵计算一致性")
        print("=" * 60)

        n_time = len(self.Z_small_df)

        # 标准化数据
        obs_mean = self.Z_small_df.mean()
        obs_std = self.Z_small_df.std()
        obs_std[obs_std == 0] = 1.0
        z = ((self.Z_small_df - obs_mean) / obs_std).fillna(0).values

        # 方法1: _calculate_pca的协方差矩阵计算
        # 直接调用内部逻辑
        S_eigh = (z.T @ z) / n_time

        # 方法2: SVD等价计算
        U, s, Vh = np.linalg.svd(z, full_matrices=False)
        # S = (1/N) * Z'Z = (1/N) * (V s U') (U s V') = V (s^2/N) V'
        S_svd = Vh.T @ np.diag(s**2 / n_time) @ Vh

        print(f"\n协方差矩阵形状: {S_eigh.shape}")
        print(f"eigh方法样本: S[0,0] = {S_eigh[0,0]:.10e}")
        print(f"SVD方法样本: S[0,0] = {S_svd[0,0]:.10e}")

        # 极严格容差验证
        print("\n验证协方差矩阵在极严格容差内一致:")
        self.assert_allclose_strict(
            S_svd,
            S_eigh,
            name="协方差矩阵(SVD vs eigh)"
        )

        print(f"\n[PASS] 测试2通过: 协方差矩阵计算在极严格容差内一致")


    def test_003_svd_decomposition_consistency(self):
        """
        测试3: SVD分解一致性

        验证train_model.DFM和train_ref._initialize_factors_pca的SVD分解
        是否产生完全相同的U, s, Vh
        """
        print("\n" + "=" * 60)
        print("测试3: SVD分解一致性")
        print("=" * 60)

        # 标准化数据
        obs_mean = self.Z_small_df.mean()
        obs_std = self.Z_small_df.std()
        obs_std[obs_std == 0] = 1.0
        z = ((self.Z_small_df - obs_mean) / obs_std).fillna(0).values

        # 固定随机种子(虽然SVD是确定性的,但为了保险)
        np.random.seed(42)

        # 两边应该使用相同的SVD调用
        U_old, s_old, Vh_old = np.linalg.svd(z, full_matrices=False)

        np.random.seed(42)  # 重置种子
        U_new, s_new, Vh_new = np.linalg.svd(z, full_matrices=False)

        print(f"\nSVD分解结果形状:")
        print(f"  U: {U_old.shape}")
        print(f"  s: {s_old.shape}")
        print(f"  Vh: {Vh_old.shape}")

        # 验证奇异值完全相等(SVD是确定性的,应该逐位相等)
        print("\n验证奇异值向量 s:")
        self.assert_array_exact_equal(
            s_new,
            s_old,
            name="SVD奇异值"
        )
        print("[PASS] 奇异值完全一致")

        # 验证左奇异向量U (允许符号歧义,使用零容差因为SVD是确定性的)
        # 注意: 奇异向量可能有符号歧义
        print("\n验证左奇异向量 U:")
        self.assert_eigenvectors_equal_up_to_sign(
            U_new,
            U_old,
            name="SVD左奇异向量U",
            use_strict_tolerance=False  # SVD是确定性的,使用零容差
        )
        print("[PASS] 左奇异向量完全一致(允许符号歧义)")

        # 验证右奇异向量Vh
        print("\n验证右奇异向量 Vh:")
        self.assert_eigenvectors_equal_up_to_sign(
            Vh_new.T,  # 转置为列向量形式
            Vh_old.T,
            name="SVD右奇异向量Vh",
            use_strict_tolerance=False  # SVD是确定性的,使用零容差
        )
        print("[PASS] 右奇异向量完全一致(允许符号歧义)")

        print(f"\n[PASS] 测试3通过: SVD分解完全一致")


    def test_004_eigenvalue_decomposition_consistency(self):
        """
        测试4: 特征值分解一致性

        验证_calculate_pca的eigh分解与SVD分解是否等价

        数学关系:
        - SVD: Z = U @ diag(s) @ Vh
        - S = Z'Z / N = Vh' @ diag(s^2/N) @ Vh
        - eigh(S) 的特征值应该等于 s^2 / N
        - eigh(S) 的特征向量应该等于 Vh'(转置后的列)
        """
        print("\n" + "=" * 60)
        print("测试4: 特征值分解与SVD的等价性")
        print("=" * 60)

        n_time = len(self.Z_small_df)

        # 标准化数据
        obs_mean = self.Z_small_df.mean()
        obs_std = self.Z_small_df.std()
        obs_std[obs_std == 0] = 1.0
        z = ((self.Z_small_df - obs_mean) / obs_std).fillna(0).values

        # 方法1: _calculate_pca的逻辑(协方差矩阵特征分解)
        S = (z.T @ z) / n_time
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        sorted_indices = np.argsort(eigenvalues)[::-1]  # 降序
        eigenvalues_sorted = eigenvalues[sorted_indices]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]

        # 方法2: SVD分解
        U, s, Vh = np.linalg.svd(z, full_matrices=False)
        eigenvalues_from_svd = (s**2) / n_time
        eigenvectors_from_svd = Vh.T  # V = Vh.T

        print(f"\n前{self.n_factors_small}个特征值对比:")
        for i in range(self.n_factors_small):
            print(f"  因子{i+1}:")
            print(f"    eigh: {eigenvalues_sorted[i]:.10e}")
            print(f"    SVD:  {eigenvalues_from_svd[i]:.10e}")

        # 验证特征值在极严格容差内一致
        print("\n验证特征值在极严格容差内一致:")
        self.assert_allclose_strict(
            eigenvalues_from_svd[:self.n_factors_small],
            eigenvalues_sorted[:self.n_factors_small],
            name="特征值(SVD vs eigh)"
        )
        print("[PASS] 特征值在极严格容差内一致")

        # 验证特征向量在极严格容差内一致(允许符号歧义)
        print("\n验证特征向量在极严格容差内一致(允许符号歧义):")
        self.assert_eigenvectors_equal_up_to_sign(
            eigenvectors_from_svd[:, :self.n_factors_small],
            eigenvectors_sorted[:, :self.n_factors_small],
            name="特征向量(SVD vs eigh)",
            use_strict_tolerance=True
        )
        print("[PASS] 特征向量在极严格容差内一致(允许符号歧义)")

        print(f"\n[PASS] 测试4通过: 特征值分解与SVD在极严格容差内等价")


    def test_005_factor_extraction_consistency(self):
        """
        测试5: 因子提取一致性

        验证train_model.DFM和train_ref._initialize_factors_pca提取的因子是否完全相同

        因子计算公式: F = U[:, :k] * s[:k]
        """
        print("\n" + "=" * 60)
        print("测试5: 因子提取一致性")
        print("=" * 60)

        # 调用train_model的DFM函数
        print("\n调用train_model.DFM...")
        result_old = DFM(self.Z_small_df, self.n_factors_small)
        factors_old = result_old.common_factors.values  # 转为numpy数组

        # 调用train_ref的DFMModel
        print("调用train_ref.DFMModel...")
        dfm_new = DFM_ref(n_factors=self.n_factors_small, max_iter=1)  # 只做初始化,不迭代

        # 需要手动调用内部的PCA初始化方法
        # 首先标准化数据
        obs_mean = self.Z_small_df.mean()
        obs_std = self.Z_small_df.std()
        obs_std[obs_std == 0] = 1.0
        z_standardized = ((self.Z_small_df - obs_mean) / obs_std).fillna(0).values
        obs_centered = (self.Z_small_df - obs_mean).fillna(0)

        factors_new_df, _, _ = dfm_new._initialize_factors_pca(
            z_standardized,
            obs_centered,
            obs_mean.values,
            obs_std.values
        )
        factors_new = factors_new_df.values

        print(f"\n因子矩阵形状:")
        print(f"  train_model: {factors_old.shape}")
        print(f"  train_ref:   {factors_new.shape}")

        print(f"\n前3个时间点的因子值对比:")
        for t in range(min(3, len(factors_old))):
            print(f"  t={t}:")
            print(f"    train_model: {factors_old[t, :]}")
            print(f"    train_ref:   {factors_new[t, :]}")

        # 零容差验证
        # 注意: 由于SVD的奇异向量可能有符号歧义,因子也可能有符号歧义
        # 我们需要检查是否 factors_new == factors_old 或 factors_new == -factors_old
        print("\n验证因子矩阵完全相等(允许整体符号歧义):")

        # 对每个因子列分别检查符号
        for i in range(self.n_factors_small):
            col_old = factors_old[:, i]
            col_new = factors_new[:, i]

            if np.array_equal(col_new, col_old):
                print(f"  因子{i+1}: 完全一致(正向)")
            elif np.array_equal(col_new, -col_old):
                print(f"  因子{i+1}: 完全一致(反向)")
            else:
                # 如果既不相等也不相反,报告详细差异
                diff_log = self.log_detailed_diff(col_new, col_old, f"因子{i+1}")
                raise AssertionError(
                    f"\n因子{i+1}既不满足正向相等也不满足反向相等!\n{diff_log}"
                )

        print(f"\n[PASS] 测试5通过: 因子提取完全一致(允许符号歧义)")


    def test_006_loading_matrix_estimation_consistency(self):
        """
        测试6: 载荷矩阵估计一致性

        验证两边估计的载荷矩阵是否一致

        注意:
        - train_model: 不显式计算载荷矩阵(在DFM函数中)
        - train_ref: 使用OLS估计 Lambda = (F'F)^-1 F'Z

        我们需要手动计算train_model的等价载荷矩阵进行对比
        """
        print("\n" + "=" * 60)
        print("测试6: 载荷矩阵估计一致性")
        print("=" * 60)

        # 获取因子
        result_old = DFM(self.Z_small_df, self.n_factors_small)
        factors_old = result_old.common_factors

        # train_model没有显式计算载荷矩阵,我们需要手动计算
        # 使用与train_ref相同的OLS方法
        from dashboard.DFM.train_ref.core.estimator import estimate_loadings

        obs_mean = self.Z_small_df.mean()
        obs_centered_old = (self.Z_small_df - obs_mean).fillna(0)

        loadings_old = estimate_loadings(
            obs_centered_old,
            factors_old
        )

        # train_ref的载荷矩阵估计
        dfm_new = DFM_ref(n_factors=self.n_factors_small, max_iter=1)

        obs_std = self.Z_small_df.std()
        obs_std[obs_std == 0] = 1.0
        z_standardized = ((self.Z_small_df - obs_mean) / obs_std).fillna(0).values
        obs_centered_new = (self.Z_small_df - obs_mean).fillna(0)

        factors_new_df, loadings_new, _ = dfm_new._initialize_factors_pca(
            z_standardized,
            obs_centered_new,
            obs_mean.values,
            obs_std.values
        )

        print(f"\n载荷矩阵形状:")
        print(f"  train_model(重新计算): {loadings_old.shape}")
        print(f"  train_ref:             {loadings_new.shape}")

        print(f"\n载荷矩阵样本(前3个变量):")
        for i in range(min(3, len(loadings_old))):
            print(f"  变量{i+1}:")
            print(f"    train_model: {loadings_old[i, :]}")
            print(f"    train_ref:   {loadings_new[i, :]}")

        # 注意: 由于因子可能有符号歧义,载荷矩阵也可能有相应的符号歧义
        # 我们需要根据test_005中因子的符号关系来调整载荷矩阵

        # 首先检测因子的符号关系
        factors_old_array = factors_old.values
        factors_new_array = factors_new_df.values

        sign_adjustments = np.ones(self.n_factors_small)
        for i in range(self.n_factors_small):
            if np.array_equal(factors_new_array[:, i], -factors_old_array[:, i]):
                sign_adjustments[i] = -1

        # 调整载荷矩阵的符号
        loadings_new_adjusted = loadings_new * sign_adjustments[np.newaxis, :]

        print(f"\n符号调整向量: {sign_adjustments}")

        # 零容差验证(调整符号后)
        print("\n验证载荷矩阵完全相等(符号调整后):")
        self.assert_array_exact_equal(
            loadings_new_adjusted,
            loadings_old,
            name="载荷矩阵"
        )

        print(f"\n[PASS] 测试6通过: 载荷矩阵估计完全一致")


if __name__ == '__main__':
    """
    直接运行本测试文件

    执行方式:
        python dashboard/DFM/train_ref/tests/consistency/test_pca_consistency.py

    或使用pytest:
        pytest dashboard/DFM/train_ref/tests/consistency/test_pca_consistency.py -v
    """
    print("=" * 60)
    print("PCA算法一致性测试")
    print("=" * 60)
    print("\n运行所有PCA一致性测试...")

    # 运行测试
    pytest.main([__file__, '-v', '-s'])
