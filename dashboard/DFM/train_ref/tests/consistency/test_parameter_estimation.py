# -*- coding: utf-8 -*-
"""
参数估计对比测试

不依赖baseline，测试train_ref内部的参数估计一致性和稳定性。
验证EM算法估计的A, Q, H, R矩阵的数值稳定性。
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.DFM.train_ref.core.factor_model import DFMModel
from dashboard.DFM.data_prep import prepare_data


class TestParameterEstimation:
    """参数估计对比测试

    测试内容：
    1. 相同输入产生相同参数估计
    2. 参数矩阵的数值特性（正定性、有界性等）
    3. 收敛迭代次数的稳定性
    """

    @pytest.fixture(scope="class")
    def prepared_data(self):
        """准备测试数据（所有测试共用）"""
        data_path = PROJECT_ROOT / "data" / "经济数据库1017.xlsx"

        if not data_path.exists():
            pytest.skip(f"数据文件不存在: {data_path}")

        # 使用data_prep准备数据
        processed_data, _, _, _ = prepare_data(
            excel_path=str(data_path),
            target_freq='W-FRI',
            target_sheet_name='工业增加值同比增速_月度_同花顺',
            target_variable_name='规模以上工业增加值:当月同比',
            consecutive_nan_threshold=10,
            data_start_date='2020-01-01',
            data_end_date='2025-07-03'
        )

        if processed_data is None:
            pytest.skip("数据预处理失败")

        return processed_data

    def test_parameter_estimation_reproducibility(self, prepared_data):
        """测试参数估计的可重现性：相同输入产生相同输出"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        # 第一次估计
        np.random.seed(42)
        model1 = DFMModel(n_factors=2, max_lags=1, max_iter=10, tolerance=1e-6)
        result1 = model1.fit(test_data)

        # 第二次估计（相同种子）
        np.random.seed(42)
        model2 = DFMModel(n_factors=2, max_lags=1, max_iter=10, tolerance=1e-6)
        result2 = model2.fit(test_data)

        # 验证参数一致
        np.testing.assert_allclose(result1.transition_matrix, result2.transition_matrix, rtol=1e-10, atol=1e-10,
                                   err_msg="转移矩阵A不一致")
        np.testing.assert_allclose(result1.process_noise_cov, result2.process_noise_cov, rtol=1e-10, atol=1e-10,
                                   err_msg="状态噪声协方差Q不一致")
        np.testing.assert_allclose(result1.loadings, result2.loadings, rtol=1e-10, atol=1e-10,
                                   err_msg="观测矩阵H不一致")
        np.testing.assert_allclose(result1.measurement_noise_cov, result2.measurement_noise_cov, rtol=1e-10, atol=1e-10,
                                   err_msg="观测噪声协方差R不一致")

        assert result1.n_iter == result2.n_iter, \
            f"迭代次数不一致: {result1.n_iter} vs {result2.n_iter}"

    def test_transition_matrix_properties(self, prepared_data):
        """测试转移矩阵A的数值特性"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        np.random.seed(42)
        model = DFMModel(n_factors=2, max_lags=1, max_iter=10, tolerance=1e-6)
        result = model.fit(test_data)

        A = result.transition_matrix

        # 验证A是方阵
        assert A.shape[0] == A.shape[1], f"A不是方阵: {A.shape}"
        assert A.shape[0] == 2, f"A维度错误: {A.shape[0]} (期望2)"

        # 验证A的元素有界
        assert np.all(np.abs(A) < 10), f"A的元素过大: max={np.max(np.abs(A))}"

        # 验证A不包含NaN或Inf
        assert np.all(np.isfinite(A)), "A包含NaN或Inf"

        print(f"\n转移矩阵A:\n{A}")
        print(f"特征值: {np.linalg.eigvals(A)}")

    def test_covariance_matrices_properties(self, prepared_data):
        """测试协方差矩阵Q和R的数值特性"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        np.random.seed(42)
        model = DFMModel(n_factors=2, max_lags=1, max_iter=10, tolerance=1e-6)
        result = model.fit(test_data)

        Q = result.process_noise_cov
        R = result.measurement_noise_cov

        # 验证Q的特性
        assert Q.shape[0] == Q.shape[1], f"Q不是方阵: {Q.shape}"
        assert Q.shape[0] == 2, f"Q维度错误: {Q.shape[0]}"
        assert np.allclose(Q, Q.T, atol=1e-10), "Q不对称"

        Q_eigvals = np.linalg.eigvalsh(Q)
        assert np.all(Q_eigvals > -1e-10), f"Q不是半正定的: 最小特征值={np.min(Q_eigvals)}"

        # 验证R的特性
        assert R.shape[0] == R.shape[1], f"R不是方阵: {R.shape}"
        assert R.shape[0] == len(selected_vars), f"R维度错误: {R.shape[0]}"

        # R应该是对角矩阵
        R_diag = np.diag(np.diag(R))
        assert np.allclose(R, R_diag, atol=1e-10), "R不是对角矩阵"

        R_diagonal = np.diag(R)
        assert np.all(R_diagonal > 0), f"R的对角元素有非正值: min={np.min(R_diagonal)}"

        print(f"\n状态噪声协方差Q对角线: {np.diag(Q)}")
        print(f"观测噪声协方差R对角线前5个: {R_diagonal[:5]}")

    def test_loading_matrix_properties(self, prepared_data):
        """测试载荷矩阵H的数值特性"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        np.random.seed(42)
        model = DFMModel(n_factors=2, max_lags=1, max_iter=10, tolerance=1e-6)
        result = model.fit(test_data)

        H = result.loadings

        # 验证H的形状
        assert H.shape == (len(selected_vars), 2), f"H形状错误: {H.shape}"

        # 验证H不包含NaN或Inf
        assert np.all(np.isfinite(H)), "H包含NaN或Inf"

        # 验证H的元素有界
        assert np.all(np.abs(H) < 1000), f"H的元素过大: max={np.max(np.abs(H))}"

        print(f"\n载荷矩阵H形状: {H.shape}")
        print(f"H的统计: mean={np.mean(np.abs(H)):.2f}, std={np.std(H):.2f}, max={np.max(np.abs(H)):.2f}")

    def test_convergence_stability(self, prepared_data):
        """测试EM算法的迭代行为稳定性

        注意：此测试不强制要求收敛，因为收敛与否取决于数据质量和参数配置。
        测试目标是验证算法能够稳定运行，不会崩溃或产生无效结果。
        """
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        print(f"\n测试数据形状: {test_data.shape}")

        # 测试不同初始化产生的稳定运行
        seeds = [42, 123, 456]
        results = []

        for seed in seeds:
            np.random.seed(seed)
            model = DFMModel(n_factors=1, max_lags=1, max_iter=20, tolerance=1e-6)
            result = model.fit(test_data)
            results.append(result)

        # 验证所有运行都能产生有效结果（不崩溃）
        for i, result in enumerate(results):
            assert np.all(np.isfinite(result.transition_matrix)), f"种子{seeds[i]}产生了无效的转移矩阵"
            assert np.all(np.isfinite(result.loadings)), f"种子{seeds[i]}产生了无效的载荷矩阵"
            assert np.isfinite(result.loglikelihood), f"种子{seeds[i]}产生了无效的对数似然"

        # 验证迭代次数在合理范围内
        iterations = [r.n_iter for r in results]
        assert all(1 <= it <= 20 for it in iterations), f"迭代次数异常: {iterations}"

        # 记录收敛情况（仅作参考）
        convergence_status = [(seeds[i], r.converged) for i, r in enumerate(results)]
        converged_count = sum(1 for _, converged in convergence_status if converged)
        print(f"收敛情况（参考）: {convergence_status}")
        print(f"收敛率（参考）: {converged_count}/{len(seeds)}")
        print(f"迭代次数: {iterations}")
        print(f"对数似然值: {[f'{r.loglikelihood:.2f}' for r in results]}")

    def test_different_factor_numbers(self, prepared_data):
        """测试不同因子数的参数估计"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:8]
        test_data = prepared_data[selected_vars].dropna()

        for k in [1, 2, 3]:
            np.random.seed(42)
            model = DFMModel(n_factors=k, max_lags=1, max_iter=10, tolerance=1e-6)
            result = model.fit(test_data)

            # 验证参数维度正确
            assert result.transition_matrix.shape == (k, k), f"k={k}时A维度错误"
            assert result.process_noise_cov.shape == (k, k), f"k={k}时Q维度错误"
            assert result.loadings.shape == (len(selected_vars), k), f"k={k}时H维度错误"

            # 验证参数有效
            assert np.all(np.isfinite(result.transition_matrix)), f"k={k}时A包含无效值"
            assert np.all(np.isfinite(result.process_noise_cov)), f"k={k}时Q包含无效值"
            assert np.all(np.isfinite(result.loadings)), f"k={k}时H包含无效值"

            print(f"\nk={k}: iterations={result.n_iter}, converged={result.converged}")

    def test_single_factor_model(self, prepared_data):
        """专门测试单因子模型的参数估计"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        np.random.seed(42)
        model = DFMModel(n_factors=1, max_lags=1, max_iter=10, tolerance=1e-6)
        result = model.fit(test_data)

        # 单因子时A应该是1x1矩阵
        A = result.transition_matrix
        assert A.shape == (1, 1), f"单因子时A形状错误: {A.shape}"

        # 单因子时Q应该是1x1矩阵
        Q = result.process_noise_cov
        assert Q.shape == (1, 1), f"单因子时Q形状错误: {Q.shape}"

        # H应该是Nx1矩阵
        H = result.loadings
        assert H.shape == (len(selected_vars), 1), f"单因子时H形状错误: {H.shape}"

        print(f"\n单因子模型参数:")
        print(f"  A = {A[0, 0]:.6f}")
        print(f"  Q = {Q[0, 0]:.6f}")
        print(f"  H (前3个): {H[:3, 0]}")
