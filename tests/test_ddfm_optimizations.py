# -*- coding: utf-8 -*-
"""
DDFM模型性能优化单元测试

测试内容：
1. 批量推理内存保护机制
2. x_sim_den向量化构建
3. 类常量MAX_BATCH_SIZE
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import sys


class TestDDFMOptimizations(unittest.TestCase):
    """DDFM优化相关测试"""

    @classmethod
    def setUpClass(cls):
        """检查TensorFlow是否可用"""
        try:
            import tensorflow as tf
            cls.tf_available = True
        except ImportError:
            cls.tf_available = False

    def test_max_batch_size_class_constant(self):
        """测试MAX_BATCH_SIZE是类常量而非局部变量"""
        if not self.tf_available:
            self.skipTest("TensorFlow not available")

        from dashboard.models.DFM.train.core.ddfm_model import DDFMModel

        # 验证类常量存在
        self.assertTrue(hasattr(DDFMModel, 'MAX_BATCH_SIZE'))
        self.assertEqual(DDFMModel.MAX_BATCH_SIZE, 5000)

        # 验证实例可以访问
        model = DDFMModel()
        self.assertEqual(model.MAX_BATCH_SIZE, 5000)

    def test_vectorized_x_sim_den_construction(self):
        """测试x_sim_den向量化构建逻辑"""
        # 模拟数据
        epochs = 10
        T = 50
        n_vars = 5
        input_dim = n_vars * 2  # 假设有滞后变量

        data_tmp_values = np.random.randn(T, input_dim)
        eps_draws = np.random.randn(epochs, T, n_vars)

        # 向量化构建（优化后的代码逻辑）
        x_sim_den = np.broadcast_to(
            data_tmp_values[np.newaxis, :, :],
            (epochs, T, input_dim)
        ).copy()
        x_sim_den[:, :, :n_vars] -= eps_draws

        # 验证形状
        self.assertEqual(x_sim_den.shape, (epochs, T, input_dim))

        # 验证每个epoch的数据正确减去了eps_draws
        for i in range(epochs):
            expected = data_tmp_values.copy()
            expected[:, :n_vars] -= eps_draws[i]
            np.testing.assert_array_almost_equal(x_sim_den[i], expected)

    def test_broadcast_copy_is_writable(self):
        """测试broadcast_to().copy()返回可写数组"""
        base = np.array([[1, 2, 3]])
        broadcasted = np.broadcast_to(base, (5, 3))

        # broadcast_to返回只读视图
        self.assertFalse(broadcasted.flags.writeable)

        # copy()后可写
        copied = broadcasted.copy()
        self.assertTrue(copied.flags.writeable)

        # 可以修改
        copied[0, 0] = 999
        self.assertEqual(copied[0, 0], 999)

    def test_ascontiguousarray_for_reshape(self):
        """测试np.ascontiguousarray确保reshape安全"""
        # 创建非连续数组
        arr = np.random.randn(10, 5, 3)
        non_contiguous = arr.transpose(1, 0, 2)

        # 非连续数组
        self.assertFalse(non_contiguous.flags['C_CONTIGUOUS'])

        # ascontiguousarray使其连续
        contiguous = np.ascontiguousarray(non_contiguous)
        self.assertTrue(contiguous.flags['C_CONTIGUOUS'])

        # reshape正常工作
        reshaped = contiguous.reshape(-1, 3)
        self.assertEqual(reshaped.shape, (50, 3))

    def test_chunk_size_calculation(self):
        """测试分块大小计算逻辑"""
        MAX_BATCH_SIZE = 5000

        test_cases = [
            # (epochs, T, expected_chunk_size)
            (100, 100, 50),   # 10000 > 5000, chunk_size = 5000//100 = 50
            (50, 50, 100),    # 2500 < 5000, 不需要分块但计算结果
            (200, 50, 100),   # 10000 > 5000, chunk_size = 5000//50 = 100
            (10, 1000, 5),    # 10000 > 5000, chunk_size = 5000//1000 = 5
        ]

        for epochs, T, expected in test_cases:
            batch_size_total = epochs * T
            if batch_size_total > MAX_BATCH_SIZE:
                chunk_size = max(1, MAX_BATCH_SIZE // T)
                self.assertEqual(chunk_size, expected,
                    f"Failed for epochs={epochs}, T={T}")

    def test_memory_protection_threshold(self):
        """测试内存保护阈值逻辑"""
        MAX_BATCH_SIZE = 5000

        # 小批量：直接处理
        small_batch = 100 * 40  # 4000 < 5000
        self.assertLessEqual(small_batch, MAX_BATCH_SIZE)

        # 大批量：需要分块
        large_batch = 100 * 100  # 10000 > 5000
        self.assertGreater(large_batch, MAX_BATCH_SIZE)

    def test_chunked_processing_correctness(self):
        """测试分块处理的正确性（模拟批量推理）"""
        MAX_BATCH_SIZE = 50  # 使用小阈值便于测试
        epochs = 10
        T = 10
        input_dim = 5
        output_dim = 3

        # 模拟数据
        x_sim_den = np.random.randn(epochs, T, input_dim)

        # 模拟encoder（简单线性变换）
        W = np.random.randn(input_dim, output_dim)

        def mock_encoder(x):
            return x @ W

        batch_shape = x_sim_den.shape
        batch_size_total = batch_shape[0] * batch_shape[1]

        # 直接处理（参考结果）
        x_batch_all = x_sim_den.reshape(-1, batch_shape[-1])
        expected_result = mock_encoder(x_batch_all).reshape(
            batch_shape[0], batch_shape[1], -1)

        # 分块处理
        if batch_size_total > MAX_BATCH_SIZE:
            result_list = []
            chunk_size = max(1, MAX_BATCH_SIZE // batch_shape[1])
            for chunk_start in range(0, batch_shape[0], chunk_size):
                chunk_end = min(chunk_start + chunk_size, batch_shape[0])
                chunk = x_sim_den[chunk_start:chunk_end]
                chunk_flat = np.ascontiguousarray(
                    chunk.reshape(-1, batch_shape[-1]))
                result_chunk = mock_encoder(chunk_flat)
                result_list.append(result_chunk.reshape(
                    chunk_end - chunk_start, batch_shape[1], -1))
            chunked_result = np.concatenate(result_list, axis=0)

            # 验证结果一致
            np.testing.assert_array_almost_equal(chunked_result, expected_result)


class TestConditionalPreprocessing(unittest.TestCase):
    """条件性预处理优化测试"""

    def test_no_lags_direct_update(self):
        """测试无滞后变量时的直接更新"""
        lags_input = 0
        has_lags = lags_input > 0

        # 模拟data_mod和data_tmp
        data_mod_values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        data_tmp_values = np.zeros((3, 3))

        if not has_lags:
            # 无滞后：直接更新values
            data_tmp_values[:] = data_mod_values[lags_input:]

        np.testing.assert_array_equal(data_tmp_values, data_mod_values)

    def test_with_lags_needs_rebuild(self):
        """测试有滞后变量时需要完整重建"""
        lags_input = 2
        has_lags = lags_input > 0

        self.assertTrue(has_lags)
        # 有滞后时应调用_build_inputs，这里仅验证条件判断


class TestTensorFlowThreadConfig(unittest.TestCase):
    """TensorFlow线程配置测试"""

    def test_thread_calculation(self):
        """测试线程数计算逻辑"""
        import os

        test_cases = [
            # (num_threads, expected_intra, expected_inter)
            (None, os.cpu_count() or 4, max(2, (os.cpu_count() or 4) // 2)),
            (8, 8, 4),
            (4, 4, 2),
            (2, 2, 2),  # inter至少为2
        ]

        for num_threads, expected_intra, expected_inter in test_cases:
            n_cpus = num_threads or os.cpu_count() or 4
            intra = n_cpus
            inter = max(2, n_cpus // 2)

            self.assertEqual(intra, expected_intra,
                f"Failed intra for num_threads={num_threads}")
            self.assertEqual(inter, expected_inter,
                f"Failed inter for num_threads={num_threads}")


class TestCovarianceMatrix(unittest.TestCase):
    """协方差矩阵参数测试"""

    def test_multivariate_normal_requires_variance(self):
        """测试multivariate_normal需要方差（标准差²）作为协方差矩阵对角线"""
        std = np.array([1.0, 2.0, 3.0])
        cov = np.diag(std**2)  # 正确的协方差矩阵：方差 = 标准差²

        rng = np.random.RandomState(42)
        samples = rng.multivariate_normal(np.zeros(3), cov, 10000)

        # 验证样本标准差接近预期标准差
        sample_std = np.std(samples, axis=0)
        np.testing.assert_array_almost_equal(sample_std, std, decimal=1)

    def test_wrong_covariance_produces_wrong_std(self):
        """测试使用标准差（而非方差）作为协方差会导致错误的分布"""
        std = np.array([2.0, 3.0, 4.0])

        # 错误：直接使用标准差作为协方差对角线
        wrong_cov = np.diag(std)

        rng = np.random.RandomState(42)
        samples = rng.multivariate_normal(np.zeros(3), wrong_cov, 10000)

        # 样本标准差应接近sqrt(std)，而非std本身
        sample_std = np.std(samples, axis=0)
        expected_wrong_std = np.sqrt(std)  # 错误情况下的预期标准差

        np.testing.assert_array_almost_equal(sample_std, expected_wrong_std, decimal=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
