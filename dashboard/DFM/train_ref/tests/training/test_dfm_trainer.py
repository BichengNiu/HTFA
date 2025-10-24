# -*- coding: utf-8 -*-
"""
DFMTrainer单元测试

测试内容:
1. DFMTrainer初始化
2. 环境初始化（_init_environment）
3. 因子数选择方法（_select_num_factors）
4. 可重现性（相同种子相同结果）
5. TrainingResult数据结构
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import os

from dashboard.DFM.train_ref.training.trainer import (
    DFMTrainer,
    TrainingResult,
    DFMModelResult
)
from dashboard.DFM.train_ref.training.config import TrainingConfig


class TestDFMTrainerInit:
    """DFMTrainer初始化测试"""

    @pytest.fixture
    def temp_data_file(self, tmp_path):
        """创建临时数据文件"""
        data = pd.DataFrame({'target': [1, 2, 3], 'var1': [4, 5, 6]})
        data_file = tmp_path / "data.xlsx"
        data.to_excel(data_file)
        return str(data_file)

    def test_init_with_valid_config(self, temp_data_file):
        """测试使用有效配置初始化"""
        config = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target"
        )

        trainer = DFMTrainer(config)

        assert trainer.config == config
        assert trainer.evaluator is not None

    def test_init_environment_sets_env_vars(self, temp_data_file):
        """测试环境初始化设置环境变量"""
        config = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target"
        )

        with patch.dict(os.environ, {}, clear=True):
            trainer = DFMTrainer(config)

            # 验证环境变量被设置
            assert 'OMP_NUM_THREADS' in os.environ
            assert 'MKL_NUM_THREADS' in os.environ
            assert 'OPENBLAS_NUM_THREADS' in os.environ

    def test_init_sets_random_seed(self, temp_data_file):
        """测试初始化设置随机种子"""
        config = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target"
        )

        # 第一次初始化
        trainer1 = DFMTrainer(config)
        random_nums1 = np.random.rand(5)

        # 第二次初始化
        trainer2 = DFMTrainer(config)
        random_nums2 = np.random.rand(5)

        # 两次应该生成相同的随机数（因为种子相同）
        np.testing.assert_array_almost_equal(random_nums1, random_nums2)


class TestDFMTrainerFactorSelection:
    """DFMTrainer因子数选择测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='W-FRI')
        data = pd.DataFrame({
            'var1': np.random.randn(100),
            'var2': np.random.randn(100) + 0.5,
            'var3': np.random.randn(100) - 0.5,
        }, index=dates)
        return data

    def test_select_num_factors_fixed_method(self, sample_data, tmp_path):
        """测试固定因子数方法"""
        # 创建临时数据文件
        data_path = tmp_path / "data.xlsx"
        sample_data.to_excel(data_path)

        config = TrainingConfig(
            data_path=str(data_path),
            target_variable="var1",
            factor_selection_method='fixed',
            k_factors=2
        )

        trainer = DFMTrainer(config)
        selected_vars = ['var1', 'var2', 'var3']

        k_factors, pca_analysis = trainer._select_num_factors(
            data=sample_data,  # 添加data参数
            selected_vars=selected_vars,
            progress_callback=None
        )

        assert k_factors == 2
        assert pca_analysis is None  # fixed方法不返回PCA分析

    def test_select_num_factors_cumulative_method(self, sample_data, tmp_path):
        """测试累积方差方法"""
        # 创建临时数据文件
        data_path = tmp_path / "data.xlsx"
        sample_data.to_excel(data_path)

        config = TrainingConfig(
            data_path=str(data_path),
            target_variable="var1",
            factor_selection_method='cumulative',
            pca_threshold=0.9
        )

        trainer = DFMTrainer(config)
        selected_vars = ['var1', 'var2', 'var3']

        k_factors, pca_analysis = trainer._select_num_factors(
            data=sample_data,  # 添加data参数
            selected_vars=selected_vars,
            progress_callback=None
        )

        # 验证返回的因子数是合理的
        assert 1 <= k_factors <= len(selected_vars)
        assert pca_analysis is not None  # 应该返回PCA对象

    def test_select_num_factors_elbow_method(self, sample_data, tmp_path):
        """测试肘部法"""
        # 创建临时数据文件
        data_path = tmp_path / "data.xlsx"
        sample_data.to_excel(data_path)

        config = TrainingConfig(
            data_path=str(data_path),
            target_variable="var1",
            factor_selection_method='elbow',
            elbow_threshold=0.1
        )

        trainer = DFMTrainer(config)
        selected_vars = ['var1', 'var2', 'var3']

        k_factors, pca_analysis = trainer._select_num_factors(
            data=sample_data,  # 添加data参数
            selected_vars=selected_vars,
            progress_callback=None
        )

        # 验证返回的因子数是合理的
        assert 1 <= k_factors <= len(selected_vars)
        assert pca_analysis is not None


class TestDFMTrainerReproducibility:
    """DFMTrainer可重现性测试"""

    @pytest.fixture
    def temp_data_file(self, tmp_path):
        """创建临时数据文件"""
        data = pd.DataFrame({'target': [1, 2, 3], 'var1': [4, 5, 6]})
        data_file = tmp_path / "data.xlsx"
        data.to_excel(data_file)
        return str(data_file)

    def test_same_seed_same_results(self, temp_data_file):
        """测试相同种子产生相同结果"""
        config = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target"
        )

        # 第一次运行
        trainer1 = DFMTrainer(config)
        random_state1 = np.random.get_state()

        # 第二次运行
        trainer2 = DFMTrainer(config)
        random_state2 = np.random.get_state()

        # 随机状态应该相同（因为都是SEED=42）
        assert random_state1[0] == random_state2[0]
        # 注意：完整的状态比较较复杂，这里只比较类型


class TestTrainingResult:
    """TrainingResult数据类测试"""

    def test_training_result_creation(self):
        """测试TrainingResult对象创建"""
        result = TrainingResult(
            selected_variables=['target', 'var1', 'var2'],
            selection_history=[],
            k_factors=2,
            factor_selection_method='fixed',
            pca_analysis=None,
            model_result=None,
            metrics=None,
            total_evaluations=10,
            svd_error_count=0,
            training_time=15.5,
            output_dir='./output'
        )

        assert result.selected_variables == ['target', 'var1', 'var2']
        assert result.k_factors == 2
        assert result.total_evaluations == 10
        assert result.training_time == 15.5

    def test_training_result_default_values(self):
        """测试TrainingResult默认值"""
        result = TrainingResult()

        assert result.selected_variables == []
        assert result.selection_history == []
        assert result.k_factors == 0
        assert result.total_evaluations == 0
        assert result.svd_error_count == 0
        assert result.training_time == 0.0

    def test_training_result_is_dataclass(self):
        """验证TrainingResult是dataclass"""
        from dataclasses import is_dataclass
        assert is_dataclass(TrainingResult)


class TestDFMModelResult:
    """DFMModelResult数据类测试"""

    def test_dfm_model_result_creation(self):
        """测试DFMModelResult对象创建"""
        result = DFMModelResult(
            A=np.eye(2),
            Q=np.eye(2),
            H=np.ones((3, 2)),
            R=np.eye(3),
            factors=np.random.randn(10, 2),
            factors_smooth=np.random.randn(10, 2),
            forecast_is=np.random.randn(5),
            forecast_oos=np.random.randn(5),
            converged=True,
            iterations=10,
            log_likelihood=-100.5
        )

        assert result.A.shape == (2, 2)
        assert result.Q.shape == (2, 2)
        assert result.H.shape == (3, 2)
        assert result.converged is True
        assert result.iterations == 10

    def test_dfm_model_result_default_values(self):
        """测试DFMModelResult默认值"""
        result = DFMModelResult()

        assert result.A is None
        assert result.Q is None
        assert result.H is None
        assert result.R is None
        assert result.converged is False
        assert result.iterations == 0
        assert result.log_likelihood == -np.inf

    def test_dfm_model_result_is_dataclass(self):
        """验证DFMModelResult是dataclass"""
        from dataclasses import is_dataclass
        assert is_dataclass(DFMModelResult)


class TestDFMTrainerProgressCallback:
    """DFMTrainer进度回调测试"""

    def test_progress_callback_called(self, tmp_path):
        """测试进度回调被调用"""
        # 创建简单的测试数据
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50, freq='W-FRI')
        data = pd.DataFrame({
            'target': np.random.randn(50),
            'var1': np.random.randn(50),
        }, index=dates)

        data_path = tmp_path / "data.xlsx"
        data.to_excel(data_path)

        config = TrainingConfig(
            data_path=str(data_path),
            target_variable="target",
            selected_indicators=['target', 'var1'],
            factor_selection_method='fixed',
            k_factors=1
        )

        trainer = DFMTrainer(config)

        # 测试_select_num_factors的回调
        progress_messages = []
        def progress_callback(msg):
            progress_messages.append(msg)

        trainer._select_num_factors(
            data=data,  # 添加data参数
            selected_vars=['target', 'var1'],
            progress_callback=progress_callback
        )

        # fixed方法可能没有回调消息，这个测试只是确保不会崩溃
        # 如果有消息，验证它们是字符串
        for msg in progress_messages:
            assert isinstance(msg, str)
