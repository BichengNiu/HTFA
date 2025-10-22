# -*- coding: utf-8 -*-
"""
TrainingConfig单元测试

测试内容:
1. 配置对象创建
2. 默认值设置
3. 配置验证逻辑
4. 边界情况处理
"""
import pytest
from dataclasses import is_dataclass

from dashboard.DFM.train_ref.training.config import TrainingConfig


class TestTrainingConfigCreation:
    """TrainingConfig创建测试"""

    @pytest.fixture
    def temp_data_file(self, tmp_path):
        """创建临时数据文件"""
        import pandas as pd
        data = pd.DataFrame({'target': [1, 2, 3], 'var1': [4, 5, 6]})
        data_file = tmp_path / "data.xlsx"
        data.to_excel(data_file)
        return str(data_file)

    def test_training_config_is_dataclass(self):
        """验证TrainingConfig是dataclass"""
        assert is_dataclass(TrainingConfig)

    def test_training_config_creation_minimal(self, temp_data_file):
        """测试最小配置创建"""
        config = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target"
        )

        assert config.data_path == temp_data_file
        assert config.target_variable == "target"

    def test_training_config_creation_full(self, temp_data_file):
        """测试完整配置创建"""
        config = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target",
            selected_indicators=['var1', 'var2'],
            train_start='2020-01-01',
            train_end='2020-12-31',
            validation_start='2021-01-01',
            validation_end='2021-06-30',
            target_freq='W-FRI',
            k_factors=3,
            max_iterations=50,
            tolerance=1e-8,
            enable_variable_selection=True,
            variable_selection_method='backward',
            min_variables_after_selection=2,
            factor_selection_method='cumulative',
            pca_threshold=0.95,
            output_dir='./output'
        )

        assert config.data_path == temp_data_file
        assert config.target_variable == "target"
        assert config.selected_indicators == ['var1', 'var2']
        assert config.k_factors == 3
        assert config.max_iterations == 50
        assert config.enable_variable_selection is True


class TestTrainingConfigDefaults:
    """TrainingConfig默认值测试"""

    @pytest.fixture
    def temp_data_file(self, tmp_path):
        """创建临时数据文件"""
        import pandas as pd
        data = pd.DataFrame({'target': [1, 2, 3], 'var1': [4, 5, 6]})
        data_file = tmp_path / "data.xlsx"
        data.to_excel(data_file)
        return str(data_file)

    def test_default_model_params(self, temp_data_file):
        """测试模型参数默认值"""
        config = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target"
        )

        assert config.k_factors == 4  # 默认因子数
        assert config.max_iterations == 30
        assert config.max_lags == 1
        assert config.tolerance == 1e-6

    def test_default_selection_params(self, temp_data_file):
        """测试变量选择参数默认值"""
        config = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target"
        )

        assert config.enable_variable_selection is False
        assert config.variable_selection_method == 'backward'
        assert config.min_variables_after_selection is None

    def test_default_factor_selection_params(self, temp_data_file):
        """测试因子选择参数默认值"""
        config = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target"
        )

        assert config.factor_selection_method == 'fixed'
        assert config.pca_threshold is None or config.pca_threshold == 0.9
        assert config.elbow_threshold is None or config.elbow_threshold == 0.1

    def test_default_optimization_params(self, temp_data_file):
        """测试优化参数默认值"""
        config = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target"
        )

        assert config.use_cache is False
        assert config.use_precompute is False


class TestTrainingConfigEdgeCases:
    """TrainingConfig边界情况测试"""

    @pytest.fixture
    def temp_data_file(self, tmp_path):
        """创建临时数据文件"""
        import pandas as pd
        data = pd.DataFrame({'target': [1, 2, 3], 'var1': [4, 5, 6]})
        data_file = tmp_path / "data.xlsx"
        data.to_excel(data_file)
        return str(data_file)

    def test_empty_selected_indicators(self, temp_data_file):
        """测试空指标列表"""
        config = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target",
            selected_indicators=[]
        )

        assert config.selected_indicators == []

    def test_none_optional_fields(self, temp_data_file):
        """测试None可选字段"""
        config = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target",
            train_start=None,
            train_end=None,
            validation_start=None,
            validation_end=None
        )

        assert config.train_start is None
        assert config.train_end is None
        assert config.validation_start is None
        assert config.validation_end is None
        # output_dir会有默认值，不是None
        assert config.output_dir is not None

    def test_factor_selection_methods(self, temp_data_file):
        """测试不同因子选择方法"""
        # Fixed method
        config1 = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target",
            factor_selection_method='fixed',
            k_factors=5
        )
        assert config1.factor_selection_method == 'fixed'
        assert config1.k_factors == 5

        # Cumulative method
        config2 = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target",
            factor_selection_method='cumulative',
            pca_threshold=0.95
        )
        assert config2.factor_selection_method == 'cumulative'
        assert config2.pca_threshold == 0.95

        # Elbow method
        config3 = TrainingConfig(
            data_path=temp_data_file,
            target_variable="target",
            factor_selection_method='elbow',
            elbow_threshold=0.05
        )
        assert config3.factor_selection_method == 'elbow'
        assert config3.elbow_threshold == 0.05
