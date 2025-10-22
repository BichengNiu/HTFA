# -*- coding: utf-8 -*-
"""
端到端基础功能测试

不依赖baseline，测试train_ref的完整训练流程是否能正常运行。
这是数值一致性验证的第一步 - 确保重构代码功能完整。
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.DFM.train_ref.training.trainer import DFMTrainer
from dashboard.DFM.train_ref.training.config import TrainingConfig
from dashboard.DFM.data_prep import prepare_data


class TestEndToEndBasic:
    """端到端基础功能测试

    测试完整的训练流程能否正常运行，不对比具体数值。
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

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """创建临时输出目录"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return str(output_dir)

    def test_minimal_training_flow(self, prepared_data, temp_output_dir, tmp_path):
        """测试最小配置的完整训练流程"""
        # 创建临时数据文件
        data_file = tmp_path / "test_data.csv"
        prepared_data.to_csv(data_file)

        # 获取可用的列（排除目标变量）
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]

        # 选择前5个作为指标
        selected_indicators = available_cols[:5] if len(available_cols) >= 5 else available_cols

        # 创建最小配置
        config = TrainingConfig(
            data_path=str(data_file),
            target_variable=target_col,
            selected_indicators=selected_indicators,
            train_start='2020-01-01',
            train_end='2022-12-31',
            validation_start='2023-01-01',
            validation_end='2023-06-30',
            k_factors=2,
            max_iterations=5,  # 使用少量迭代以加快测试
            enable_variable_selection=False,
            output_dir=temp_output_dir
        )

        # 创建训练器
        trainer = DFMTrainer(config)

        # 执行训练
        result = trainer.train()

        # 验证结果结构
        assert result is not None
        assert hasattr(result, 'selected_variables')
        assert hasattr(result, 'k_factors')
        assert hasattr(result, 'model_result')
        assert hasattr(result, 'metrics')

        # 验证变量列表
        assert len(result.selected_variables) > 0

        # 验证因子数
        assert result.k_factors == 2

        # 验证模型结果
        if result.model_result is not None:
            assert hasattr(result.model_result, 'A')
            assert hasattr(result.model_result, 'Q')
            assert hasattr(result.model_result, 'H')
            assert hasattr(result.model_result, 'R')

        # 验证评估指标
        if result.metrics is not None:
            assert hasattr(result.metrics, 'is_rmse')
            assert hasattr(result.metrics, 'oos_rmse')
            assert hasattr(result.metrics, 'is_hit_rate')
            assert hasattr(result.metrics, 'oos_hit_rate')

        print(f"\n训练完成:")
        print(f"  选择变量数: {len(result.selected_variables)}")
        print(f"  因子数: {result.k_factors}")
        print(f"  训练时间: {result.training_time:.2f}秒")

    def test_variable_selection_flow(self, prepared_data, temp_output_dir, tmp_path):
        """测试带变量选择的训练流程"""
        # 创建临时数据文件
        data_file = tmp_path / "test_data.csv"
        prepared_data.to_csv(data_file)

        # 获取可用的列
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]

        # 选择前10个作为初始指标
        selected_indicators = available_cols[:10] if len(available_cols) >= 10 else available_cols

        if len(selected_indicators) < 3:
            pytest.skip("可用指标数量不足")

        # 创建配置（启用变量选择）
        config = TrainingConfig(
            data_path=str(data_file),
            target_variable=target_col,
            selected_indicators=selected_indicators,
            train_start='2020-01-01',
            train_end='2022-12-31',
            validation_start='2023-01-01',
            validation_end='2023-06-30',
            k_factors=2,
            max_iterations=5,
            enable_variable_selection=True,
            variable_selection_method='backward',
            min_variables_after_selection=3,
            output_dir=temp_output_dir
        )

        # 创建训练器
        trainer = DFMTrainer(config)

        # 执行训练
        result = trainer.train()

        # 验证结果
        assert result is not None
        assert len(result.selected_variables) >= 3
        assert len(result.selected_variables) <= len(selected_indicators)

        # 验证选择历史
        assert len(result.selection_history) > 0

        print(f"\n变量选择完成:")
        print(f"  初始变量数: {len(selected_indicators)}")
        print(f"  最终变量数: {len(result.selected_variables)}")
        print(f"  选择轮数: {len(result.selection_history)}")

    def test_different_factor_numbers(self, prepared_data, temp_output_dir, tmp_path):
        """测试不同因子数的训练"""
        # 创建临时数据文件
        data_file = tmp_path / "test_data.csv"
        prepared_data.to_csv(data_file)

        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_indicators = available_cols[:5] if len(available_cols) >= 5 else available_cols

        for k_factors in [1, 2, 3]:
            config = TrainingConfig(
                data_path=str(data_file),
                target_variable=target_col,
                selected_indicators=selected_indicators,
                train_start='2020-01-01',
                train_end='2022-12-31',
                validation_start='2023-01-01',
                validation_end='2023-06-30',
                k_factors=k_factors,
                max_iterations=5,
                enable_variable_selection=False,
                output_dir=temp_output_dir
            )

            trainer = DFMTrainer(config)
            result = trainer.train()

            assert result is not None
            assert result.k_factors == k_factors

            print(f"\nk={k_factors} 训练完成")

    def test_reproducibility(self, prepared_data, temp_output_dir, tmp_path):
        """测试可重现性 - 相同配置应产生相同结果"""
        # 创建临时数据文件
        data_file = tmp_path / "test_data.csv"
        prepared_data.to_csv(data_file)

        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_indicators = available_cols[:5] if len(available_cols) >= 5 else available_cols

        config = TrainingConfig(
            data_path=str(data_file),
            target_variable=target_col,
            selected_indicators=selected_indicators,
            train_start='2020-01-01',
            train_end='2022-12-31',
            validation_start='2023-01-01',
            validation_end='2023-06-30',
            k_factors=2,
            max_iterations=5,
            enable_variable_selection=False,
            output_dir=temp_output_dir
        )

        # 第一次训练
        trainer1 = DFMTrainer(config)
        result1 = trainer1.train()

        # 第二次训练（相同配置）
        trainer2 = DFMTrainer(config)
        result2 = trainer2.train()

        # 验证结果一致
        assert result1.k_factors == result2.k_factors
        assert len(result1.selected_variables) == len(result2.selected_variables)

        # 如果有模型结果，验证参数矩阵近似相等
        if result1.model_result and result2.model_result:
            if result1.model_result.A is not None and result2.model_result.A is not None:
                np.testing.assert_allclose(
                    result1.model_result.A,
                    result2.model_result.A,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg="两次训练的A矩阵不一致"
                )

        print("\n可重现性验证通过")
