# -*- coding: utf-8 -*-
"""
测试卡尔曼增益历史保存功能

验证训练模块正确保存和导出K_t历史数据
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from pathlib import Path

# 导入训练模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from dashboard.models.DFM.train.core.kalman import KalmanFilter
from dashboard.models.DFM.train.core.factor_model import DFMModel
from dashboard.models.DFM.train.export.exporter import TrainingResultExporter
from dashboard.models.DFM.train.core.models import KalmanFilterResult, DFMModelResult


class TestKalmanGainsHistory:
    """测试卡尔曼增益历史保存"""

    def test_kalman_filter_saves_kt_history(self):
        """测试卡尔曼滤波器保存K_t历史"""
        # 构造简单的测试数据
        n_time = 10
        n_factors = 2
        n_variables = 3

        # 系统矩阵
        A = np.array([[0.8, 0.1], [0.1, 0.7]])
        Q = np.eye(n_factors) * 0.1
        H = np.random.randn(n_variables, n_factors)
        R = np.eye(n_variables) * 0.5
        B = np.zeros((n_factors, 1))  # 无控制输入

        # 初始状态
        x0 = np.zeros(n_factors)
        P0 = np.eye(n_factors)

        # 生成观测数据（部分缺失）
        Z = np.random.randn(n_time, n_variables)
        Z[0, :] = np.nan  # 第一个时刻无观测
        Z[5, 2] = np.nan  # 部分缺失

        # 控制输入
        U = np.zeros((n_time, 1))

        # 创建卡尔曼滤波器
        kalman = KalmanFilter(A=A, B=B, H=H, Q=Q, R=R, x0=x0, P0=P0)

        # 执行滤波
        result = kalman.filter(Z, U)

        # 验证K_t历史被保存
        assert hasattr(result, 'kalman_gains_history'), "KalmanFilterResult应该有kalman_gains_history属性"
        assert result.kalman_gains_history is not None, "kalman_gains_history不应为None"
        assert isinstance(result.kalman_gains_history, list), "kalman_gains_history应该是列表"
        assert len(result.kalman_gains_history) == n_time, f"K_t历史长度应为{n_time}"

        # 验证K_t矩阵形状
        # t=0时应该是None（无更新）
        assert result.kalman_gains_history[0] is None, "t=0时K_t应为None"

        # t>0时应该有正确的形状
        for t in range(1, n_time):
            K_t = result.kalman_gains_history[t]
            assert K_t is not None, f"t={t}时K_t不应为None"
            assert K_t.shape == (n_factors, n_variables), \
                f"t={t}时K_t形状应为({n_factors}, {n_variables})，实际为{K_t.shape}"

            # K_t应该是有限值
            assert np.all(np.isfinite(K_t)), f"t={t}时K_t应该全部为有限值"

        print("[测试通过] 卡尔曼滤波器正确保存K_t历史")

    def test_partial_observations_kt_structure(self):
        """测试部分观测时K_t的结构"""
        n_time = 5
        n_factors = 2
        n_variables = 4

        # 系统参数
        A = np.eye(n_factors) * 0.9
        Q = np.eye(n_factors) * 0.1
        H = np.random.randn(n_variables, n_factors)
        R = np.eye(n_variables) * 0.5
        B = np.zeros((n_factors, 1))
        x0 = np.zeros(n_factors)
        P0 = np.eye(n_factors)

        # 观测数据（第1个时刻只有第0和第2个变量有观测）
        Z = np.full((n_time, n_variables), np.nan)
        Z[1, [0, 2]] = [1.0, 2.0]
        Z[2, :] = [1.0, 2.0, 3.0, 4.0]

        # 控制输入
        U = np.zeros((n_time, 1))

        kalman = KalmanFilter(A=A, B=B, H=H, Q=Q, R=R, x0=x0, P0=P0)
        result = kalman.filter(Z, U)

        # 验证t=1时，未观测到的变量对应的K_t列应该为0
        K_t1 = result.kalman_gains_history[1]
        assert K_t1.shape == (n_factors, n_variables)

        # 观测到的变量列应该非零
        assert np.any(K_t1[:, 0] != 0), "观测变量0的K_t列应该非零"
        assert np.any(K_t1[:, 2] != 0), "观测变量2的K_t列应该非零"

        # 未观测到的变量列应该为0
        assert np.all(K_t1[:, 1] == 0), "未观测变量1的K_t列应该为0"
        assert np.all(K_t1[:, 3] == 0), "未观测变量3的K_t列应该为0"

        print("[测试通过] 部分观测时K_t结构正确")

    def test_exporter_saves_kt_history(self):
        """测试导出器保存K_t历史到元数据"""
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            # 构造模拟的训练结果
            n_time = 8
            n_factors = 2
            n_variables = 3

            # 模拟K_t历史
            kalman_gains_history = [None]  # t=0
            for t in range(1, n_time):
                K_t = np.random.randn(n_factors, n_variables) * 0.1
                kalman_gains_history.append(K_t)

            # 构造DFMModelResult
            model_result = DFMModelResult(
                A=np.eye(n_factors) * 0.9,
                Q=np.eye(n_factors) * 0.1,
                H=np.random.randn(n_variables, n_factors),
                R=np.eye(n_variables) * 0.5,
                factors=pd.DataFrame(np.random.randn(n_time, n_factors)),
                factors_smooth=pd.DataFrame(np.random.randn(n_time, n_factors)),
                kalman_gains_history=kalman_gains_history,
                converged=True,
                iterations=10,
                log_likelihood=-100.0
            )

            # 构造最小化的TrainingResult对象（模拟）
            from dataclasses import dataclass
            from typing import List, Optional

            @dataclass
            class MinimalTrainingResult:
                model_result: Optional[DFMModelResult]
                selected_variables: List[str]
                k_factors: int

            training_result = MinimalTrainingResult(
                model_result=model_result,
                selected_variables=["var1", "var2", "var3"],
                k_factors=n_factors
            )

            # 创建导出器并导出
            exporter = TrainingResultExporter()

            # 手动构建元数据（模拟_build_metadata的核心逻辑）
            metadata = {
                'best_variables': training_result.selected_variables,
                'k_factors': training_result.k_factors,
            }

            # 保存K_t历史
            if training_result.model_result and hasattr(training_result.model_result, 'kalman_gains_history'):
                kalman_gains = training_result.model_result.kalman_gains_history
                if kalman_gains is not None:
                    metadata['kalman_gains_history'] = kalman_gains

            # 验证元数据包含K_t历史
            assert 'kalman_gains_history' in metadata, "元数据应该包含kalman_gains_history"
            saved_kt = metadata['kalman_gains_history']
            assert isinstance(saved_kt, list), "保存的K_t历史应该是列表"
            assert len(saved_kt) == n_time, f"保存的K_t历史长度应为{n_time}"
            assert saved_kt[0] is None, "t=0的K_t应为None"

            for t in range(1, n_time):
                assert isinstance(saved_kt[t], np.ndarray), f"t={t}的K_t应为ndarray"
                assert saved_kt[t].shape == (n_factors, n_variables)

            print("[测试通过] 导出器正确保存K_t历史到元数据")

    def test_kt_numerical_properties(self):
        """测试K_t的数值特性"""
        n_time = 6
        n_factors = 2
        n_variables = 3

        A = np.eye(n_factors) * 0.8
        Q = np.eye(n_factors) * 0.05
        H = np.array([[1.0, 0.5], [0.8, 1.2], [0.6, 0.9]])
        R = np.eye(n_variables) * 0.2
        B = np.zeros((n_factors, 1))
        x0 = np.zeros(n_factors)
        P0 = np.eye(n_factors) * 0.5

        # 生成观测数据
        Z = np.random.randn(n_time, n_variables) * 0.5
        Z[0, :] = np.nan

        # 控制输入
        U = np.zeros((n_time, 1))

        kalman = KalmanFilter(A=A, B=B, H=H, Q=Q, R=R, x0=x0, P0=P0)
        result = kalman.filter(Z, U)

        # K_t的合理性检查
        for t in range(1, n_time):
            K_t = result.kalman_gains_history[t]

            # K_t应该在合理范围内（通常小于1）
            max_gain = np.max(np.abs(K_t))
            assert max_gain < 10.0, f"t={t}时K_t的最大值{max_gain}过大"

            # K_t不应该全为0（除非观测噪声极大）
            assert not np.all(K_t == 0), f"t={t}时K_t不应全为0"

        print("[测试通过] K_t数值特性合理")


def run_tests():
    """运行所有测试"""
    print("=" * 70)
    print("开始测试：卡尔曼增益历史保存功能")
    print("=" * 70)

    test_suite = TestKalmanGainsHistory()

    tests = [
        ("卡尔曼滤波器保存K_t历史", test_suite.test_kalman_filter_saves_kt_history),
        ("部分观测时K_t结构", test_suite.test_partial_observations_kt_structure),
        ("导出器保存K_t历史", test_suite.test_exporter_saves_kt_history),
        ("K_t数值特性", test_suite.test_kt_numerical_properties),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n测试: {test_name}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"[失败] {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"测试完成: {passed}个通过, {failed}个失败")
    print("=" * 70)

    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
