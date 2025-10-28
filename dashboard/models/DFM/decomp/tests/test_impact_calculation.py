# -*- coding: utf-8 -*-
"""
测试影响分解计算公式

验证decomp模块使用正确的公式计算数据更新对目标变量的影响
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from dashboard.models.DFM.decomp.core.model_loader import SavedNowcastData
from dashboard.models.DFM.decomp.core.nowcast_extractor import NowcastExtractor
from dashboard.models.DFM.decomp.core.impact_analyzer import ImpactAnalyzer, DataRelease, ImpactResult


class TestImpactCalculationFormula:
    """测试影响计算公式的正确性"""

    def setup_method(self):
        """为每个测试方法准备数据"""
        # 模型参数
        self.n_factors = 2
        self.n_variables = 4
        self.n_time = 10

        # 因子载荷矩阵 H (n_variables, n_factors)
        self.H = np.array([
            [1.0, 0.5],   # 目标变量
            [0.8, 1.2],
            [0.6, 0.9],
            [1.1, 0.7]
        ])

        # 卡尔曼增益历史 (n_factors, n_variables)
        self.kalman_gains_history = []
        self.kalman_gains_history.append(None)  # t=0

        for t in range(1, self.n_time):
            # 生成合理的K_t矩阵
            K_t = np.array([
                [0.3, 0.2, 0.15, 0.25],
                [0.25, 0.3, 0.2, 0.15]
            ])
            self.kalman_gains_history.append(K_t)

        # 变量映射
        self.variable_names = ["目标变量", "变量1", "变量2", "变量3"]
        self.variable_index_map = {name: idx for idx, name in enumerate(self.variable_names)}
        self.target_variable_index = 0

        # 时间序列索引
        self.date_index = pd.date_range('2024-01-01', periods=self.n_time, freq='M')

    def create_saved_nowcast_data(self) -> SavedNowcastData:
        """创建模拟的SavedNowcastData对象"""
        data = SavedNowcastData()
        data.factor_loadings = self.H
        data.kalman_gains_history = self.kalman_gains_history
        data.target_variable_index = self.target_variable_index
        data.variable_index_map = self.variable_index_map

        # 模拟因子序列
        factor_data = np.random.randn(self.n_time, self.n_factors) * 0.5
        data.factor_series = pd.DataFrame(
            factor_data,
            index=self.date_index,
            columns=[f'Factor_{i}' for i in range(self.n_factors)]
        )

        # 模拟nowcast序列
        nowcast_values = np.dot(factor_data, self.H[self.target_variable_index, :])
        data.nowcast_series = pd.Series(
            nowcast_values,
            index=self.date_index,
            name='nowcast'
        )

        return data

    def test_impact_formula_basic(self):
        """测试基本的影响计算公式: Δy = λ_y' × K_t[:, i] × v_i"""
        data = self.create_saved_nowcast_data()
        extractor = NowcastExtractor(data)
        analyzer = ImpactAnalyzer(extractor)

        # 创建数据释放事件
        release_date = self.date_index[5]
        variable_name = "变量1"
        variable_index = self.variable_index_map[variable_name]

        observed_value = 2.0
        expected_value = 1.5
        innovation = observed_value - expected_value  # v_i = 0.5

        release = DataRelease(
            timestamp=release_date,
            variable_name=variable_name,
            observed_value=observed_value,
            expected_value=expected_value
        )

        # 计算影响
        result = analyzer.calculate_single_release_impact(release)

        # 手动验证计算
        lambda_y = self.H[self.target_variable_index, :]  # (n_factors,)
        K_t = self.kalman_gains_history[5]  # (n_factors, n_variables)
        K_col = K_t[:, variable_index]  # (n_factors,)

        # 正确公式: Δy = λ_y' × K_col × v_i
        expected_impact = np.dot(lambda_y, K_col) * innovation

        # 验证结果
        assert result.impact_on_target is not None, "影响值不应为None"
        np.testing.assert_almost_equal(
            result.impact_on_target,
            expected_impact,
            decimal=10,
            err_msg="影响计算与理论值不符"
        )

        print(f"[测试通过] 基本影响公式计算正确")
        print(f"  λ_y = {lambda_y}")
        print(f"  K_col = {K_col}")
        print(f"  v_i = {innovation}")
        print(f"  Δy = {result.impact_on_target:.6f} (理论值: {expected_impact:.6f})")

    def test_impact_formula_different_variables(self):
        """测试不同变量的影响计算"""
        data = self.create_saved_nowcast_data()
        extractor = NowcastExtractor(data)
        analyzer = ImpactAnalyzer(extractor)

        release_date = self.date_index[3]
        innovation = 1.0

        impacts = {}

        for var_name in ["变量1", "变量2", "变量3"]:
            var_index = self.variable_index_map[var_name]

            release = DataRelease(
                timestamp=release_date,
                variable_name=var_name,
                observed_value=1.0,
                expected_value=0.0
            )

            result = analyzer.calculate_single_release_impact(release)

            # 手动计算
            lambda_y = self.H[self.target_variable_index, :]
            K_t = self.kalman_gains_history[3]
            K_col = K_t[:, var_index]
            expected = np.dot(lambda_y, K_col) * innovation

            np.testing.assert_almost_equal(result.impact_on_target, expected, decimal=10)
            impacts[var_name] = result.impact_on_target

        print(f"[测试通过] 不同变量影响计算正确")
        for var_name, impact in impacts.items():
            print(f"  {var_name}: {impact:.6f}")

    def test_impact_sign_consistency(self):
        """测试影响符号的一致性"""
        data = self.create_saved_nowcast_data()
        extractor = NowcastExtractor(data)
        analyzer = ImpactAnalyzer(extractor)

        release_date = self.date_index[4]
        variable_name = "变量1"

        # 测试正向新息
        release_pos = DataRelease(
            timestamp=release_date,
            variable_name=variable_name,
            observed_value=1.0,
            expected_value=0.0
        )
        result_pos = analyzer.calculate_single_release_impact(release_pos)

        # 测试负向新息
        release_neg = DataRelease(
            timestamp=release_date,
            variable_name=variable_name,
            observed_value=-1.0,
            expected_value=0.0
        )
        result_neg = analyzer.calculate_single_release_impact(release_neg)

        # 符号应该相反
        assert np.sign(result_pos.impact_on_target) == -np.sign(result_neg.impact_on_target), \
            "正负新息的影响符号应该相反"

        # 幅度应该相同
        np.testing.assert_almost_equal(
            np.abs(result_pos.impact_on_target),
            np.abs(result_neg.impact_on_target),
            decimal=10,
            err_msg="正负新息的影响幅度应该相同"
        )

        print(f"[测试通过] 影响符号一致性正确")
        print(f"  正向新息影响: {result_pos.impact_on_target:.6f}")
        print(f"  负向新息影响: {result_neg.impact_on_target:.6f}")

    def test_impact_zero_innovation(self):
        """测试零新息时影响为零"""
        data = self.create_saved_nowcast_data()
        extractor = NowcastExtractor(data)
        analyzer = ImpactAnalyzer(extractor)

        release = DataRelease(
            timestamp=self.date_index[2],
            variable_name="变量2",
            observed_value=1.5,
            expected_value=1.5  # 无新息
        )

        result = analyzer.calculate_single_release_impact(release)

        np.testing.assert_almost_equal(
            result.impact_on_target,
            0.0,
            decimal=10,
            err_msg="零新息时影响应为0"
        )

        print(f"[测试通过] 零新息时影响为0")

    def test_impact_with_factor_state_change(self):
        """测试通过因子状态变化验证影响计算"""
        data = self.create_saved_nowcast_data()
        extractor = NowcastExtractor(data)
        analyzer = ImpactAnalyzer(extractor)

        release_date = self.date_index[6]
        variable_name = "变量2"
        variable_index = self.variable_index_map[variable_name]

        observed = 3.0
        expected = 2.0
        innovation = observed - expected

        release = DataRelease(
            timestamp=release_date,
            variable_name=variable_name,
            observed_value=observed,
            expected_value=expected
        )

        result = analyzer.calculate_single_release_impact(release)

        # 分步验证
        lambda_y = self.H[self.target_variable_index, :]
        K_t = self.kalman_gains_history[6]
        K_col = K_t[:, variable_index]

        # 因子状态变化
        delta_f = K_col * innovation
        print(f"  因子状态变化 Δf = {delta_f}")

        # 传递到目标变量
        impact_via_factors = np.dot(lambda_y, delta_f)
        print(f"  通过因子传递的影响 = {impact_via_factors:.6f}")

        np.testing.assert_almost_equal(
            result.impact_on_target,
            impact_via_factors,
            decimal=10
        )

        print(f"[测试通过] 通过因子状态变化验证影响计算")

    def test_impact_result_contains_calculation_details(self):
        """测试影响结果包含计算细节"""
        data = self.create_saved_nowcast_data()
        extractor = NowcastExtractor(data)
        analyzer = ImpactAnalyzer(extractor)

        release = DataRelease(
            timestamp=self.date_index[7],
            variable_name="变量3",
            observed_value=2.5,
            expected_value=2.0
        )

        result = analyzer.calculate_single_release_impact(release)

        # 验证结果包含必要的计算细节
        assert result.release.timestamp == release.timestamp
        assert result.release.variable_name == release.variable_name
        innovation = release.observed_value - release.expected_value
        assert result.impact_on_target is not None

        # 验证计算细节
        assert result.calculation_details is not None
        details = result.calculation_details

        assert 'factor_loading_vector' in details, "应包含λ_y (factor_loading_vector)"
        assert 'kalman_gain_vector' in details, "应包含K列向量 (kalman_gain_vector)"
        assert 'factor_state_change' in details, "应包含因子变化Δf (factor_state_change)"

        print(f"[测试通过] 影响结果包含完整的计算细节")
        print(f"  λ_y: {details['factor_loading_vector']}")
        print(f"  K_col: {details['kalman_gain_vector']}")
        print(f"  Δf: {details['factor_state_change']}")


class TestImpactAnalyzerEdgeCases:
    """测试影响分析器的边界情况"""

    def test_missing_kt_history_raises_error(self):
        """测试缺失K_t历史时应抛出错误"""
        data = SavedNowcastData()
        data.factor_loadings = np.random.randn(3, 2)
        data.kalman_gains_history = None  # 缺失K_t历史
        data.target_variable_index = 0
        data.variable_index_map = {"var1": 0, "var2": 1, "var3": 2}

        extractor = NowcastExtractor(data)
        analyzer = ImpactAnalyzer(extractor)

        date = pd.Timestamp('2024-06-01')
        release = DataRelease(
            timestamp=date,
            variable_name="var1",
            observed_value=1.0,
            expected_value=0.5
        )

        # 应该抛出ComputationError
        from dashboard.models.DFM.decomp.utils.exceptions import ComputationError
        with pytest.raises(ComputationError, match="影响分解功能需要卡尔曼增益历史数据"):
            analyzer.calculate_single_release_impact(release)

        print(f"[测试通过] 缺失K_t历史时正确抛出错误")

    def test_invalid_variable_name_raises_error(self):
        """测试无效变量名时应抛出错误"""
        data = SavedNowcastData()
        data.factor_loadings = np.random.randn(3, 2)
        data.kalman_gains_history = [None, np.random.randn(2, 3)]
        data.target_variable_index = 0
        data.variable_index_map = {"var1": 0, "var2": 1, "var3": 2}

        extractor = NowcastExtractor(data)
        analyzer = ImpactAnalyzer(extractor)

        date = pd.Timestamp('2024-06-01')
        release = DataRelease(
            timestamp=date,
            variable_name="invalid_var",  # 不存在的变量
            observed_value=1.0,
            expected_value=0.5
        )

        # 应该抛出ComputationError
        from dashboard.models.DFM.decomp.utils.exceptions import ComputationError
        with pytest.raises(ComputationError, match="变量.*不在变量映射中"):
            analyzer.calculate_single_release_impact(release)

        print(f"[测试通过] 无效变量名时正确抛出错误")


def run_tests():
    """运行所有测试"""
    print("=" * 70)
    print("开始测试：影响分解计算公式")
    print("=" * 70)

    test_formula = TestImpactCalculationFormula()
    test_edge = TestImpactAnalyzerEdgeCases()

    tests = [
        ("基本影响公式", test_formula.test_impact_formula_basic),
        ("不同变量影响", test_formula.test_impact_formula_different_variables),
        ("影响符号一致性", test_formula.test_impact_sign_consistency),
        ("零新息影响", test_formula.test_impact_zero_innovation),
        ("因子状态变化验证", test_formula.test_impact_with_factor_state_change),
        ("计算细节完整性", test_formula.test_impact_result_contains_calculation_details),
        ("缺失K_t历史", test_edge.test_missing_kt_history_raises_error),
        ("无效变量名", test_edge.test_invalid_variable_name_raises_error),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n测试: {test_name}")
            # 为每个测试创建新实例（如果是TestImpactCalculationFormula的方法）
            if hasattr(test_func, '__self__') and isinstance(test_func.__self__, TestImpactCalculationFormula):
                test_func.__self__.setup_method()
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
