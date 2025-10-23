# -*- coding: utf-8 -*-
"""
对比测试基类

提供baseline加载、结果对比等工具函数，供所有一致性测试使用。
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ConsistencyTestBase:
    """对比测试基类

    提供baseline结果加载和对比工具函数。
    """

    # 项目根目录
    PROJECT_ROOT = Path(__file__).resolve().parents[5]

    # Baseline目录
    BASELINE_DIR = PROJECT_ROOT / "dashboard" / "DFM" / "train_ref" / "tests" / "consistency" / "baseline"

    # 随机种子（与baseline生成保持一致）
    SEED = 42

    @classmethod
    def setup_seed(cls, seed: Optional[int] = None):
        """设置随机种子，确保可重现性

        Args:
            seed: 随机种子，默认使用类属性SEED
        """
        if seed is None:
            seed = cls.SEED

        np.random.seed(seed)
        logger.info(f"随机种子已设置: {seed}")

    @classmethod
    def load_baseline_config(cls, case_id: str) -> Dict[str, Any]:
        """加载baseline配置

        Args:
            case_id: 测试案例ID（如'case_1'）

        Returns:
            配置字典

        Raises:
            FileNotFoundError: 配置文件不存在
        """
        config_path = cls.BASELINE_DIR / case_id / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Baseline配置不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def load_baseline_parameters(cls, case_id: str) -> Dict[str, np.ndarray]:
        """加载baseline模型参数

        Args:
            case_id: 测试案例ID

        Returns:
            参数字典 {'A': array, 'Q': array, 'H': array, 'R': array}

        Raises:
            FileNotFoundError: 参数文件不存在
        """
        params_path = cls.BASELINE_DIR / case_id / "parameters.pkl"

        if not params_path.exists():
            raise FileNotFoundError(f"Baseline参数不存在: {params_path}")

        with open(params_path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def load_baseline_states(cls, case_id: str) -> Dict[str, np.ndarray]:
        """加载baseline状态估计

        Args:
            case_id: 测试案例ID

        Returns:
            状态字典 {'filtered': array, 'smoothed': array, ...}

        Raises:
            FileNotFoundError: 状态文件不存在
        """
        states_path = cls.BASELINE_DIR / case_id / "states.pkl"

        if not states_path.exists():
            raise FileNotFoundError(f"Baseline状态不存在: {states_path}")

        with open(states_path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def load_baseline_predictions(cls, case_id: str) -> Dict[str, np.ndarray]:
        """加载baseline预测结果

        Args:
            case_id: 测试案例ID

        Returns:
            预测字典 {'forecast_is': array, 'forecast_oos': array}

        Raises:
            FileNotFoundError: 预测文件不存在
        """
        preds_path = cls.BASELINE_DIR / case_id / "predictions.pkl"

        if not preds_path.exists():
            raise FileNotFoundError(f"Baseline预测不存在: {preds_path}")

        with open(preds_path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def load_baseline_metrics(cls, case_id: str) -> Dict[str, float]:
        """加载baseline评估指标

        Args:
            case_id: 测试案例ID

        Returns:
            指标字典 {'is_rmse': float, 'oos_rmse': float, ...}

        Raises:
            FileNotFoundError: 指标文件不存在
        """
        metrics_path = cls.BASELINE_DIR / case_id / "metrics.json"

        if not metrics_path.exists():
            raise FileNotFoundError(f"Baseline指标不存在: {metrics_path}")

        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def assert_arrays_close(
        actual: np.ndarray,
        expected: np.ndarray,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        name: str = "array"
    ):
        """断言两个数组近似相等

        Args:
            actual: 实际值
            expected: 期望值
            rtol: 相对容差
            atol: 绝对容差
            name: 数组名称（用于错误消息）

        Raises:
            AssertionError: 数组不近似相等
        """
        if actual.shape != expected.shape:
            raise AssertionError(
                f"{name} 形状不匹配: actual={actual.shape}, expected={expected.shape}"
            )

        if not np.allclose(actual, expected, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(actual - expected))
            rel_diff = np.max(np.abs((actual - expected) / (expected + 1e-10)))

            raise AssertionError(
                f"{name} 数值差异过大:\n"
                f"  最大绝对差异: {max_diff:.2e} (容差: {atol:.2e})\n"
                f"  最大相对差异: {rel_diff:.2e} (容差: {rtol:.2e})\n"
                f"  actual shape: {actual.shape}\n"
                f"  expected shape: {expected.shape}"
            )

        logger.info(f"{name} 验证通过")

    @staticmethod
    def assert_scalars_close(
        actual: float,
        expected: float,
        rtol: float = 1e-4,
        atol: float = 1e-4,
        name: str = "scalar"
    ):
        """断言两个标量近似相等

        Args:
            actual: 实际值
            expected: 期望值
            rtol: 相对容差
            atol: 绝对容差
            name: 标量名称（用于错误消息）

        Raises:
            AssertionError: 标量不近似相等
        """
        abs_diff = abs(actual - expected)
        rel_diff = abs_diff / (abs(expected) + 1e-10)

        if abs_diff > atol and rel_diff > rtol:
            raise AssertionError(
                f"{name} 数值差异过大:\n"
                f"  actual: {actual}\n"
                f"  expected: {expected}\n"
                f"  绝对差异: {abs_diff:.2e} (容差: {atol:.2e})\n"
                f"  相对差异: {rel_diff:.2e} (容差: {rtol:.2e})"
            )

        logger.info(f"{name} 验证通过: {actual:.6f} ≈ {expected:.6f}")

    @staticmethod
    def compute_l2_norm_diff(
        actual: np.ndarray,
        expected: np.ndarray
    ) -> float:
        """计算两个矩阵的L2范数差异

        Args:
            actual: 实际矩阵
            expected: 期望矩阵

        Returns:
            L2范数差异
        """
        diff = actual - expected
        return np.linalg.norm(diff, ord='fro')

    @staticmethod
    def compute_relative_error(
        actual: np.ndarray,
        expected: np.ndarray
    ) -> float:
        """计算相对误差

        Args:
            actual: 实际值
            expected: 期望值

        Returns:
            相对误差（百分比）
        """
        abs_diff = np.abs(actual - expected)
        abs_expected = np.abs(expected) + 1e-10
        rel_error = np.mean(abs_diff / abs_expected) * 100
        return rel_error

    @classmethod
    def check_baseline_exists(cls, case_id: str) -> Tuple[bool, str]:
        """检查baseline是否存在

        Args:
            case_id: 测试案例ID

        Returns:
            (是否存在, 消息)
        """
        case_dir = cls.BASELINE_DIR / case_id

        if not case_dir.exists():
            return False, f"案例目录不存在: {case_dir}"

        required_files = ['config.json']
        optional_files = ['parameters.pkl', 'states.pkl', 'predictions.pkl', 'metrics.json']

        missing_required = []
        missing_optional = []

        for fname in required_files:
            if not (case_dir / fname).exists():
                missing_required.append(fname)

        for fname in optional_files:
            if not (case_dir / fname).exists():
                missing_optional.append(fname)

        if missing_required:
            return False, f"缺少必需文件: {missing_required}"

        if missing_optional:
            logger.warning(f"缺少可选文件: {missing_optional}")

        return True, "Baseline完整"

    # ========================================================================
    # 零容差对比工具函数 (新增)
    # ========================================================================

    @staticmethod
    def assert_exact_equality(
        actual: float,
        expected: float,
        name: str = "scalar"
    ):
        """断言两个标量完全相等(零容差)

        Args:
            actual: 实际值
            expected: 期望值
            name: 标量名称(用于错误消息)

        Raises:
            AssertionError: 标量不完全相等
        """
        if actual != expected:
            raise AssertionError(
                f"{name} 数值不完全相等(零容差):\n"
                f"  actual:   {actual!r}\n"
                f"  expected: {expected!r}\n"
                f"  差异:     {actual - expected!r}\n"
                f"  提示: 要求完全相等, 不允许任何容差"
            )

        logger.info(f"{name} 验证通过(完全相等): {actual!r} == {expected!r}")

    @staticmethod
    def assert_array_exact_equal(
        actual: np.ndarray,
        expected: np.ndarray,
        name: str = "array"
    ):
        """断言两个数组完全相等(逐位比较, 零容差)

        Args:
            actual: 实际数组
            expected: 期望数组
            name: 数组名称(用于错误消息)

        Raises:
            AssertionError: 数组不完全相等
        """
        # 形状检查
        if actual.shape != expected.shape:
            raise AssertionError(
                f"{name} 形状不匹配:\n"
                f"  actual shape:   {actual.shape}\n"
                f"  expected shape: {expected.shape}"
            )

        # 完全相等检查(使用np.array_equal)
        if not np.array_equal(actual, expected):
            # 查找首个差异位置
            diff_mask = (actual != expected)
            first_diff_idx = np.unravel_index(np.argmax(diff_mask), actual.shape)

            raise AssertionError(
                f"{name} 数值不完全相等(零容差):\n"
                f"  形状: {actual.shape}\n"
                f"  首个差异位置: {first_diff_idx}\n"
                f"    actual[{first_diff_idx}]:   {actual[first_diff_idx]!r}\n"
                f"    expected[{first_diff_idx}]: {expected[first_diff_idx]!r}\n"
                f"    差异: {actual[first_diff_idx] - expected[first_diff_idx]!r}\n"
                f"  提示: 要求逐位完全相等, 不允许任何容差"
            )

        logger.info(f"{name} 验证通过(完全相等): shape={actual.shape}")

    @staticmethod
    def assert_matrix_exact_equal(
        actual: np.ndarray,
        expected: np.ndarray,
        name: str = "matrix"
    ):
        """断言两个矩阵完全相等(零容差)

        Args:
            actual: 实际矩阵
            expected: 期望矩阵
            name: 矩阵名称(用于错误消息)

        Raises:
            AssertionError: 矩阵不完全相等
        """
        # 委托给assert_array_exact_equal
        ConsistencyTestBase.assert_array_exact_equal(actual, expected, name)

    @staticmethod
    def assert_dataframe_exact_equal(
        actual: pd.DataFrame,
        expected: pd.DataFrame,
        name: str = "dataframe"
    ):
        """断言两个DataFrame完全相等(零容差)

        Args:
            actual: 实际DataFrame
            expected: 期望DataFrame
            name: DataFrame名称(用于错误消息)

        Raises:
            AssertionError: DataFrame不完全相等
        """
        # 列名检查
        if not actual.columns.equals(expected.columns):
            raise AssertionError(
                f"{name} 列名不匹配:\n"
                f"  actual:   {list(actual.columns)}\n"
                f"  expected: {list(expected.columns)}"
            )

        # 索引检查
        if not actual.index.equals(expected.index):
            raise AssertionError(
                f"{name} 索引不匹配:\n"
                f"  actual shape:   {actual.index.shape}\n"
                f"  expected shape: {expected.index.shape}"
            )

        # 数值完全相等检查
        ConsistencyTestBase.assert_array_exact_equal(
            actual.values,
            expected.values,
            name=f"{name}.values"
        )

        logger.info(f"{name} 验证通过(完全相等): shape={actual.shape}")

    @staticmethod
    def log_detailed_diff(
        actual: np.ndarray,
        expected: np.ndarray,
        name: str = "array",
        max_entries: int = 10
    ) -> str:
        """记录详细的数组差异信息

        Args:
            actual: 实际数组
            expected: 期望数组
            name: 数组名称
            max_entries: 最多记录多少个差异位置

        Returns:
            详细差异信息字符串
        """
        if actual.shape != expected.shape:
            return f"{name} 形状不匹配: {actual.shape} vs {expected.shape}"

        diff = actual - expected
        diff_mask = (actual != expected)
        n_diff = np.sum(diff_mask)

        if n_diff == 0:
            return f"{name} 完全相等(无差异)"

        # 统计信息
        max_abs_diff = np.max(np.abs(diff))
        mean_abs_diff = np.mean(np.abs(diff[diff_mask]))
        std_abs_diff = np.std(np.abs(diff[diff_mask]))

        msg = [
            f"{name} 差异统计:",
            f"  形状: {actual.shape}",
            f"  差异元素数: {n_diff} / {actual.size} ({n_diff/actual.size*100:.2f}%)",
            f"  最大绝对差异: {max_abs_diff:.2e}",
            f"  平均绝对差异: {mean_abs_diff:.2e}",
            f"  标准差: {std_abs_diff:.2e}",
            ""
        ]

        # 记录前max_entries个差异位置
        diff_indices = np.where(diff_mask)
        n_show = min(len(diff_indices[0]), max_entries)

        if n_show > 0:
            msg.append(f"前{n_show}个差异位置:")
            for i in range(n_show):
                idx = tuple(d[i] for d in diff_indices)
                msg.append(
                    f"  [{idx}]: actual={actual[idx]!r}, "
                    f"expected={expected[idx]!r}, "
                    f"diff={diff[idx]!r}"
                )

        return "\n".join(msg)

    @staticmethod
    def load_simulated_dataset(dataset_name: str, fixtures_dir: Optional[Path] = None) -> Dict[str, Any]:
        """加载模拟数据集

        Args:
            dataset_name: 数据集名称(如'small', 'medium')
            fixtures_dir: fixtures目录路径,默认为consistency/fixtures/

        Returns:
            数据集字典,包含Z, true_factors, true_Lambda等

        Raises:
            FileNotFoundError: 数据集文件不存在
        """
        if fixtures_dir is None:
            fixtures_dir = Path(__file__).parent / "fixtures"

        dataset_path = fixtures_dir / f"{dataset_name}_dataset.npz"

        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集不存在: {dataset_path}")

        # 加载npz文件
        data = np.load(dataset_path, allow_pickle=True)

        # 构建返回字典
        result = {
            'Z': pd.DataFrame(
                data['Z'],
                columns=data['Z_columns'].tolist()
            ),
            'true_factors': pd.DataFrame(
                data['true_factors'],
                columns=data['factor_columns'].tolist()
            ),
            'true_Lambda': data['true_Lambda'],
            'true_A': data['true_A'],
            'true_Q': data['true_Q'],
            'true_R': data['true_R'],
            'n_time': int(data['n_time']),
            'n_obs': int(data['n_obs']),
            'n_factors': int(data['n_factors'])
        }

        logger.info(f"加载模拟数据集: {dataset_name}, Z shape={result['Z'].shape}")

        return result

    @staticmethod
    def assert_allclose_strict(
        actual: np.ndarray,
        expected: np.ndarray,
        name: str = "array",
        rtol: float = 1e-10,
        atol: float = 1e-14
    ):
        """断言两个数组在极严格容差内接近

        使用比NumPy默认值严格10万倍的容差标准:
        - rtol=1e-10 (默认1e-5的10万倍严格)
        - atol=1e-14 (默认1e-8的100万倍严格)

        此标准能检测所有实质性算法差异,同时允许浮点数运算的固有误差(~1e-15)

        Args:
            actual: 实际数组
            expected: 期望数组
            name: 数组名称(用于错误消息)
            rtol: 相对误差容忍度
            atol: 绝对误差容忍度

        Raises:
            AssertionError: 数组不在容差范围内接近
        """
        # 形状检查
        if actual.shape != expected.shape:
            raise AssertionError(
                f"{name} 形状不匹配:\n"
                f"  actual shape:   {actual.shape}\n"
                f"  expected shape: {expected.shape}"
            )

        # 数值接近检查
        if not np.allclose(actual, expected, rtol=rtol, atol=atol):
            # 计算差异统计
            diff = np.abs(actual - expected)
            max_diff = np.max(diff)
            max_diff_idx = np.unravel_index(np.argmax(diff), actual.shape)

            # 计算相对误差
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diff = diff / np.abs(expected)
                rel_diff[~np.isfinite(rel_diff)] = 0  # 处理除零情况
                max_rel_diff = np.max(rel_diff)

            raise AssertionError(
                f"\n{'='*60}\n"
                f"极严格容差验证失败: {name}\n"
                f"{'='*60}\n"
                f"容差标准:\n"
                f"  相对误差容忍度(rtol): {rtol} (比默认值1e-5严格{1e-5/rtol:.0f}倍)\n"
                f"  绝对误差容忍度(atol): {atol} (比默认值1e-8严格{1e-8/atol:.0f}倍)\n"
                f"\n实际差异:\n"
                f"  最大绝对差异: {max_diff:.6e} at {max_diff_idx}\n"
                f"  最大相对差异: {max_rel_diff:.6e}\n"
                f"  actual[{max_diff_idx}]:   {actual[max_diff_idx]!r}\n"
                f"  expected[{max_diff_idx}]: {expected[max_diff_idx]!r}\n"
                f"\n建议:\n"
                f"  - 如果差异在1e-15量级,可能是浮点数固有误差\n"
                f"  - 如果差异超过1e-14,需要检查算法实现差异\n"
                f"{'='*60}"
            )

        logger.info(
            f"{name} 验证通过(极严格容差): shape={actual.shape}, "
            f"max_diff={np.max(np.abs(actual - expected)):.6e}"
        )

    @staticmethod
    def assert_eigenvectors_equal_up_to_sign(
        actual: np.ndarray,
        expected: np.ndarray,
        name: str = "eigenvectors",
        use_strict_tolerance: bool = True
    ):
        """断言特征向量/奇异向量在允许符号歧义的情况下相等

        数学背景:
        如果v是特征向量,则-v也是特征向量(特征向量的符号是任意的)
        因此需要检查: v1 == v2 OR v1 == -v2

        Args:
            actual: 实际特征向量矩阵 (n_dim, n_vectors)
            expected: 期望特征向量矩阵 (n_dim, n_vectors)
            name: 矩阵名称
            use_strict_tolerance: 是否使用严格容差(True)或零容差(False)

        Raises:
            AssertionError: 特征向量既不正向相等也不反向相等
        """
        # 形状检查
        if actual.shape != expected.shape:
            raise AssertionError(
                f"{name} 形状不匹配:\n"
                f"  actual shape:   {actual.shape}\n"
                f"  expected shape: {expected.shape}"
            )

        n_vectors = actual.shape[1]

        for i in range(n_vectors):
            v_actual = actual[:, i]
            v_expected = expected[:, i]

            # 检查正向或反向相等
            if use_strict_tolerance:
                # 使用极严格容差
                positive_match = np.allclose(v_actual, v_expected, rtol=1e-10, atol=1e-14)
                negative_match = np.allclose(v_actual, -v_expected, rtol=1e-10, atol=1e-14)
            else:
                # 使用零容差
                positive_match = np.array_equal(v_actual, v_expected)
                negative_match = np.array_equal(v_actual, -v_expected)

            if not (positive_match or negative_match):
                # 计算两种方向的差异
                diff_positive = np.max(np.abs(v_actual - v_expected))
                diff_negative = np.max(np.abs(v_actual + v_expected))

                raise AssertionError(
                    f"\n{'='*60}\n"
                    f"特征向量符号歧义验证失败: {name}第{i}列\n"
                    f"{'='*60}\n"
                    f"既不满足 v1 ≈ v2, 也不满足 v1 ≈ -v2\n"
                    f"\n差异分析:\n"
                    f"  正向差异(v1 - v2):  max={diff_positive:.6e}\n"
                    f"  反向差异(v1 + v2):  max={diff_negative:.6e}\n"
                    f"\n向量样本:\n"
                    f"  actual[:5]:   {v_actual[:5]}\n"
                    f"  expected[:5]: {v_expected[:5]}\n"
                    f"  -expected[:5]: {-v_expected[:5]}\n"
                    f"{'='*60}"
                )

        logger.info(
            f"{name} 验证通过(允许符号歧义): shape={actual.shape}, "
            f"{'strict_tolerance' if use_strict_tolerance else 'zero_tolerance'}"
        )
