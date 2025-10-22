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
