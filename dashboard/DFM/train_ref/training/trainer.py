# -*- coding: utf-8 -*-
"""
DFM训练器 - 合并评估器和流水线逻辑(方案B精简架构)

包含:
- ModelEvaluator: 模型评估器(内部类)
- DFMTrainer: 主训练器(包含两阶段训练流程)

参考:
- dashboard/DFM/train_model/dfm_core.py (评估逻辑)
- dashboard/DFM/train_model/tune_dfm.py (训练流程)
"""

import os
import sys
import time
import random
import logging
import multiprocessing
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ==================== 数据类定义 ====================

@dataclass
class EvaluationMetrics:
    """评估指标"""
    is_rmse: float = np.inf
    oos_rmse: float = np.inf
    is_hit_rate: float = -np.inf
    oos_hit_rate: float = -np.inf
    is_correlation: float = -np.inf
    oos_correlation: float = -np.inf

    def to_tuple(self) -> Tuple:
        """转换为9元组格式(兼容train_model)"""
        return (
            self.is_rmse,
            self.oos_rmse,
            None,  # placeholder
            None,  # placeholder
            self.is_hit_rate,
            self.oos_hit_rate,
            False,  # is_svd_error
            None,  # placeholder
            None   # placeholder
        )


@dataclass
class DFMModelResult:
    """DFM模型结果"""
    # EM估计参数
    A: np.ndarray = None  # 状态转移矩阵
    Q: np.ndarray = None  # 状态噪声协方差
    H: np.ndarray = None  # 观测矩阵(因子载荷)
    R: np.ndarray = None  # 观测噪声协方差

    # 卡尔曼滤波结果
    factors: np.ndarray = None  # 因子时间序列
    factors_smooth: np.ndarray = None  # 平滑因子

    # 预测结果
    forecast_is: np.ndarray = None  # 样本内预测
    forecast_oos: np.ndarray = None  # 样本外预测

    # 其他信息
    converged: bool = False
    iterations: int = 0
    log_likelihood: float = -np.inf


@dataclass
class TrainingResult:
    """训练结果"""
    # 变量选择结果
    selected_variables: List[str] = field(default_factory=list)
    selection_history: List[Dict] = field(default_factory=list)

    # 因子数选择结果
    k_factors: int = 0
    factor_selection_method: str = 'fixed'
    pca_analysis: Optional[Dict] = None

    # 模型结果
    model_result: Optional[DFMModelResult] = None

    # 评估指标
    metrics: Optional[EvaluationMetrics] = None

    # 训练统计
    total_evaluations: int = 0
    svd_error_count: int = 0
    training_time: float = 0.0

    # 数据统计
    target_mean_original: float = 0.0
    target_std_original: float = 1.0

    # 输出路径
    output_dir: Optional[str] = None


# ==================== ModelEvaluator (内部类) ====================

class ModelEvaluator:
    """
    模型评估器(内部类)

    功能:
    - calculate_rmse: RMSE计算
    - calculate_hit_rate: 命中率计算
    - calculate_correlation: 相关系数计算
    - evaluate: 完整评估流程
    """

    def __init__(self):
        """初始化评估器"""
        pass

    def calculate_rmse(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> float:
        """
        计算RMSE

        Args:
            predictions: 预测值
            actuals: 实际值

        Returns:
            RMSE值
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return np.inf

        if len(predictions) != len(actuals):
            logger.warning(f"预测值和实际值长度不一致: {len(predictions)} vs {len(actuals)}")
            min_len = min(len(predictions), len(actuals))
            predictions = predictions[:min_len]
            actuals = actuals[:min_len]

        # 移除NaN值
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
        if not valid_mask.any():
            return np.inf

        predictions_clean = predictions[valid_mask]
        actuals_clean = actuals[valid_mask]

        mse = np.mean((predictions_clean - actuals_clean) ** 2)
        rmse = np.sqrt(mse)

        return float(rmse)

    def calculate_hit_rate(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        previous_values: np.ndarray
    ) -> float:
        """
        计算命中率(方向预测准确率)

        Args:
            predictions: 预测值
            actuals: 实际值
            previous_values: 前一期值

        Returns:
            命中率(0-100)
        """
        if len(predictions) == 0 or len(actuals) == 0 or len(previous_values) == 0:
            return -np.inf

        # 确保长度一致
        min_len = min(len(predictions), len(actuals), len(previous_values))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]
        previous_values = previous_values[:min_len]

        # 移除NaN值
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals) | np.isnan(previous_values))
        if not valid_mask.any():
            return -np.inf

        predictions_clean = predictions[valid_mask]
        actuals_clean = actuals[valid_mask]
        previous_clean = previous_values[valid_mask]

        # 计算方向
        pred_direction = np.sign(predictions_clean - previous_clean)
        actual_direction = np.sign(actuals_clean - previous_clean)

        # 计算命中率
        hits = (pred_direction == actual_direction).sum()
        total = len(pred_direction)

        if total == 0:
            return -np.inf

        hit_rate = (hits / total) * 100.0

        return float(hit_rate)

    def calculate_correlation(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> float:
        """
        计算相关系数

        Args:
            predictions: 预测值
            actuals: 实际值

        Returns:
            相关系数
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return -np.inf

        # 确保长度一致
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]

        # 移除NaN值
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
        if not valid_mask.any():
            return -np.inf

        predictions_clean = predictions[valid_mask]
        actuals_clean = actuals[valid_mask]

        if len(predictions_clean) < 2:
            return -np.inf

        # 计算相关系数
        corr = np.corrcoef(predictions_clean, actuals_clean)[0, 1]

        return float(corr) if np.isfinite(corr) else -np.inf

    def evaluate(
        self,
        model_result: DFMModelResult,
        target_data: pd.Series,
        train_end_date: str,
        validation_start: Optional[str] = None,
        validation_end: Optional[str] = None
    ) -> EvaluationMetrics:
        """
        完整评估流程

        Args:
            model_result: 模型结果
            target_data: 目标变量数据
            train_end_date: 训练期结束日期
            validation_start: 验证期开始日期
            validation_end: 验证期结束日期

        Returns:
            EvaluationMetrics对象
        """
        metrics = EvaluationMetrics()

        try:
            # 分割样本内和样本外
            train_data = target_data.loc[:train_end_date]

            if validation_start and validation_end:
                val_data = target_data.loc[validation_start:validation_end]
            else:
                # 如果没有指定验证期,使用训练期之后的所有数据
                val_data = target_data.loc[train_end_date:]
                if len(val_data) > 0:
                    val_data = val_data.iloc[1:]  # 排除训练期最后一天

            # 样本内评估
            if model_result.forecast_is is not None and len(train_data) > 0:
                # 对齐长度
                min_len = min(len(model_result.forecast_is), len(train_data))
                forecast_is = model_result.forecast_is[:min_len]
                actual_is = train_data.values[:min_len]

                # RMSE
                metrics.is_rmse = self.calculate_rmse(forecast_is, actual_is)

                # 相关系数
                metrics.is_correlation = self.calculate_correlation(forecast_is, actual_is)

                # 命中率(需要前一期值)
                if len(actual_is) > 1:
                    previous_is = np.concatenate([[np.nan], actual_is[:-1]])
                    metrics.is_hit_rate = self.calculate_hit_rate(
                        forecast_is[1:], actual_is[1:], previous_is[1:]
                    )

            # 样本外评估
            if model_result.forecast_oos is not None and len(val_data) > 0:
                # 对齐长度
                min_len = min(len(model_result.forecast_oos), len(val_data))
                forecast_oos = model_result.forecast_oos[:min_len]
                actual_oos = val_data.values[:min_len]

                # RMSE
                metrics.oos_rmse = self.calculate_rmse(forecast_oos, actual_oos)

                # 相关系数
                metrics.oos_correlation = self.calculate_correlation(forecast_oos, actual_oos)

                # 命中率(需要前一期值)
                if len(actual_oos) > 1:
                    # 使用训练期最后一个值作为第一个previous值
                    if len(train_data) > 0:
                        first_previous = train_data.values[-1]
                        previous_oos = np.concatenate([[first_previous], actual_oos[:-1]])
                    else:
                        previous_oos = np.concatenate([[np.nan], actual_oos[:-1]])

                    metrics.oos_hit_rate = self.calculate_hit_rate(
                        forecast_oos, actual_oos, previous_oos
                    )

        except Exception as e:
            logger.error(f"评估过程出错: {e}")
            import traceback
            traceback.print_exc()

        return metrics


# ==================== DFMTrainer (主训练器) ====================

class DFMTrainer:
    """
    DFM主训练器(合并pipeline逻辑)

    两阶段训练流程:
    1. 阶段1: 变量选择(固定k=块数)
    2. 阶段2: 因子数选择(PCA/Elbow/Fixed)
    3. 最终训练: 使用选定变量和因子数训练模型

    环境初始化:
    - 多线程BLAS配置
    - 随机种子设置
    - 静默模式控制
    """

    def __init__(self, config: 'TrainingConfig'):
        """
        初始化训练器

        Args:
            config: 训练配置对象(TrainingConfig)
        """
        self.config = config
        self.evaluator = ModelEvaluator()

        # 环境初始化
        self._init_environment()

        # 训练状态
        self.total_evaluations = 0
        self.svd_error_count = 0

    def _init_environment(self):
        """
        环境初始化和可重现性控制

        配置:
        - 多线程BLAS: 使用所有CPU核心
        - 随机种子: 42
        - 静默模式: 通过环境变量控制
        """
        # 1. 多线程BLAS配置
        cpu_count = multiprocessing.cpu_count()
        os.environ['OMP_NUM_THREADS'] = str(cpu_count)
        os.environ['MKL_NUM_THREADS'] = str(cpu_count)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_count)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)

        logger.info(f"配置多线程BLAS: {cpu_count}个线程")

        # 2. 随机种子设置
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)

        logger.info(f"设置随机种子: {SEED}")

        # 3. 静默模式(可选)
        silent_mode = os.getenv('DFM_SILENT_WARNINGS', 'false').lower() == 'true'
        if silent_mode:
            logger.info("静默模式已启用")

    def _load_and_validate_data(
        self,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        加载和验证数据

        Returns:
            (data, target_data, predictor_vars)
        """
        if progress_callback:
            progress_callback("[TRAIN] 加载数据...")

        logger.info(f"加载数据: {self.config.data_path}")

        # 加载数据 - 支持Excel和CSV格式
        file_path = Path(self.config.data_path)
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            data = pd.read_excel(self.config.data_path, index_col=0, parse_dates=True)
        elif file_path.suffix.lower() == '.csv':
            data = pd.read_csv(self.config.data_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

        # 验证目标变量
        if self.config.target_variable not in data.columns:
            raise ValueError(
                f"目标变量'{self.config.target_variable}'不在数据中. "
                f"可用列: {list(data.columns[:5])}..."
            )

        target_data = data[self.config.target_variable]

        # 确定预测变量
        if self.config.selected_indicators:
            predictor_vars = [
                v for v in self.config.selected_indicators
                if v != self.config.target_variable and v in data.columns
            ]
        else:
            predictor_vars = [
                v for v in data.columns
                if v != self.config.target_variable
            ]

        # 数据质量检查和清理
        initial_count = len(predictor_vars)
        valid_predictor_vars = []
        removed_vars = []

        for var in predictor_vars:
            var_data = data[var]
            valid_count = var_data.notna().sum()

            # 过滤全NaN或有效数据过少的列
            if valid_count < 10:  # 至少需要10个有效数据点
                removed_vars.append((var, valid_count))
                logger.warning(
                    f"移除变量'{var}': 有效数据点({valid_count}) < 最小要求(10)"
                )
            else:
                valid_predictor_vars.append(var)

        predictor_vars = valid_predictor_vars

        if removed_vars:
            logger.info(
                f"数据清理: 移除了{len(removed_vars)}个无效变量, "
                f"剩余{len(predictor_vars)}个有效变量"
            )
            if progress_callback:
                progress_callback(
                    f"[TRAIN] 数据清理: 移除{len(removed_vars)}个无效变量"
                )

        # 验证是否还有足够的预测变量
        if len(predictor_vars) < 2:
            raise ValueError(
                f"有效预测变量不足({len(predictor_vars)}个), "
                f"至少需要2个有效变量进行DFM建模"
            )

        logger.info(
            f"数据加载完成: {data.shape}, "
            f"有效预测变量数: {len(predictor_vars)}"
        )

        if progress_callback:
            progress_callback(
                f"[TRAIN] 数据加载完成: {data.shape[0]}行, "
                f"{len(predictor_vars)}个有效预测变量"
            )

        return data, target_data, predictor_vars

    def _run_variable_selection(
        self,
        data: pd.DataFrame,
        target_data: pd.Series,
        predictor_vars: List[str],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[List[str], List[Dict]]:
        """
        阶段1: 变量选择(固定k=块数)

        Returns:
            (selected_vars, selection_history)
        """
        # 检查是否启用变量选择
        if not self.config.enable_variable_selection:
            logger.info("跳过变量选择,使用全部变量")
            if progress_callback:
                progress_callback("[TRAIN] 跳过变量选择")
            return predictor_vars, []

        logger.info("=" * 60)
        logger.info("阶段1: 变量选择")
        logger.info("=" * 60)

        if progress_callback:
            progress_callback("[TRAIN] 阶段1: 开始变量选择")

        # 导入BackwardSelector
        from dashboard.DFM.train_ref.selection import BackwardSelector

        # 创建选择器
        selector = BackwardSelector(
            evaluator_func=self._evaluate_dfm_for_selection,
            criterion='rmse',
            min_variables=self.config.min_variables_after_selection or 1
        )

        # 执行选择
        initial_vars = [self.config.target_variable] + predictor_vars

        # 计算合理的k_factors: 应该小于变量数以允许变量移除
        # 使用变量数的一半，或至少为2
        k_for_selection = max(2, min(len(predictor_vars) // 2, len(predictor_vars) - 2))
        logger.info(f"变量选择使用k_factors={k_for_selection} (变量数: {len(predictor_vars)})")

        selection_result = selector.select(
            initial_variables=initial_vars,
            target_variable=self.config.target_variable,
            full_data=data,
            params={'k_factors': k_for_selection},  # 使用合理的k值
            validation_start=self.config.validation_start,
            validation_end=self.config.validation_end,
            target_freq=self.config.target_freq,
            train_end_date=self.config.train_end,
            target_mean_original=target_data.mean(),
            target_std_original=target_data.std(),
            max_iter=self.config.max_iterations,
            max_lags=self.config.max_lags,
            progress_callback=progress_callback
        )

        # 更新统计
        self.total_evaluations += selection_result.total_evaluations
        self.svd_error_count += selection_result.svd_error_count

        # 提取选定的预测变量
        selected_vars = [
            v for v in selection_result.selected_variables
            if v != self.config.target_variable
        ]

        logger.info(
            f"变量选择完成: {len(predictor_vars)} -> {len(selected_vars)}个变量"
        )

        if progress_callback:
            progress_callback(
                f"[TRAIN] 变量选择完成: 保留{len(selected_vars)}个变量"
            )

        return selected_vars, selection_result.selection_history

    def _select_num_factors(
        self,
        data: pd.DataFrame,
        selected_vars: List[str],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[int, Optional[Dict]]:
        """
        阶段2: 因子数选择

        Returns:
            (k_factors, pca_analysis)
        """
        logger.info("=" * 60)
        logger.info("阶段2: 因子数选择")
        logger.info("=" * 60)

        if progress_callback:
            progress_callback("[TRAIN] 阶段2: 开始因子数选择")

        method = self.config.factor_selection_method

        # 方法1: 固定因子数
        if method == 'fixed':
            k = self.config.k_factors
            logger.info(f"使用固定因子数: k={k}")
            if progress_callback:
                progress_callback(f"[TRAIN] 使用固定因子数: k={k}")
            return k, None

        # 方法2/3: 基于PCA
        from sklearn.decomposition import PCA

        # 准备数据(填充NaN)
        data_for_pca = data[selected_vars].fillna(0)

        # PCA分析
        pca = PCA()
        pca.fit(data_for_pca)

        explained_variance = pca.explained_variance_ratio_
        cumsum_variance = np.cumsum(explained_variance)

        pca_analysis = {
            'explained_variance': explained_variance,
            'cumsum_variance': cumsum_variance,
            'eigenvalues': pca.explained_variance_
        }

        # 方法2: 累积方差贡献率
        if method == 'cumulative':
            threshold = self.config.pca_threshold or 0.9
            k = np.argmax(cumsum_variance >= threshold) + 1
            logger.info(
                f"PCA累积方差方法: 阈值={threshold:.1%}, k={k}, "
                f"累积方差={cumsum_variance[k-1]:.1%}"
            )
            if progress_callback:
                progress_callback(
                    f"[TRAIN] PCA选择因子数: k={k} "
                    f"(累积方差={cumsum_variance[k-1]:.1%})"
                )

        # 方法3: Elbow方法
        elif method == 'elbow':
            threshold = self.config.elbow_threshold or 0.1
            marginal_variance = np.diff(explained_variance)
            k = np.argmax(marginal_variance < threshold) + 1
            logger.info(f"Elbow方法: 阈值={threshold:.1%}, k={k}")
            if progress_callback:
                progress_callback(f"[TRAIN] Elbow选择因子数: k={k}")

        else:
            raise ValueError(f"未知的因子选择方法: {method}")

        # 确保k在合理范围内
        k = max(1, min(k, len(selected_vars) - 1))
        logger.info(f"因子数选择完成: k={k}")

        return k, pca_analysis

    def _train_final_model(
        self,
        data: pd.DataFrame,
        target_data: pd.Series,
        selected_vars: List[str],
        k_factors: int,
        progress_callback: Optional[Callable] = None
    ) -> DFMModelResult:
        """
        最终模型训练

        Returns:
            DFMModelResult对象
        """
        logger.info("=" * 60)
        logger.info("最终模型训练")
        logger.info("=" * 60)

        if progress_callback:
            progress_callback(
                f"[TRAIN] 开始最终训练: k={k_factors}, "
                f"{len(selected_vars)}个变量"
            )

        logger.info(f"训练参数: k={k_factors}, max_iter={self.config.max_iterations}")

        # 准备数据
        predictor_data = data[selected_vars]

        # 使用DFMModel进行EM估计
        from dashboard.DFM.train_ref.core.factor_model import DFMModel

        dfm = DFMModel(
            n_factors=k_factors,
            max_lags=1,  # 使用1阶滞后（匹配train_model默认配置）
            max_iter=self.config.max_iterations,
            tolerance=self.config.tolerance
        )

        logger.info("开始DFM模型EM参数估计...")

        # 拟合模型
        dfm_results = dfm.fit(
            data=predictor_data,
            train_end=self.config.train_end
        )

        # 转换为DFMModelResult格式（先不填充预测值）
        model_result = DFMModelResult(
            A=dfm_results.transition_matrix,
            Q=dfm_results.process_noise_cov,
            H=dfm_results.loadings,  # 因子载荷即为观测矩阵
            R=dfm_results.measurement_noise_cov,
            factors=dfm_results.factors.values.T,  # (n_factors, n_time)
            factors_smooth=dfm_results.factors.values.T,  # 平滑因子
            converged=dfm_results.converged,
            iterations=dfm_results.n_iter,
            log_likelihood=dfm_results.loglikelihood
        )

        logger.info(
            f"模型训练完成: 收敛={model_result.converged}, "
            f"迭代={model_result.iterations}次, "
            f"LogLik={model_result.log_likelihood:.2f}"
        )

        # 生成目标变量的预测
        model_result = self._generate_target_forecast(
            model_result=model_result,
            target_data=target_data,
            progress_callback=progress_callback
        )

        return model_result

    def _generate_target_forecast(
        self,
        model_result: DFMModelResult,
        target_data: pd.Series,
        progress_callback: Optional[Callable] = None
    ) -> DFMModelResult:
        """
        生成目标变量的预测值

        通过将目标变量回归到因子上，得到目标变量的因子载荷，
        然后使用因子和载荷进行预测。

        Args:
            model_result: DFM模型结果
            target_data: 目标变量数据
            progress_callback: 进度回调

        Returns:
            更新了forecast_is和forecast_oos的DFMModelResult
        """
        try:
            if progress_callback:
                progress_callback("[TRAIN] 生成目标变量预测...")

            logger.info("开始生成目标变量预测...")

            # 提取因子（时间 x 因子数）
            factors = model_result.factors.T  # (n_time, n_factors)

            # 对齐目标变量和因子的时间索引
            # 确保使用相同的时间范围
            common_index = target_data.index[:factors.shape[0]]
            y = target_data.loc[common_index].values
            X = factors[:len(common_index), :]

            # 移除NaN值（同时移除y和X中对应的行）
            # 处理因子矩阵中可能存在的NaN
            y_isnan = np.isnan(y)
            X_has_nan = np.any(np.isnan(X), axis=1)
            valid_mask = ~(y_isnan | X_has_nan)

            if not valid_mask.any():
                logger.error("没有找到有效的数据点用于回归")
                raise ValueError("目标变量和因子都没有有效的对齐数据点")

            y_clean = y[valid_mask]
            X_clean = X[valid_mask, :]

            logger.info(
                f"回归数据准备: 总样本{len(y)}, "
                f"有效样本{len(y_clean)} ({100*len(y_clean)/len(y):.1f}%)"
            )

            # 回归：y = X @ beta + epsilon
            # 使用最小二乘法估计beta（目标变量的因子载荷）
            from numpy.linalg import lstsq
            beta, residuals, rank, s = lstsq(X_clean, y_clean, rcond=None)

            logger.info(
                f"目标变量因子载荷: {beta}, "
                f"回归残差: {np.sqrt(residuals[0]/len(y_clean)) if len(residuals) > 0 else 'N/A'}"
            )

            # 生成完整预测（包含NaN位置）
            forecast_full = X @ beta

            # 分割样本内和样本外（使用更健壮的日期查找）
            # 使用asof方法找到最接近的日期
            train_end_date = pd.to_datetime(self.config.train_end)
            train_data_filtered = target_data[:train_end_date]
            train_end_idx = len(train_data_filtered) - 1

            # 样本内预测
            forecast_is = forecast_full[:train_end_idx + 1]

            # 样本外预测
            if self.config.validation_start and self.config.validation_end:
                val_start_date = pd.to_datetime(self.config.validation_start)
                val_end_date = pd.to_datetime(self.config.validation_end)

                val_data_filtered = target_data[val_start_date:val_end_date]
                if len(val_data_filtered) > 0:
                    val_start_idx = target_data.index.get_loc(val_data_filtered.index[0])
                    val_end_idx = target_data.index.get_loc(val_data_filtered.index[-1])

                    if val_start_idx < len(forecast_full) and val_end_idx < len(forecast_full):
                        forecast_oos = forecast_full[val_start_idx:val_end_idx + 1]
                    else:
                        forecast_oos = None
                else:
                    forecast_oos = None
            else:
                forecast_oos = forecast_full[train_end_idx + 1:] if train_end_idx + 1 < len(forecast_full) else None

            # 更新model_result
            model_result.forecast_is = forecast_is
            model_result.forecast_oos = forecast_oos if forecast_oos is not None and len(forecast_oos) > 0 else None

            logger.info(
                f"预测生成完成: IS={len(forecast_is) if forecast_is is not None else 0}, "
                f"OOS={len(forecast_oos) if forecast_oos is not None else 0}"
            )

        except Exception as e:
            logger.error(f"生成目标变量预测时出错: {e}")
            import traceback
            traceback.print_exc()
            # 保持forecast为None（暂时保留异常处理以便调试）

        return model_result

    def _evaluate_model(
        self,
        model_result: DFMModelResult,
        target_data: pd.Series,
        progress_callback: Optional[Callable] = None
    ) -> EvaluationMetrics:
        """
        评估模型性能

        Returns:
            EvaluationMetrics对象
        """
        if progress_callback:
            progress_callback("[TRAIN] 评估模型性能...")

        logger.info("评估模型性能...")

        metrics = self.evaluator.evaluate(
            model_result=model_result,
            target_data=target_data,
            train_end_date=self.config.train_end,
            validation_start=self.config.validation_start,
            validation_end=self.config.validation_end
        )

        logger.info(
            f"评估完成: IS_RMSE={metrics.is_rmse:.4f}, "
            f"OOS_RMSE={metrics.oos_rmse:.4f}"
        )

        return metrics

    def _build_training_result(
        self,
        selected_vars: List[str],
        selection_history: List[Dict],
        k_factors: int,
        pca_analysis: Optional[Dict],
        model_result: DFMModelResult,
        metrics: EvaluationMetrics,
        start_time: float
    ) -> TrainingResult:
        """构建训练结果对象"""
        training_time = time.time() - start_time

        result = TrainingResult(
            selected_variables=[self.config.target_variable] + selected_vars,
            selection_history=selection_history,
            k_factors=k_factors,
            factor_selection_method=self.config.factor_selection_method,
            pca_analysis=pca_analysis,
            model_result=model_result,
            metrics=metrics,
            total_evaluations=self.total_evaluations,
            svd_error_count=self.svd_error_count,
            training_time=training_time,
            output_dir=self.config.output_dir
        )

        return result

    def _print_training_summary(
        self,
        result: TrainingResult,
        progress_callback: Optional[Callable] = None
    ):
        """打印训练摘要"""
        summary = f"""
========== 训练摘要 ==========
变量数: {len(result.selected_variables) - 1}
因子数: {result.k_factors}
迭代次数: {result.model_result.iterations}
收敛: {result.model_result.converged}

样本内RMSE: {result.metrics.is_rmse:.4f}
样本外RMSE: {result.metrics.oos_rmse:.4f}
样本内命中率: {result.metrics.is_hit_rate:.2f}%
样本外命中率: {result.metrics.oos_hit_rate:.2f}%

总评估次数: {result.total_evaluations}
SVD错误: {result.svd_error_count}
训练时间: {result.training_time:.2f}秒
=============================
"""
        logger.info(summary)

        if progress_callback:
            progress_callback(f"[TRAIN] {summary}")

    def _evaluate_dfm_for_selection(self, variables: List[str], **kwargs) -> Tuple:
        """
        为变量选择提供的评估函数

        Args:
            variables: 变量列表（包含目标变量）
            **kwargs: 其他参数（full_data, params等）

        Returns:
            9元组: (is_rmse, oos_rmse, _, _, is_hit_rate, oos_hit_rate, is_svd_error, _, _)
        """
        try:
            # 提取参数
            full_data = kwargs.get('full_data')
            k_factors = kwargs.get('params', {}).get('k_factors', 2)
            max_iter = kwargs.get('max_iter', self.config.max_iterations)

            # 提取目标变量和预测变量
            target_var = self.config.target_variable
            predictor_vars = [v for v in variables if v != target_var]

            if len(predictor_vars) == 0:
                logger.warning("预测变量为空，返回无穷大RMSE")
                return (np.inf, np.inf, None, None, -np.inf, -np.inf, False, None, None)

            # 准备数据
            predictor_data = full_data[predictor_vars]
            target_data = full_data[target_var]

            # 使用DFMModel训练
            from dashboard.DFM.train_ref.core.factor_model import DFMModel

            dfm = DFMModel(
                n_factors=k_factors,
                max_lags=1,
                max_iter=max_iter,
                tolerance=self.config.tolerance
            )

            dfm_results = dfm.fit(
                data=predictor_data,
                train_end=self.config.train_end
            )

            # 转换为DFMModelResult
            model_result = DFMModelResult(
                A=dfm_results.transition_matrix,
                Q=dfm_results.process_noise_cov,
                H=dfm_results.loadings,
                R=dfm_results.measurement_noise_cov,
                factors=dfm_results.factors.values.T,
                factors_smooth=dfm_results.factors.values.T,
                converged=dfm_results.converged,
                iterations=dfm_results.n_iter,
                log_likelihood=dfm_results.loglikelihood
            )

            # 生成目标变量预测
            model_result = self._generate_target_forecast(
                model_result=model_result,
                target_data=target_data,
                progress_callback=None
            )

            # 评估模型
            metrics = self.evaluator.evaluate(
                model_result=model_result,
                target_data=target_data,
                train_end_date=self.config.train_end,
                validation_start=self.config.validation_start,
                validation_end=self.config.validation_end
            )

            # 返回9元组
            return metrics.to_tuple()

        except Exception as e:
            logger.error(f"变量选择评估失败: {e}")
            return (np.inf, np.inf, None, None, -np.inf, -np.inf, True, None, None)

    def train(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> TrainingResult:
        """
        完整两阶段训练流程

        Args:
            progress_callback: 进度回调函数,签名为 (message: str) -> None

        Returns:
            TrainingResult对象
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("开始DFM模型训练")
        logger.info("=" * 60)

        if progress_callback:
            progress_callback("[TRAIN] 开始DFM模型训练")

        try:
            # 步骤1: 加载和验证数据
            data, target_data, predictor_vars = self._load_and_validate_data(progress_callback)

            # 步骤2: 阶段1变量选择
            selected_vars, selection_history = self._run_variable_selection(
                data, target_data, predictor_vars, progress_callback
            )

            # 步骤3: 阶段2因子数选择
            k_factors, pca_analysis = self._select_num_factors(
                data, selected_vars, progress_callback
            )

            # 步骤4: 最终模型训练
            model_result = self._train_final_model(
                data, target_data, selected_vars, k_factors, progress_callback
            )

            # 步骤5: 模型评估
            metrics = self._evaluate_model(
                model_result, target_data, progress_callback
            )

            # 步骤6: 构建结果
            result = self._build_training_result(
                selected_vars, selection_history, k_factors, pca_analysis,
                model_result, metrics, start_time
            )

            # 步骤7: 打印摘要
            self._print_training_summary(result, progress_callback)

            logger.info("训练完成!")
            if progress_callback:
                progress_callback("[TRAIN] 训练完成!")

            return result

        except Exception as e:
            logger.error(f"训练过程出错: {e}")
            import traceback
            traceback.print_exc()

            if progress_callback:
                progress_callback(f"[ERROR] 训练失败: {e}")

            raise


# ==================== 导出 ====================

__all__ = [
    'ModelEvaluator',
    'DFMTrainer',
    'EvaluationMetrics',
    'DFMModelResult',
    'TrainingResult',
]
