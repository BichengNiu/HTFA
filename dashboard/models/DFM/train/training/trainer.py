# -*- coding: utf-8 -*-
"""
DFM训练器 - 简化版（真正的轻量级协调器）

仅负责协调训练流程，直接调用底层函数，避免不必要的包装
"""

import time
import numpy as np
import pandas as pd
from typing import Optional, Callable
from dashboard.models.DFM.train.utils.logger import get_logger

# 导入数据模型
from dashboard.models.DFM.train.core.models import TrainingResult

# 导入统一训练和评估函数
from dashboard.models.DFM.train.training.model_ops import train_dfm_with_forecast, evaluate_model_performance

# 导入流程步骤
from dashboard.models.DFM.train.utils.data_utils import load_and_validate_data
from dashboard.models.DFM.train.training.evaluator_strategy import create_variable_selection_evaluator
from dashboard.models.DFM.train.export.exporter import TrainingResultExporter

# 导入核心功能
from dashboard.models.DFM.train.selection.backward_selector import BackwardSelector
from dashboard.models.DFM.train.core.pca_utils import select_num_factors

# 导入格式化工具
from dashboard.models.DFM.train.utils.formatting import print_training_summary, format_training_config

# 导入环境配置
from dashboard.models.DFM.train.utils.environment import setup_training_environment

logger = get_logger(__name__)


class DFMTrainer:
    """
    DFM主训练器（轻量级协调器）

    两阶段训练流程:
    1. 阶段1: 变量选择(可选)
    2. 阶段2: 因子数选择
    3. 最终训练: 使用选定变量和因子数训练模型
    """

    def __init__(self, config: 'TrainingConfig'):
        """
        初始化训练器

        Args:
            config: 训练配置对象(TrainingConfig)
        """
        self.config = config

        # 环境初始化
        setup_training_environment(
            seed=42,
            silent_mode=False,
            enable_debug_logging=True
        )

        # 训练统计
        self.total_evaluations = 0
        self.svd_error_count = 0

    def train(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
        enable_export: bool = True,
        export_dir: Optional[str] = None
    ) -> TrainingResult:
        """
        完整两阶段训练流程

        Args:
            progress_callback: 进度回调函数,签名为 (message: str) -> None
            enable_export: 是否导出结果文件(模型、元数据)
            export_dir: 导出目录(None=使用临时目录)

        Returns:
            TrainingResult对象
        """
        start_time = time.time()

        try:
            # 步骤1: 加载和验证数据
            data, target_data, predictor_vars = load_and_validate_data(
                data_path=self.config.data_path,
                target_variable=self.config.target_variable,
                selected_indicators=self.config.selected_indicators,
                progress_callback=progress_callback
            )

            # 步骤1.5: 应用去趋势（如果启用）
            detrend_handler = None
            if self.config.enable_detrend:
                from dashboard.models.DFM.train.preprocessing.detrend_handler import DetrendHandler

                detrend_handler = DetrendHandler(method=self.config.detrend_method)

                # 确定需要去趋势的变量（包括目标变量）
                detrend_vars = self.config.detrend_variables
                if detrend_vars is None:
                    detrend_vars = [self.config.target_variable] + predictor_vars

                # 对数据进行去趋势（基于完整数据拟合趋势）
                data = detrend_handler.fit_and_transform(data, detrend_vars)
                target_data = data[self.config.target_variable]  # 更新为残差

                if progress_callback:
                    progress_callback(f"已对 {len(detrend_vars)} 个变量进行线性去趋势")

                logger.info(f"去趋势完成: {len(detrend_handler.trend_params)} 个变量")

            # 输出训练配置摘要
            # 根据training_start切分训练数据
            train_data = data.loc[self.config.training_start:self.config.train_end]
            val_data = data.loc[self.config.validation_start:self.config.validation_end]

            config_summary = format_training_config(
                train_start=self.config.training_start,
                train_end=self.config.train_end,
                validation_start=self.config.validation_start,
                validation_end=self.config.validation_end,
                train_samples=len(train_data),
                validation_samples=len(val_data),
                initial_vars=len(predictor_vars),
                k_factors=self.config.k_factors
            )

            logger.info(config_summary)
            if progress_callback:
                progress_callback(config_summary.strip())

            # 步骤2: 阶段1变量选择
            if self.config.enable_variable_selection:
                # 创建变量筛选专用评估器（使用下月配对RMSE）
                evaluator = create_variable_selection_evaluator(self.config)

                # 创建变量选择器（传递并行配置）
                selector = BackwardSelector(
                    evaluator_func=evaluator,
                    criterion='rmse',
                    min_variables=self.config.min_variables_after_selection or 1,
                    parallel_config=self.config.get_parallel_config()
                )

                # 根据因子选择策略确定k_factors用于变量选择
                if self.config.factor_selection_method == 'fixed':
                    # 如果使用固定因子数策略，直接使用用户设置的k_factors
                    k_for_selection = self.config.k_factors
                else:  # cumulative
                    # 如果使用累积方差贡献策略，计算合理的k_factors用于变量选择
                    # 最终的k_factors会在阶段2通过PCA确定
                    k_for_selection = max(2, min(len(predictor_vars) // 2, len(predictor_vars) - 2))

                # 执行变量选择
                initial_vars = [self.config.target_variable] + predictor_vars
                selection_result = selector.select(
                    initial_variables=initial_vars,
                    target_variable=self.config.target_variable,
                    full_data=data,
                    params={'k_factors': k_for_selection},
                    validation_start=self.config.validation_start,
                    validation_end=self.config.validation_end,
                    target_freq=self.config.target_freq,
                    training_start_date=self.config.training_start,
                    train_end_date=self.config.train_end,
                    target_mean_original=target_data.mean(),
                    target_std_original=target_data.std(),
                    max_iter=self.config.max_iterations,
                    max_lags=1,
                    progress_callback=progress_callback
                )

                # 提取选定的预测变量
                selected_vars = [
                    v for v in selection_result.selected_variables
                    if v != self.config.target_variable
                ]
                selection_history = selection_result.selection_history

                # 更新统计
                self.total_evaluations += selection_result.total_evaluations
                self.svd_error_count += selection_result.svd_error_count
            else:
                selected_vars = predictor_vars
                selection_history = []

            # 步骤3: 阶段2因子数选择
            print(f"[FACTOR_SELECTION] 输入参数: method={self.config.factor_selection_method}, fixed_k={self.config.k_factors}, pca_threshold={self.config.pca_threshold or 0.9}, kaiser_threshold={self.config.kaiser_threshold or 1.0}")
            k_factors, pca_analysis = select_num_factors(
                data=data,
                selected_vars=selected_vars,
                method=self.config.factor_selection_method,
                fixed_k=self.config.k_factors,
                pca_threshold=self.config.pca_threshold or 0.9,
                kaiser_threshold=self.config.kaiser_threshold or 1.0,
                train_end=self.config.train_end,
                progress_callback=progress_callback
            )
            print(f"[FACTOR_SELECTION] 输出结果: k_factors={k_factors}")

            # 步骤4: 最终模型训练（直接调用）
            # 准备数据并训练
            predictor_data = data[selected_vars]

            model_result = train_dfm_with_forecast(
                predictor_data=predictor_data,
                target_data=target_data,
                k_factors=k_factors,
                training_start=self.config.training_start,
                train_end=self.config.train_end,
                validation_start=self.config.validation_start,
                validation_end=self.config.validation_end,
                max_iter=self.config.max_iterations,
                max_lags=1,
                tolerance=self.config.tolerance,
                progress_callback=progress_callback
            )

            # 步骤5: 模型评估（残差空间，用于内部模型选择）
            metrics = evaluate_model_performance(
                model_result=model_result,
                target_data=target_data,
                train_end=self.config.train_end,
                validation_start=self.config.validation_start,
                validation_end=self.config.validation_end
            )

            # 步骤5.5: 如果启用了去趋势，还原预测值并计算原始值指标
            if detrend_handler and detrend_handler.has_params(self.config.target_variable):
                logger.info("还原预测值到原始水平...")

                # 还原样本内预测
                if model_result.forecast_is is not None:
                    train_dates = data.loc[self.config.training_start:self.config.train_end].index
                    forecast_is_residual = pd.Series(model_result.forecast_is, index=train_dates)
                    forecast_is_original = detrend_handler.inverse_transform(
                        forecast_is_residual,
                        self.config.target_variable
                    )
                    model_result.forecast_is_original = forecast_is_original.values

                # 还原样本外预测
                if model_result.forecast_oos is not None:
                    val_dates = data.loc[self.config.validation_start:self.config.validation_end].index
                    forecast_oos_residual = pd.Series(model_result.forecast_oos, index=val_dates)
                    forecast_oos_original = detrend_handler.inverse_transform(
                        forecast_oos_residual,
                        self.config.target_variable
                    )
                    model_result.forecast_oos_original = forecast_oos_original.values

                # 还原观察期预测（2025-11-15新增）
                if model_result.forecast_observation is not None:
                    val_end_date = pd.to_datetime(self.config.validation_end)
                    data_end_date = data.index.max()

                    if data_end_date > val_end_date:
                        # 获取观察期日期范围（validation_end之后到数据结束）
                        observation_dates = data.loc[val_end_date + pd.Timedelta(days=1):data_end_date].index

                        if len(observation_dates) == len(model_result.forecast_observation):
                            forecast_observation_residual = pd.Series(
                                model_result.forecast_observation,
                                index=observation_dates
                            )
                            forecast_observation_original = detrend_handler.inverse_transform(
                                forecast_observation_residual,
                                self.config.target_variable
                            )
                            model_result.forecast_observation_original = forecast_observation_original.values

                            logger.info(f"观察期预测值已还原: {len(forecast_observation_original)} 个数据点，范围 {observation_dates.min()} 到 {observation_dates.max()}")
                        else:
                            logger.warning(f"观察期日期数量不匹配: observation_dates={len(observation_dates)}, forecast_observation={len(model_result.forecast_observation)}")
                    else:
                        logger.debug("数据结束日期未超出validation_end，无需还原观察期数据")

                # 计算原始值空间的评估指标
                from dashboard.models.DFM.train.evaluation.metrics import (
                    calculate_next_month_rmse,
                    calculate_next_month_mae,
                    calculate_next_month_hit_rate
                )

                # 还原目标变量真实值（从残差还原）
                # 注意：这里需要还原整个target_data序列
                target_original_series = detrend_handler.inverse_transform(
                    target_data,
                    self.config.target_variable
                )

                # 计算IS指标
                if model_result.forecast_is_original is not None:
                    train_dates = data.loc[self.config.training_start:self.config.train_end].index
                    forecast_is_series = pd.Series(model_result.forecast_is_original, index=train_dates)
                    target_is = target_original_series.loc[:self.config.train_end]

                    metrics.is_rmse_original = calculate_next_month_rmse(forecast_is_series, target_is)
                    metrics.is_mae_original = calculate_next_month_mae(forecast_is_series, target_is)
                    metrics.is_hit_rate_original = calculate_next_month_hit_rate(forecast_is_series, target_is)

                    logger.debug(f"IS原始值指标: RMSE={metrics.is_rmse_original:.4f}, MAE={metrics.is_mae_original:.4f}, Hit Rate={metrics.is_hit_rate_original:.2f}%")

                # 计算OOS指标
                if model_result.forecast_oos_original is not None:
                    val_dates = data.loc[self.config.validation_start:self.config.validation_end].index
                    forecast_oos_series = pd.Series(model_result.forecast_oos_original, index=val_dates)
                    target_oos = target_original_series.loc[self.config.validation_start:self.config.validation_end]

                    metrics.oos_rmse_original = calculate_next_month_rmse(forecast_oos_series, target_oos)
                    metrics.oos_mae_original = calculate_next_month_mae(forecast_oos_series, target_oos)
                    metrics.oos_hit_rate_original = calculate_next_month_hit_rate(forecast_oos_series, target_oos)

                    logger.debug(f"OOS原始值指标: RMSE={metrics.oos_rmse_original:.4f}, MAE={metrics.oos_mae_original:.4f}, Hit Rate={metrics.oos_hit_rate_original:.2f}%")

                # 计算观察期指标（2025-11-15新增）
                if model_result.forecast_observation_original is not None:
                    val_end_date = pd.to_datetime(self.config.validation_end)
                    data_end_date = data.index.max()

                    if data_end_date > val_end_date:
                        observation_dates = data.loc[val_end_date + pd.Timedelta(days=1):data_end_date].index
                        forecast_obs_series = pd.Series(model_result.forecast_observation_original, index=observation_dates)
                        target_obs = target_original_series.loc[val_end_date + pd.Timedelta(days=1):data_end_date]

                        if len(target_obs) > 0:
                            metrics.obs_rmse_original = calculate_next_month_rmse(forecast_obs_series, target_obs)
                            metrics.obs_mae_original = calculate_next_month_mae(forecast_obs_series, target_obs)
                            metrics.obs_hit_rate_original = calculate_next_month_hit_rate(forecast_obs_series, target_obs)

                            logger.info(f"观察期评估指标计算完成: RMSE={metrics.obs_rmse_original:.4f}, MAE={metrics.obs_mae_original:.4f}, 胜率={metrics.obs_hit_rate_original:.2f}%")
                        else:
                            logger.warning("观察期没有目标变量真实值，无法计算评估指标")

                if progress_callback:
                    progress_callback("预测值已还原到原始水平，并计算原始值空间评估指标")

            # 步骤6: 构建结果（直接调用）
            training_time = time.time() - start_time

            result = TrainingResult.build(
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

            # 步骤7: 打印摘要（直接调用）
            print_training_summary(result, progress_callback, logger)

            # 步骤8: 导出结果文件（直接调用）
            if enable_export:
                try:
                    exporter = TrainingResultExporter()
                    file_paths = exporter.export_all(
                        result,
                        self.config,
                        output_dir=export_dir,
                        prepared_data=data,  # 传递完整观测数据用于新闻分析
                        detrend_handler=detrend_handler  # 传递去趋势处理器
                    )

                    result.export_files = file_paths

                    success_count = len([p for p in file_paths.values() if p])

                    if progress_callback:
                        progress_callback(f"结果文件导出完成 (成功 {success_count}/2 个)")

                except Exception as e:
                    logger.warning(f"文件导出失败: {e}", exc_info=True)
                    if progress_callback:
                        progress_callback(f"文件导出失败: {e}")

                    result.export_files = None

            return result

        except Exception as e:
            logger.error(f"训练过程出错: {e}")
            import traceback
            traceback.print_exc()

            if progress_callback:
                progress_callback(f"训练失败: {e}")

            raise


__all__ = [
    'DFMTrainer',
]
