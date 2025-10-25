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
from dashboard.models.DFM.train.training.evaluator_strategy import create_dfm_evaluator
from dashboard.models.DFM.train.export.exporter import TrainingResultExporter

# 导入核心功能
from dashboard.models.DFM.train.selection.backward_selector import BackwardSelector
from dashboard.models.DFM.train.core.pca_utils import select_num_factors

# 导入格式化工具
from dashboard.models.DFM.train.utils.formatting import print_training_summary

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
            enable_export: 是否导出结果文件(模型、元数据、Excel报告)
            export_dir: 导出目录(None=使用临时目录)

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
            data, target_data, predictor_vars = load_and_validate_data(
                data_path=self.config.data_path,
                target_variable=self.config.target_variable,
                selected_indicators=self.config.selected_indicators,
                progress_callback=progress_callback
            )

            # 步骤2: 阶段1变量选择
            if self.config.enable_variable_selection:
                if progress_callback:
                    progress_callback("[SELECTION] 阶段1: 开始变量选择")

                logger.info("=" * 60)
                logger.info("阶段1: 变量选择")
                logger.info("=" * 60)

                # 创建评估器（函数式接口）
                evaluator = create_dfm_evaluator(self.config)

                # 创建变量选择器
                selector = BackwardSelector(
                    evaluator_func=evaluator,
                    criterion='rmse',
                    min_variables=self.config.min_variables_after_selection or 1
                )

                # 计算合理的k_factors用于变量选择
                k_for_selection = max(2, min(len(predictor_vars) // 2, len(predictor_vars) - 2))
                logger.info(f"变量选择使用k_factors={k_for_selection} (变量数: {len(predictor_vars)})")

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

                logger.info(
                    f"变量选择完成: {len(predictor_vars)} -> {len(selected_vars)}个变量"
                )
                if progress_callback:
                    progress_callback(
                        f"[SELECTION] 变量选择完成: 保留{len(selected_vars)}个变量"
                    )
            else:
                logger.info("跳过变量选择,使用全部变量")
                if progress_callback:
                    progress_callback("[SELECTION] 跳过变量选择")
                selected_vars = predictor_vars
                selection_history = []

            # 步骤3: 阶段2因子数选择
            k_factors, pca_analysis = select_num_factors(
                data=data,
                selected_vars=selected_vars,
                method=self.config.factor_selection_method,
                fixed_k=self.config.k_factors,
                pca_threshold=self.config.pca_threshold or 0.9,
                elbow_threshold=self.config.elbow_threshold or 0.1,
                progress_callback=progress_callback
            )

            # 步骤4: 最终模型训练（直接调用）
            logger.info("=" * 60)
            logger.info("最终模型训练")
            logger.info("=" * 60)

            if progress_callback:
                progress_callback(
                    f"[TRAIN] 开始最终训练: k={k_factors}, {len(selected_vars)}个变量"
                )

            logger.info(f"训练参数: k={k_factors}, max_iter={self.config.max_iterations}")

            # 准备数据并训练
            predictor_data = data[selected_vars]

            model_result = train_dfm_with_forecast(
                predictor_data=predictor_data,
                target_data=target_data,
                k_factors=k_factors,
                train_end=self.config.train_end,
                validation_start=self.config.validation_start,
                validation_end=self.config.validation_end,
                max_iter=self.config.max_iterations,
                max_lags=1,
                tolerance=self.config.tolerance,
                progress_callback=progress_callback
            )

            logger.info(
                f"模型训练完成: 收敛={model_result.converged}, "
                f"迭代={model_result.iterations}次, "
                f"LogLik={model_result.log_likelihood:.2f}"
            )

            # 步骤5: 模型评估（直接调用）
            logger.info("评估模型性能...")
            if progress_callback:
                progress_callback("[TRAIN] 评估模型性能...")

            metrics = evaluate_model_performance(
                model_result=model_result,
                target_data=target_data,
                train_end=self.config.train_end,
                validation_start=self.config.validation_start,
                validation_end=self.config.validation_end
            )

            logger.info(
                f"评估完成: IS_RMSE={metrics.is_rmse:.4f}, "
                f"OOS_RMSE={metrics.oos_rmse:.4f}"
            )

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
                logger.info("开始导出训练结果文件")
                if progress_callback:
                    progress_callback("[EXPORT] 开始导出训练结果文件")

                try:
                    exporter = TrainingResultExporter()
                    file_paths = exporter.export_all(
                        result,
                        self.config,
                        output_dir=export_dir
                    )

                    result.export_files = file_paths

                    success_count = len([p for p in file_paths.values() if p])
                    logger.info(f"文件导出完成,成功 {success_count}/3 个")

                    if progress_callback:
                        progress_callback(f"[EXPORT] 文件导出完成,成功 {success_count}/3 个")

                except Exception as e:
                    logger.warning(f"文件导出失败: {e}", exc_info=True)
                    if progress_callback:
                        progress_callback(f"[EXPORT] 文件导出失败: {e}")

                    result.export_files = None

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


__all__ = [
    'DFMTrainer',
]
