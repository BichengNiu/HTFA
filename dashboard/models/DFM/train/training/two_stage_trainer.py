# -*- coding: utf-8 -*-
"""
二次估计法训练器

实现两阶段DFM估计：
1. 第一阶段：对各行业分别估计DFM模型，得到分行业nowcasting值
2. 第二阶段：以分行业nowcasting值为预测变量，估计总量模型
"""

import time
import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict, List
from pathlib import Path

from dashboard.models.DFM.train.utils.logger import get_logger
from dashboard.models.DFM.train.utils.environment import setup_training_environment
from dashboard.models.DFM.train.utils.industry_data_processor import IndustryDataProcessor
from dashboard.models.DFM.train.core.models import TrainingResult, TwoStageTrainingResult
from dashboard.models.DFM.train.training.config import TrainingConfig
from dashboard.models.DFM.train.training.trainer import DFMTrainer

logger = get_logger(__name__)


class TwoStageTrainer:
    """
    二次估计法训练器

    训练流程：
    1. 第一阶段：循环训练各行业DFM模型
    2. 提取各行业nowcasting序列
    3. 构建第二阶段输入数据
    4. 第二阶段：训练总量DFM模型
    """

    def __init__(self, config: TrainingConfig):
        """
        初始化二次估计法训练器

        Args:
            config: 训练配置对象（必须设置estimation_method='two_stage'）
        """
        if config.estimation_method != 'two_stage':
            raise ValueError(f"配置的estimation_method必须为'two_stage'，当前值: {config.estimation_method}")

        if not config.industry_k_factors:
            raise ValueError("二次估计法需要设置各行业因子数（industry_k_factors不能为空）")

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
    ) -> TwoStageTrainingResult:
        """
        执行二次估计法完整训练流程

        Args:
            progress_callback: 进度回调函数
            enable_export: 是否导出结果文件
            export_dir: 导出目录（None=使用配置中的output_dir）

        Returns:
            TwoStageTrainingResult对象
        """
        total_start_time = time.time()

        try:
            # 加载数据并初始化处理器
            logger.info("加载训练数据...")
            if progress_callback:
                progress_callback("正在加载训练数据...")

            data = pd.read_csv(self.config.data_path, index_col=0, parse_dates=True)
            var_industry_map = self.config.industry_map or {}

            processor = IndustryDataProcessor(data, var_industry_map)
            industry_list = processor.get_industry_list()

            if not industry_list:
                raise ValueError("未能从数据中识别到任何行业信息")

            logger.info(f"识别到 {len(industry_list)} 个行业: {', '.join(industry_list[:5])}...")
            if progress_callback:
                progress_callback(f"识别到 {len(industry_list)} 个行业")

            # 第一阶段：训练分行业模型
            first_stage_start = time.time()

            if progress_callback:
                progress_callback(f"第一阶段：开始训练 {len(industry_list)} 个行业模型...")

            first_stage_results = self._train_industry_models(
                processor,
                industry_list,
                progress_callback
            )

            first_stage_time = time.time() - first_stage_start

            logger.info(f"第一阶段完成，成功训练 {len(first_stage_results)} 个行业模型，耗时 {first_stage_time:.1f} 秒")
            if progress_callback:
                progress_callback(f"第一阶段完成：{len(first_stage_results)}/{len(industry_list)} 个行业模型训练成功")

            # 检查第一阶段训练结果
            if len(first_stage_results) < 3:
                raise ValueError(f"第一阶段成功训练的行业模型不足3个（实际{len(first_stage_results)}个），无法进行第二阶段训练")

            # 提取分行业nowcasting序列
            industry_nowcast_df = self._extract_industry_nowcasts(first_stage_results)

            logger.info(f"提取分行业nowcasting序列：{industry_nowcast_df.shape[0]}个时间点，{industry_nowcast_df.shape[1]}个行业")

            # 第二阶段：构建数据并训练总量模型
            second_stage_start = time.time()

            if progress_callback:
                progress_callback("第二阶段：开始训练总量模型...")

            second_stage_data = self._build_second_stage_data(
                data,
                industry_nowcast_df
            )

            second_stage_result = self._train_second_stage_model(
                second_stage_data,
                progress_callback
            )

            second_stage_time = time.time() - second_stage_start

            logger.info(f"第二阶段完成，耗时 {second_stage_time:.1f} 秒")
            if progress_callback:
                progress_callback("第二阶段训练完成")

            # 构建二次估计法结果对象
            result = TwoStageTrainingResult.build(
                first_stage_results=first_stage_results,
                second_stage_result=second_stage_result,
                industry_nowcast_df=industry_nowcast_df,
                industry_k_factors_used=self.config.industry_k_factors,
                first_stage_time=first_stage_time,
                second_stage_time=second_stage_time,
                output_dir=export_dir or self.config.output_dir
            )

            total_time = time.time() - total_start_time
            logger.info(f"二次估计法训练完成，总耗时 {total_time:.1f} 秒")

            # 导出结果
            if enable_export:
                from dashboard.models.DFM.train.export.exporter import TrainingResultExporter
                exporter = TrainingResultExporter()

                export_output_dir = export_dir or self.config.output_dir
                export_files = exporter.export_two_stage_results(
                    result,
                    self.config,
                    export_output_dir
                )

                result.export_files = export_files
                logger.info(f"结果已导出到: {export_output_dir}")

            return result

        except Exception as e:
            logger.error(f"二次估计法训练失败: {str(e)}", exc_info=True)
            raise

    def _train_industry_models(
        self,
        processor: IndustryDataProcessor,
        industry_list: List[str],
        progress_callback: Optional[Callable[[str], None]]
    ) -> Dict[str, TrainingResult]:
        """
        第一阶段：训练各行业DFM模型

        Args:
            processor: 行业数据处理器
            industry_list: 行业列表
            progress_callback: 进度回调

        Returns:
            行业名→训练结果的字典
        """
        results = {}
        failed_industries = []

        for idx, industry in enumerate(industry_list, 1):
            try:
                if progress_callback:
                    progress_callback(f"第一阶段 [{idx}/{len(industry_list)}]: 训练 {industry} ...")

                logger.info(f"开始训练行业模型: {industry}")

                # 验证行业数据
                is_valid, error_msg = processor.validate_industry_data(industry, min_predictors=1)
                if not is_valid:
                    logger.warning(f"跳过行业 {industry}: {error_msg}")
                    failed_industries.append(industry)
                    continue

                # 获取行业因子数
                k_factors = self.config.industry_k_factors.get(industry)
                if not k_factors:
                    logger.warning(f"跳过行业 {industry}: 未设置因子数")
                    failed_industries.append(industry)
                    continue

                # 构建行业训练数据
                industry_data = processor.build_industry_training_data(industry)
                target_col = processor.get_industry_target_column(industry)
                predictor_cols = processor.get_industry_predictors(industry)

                # 创建临时CSV文件保存行业数据
                temp_data_path = Path(self.config.output_dir) / f"temp_{industry}_data.csv"
                temp_data_path.parent.mkdir(parents=True, exist_ok=True)
                industry_data.to_csv(temp_data_path, encoding='utf-8-sig')

                # 创建行业训练配置
                industry_config = TrainingConfig(
                    data_path=str(temp_data_path),
                    target_variable=target_col,
                    selected_indicators=predictor_cols,
                    target_freq=self.config.target_freq,
                    k_factors=k_factors,
                    max_iterations=self.config.max_iterations,
                    max_lags=self.config.max_lags,
                    tolerance=self.config.tolerance,
                    training_start=self.config.training_start,
                    train_end=self.config.train_end,
                    validation_start=self.config.validation_start,
                    validation_end=self.config.validation_end,
                    factor_selection_method='fixed',  # 行业模型使用固定因子数
                    enable_variable_selection=False,  # 行业模型不进行变量选择
                    enable_parallel=False,  # 行业模型禁用并行（外层已并行）
                    output_dir=str(Path(self.config.output_dir) / "first_stage_temp"),
                    industry_map={}
                )

                # 训练行业模型
                trainer = DFMTrainer(industry_config)
                industry_result = trainer.train(
                    progress_callback=None,  # 避免嵌套进度条
                    enable_export=False  # 第一阶段不导出
                )

                results[industry] = industry_result
                logger.info(f"行业 {industry} 训练完成，RMSE(oos)={industry_result.metrics.oos_rmse:.4f}")

                # 删除临时数据文件
                if temp_data_path.exists():
                    temp_data_path.unlink()

            except Exception as e:
                logger.error(f"行业 {industry} 训练失败: {str(e)}")
                failed_industries.append(industry)
                continue

        if failed_industries:
            logger.warning(f"以下 {len(failed_industries)} 个行业训练失败: {', '.join(failed_industries)}")

        return results

    def _extract_industry_nowcasts(
        self,
        first_stage_results: Dict[str, TrainingResult]
    ) -> pd.DataFrame:
        """
        从第一阶段结果中提取分行业nowcasting序列

        Args:
            first_stage_results: 第一阶段训练结果

        Returns:
            分行业nowcasting的DataFrame，索引为日期，列为行业名
        """
        nowcast_dict = {}

        for industry, result in first_stage_results.items():
            if result.model_result and result.model_result.forecast_oos is not None:
                # 提取样本外预测序列
                nowcast_series = result.model_result.forecast_oos
                nowcast_dict[f"nowcast_{industry}"] = nowcast_series
            else:
                logger.warning(f"行业 {industry} 没有有效的nowcasting结果")

        if not nowcast_dict:
            raise ValueError("没有任何行业生成有效的nowcasting序列")

        # 合并为DataFrame
        nowcast_df = pd.DataFrame(nowcast_dict)

        return nowcast_df

    def _build_second_stage_data(
        self,
        original_data: pd.DataFrame,
        industry_nowcast_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        构建第二阶段训练数据

        Args:
            original_data: 原始完整数据
            industry_nowcast_df: 分行业nowcasting序列

        Returns:
            第二阶段训练数据（目标变量 + nowcasting + 额外指标）
        """
        # 确保目标变量存在
        if self.config.target_variable not in original_data.columns:
            raise ValueError(f"目标变量 {self.config.target_variable} 不存在于原始数据中")

        # 基础列：目标变量
        columns_to_include = [self.config.target_variable]

        # 添加分行业nowcasting列（与原始数据时间索引对齐）
        second_stage_data = original_data[[self.config.target_variable]].copy()

        for col in industry_nowcast_df.columns:
            # 通过索引对齐合并
            second_stage_data = second_stage_data.join(industry_nowcast_df[[col]], how='left')

        # 添加第二阶段额外预测变量
        if self.config.second_stage_extra_predictors:
            for var in self.config.second_stage_extra_predictors:
                if var in original_data.columns and var not in second_stage_data.columns:
                    second_stage_data[var] = original_data[var]
                elif var not in original_data.columns:
                    logger.warning(f"额外预测变量 {var} 不存在于原始数据中，已跳过")

        # 删除全为NaN的列
        second_stage_data = second_stage_data.dropna(axis=1, how='all')

        logger.info(f"第二阶段数据构建完成：{second_stage_data.shape[0]}行，{second_stage_data.shape[1]}列")

        return second_stage_data

    def _train_second_stage_model(
        self,
        second_stage_data: pd.DataFrame,
        progress_callback: Optional[Callable[[str], None]]
    ) -> TrainingResult:
        """
        第二阶段：训练总量DFM模型

        Args:
            second_stage_data: 第二阶段训练数据
            progress_callback: 进度回调

        Returns:
            TrainingResult对象
        """
        # 创建临时CSV文件保存第二阶段数据
        temp_data_path = Path(self.config.output_dir) / "temp_second_stage_data.csv"
        temp_data_path.parent.mkdir(parents=True, exist_ok=True)
        second_stage_data.to_csv(temp_data_path, encoding='utf-8-sig')

        # 第二阶段预测变量：除目标变量外的所有列
        second_stage_predictors = [
            col for col in second_stage_data.columns
            if col != self.config.target_variable
        ]

        # 创建第二阶段训练配置（使用与一次估计法相同的参数）
        second_stage_config = TrainingConfig(
            data_path=str(temp_data_path),
            target_variable=self.config.target_variable,
            selected_indicators=second_stage_predictors,
            target_freq=self.config.target_freq,
            k_factors=self.config.k_factors,
            max_iterations=self.config.max_iterations,
            max_lags=self.config.max_lags,
            tolerance=self.config.tolerance,
            training_start=self.config.training_start,
            train_end=self.config.train_end,
            validation_start=self.config.validation_start,
            validation_end=self.config.validation_end,
            factor_selection_method=self.config.factor_selection_method,
            pca_threshold=self.config.pca_threshold,
            enable_variable_selection=self.config.enable_variable_selection,
            variable_selection_method=self.config.variable_selection_method,
            min_variables_after_selection=self.config.min_variables_after_selection,
            enable_parallel=self.config.enable_parallel,
            n_jobs=self.config.n_jobs,
            parallel_backend=self.config.parallel_backend,
            min_variables_for_parallel=self.config.min_variables_for_parallel,
            output_dir=self.config.output_dir,
            industry_map={}  # 第二阶段不需要行业映射
        )

        # 训练第二阶段模型
        trainer = DFMTrainer(second_stage_config)
        second_stage_result = trainer.train(
            progress_callback=progress_callback,
            enable_export=False  # 先不导出，由TwoStageTrainer统一导出
        )

        # 删除临时数据文件
        if temp_data_path.exists():
            temp_data_path.unlink()

        return second_stage_result
