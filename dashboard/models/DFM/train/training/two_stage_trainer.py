# -*- coding: utf-8 -*-
"""
二次估计法训练器

实现两阶段DFM估计：
1. 第一阶段：对各行业分别估计DFM模型，得到分行业nowcasting值
2. 第二阶段：以分行业nowcasting值为预测变量，估计总量模型
"""

import time
import traceback
import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict, List, Tuple, Any
from pathlib import Path

from dashboard.models.DFM.train.utils.logger import get_logger
from dashboard.models.DFM.train.utils.environment import setup_training_environment
from dashboard.models.DFM.train.utils.industry_data_processor import IndustryDataProcessor
from dashboard.models.DFM.train.core.models import TrainingResult, TwoStageTrainingResult
from dashboard.models.DFM.train.training.config import TrainingConfig
from dashboard.models.DFM.train.training.trainer import DFMTrainer

logger = get_logger(__name__)


def _train_single_industry(
    industry: str,
    data: pd.DataFrame,
    industry_map: Dict[str, str],
    industry_k_factors: Dict[str, int],
    config_dict: Dict[str, Any],
    idx: int,
    total: int
) -> Tuple[str, Optional[TrainingResult]]:
    """
    训练单个行业模型（顶层可序列化函数）

    此函数设计为完全可序列化，所有参数均为基本类型或可序列化对象，
    用于joblib并行执行。

    Args:
        industry: 行业名称
        data: 完整数据DataFrame
        industry_map: 行业映射字典（变量名 -> 行业名）
        industry_k_factors: 各行业因子数映射
        config_dict: 训练配置字典（可序列化）
        idx: 当前行业索引（用于日志）
        total: 总行业数（用于日志）

    Returns:
        (industry, TrainingResult): 行业名和训练结果的元组
        如果训练失败，返回(industry, None)
    """
    import uuid
    import os
    from pathlib import Path

    try:
        # 1. 创建IndustryDataProcessor
        processor = IndustryDataProcessor(data, industry_map)

        # 2. 验证行业数据
        is_valid, error_msg = processor.validate_industry_data(industry, min_predictors=1)
        if not is_valid:
            logger.warning(f"跳过行业 {industry}: {error_msg}")
            return (industry, None)

        # 3. 获取行业因子数
        k_factors = industry_k_factors.get(industry)
        if not k_factors:
            logger.warning(f"跳过行业 {industry}: 未设置因子数")
            return (industry, None)

        # 4. 构建行业训练数据
        industry_data = processor.build_industry_training_data(industry)
        target_col = processor.get_industry_target_column(industry)
        predictor_cols = processor.get_industry_predictors(industry, exclude_target=True)

        if not predictor_cols:
            logger.warning(f"跳过行业 {industry}: 没有有效的预测变量")
            return (industry, None)

        # 5. 创建唯一临时文件（避免并发冲突）
        pid = os.getpid()
        unique_id = uuid.uuid4().hex[:8]
        output_dir = config_dict.get('output_dir', 'dfm_output')
        temp_data_path = Path(output_dir) / f"temp_{industry}_{pid}_{unique_id}.csv"
        temp_data_path.parent.mkdir(parents=True, exist_ok=True)
        industry_data.to_csv(temp_data_path, encoding='utf-8-sig')

        # 6. 创建行业训练配置
        industry_config = TrainingConfig(
            data_path=str(temp_data_path),
            target_variable=target_col,
            selected_indicators=predictor_cols,
            target_freq=config_dict['target_freq'],
            k_factors=k_factors,
            max_iterations=config_dict['max_iterations'],
            max_lags=config_dict['max_lags'],
            tolerance=config_dict['tolerance'],
            training_start=config_dict['training_start'],
            train_end=config_dict['train_end'],
            validation_start=config_dict['validation_start'],
            validation_end=config_dict['validation_end'],
            factor_selection_method='fixed',
            enable_variable_selection=False,
            enable_parallel=False,  # 禁用内层并行，避免嵌套并行
            output_dir=str(Path(output_dir) / "first_stage_temp"),
            industry_map={}
        )

        # 7. 训练行业模型
        trainer = DFMTrainer(industry_config)
        industry_result = trainer.train(
            progress_callback=None,  # 避免子进程输出混乱
            enable_export=False  # 第一阶段不导出
        )

        # 8. 清理临时文件
        if temp_data_path.exists():
            temp_data_path.unlink()

        logger.info(f"行业 {industry} 训练完成，RMSE(oos)={industry_result.metrics.oos_rmse:.4f}")
        return (industry, industry_result)

    except Exception as e:
        logger.error(f"行业 {industry} 训练失败: {str(e)}")
        import traceback
        logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
        return (industry, None)


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

        # 验证行业映射
        if config.industry_map is None or (isinstance(config.industry_map, pd.DataFrame) and config.industry_map.empty):
            raise ValueError("二次估计法需要提供行业映射（industry_map不能为空或None）")

        if not isinstance(config.industry_map, dict):
            raise ValueError(f"industry_map必须是字典类型，当前类型: {type(config.industry_map).__name__}")

        if len(config.industry_map) == 0:
            raise ValueError("industry_map不能为空字典，请确保已正确加载行业映射文件")

        logger.info(f"行业映射验证通过: {len(config.industry_map)} 个变量")

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
            industry_nowcast_df = self._extract_industry_nowcasts(first_stage_results, data)

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
        第一阶段：训练各行业DFM模型（支持并行/串行）

        Args:
            processor: 行业数据处理器
            industry_list: 行业列表
            progress_callback: 进度回调

        Returns:
            行业名→训练结果的字典
        """
        # 打印行业映射总体信息
        logger.info(f"var_industry_map总条目数: {len(self.config.industry_map)}")
        logger.debug(f"var_industry_map前5项: {dict(list(self.config.industry_map.items())[:5])}")

        # 判断是否使用并行
        use_parallel = (
            self.config.enable_first_stage_parallel
            and len(industry_list) >= self.config.min_industries_for_parallel
        )

        if use_parallel:
            logger.info(
                f"第一阶段：启用并行训练 {len(industry_list)} 个行业模型 "
                f"(n_jobs={self.config.first_stage_n_jobs}, "
                f"min_threshold={self.config.min_industries_for_parallel})"
            )
            return self._train_industry_models_parallel(
                processor, industry_list, progress_callback
            )
        else:
            logger.info(
                f"第一阶段：使用串行训练 {len(industry_list)} 个行业模型 "
                f"(并行已禁用或行业数 < {self.config.min_industries_for_parallel})"
            )
            return self._train_industry_models_serial(
                processor, industry_list, progress_callback
            )

    def _train_industry_models_parallel(
        self,
        processor: IndustryDataProcessor,
        industry_list: List[str],
        progress_callback: Optional[Callable[[str], None]]
    ) -> Dict[str, TrainingResult]:
        """
        第一阶段：并行训练各行业DFM模型

        Args:
            processor: 行业数据处理器
            industry_list: 行业列表
            progress_callback: 进度回调

        Returns:
            行业名→训练结果的字典
        """
        from joblib import Parallel, delayed

        # 主进程进度提示
        if progress_callback:
            n_jobs_display = self.config.first_stage_n_jobs if self.config.first_stage_n_jobs > 0 else 'max-1'
            progress_callback(
                f"第一阶段：并行训练 {len(industry_list)} 个行业模型 "
                f"(使用 {n_jobs_display} 个核心)..."
            )

        # 准备可序列化的配置字典
        config_dict = {
            'target_freq': self.config.target_freq,
            'max_iterations': self.config.max_iterations,
            'max_lags': self.config.max_lags,
            'tolerance': self.config.tolerance,
            'training_start': self.config.training_start,
            'train_end': self.config.train_end,
            'validation_start': self.config.validation_start,
            'validation_end': self.config.validation_end,
            'output_dir': self.config.output_dir
        }

        # 并行执行训练（失败时报错，不降级）
        results_list = Parallel(
            n_jobs=self.config.first_stage_n_jobs,
            backend=self.config.parallel_backend,
            verbose=0,
            prefer='processes'
        )(
            delayed(_train_single_industry)(
                industry,
                processor.data,  # 传递完整数据
                self.config.industry_map,
                self.config.industry_k_factors,
                config_dict,
                idx,
                len(industry_list)
            )
            for idx, industry in enumerate(industry_list, 1)
        )

        # 聚合结果
        results = {}
        failed_industries = []
        for industry, result in results_list:
            if result is not None:
                results[industry] = result
                if progress_callback:
                    progress_callback(
                        f"  [{len(results)}/{len(industry_list)}] {industry} 训练完成，"
                        f"RMSE(oos)={result.metrics.oos_rmse:.4f}"
                    )
            else:
                failed_industries.append(industry)

        if failed_industries:
            logger.warning(f"以下 {len(failed_industries)} 个行业训练失败: {', '.join(failed_industries)}")

        return results

    def _train_industry_models_serial(
        self,
        processor: IndustryDataProcessor,
        industry_list: List[str],
        progress_callback: Optional[Callable[[str], None]]
    ) -> Dict[str, TrainingResult]:
        """
        第一阶段：串行训练各行业DFM模型（原实现）

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

                # 获取行业变量信息（调试前）
                predictor_cols_with_target = processor.get_industry_predictors(industry, exclude_target=False)
                predictor_cols_without_target = processor.get_industry_predictors(industry, exclude_target=True)
                target_col = processor.get_industry_target_column(industry)

                logger.info(f"[{industry}] 目标列: {target_col}")
                logger.info(f"[{industry}] 预测变量(含目标): {len(predictor_cols_with_target)} 个")
                logger.info(f"[{industry}] 预测变量(不含目标): {len(predictor_cols_without_target)} 个")

                if len(predictor_cols_without_target) <= 5:
                    logger.info(f"[{industry}] 预测变量列表: {predictor_cols_without_target}")

                # 验证行业数据（允许单变量行业）
                is_valid, error_msg = processor.validate_industry_data(industry, min_predictors=1)
                if not is_valid:
                    logger.warning(f"跳过行业 {industry}: {error_msg}")
                    logger.warning(f"  - 目标列: {target_col}")
                    logger.warning(f"  - 预测变量数量: {len(predictor_cols_without_target)}")
                    logger.warning(f"  - 预测变量: {predictor_cols_without_target[:5] if len(predictor_cols_without_target) > 5 else predictor_cols_without_target}")
                    failed_industries.append(industry)
                    continue

                # 单变量行业警告提示
                if len(predictor_cols_without_target) == 1:
                    logger.warning(
                        f"[{industry}] 只有1个预测变量 {predictor_cols_without_target[0]}，"
                        f"模型预测能力可能有限，建议补充更多预测变量"
                    )

                # 获取行业因子数
                k_factors = self.config.industry_k_factors.get(industry)
                if not k_factors:
                    logger.warning(f"跳过行业 {industry}: 未设置因子数")
                    failed_industries.append(industry)
                    continue

                # 构建行业训练数据
                industry_data = processor.build_industry_training_data(industry)
                target_col = processor.get_industry_target_column(industry)
                # 重要：预测变量必须排除目标变量本身
                predictor_cols = processor.get_industry_predictors(industry, exclude_target=True)

                logger.info(f"[{industry}] 构建训练数据: {industry_data.shape[0]} 行, {industry_data.shape[1]} 列")
                logger.info(f"[{industry}] 将使用 {len(predictor_cols)} 个预测变量训练模型")

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
        first_stage_results: Dict[str, TrainingResult],
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        从第一阶段结果中提取分行业nowcasting序列

        Args:
            first_stage_results: 第一阶段训练结果
            data: 原始数据（用于获取时间索引）

        Returns:
            分行业nowcasting的DataFrame，索引为日期，列为行业名
        """
        nowcast_dict = {}

        # 注意：卡尔曼滤波从t=1开始预测，所以预测序列比原始数据少一个时间点
        # 使用data.index[1:]跳过第一个时间点（t=0是初始化状态，没有预测值）
        date_index = data.index[1:]  # 跳过t=0时刻

        for industry, result in first_stage_results.items():
            if result.model_result is None:
                logger.warning(f"行业 {industry} 没有模型结果，跳过")
                continue

            # 提取完整预测序列（样本内 + 样本外）
            forecast_is = result.model_result.forecast_is
            forecast_oos = result.model_result.forecast_oos

            # 合并预测序列（都是numpy数组）
            if forecast_is is not None and forecast_oos is not None:
                full_forecast = np.concatenate([forecast_is, forecast_oos])
            elif forecast_oos is not None:
                full_forecast = forecast_oos
            elif forecast_is is not None:
                full_forecast = forecast_is
            else:
                logger.warning(f"行业 {industry} 没有有效的预测序列，跳过")
                continue

            # 确保预测序列长度与时间索引一致
            if len(full_forecast) != len(date_index):
                logger.warning(f"行业 {industry} 预测序列长度({len(full_forecast)})与时间索引长度({len(date_index)})不一致，进行截断对齐")
                min_len = min(len(full_forecast), len(date_index))
                full_forecast = full_forecast[:min_len]
                current_date_index = date_index[:min_len]
            else:
                current_date_index = date_index

            # 创建带时间索引的Series
            nowcast_dict[f"nowcast_{industry}"] = pd.Series(full_forecast, index=current_date_index)

        if not nowcast_dict:
            raise ValueError("没有任何行业生成有效的nowcasting序列")

        # 合并为DataFrame
        nowcast_df = pd.DataFrame(nowcast_dict)

        logger.info(f"提取到 {len(nowcast_dict)} 个行业的nowcasting序列，时间跨度: {nowcast_df.index[0]} 至 {nowcast_df.index[-1]}")
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
            industry_map={},  # 第二阶段不需要行业映射
            # 筛选策略参数（2025-12-20修复：确保第二阶段继承父配置）
            selection_criterion=self.config.selection_criterion,
            prioritize_win_rate=self.config.prioritize_win_rate,
            rmse_tolerance_percent=self.config.rmse_tolerance_percent,
            win_rate_tolerance_percent=self.config.win_rate_tolerance_percent,
            training_weight=self.config.training_weight
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
