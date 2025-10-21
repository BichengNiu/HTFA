# -*- coding: utf-8 -*-
"""
DFM训练器Facade接口

提供统一的、简洁的API，隐藏内部实现复杂性
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from dashboard.DFM.train_ref.training.config import TrainingConfig, ModelConfig, DataConfig
from dashboard.DFM.train_ref.core.factor_model import fit_dfm, DFMResults
from dashboard.DFM.train_ref.core.estimator import estimate_target_loading
from dashboard.DFM.train_ref.evaluation.evaluator import DFMEvaluator, EvaluationResult
from dashboard.DFM.train_ref.utils.data_utils import (
    load_data,
    prepare_data,
    standardize_data,
    destandardize_series
)
from dashboard.DFM.train_ref.utils.logger import get_logger, setup_logging
from dashboard.DFM.train_ref.utils.reproducibility import ensure_reproducibility


logger = get_logger(__name__)


@dataclass
class TrainingResults:
    """训练结果

    包含模型训练的所有输出
    """
    factors: pd.DataFrame              # 因子序列
    loadings: pd.DataFrame             # 因子载荷
    nowcast: pd.Series                 # Nowcast预测
    nowcast_original_scale: pd.Series  # 原始尺度的Nowcast
    metrics: Dict[str, float]          # 评估指标
    config: TrainingConfig             # 训练配置
    success: bool                      # 是否成功
    message: str = ""                  # 消息


class DFMTrainer:
    """DFM训练器

    统一的DFM模型训练接口，提供端到端的训练流程

    使用示例:
    ```python
    from train_ref import DFMTrainer, TrainingConfig

    config = TrainingConfig(...)
    trainer = DFMTrainer(config)

    results = trainer.train()

    print(f"OOS RMSE: {results.metrics['oos_rmse']:.4f}")
    print(f"OOS Hit Rate: {results.metrics['oos_hit_rate']:.2%}")
    ```
    """

    def __init__(
        self,
        config: TrainingConfig,
        seed: Optional[int] = None,
        verbose: bool = True
    ):
        """初始化训练器

        Args:
            config: 训练配置
            seed: 随机种子
            verbose: 是否输出详细日志
        """
        self.config = config
        self.seed = ensure_reproducibility(seed)
        self.verbose = verbose

        if verbose:
            setup_logging(level='INFO')
        else:
            setup_logging(level='WARNING')

        validation_errors = config.validate()
        if validation_errors:
            raise ValueError(f"配置验证失败: {validation_errors}")

        self.data_raw_: Optional[pd.DataFrame] = None
        self.data_standardized_: Optional[pd.DataFrame] = None
        self.stats_: Optional[Dict[str, tuple]] = None
        self.results_: Optional[TrainingResults] = None

        logger.info(f"DFMTrainer初始化完成，随机种子={self.seed}")

    def train(self) -> TrainingResults:
        """执行完整训练流程

        流程:
        1. 加载数据
        2. 准备和标准化
        3. 训练DFM模型
        4. 计算Nowcast
        5. 评估性能

        Returns:
            TrainingResults: 训练结果
        """
        logger.info("="* 60)
        logger.info("开始DFM训练流程")
        logger.info("="* 60)

        try:
            self._load_and_prepare_data()

            dfm_results = self._fit_dfm_model()

            nowcast_std, nowcast_orig = self._calculate_nowcast(dfm_results)

            metrics = self._evaluate_performance(nowcast_std)

            self.results_ = TrainingResults(
                factors=dfm_results.factors,
                loadings=pd.DataFrame(
                    dfm_results.loadings,
                    index=self.config.data.selected_variables,
                    columns=[f'Factor{i+1}' for i in range(self.config.model.k_factors)]
                ),
                nowcast=nowcast_std,
                nowcast_original_scale=nowcast_orig,
                metrics=metrics,
                config=self.config,
                success=True,
                message="训练成功完成"
            )

            logger.info("="* 60)
            logger.info("训练流程完成")
            logger.info(f"OOS RMSE: {metrics.get('oos_rmse', np.nan):.4f}")
            logger.info(f"OOS Hit Rate: {metrics.get('oos_hit_rate', np.nan):.2%}")
            logger.info("="* 60)

            return self.results_

        except Exception as e:
            logger.error(f"训练失败: {e}", exc_info=True)

            self.results_ = TrainingResults(
                factors=pd.DataFrame(),
                loadings=pd.DataFrame(),
                nowcast=pd.Series(),
                nowcast_original_scale=pd.Series(),
                metrics={},
                config=self.config,
                success=False,
                message=str(e)
            )

            return self.results_

    def _load_and_prepare_data(self):
        """加载和准备数据"""
        logger.info(f"加载数据: {self.config.data.data_path}")

        self.data_raw_ = load_data(self.config.data.data_path)

        data_prepared, predictor_vars = prepare_data(
            df=self.data_raw_,
            target_variable=self.config.data.target_variable,
            selected_variables=self.config.data.selected_variables,
            end_date=self.config.data.validation_end
        )

        if not self.config.data.selected_variables:
            self.config.data.selected_variables = predictor_vars

        self.data_standardized_, self.stats_ = standardize_data(
            data=data_prepared,
            fit_data=data_prepared.loc[:self.config.data.train_end] if self.config.data.train_end else None
        )

        logger.info(
            f"数据准备完成: {self.data_standardized_.shape}, "
            f"预测变量: {len(self.config.data.selected_variables)}"
        )

    def _fit_dfm_model(self) -> DFMResults:
        """拟合DFM模型"""
        logger.info(f"拟合DFM模型: k={self.config.model.k_factors}")

        predictor_data = self.data_standardized_[self.config.data.selected_variables]

        dfm_results = fit_dfm(
            data=predictor_data,
            n_factors=self.config.model.k_factors,
            max_lags=self.config.model.max_lags,
            max_iter=self.config.model.max_iter,
            train_end=self.config.data.train_end
        )

        return dfm_results

    def _calculate_nowcast(
        self,
        dfm_results: DFMResults
    ) -> tuple[pd.Series, pd.Series]:
        """计算Nowcast"""
        logger.info("计算Nowcast预测")

        target_data_std = self.data_standardized_[self.config.data.target_variable]

        target_loading = estimate_target_loading(
            target=target_data_std,
            factors=dfm_results.factors,
            train_end=self.config.data.train_end
        )

        nowcast_std = pd.Series(
            dfm_results.factors.values @ target_loading,
            index=dfm_results.factors.index,
            name='Nowcast'
        )

        target_mean, target_std = self.stats_[self.config.data.target_variable]
        nowcast_orig = destandardize_series(nowcast_std, target_mean, target_std)

        logger.info(f"Nowcast计算完成: {len(nowcast_std)}个预测值")

        return nowcast_std, nowcast_orig

    def _evaluate_performance(self, nowcast: pd.Series) -> Dict[str, float]:
        """评估性能"""
        logger.info("评估模型性能")

        evaluator = DFMEvaluator()

        eval_result = evaluator.evaluate(
            data=self.data_standardized_,
            target_variable=self.config.data.target_variable,
            predictor_variables=self.config.data.selected_variables,
            k_factors=self.config.model.k_factors,
            train_end=self.config.data.train_end,
            validation_start=self.config.data.validation_start,
            validation_end=self.config.data.validation_end,
            max_lags=self.config.model.max_lags,
            max_iter=self.config.model.max_iter
        )

        if not eval_result.success:
            logger.warning(f"评估过程遇到问题: {eval_result.error_message}")

        metrics = {
            'is_rmse': eval_result.metrics.is_rmse,
            'is_mae': eval_result.metrics.is_mae,
            'is_hit_rate': eval_result.metrics.is_hit_rate,
            'oos_rmse': eval_result.metrics.oos_rmse,
            'oos_mae': eval_result.metrics.oos_mae,
            'oos_hit_rate': eval_result.metrics.oos_hit_rate
        }

        return metrics

    def get_results(self) -> Optional[TrainingResults]:
        """获取训练结果"""
        return self.results_

    def save_results(self, output_dir: Optional[str] = None):
        """保存训练结果

        Args:
            output_dir: 输出目录
        """
        if self.results_ is None:
            raise ValueError("尚未训练，无结果可保存")

        if output_dir is None:
            output_dir = self.config.output_dir

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.results_.factors.to_csv(output_path / 'factors.csv')
        self.results_.loadings.to_csv(output_path / 'loadings.csv')
        self.results_.nowcast.to_csv(output_path / 'nowcast.csv')

        with open(output_path / 'metrics.txt', 'w', encoding='utf-8') as f:
            f.write("DFM训练结果\n")
            f.write("="* 50 + "\n")
            for key, value in self.results_.metrics.items():
                f.write(f"{key}: {value:.4f}\n")

        logger.info(f"结果已保存到: {output_path}")
