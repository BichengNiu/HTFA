# -*- coding: utf-8 -*-
"""
DFM评估策略 - 函数式接口

提供简洁的函数式接口创建DFM评估器
"""

import numpy as np
from typing import Tuple, List, Callable
from dashboard.models.DFM.train.utils.logger import get_logger
from dashboard.models.DFM.train.training.model_ops import train_dfm_with_forecast, evaluate_model_performance

logger = get_logger(__name__)


def create_dfm_evaluator(config: 'TrainingConfig') -> Callable:
    """
    创建DFM评估器（函数式接口）

    返回一个可调用对象，用于BackwardSelector的评估回调。

    Args:
        config: 训练配置对象

    Returns:
        评估函数，签名为 (variables: List[str], **kwargs) -> Tuple[float, ...]

    Example:
        >>> evaluator = create_dfm_evaluator(config)
        >>> selector = BackwardSelector(evaluator_func=evaluator, ...)
        >>> metrics = evaluator(
        ...     variables=['target', 'var1', 'var2'],
        ...     full_data=data,
        ...     params={'k_factors': 3}
        ... )
    """
    def evaluate(variables: List[str], **kwargs) -> Tuple[float, float, None, None, float, float, bool, None, None]:
        """
        评估指定变量组合的DFM模型性能

        Args:
            variables: 变量列表（包含目标变量）
            **kwargs: 其他参数
                - full_data: 完整数据DataFrame
                - params: DFM参数字典（包含k_factors等）
                - max_iter: 最大迭代次数

        Returns:
            9元组: (is_rmse, oos_rmse, _, _, is_hit_rate, oos_hit_rate,
                   is_svd_error, _, _)
        """
        try:
            # 提取参数
            full_data = kwargs.get('full_data')
            k_factors = kwargs.get('params', {}).get('k_factors', 2)
            max_iter = kwargs.get('max_iter', config.max_iterations)
            target_var = config.target_variable

            # 分离预测变量
            predictor_vars = [v for v in variables if v != target_var]

            if len(predictor_vars) == 0:
                logger.warning("[Evaluator] 预测变量为空，返回无穷大RMSE")
                return (np.inf, np.inf, None, None, -np.inf, -np.inf, False, None, None)

            # 准备数据
            predictor_data = full_data[predictor_vars]
            target_data = full_data[target_var]

            # 训练模型
            model_result = train_dfm_with_forecast(
                predictor_data=predictor_data,
                target_data=target_data,
                k_factors=k_factors,
                train_end=config.train_end,
                validation_start=config.validation_start,
                validation_end=config.validation_end,
                max_iter=max_iter,
                max_lags=1,
                tolerance=config.tolerance,
                progress_callback=None
            )

            # 评估模型
            metrics = evaluate_model_performance(
                model_result=model_result,
                target_data=target_data,
                train_end=config.train_end,
                validation_start=config.validation_start,
                validation_end=config.validation_end
            )

            return metrics.to_tuple()

        except Exception as e:
            logger.error(f"[Evaluator] 评估失败: {e}")
            import traceback
            traceback.print_exc()
            return (np.inf, np.inf, None, None, -np.inf, -np.inf, True, None, None)

    return evaluate


__all__ = ['create_dfm_evaluator']
