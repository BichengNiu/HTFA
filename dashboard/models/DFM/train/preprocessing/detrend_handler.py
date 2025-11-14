# -*- coding: utf-8 -*-
"""
训练阶段的去趋势处理器

封装去趋势和还原逻辑，支持线性去趋势方法。
趋势参数基于完整数据拟合，可用于任意时间点的还原。

作者: Claude Code
创建时间: 2025-11-14
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DetrendHandler:
    """训练阶段去趋势处理器

    功能：
    1. 对指定变量进行线性去趋势（y_t = α + β*t + ε）
    2. 保存趋势参数（α, β, reference_date）
    3. 将残差还原到原始值水平

    特性：
    - 趋势参数基于完整数据拟合
    - 支持部分变量去趋势
    - 自动处理NaN值
    """

    def __init__(self, method: str = 'linear'):
        """初始化去趋势处理器

        Args:
            method: 去趋势方法，当前仅支持'linear'
        """
        if method != 'linear':
            raise ValueError(f"当前仅支持线性去趋势，提供的方法: {method}")

        self.method = method
        self.trend_params = {}  # {变量名: {'alpha': α, 'beta': β, 'reference_date': date, 'n_points': n}}

        logger.info(f"[DetrendHandler] 初始化: method={method}")

    def fit_and_transform(
        self,
        data: pd.DataFrame,
        variables: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """对指定变量进行去趋势（基于完整数据拟合趋势）

        Args:
            data: 原始值DataFrame（index=日期，columns=变量）
            variables: 需要去趋势的变量列表（None=全部列）

        Returns:
            残差DataFrame（去趋势后的数据，未去趋势的列保持不变）

        Raises:
            ValueError: 如果变量不存在或数据点不足
        """
        result_df = data.copy()
        variables = variables or list(data.columns)

        processed_count = 0
        skipped_count = 0

        for var in variables:
            if var not in data.columns:
                logger.warning(f"  变量 '{var}' 不存在于数据中，跳过")
                skipped_count += 1
                continue

            # 提取非NaN数据点
            series = data[var].dropna()

            if len(series) < 3:
                logger.warning(f"  变量 '{var}' 有效数据点不足({len(series)}<3)，跳过")
                skipped_count += 1
                continue

            try:
                # 拟合线性趋势: y_t = α + β*t + ε
                # t是从0开始的整数索引
                t = np.arange(len(series))
                coeffs = np.polyfit(t, series.values, deg=1)
                beta, alpha = coeffs[0], coeffs[1]  # polyfit返回[斜率, 截距]

                # 保存趋势参数
                self.trend_params[var] = {
                    'alpha': float(alpha),
                    'beta': float(beta),
                    'reference_date': str(series.index[0]),
                    'n_points': len(series)
                }

                # 计算整个序列的趋势（包括NaN位置）
                # 使用完整data的索引位置
                full_t = np.arange(len(data))
                trend_full = alpha + beta * full_t

                # 计算残差：原始值 - 趋势
                result_df[var] = data[var] - trend_full

                processed_count += 1
                logger.debug(f"  变量 '{var}' 去趋势完成: α={alpha:.6f}, β={beta:.6f}")

            except Exception as e:
                logger.error(f"  变量 '{var}' 去趋势失败: {e}")
                skipped_count += 1
                continue

        logger.info(f"[DetrendHandler] fit_and_transform 完成: 成功{processed_count}个，跳过{skipped_count}个")

        return result_df

    def inverse_transform(
        self,
        residuals: pd.Series,
        variable: str
    ) -> pd.Series:
        """将残差还原到原始值水平

        Args:
            residuals: 残差序列（index=日期）
            variable: 变量名

        Returns:
            原始值序列

        Note:
            如果变量未进行去趋势，直接返回residuals
        """
        if variable not in self.trend_params:
            logger.debug(f"  变量 '{variable}' 未进行去趋势，直接返回")
            return residuals

        params = self.trend_params[variable]
        alpha = params['alpha']
        beta = params['beta']
        ref_date = pd.to_datetime(params['reference_date'])

        # 计算时间索引（以周为单位，假设数据是周度）
        # 这里使用天数差除以7来计算周数
        t = (residuals.index - ref_date).days / 7.0

        # 还原: y = residual + (α + β*t)
        trend = alpha + beta * t.values
        original = residuals.values + trend

        result = pd.Series(original, index=residuals.index, name=variable)

        logger.debug(f"  变量 '{variable}' 还原完成: {len(result)}个数据点")

        return result

    def get_trend_params(self) -> Dict[str, Dict[str, Any]]:
        """获取所有变量的趋势参数

        Returns:
            趋势参数字典: {变量名: {'alpha': α, 'beta': β, 'reference_date': date, 'n_points': n}}
        """
        return self.trend_params.copy()

    def has_params(self, variable: str) -> bool:
        """检查变量是否有趋势参数

        Args:
            variable: 变量名

        Returns:
            是否存在趋势参数
        """
        return variable in self.trend_params

    def __repr__(self) -> str:
        return f"DetrendHandler(method={self.method}, n_variables={len(self.trend_params)})"
