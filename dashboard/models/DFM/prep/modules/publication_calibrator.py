# -*- coding: utf-8 -*-
"""
发布日期校准模块

负责根据变量的发布滞后天数校准数据日期。
从processor.py提取，遵循单一职责原则(SRP)。
"""

import pandas as pd
import logging
from typing import Dict, Optional

from dashboard.models.DFM.prep.utils.friday_utils import get_friday_with_lag
from dashboard.models.DFM.utils.text_utils import normalize_text

logger = logging.getLogger(__name__)


class PublicationCalibrator:
    """发布日期校准器

    根据频率类型采用不同的校准逻辑：
    - 日度/周度/旬度：数据日期 + 滞后天数 = 实际发布日期
    - 月度：对齐到发布月份内的周五
    - 季度：对齐到发布季度内的周五
    - 年度：对齐到发布年份内的周五
    """

    def __init__(self, publication_lag_map: Dict[str, int]):
        """初始化校准器

        Args:
            publication_lag_map: 变量名到滞后天数的映射
        """
        self.lag_map = publication_lag_map or {}

    def calibrate(self, df: pd.DataFrame, freq_type: str) -> pd.DataFrame:
        """应用发布日期校准

        Args:
            df: 输入DataFrame，索引为数据日期
            freq_type: 频率类型 ('daily', 'weekly', 'dekad', 'monthly', 'quarterly', 'yearly')

        Returns:
            pd.DataFrame: 索引为校准后日期的DataFrame
        """
        if not self.lag_map:
            return df

        if df is None or df.empty:
            return df

        logger.info(f"  应用发布日期校准 ({freq_type})...")
        calibration_count = 0
        result_columns = {}

        # 映射频率类型到周期类型
        period_type_map = {
            'daily': None,      # 直接加滞后天数
            'weekly': None,     # 直接加滞后天数
            'dekad': None,      # 直接加滞后天数
            'monthly': 'month',
            'quarterly': 'quarter',
            'yearly': 'year'
        }
        period_type = period_type_map.get(freq_type)

        for col in df.columns:
            col_norm = normalize_text(col)
            lag_days = self.lag_map.get(col_norm)

            if lag_days is not None and lag_days != 0:
                series = df[col].dropna()
                if series.empty:
                    result_columns[col] = df[col]
                    continue

                # 根据频率类型计算校准后的日期
                if period_type is None:
                    # 日度/周度/旬度：直接加滞后天数
                    new_index = series.index + pd.Timedelta(days=lag_days)
                else:
                    # 月度/季度/年度：使用friday_utils计算
                    new_index = series.index.map(
                        lambda d: get_friday_with_lag(d, lag_days, period_type)
                    )

                calibrated_series = pd.Series(series.values, index=new_index, name=col)
                result_columns[col] = calibrated_series
                calibration_count += 1
                logger.debug(f"    {col}: 滞后{lag_days}天 ({freq_type})")
            else:
                result_columns[col] = df[col]

        if calibration_count > 0:
            result_df = pd.DataFrame(result_columns)
            result_df = result_df.sort_index()
            logger.info(f"  发布日期校准完成: {calibration_count}个变量 ({freq_type})")
            return result_df
        else:
            return df


def calibrate_publication_dates(
    df: pd.DataFrame,
    freq_type: str,
    publication_lag_map: Dict[str, int]
) -> pd.DataFrame:
    """便利函数：校准发布日期

    Args:
        df: 输入DataFrame
        freq_type: 频率类型
        publication_lag_map: 变量名到滞后天数的映射

    Returns:
        pd.DataFrame: 校准后的DataFrame
    """
    calibrator = PublicationCalibrator(publication_lag_map)
    return calibrator.calibrate(df, freq_type)


__all__ = [
    'PublicationCalibrator',
    'calibrate_publication_dates'
]
