"""
数据对齐模块

负责将不同频率的数据对齐到目标频率（通常是周度）
包括：
- 目标变量对齐到最近周五
- 月度数据对齐到月末最后周五
- 日度数据转换为周度
- 周度数据对齐
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from calendar import monthrange

logger = logging.getLogger(__name__)

from dashboard.models.DFM.prep.modules.data_cleaner import DataCleaner
from dashboard.models.DFM.prep.utils.friday_utils import (
    get_nearest_friday,
    get_monthly_friday,
    get_quarterly_friday,
    get_yearly_friday,
    get_dekad_friday
)

class DataAligner:
    """数据对齐器类"""

    def __init__(self, target_freq: str = 'W-FRI', enable_borrowing: bool = True):
        self.target_freq = target_freq
        self.enable_borrowing = enable_borrowing
        self.cleaner = DataCleaner()

        # 验证目标频率
        if not target_freq.upper().endswith('-FRI'):
            raise ValueError(f"当前目标对齐逻辑仅支持周五 (W-FRI)。提供的目标频率 '{target_freq}' 无效。")

    def align_target_to_nearest_friday(self, target_values: pd.Series) -> pd.Series:
        """
        将目标变量对齐到最近的周五

        Args:
            target_values: 目标变量序列，索引为发布日期

        Returns:
            pd.Series: 对齐到周五的目标变量序列
        """
        if target_values is None or target_values.empty:
            raise ValueError("目标变量数据为空，无法对齐到周五")
        
        logger.info("目标变量对齐到最近周五...")

        temp_target_df = pd.DataFrame({'value': target_values})

        temp_target_df['nearest_friday'] = temp_target_df.index.map(get_nearest_friday)
        
        # 处理同一个目标周五的重复：保留最新发布日期的数据
        # 我们先按原始发布日期索引排序，然后分组并取最后一个
        target_series_aligned = temp_target_df.sort_index(ascending=True).groupby('nearest_friday')['value'].last()
        target_series_aligned.index.name = 'Date'
        target_series_aligned.name = target_values.name

        logger.info("目标变量对齐完成。Shape: %s", target_series_aligned.shape)

        return target_series_aligned
    
    def align_target_sheet_predictors_to_nearest_friday(self, predictors_df: pd.DataFrame) -> pd.DataFrame:
        """
        将目标表格的预测变量对齐到最近的周五

        Args:
            predictors_df: 预测变量DataFrame，索引为发布日期

        Returns:
            pd.DataFrame: 对齐到周五的预测变量DataFrame
        """
        if predictors_df is None or predictors_df.empty:
            raise ValueError("预测变量数据为空，无法对齐到周五")
        
        logger.info("目标 Sheet 预测变量对齐到最近周五...")
        
        # 确保没有重复列
        predictors_df = self.cleaner.remove_duplicate_columns(predictors_df, "[目标Sheet预测变量对齐] ")

        temp_df = predictors_df.copy()
        temp_df['nearest_friday'] = temp_df.index.map(get_nearest_friday)
        
        # 处理同一个目标周五的重复：保留最新发布日期的数据
        aligned_predictors = temp_df.sort_index(ascending=True).groupby('nearest_friday').last()
        aligned_predictors.index.name = 'Date'
        
        logger.info("目标 Sheet 预测变量对齐完成。Shape: %s", aligned_predictors.shape)

        return aligned_predictors

    def _align_periodic_to_friday(
        self,
        data: pd.DataFrame,
        freq_name: str,
        log_prefix: str,
        align_func=None
    ) -> pd.DataFrame:
        """
        将周期性数据(月度/季度/年度)对齐到所属周期内的周五（不跨期）

        Args:
            data: 数据DataFrame，索引为发布日期
            freq_name: 频率名称，用于错误消息（"月度"/"季度"/"年度"）
            log_prefix: 日志前缀
            align_func: 对齐函数，用于计算目标周五（默认使用 _get_nearest_friday）

        Returns:
            pd.DataFrame: 对齐后的数据
        """
        if data is None or data.empty:
            raise ValueError(f"{freq_name}数据为空，无法对齐到最近周五")

        logger.info("对齐%s数据到所属周期内的周五（不跨期）...", freq_name)

        # 确保没有重复列
        data = self.cleaner.remove_duplicate_columns(data, f"[{log_prefix}] ")

        # 使用指定的对齐函数，默认使用 _get_nearest_friday
        if align_func is None:
            align_func = get_nearest_friday

        temp_df = data.copy()
        temp_df['nearest_friday'] = temp_df.index.map(align_func)

        # 处理同一个目标周五的重复：保留最新发布日期的数据
        aligned = temp_df.sort_index(ascending=True).groupby('nearest_friday').last()
        aligned.index.name = 'Date'

        logger.info("对齐到周五完成。Shape: %s", aligned.shape)

        return aligned

    def align_monthly_to_last_friday(
        self,
        monthly_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        将月度数据对齐到当月内的周五（不跨月）

        Args:
            monthly_data: 月度数据DataFrame，索引为发布日期

        Returns:
            pd.DataFrame: 对齐后的数据
        """
        return self._align_periodic_to_friday(
            monthly_data, "月度", "月度对齐",
            align_func=get_monthly_friday
        )

    def align_quarterly_to_friday(
        self,
        quarterly_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        将季度数据对齐到当季内的周五（不跨季）

        Args:
            quarterly_data: 季度数据DataFrame，索引为发布日期

        Returns:
            pd.DataFrame: 对齐后的数据
        """
        return self._align_periodic_to_friday(
            quarterly_data, "季度", "季度对齐",
            align_func=get_quarterly_friday
        )

    def align_yearly_to_friday(
        self,
        yearly_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        将年度数据对齐到当年内的周五（不跨年）

        Args:
            yearly_data: 年度数据DataFrame，索引为发布日期

        Returns:
            pd.DataFrame: 对齐后的数据
        """
        return self._align_periodic_to_friday(
            yearly_data, "年度", "年度对齐",
            align_func=get_yearly_friday
        )

    def convert_daily_to_weekly(
        self,
        daily_data_list: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        将日度数据转换为周度数据（使用均值）

        Args:
            daily_data_list: 日度数据DataFrame列表

        Returns:
            pd.DataFrame: 转换后的周度数据
        """
        if not daily_data_list:
            return pd.DataFrame()

        logger.info("处理日度数据...")

        # 首先确保每个DataFrame的索引是唯一的
        cleaned_daily_list = []
        for i, df in enumerate(daily_data_list):
            # 移除重复的索引（保留第一个）
            if df.index.duplicated().any():
                logger.warning("日度数据 %d 有重复索���，正在清理...", i)
                df = df[~df.index.duplicated(keep='first')]
            cleaned_daily_list.append(df)

        # 合并所有日度数据
        df_daily_full = pd.concat(cleaned_daily_list, axis=1, join='outer', sort=True)

        # 处理重复列
        df_daily_full = self.cleaner.remove_duplicate_columns(df_daily_full, "[日度合并] ")

        if df_daily_full.empty:
            logger.warning("合并后的日度数据为空。")
            return pd.DataFrame()

        # 转换为周度（均值聚合）
        logger.info("转换日度数据为周度（均值聚合）...")
        # 零值处理已在processor全局预处理中完成
        df_daily_weekly = df_daily_full.resample(self.target_freq).mean()
        logger.info("日度->周度(均值) 完成. Shape: %s", df_daily_weekly.shape)
        return df_daily_weekly
    
    def align_weekly_data(
        self,
        weekly_data_list: List[pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        对齐周度数据到理论日期索引（基于用户指定的时间范围）

        新逻辑：
        1. 合并原始数据
        2. 生成理论日期索引（用户时间范围内的所有周五）
        3. 将原始数据映射到理论索引，包含借调逻辑

        Args:
            weekly_data_list: 周度数据DataFrame列表
            start_date: 用户设置的开始日期（必需，用于生成理论索引）
            end_date: 用户设置的结束日期（必需，用于生成理论索引）

        Returns:
            Tuple[pd.DataFrame, Dict]: (对齐后的周度数据, 借调日志字典)
        """
        borrowing_log = {}

        if not weekly_data_list:
            return pd.DataFrame(), borrowing_log

        logger.info("处理周度数据...")

        # 首先确保每个DataFrame的索引是唯一的
        cleaned_weekly_list = []
        for i, df in enumerate(weekly_data_list):
            # 移除重复的索引（保留第一个）
            if df.index.duplicated().any():
                logger.warning("周度数据 %d 有重复索引，正在清理...", i)
                df = df[~df.index.duplicated(keep='first')]
            cleaned_weekly_list.append(df)

        # 合并所有周度数据
        df_weekly_full = pd.concat(cleaned_weekly_list, axis=1, join='outer', sort=True)

        # 处理重复列
        df_weekly_full = self.cleaner.remove_duplicate_columns(df_weekly_full, "[周度合并] ")

        if df_weekly_full.empty:
            logger.warning("合并后的周度数据为空。")
            return pd.DataFrame(), borrowing_log

        # 生成理论日期索引（用户时间范围内的所有周五）
        if start_date and end_date:
            theoretical_index = generate_theoretical_index(start_date, end_date, self.target_freq)
            logger.info("理论日期索引: %s 至 %s, 共 %d 个周五", theoretical_index[0].date(), theoretical_index[-1].date(), len(theoretical_index))
        else:
            # 如果没有指定日期范围，使用数据本身的范围
            data_start = df_weekly_full.index.min()
            data_end = df_weekly_full.index.max()
            theoretical_index = generate_theoretical_index(data_start, data_end, self.target_freq)
            logger.warning("未指定日期范���，使用数据范围: %s 至 %s", theoretical_index[0].date(), theoretical_index[-1].date())

        # 对齐到理论索引（包含借调逻辑）
        logger.info("对齐周度数据到理论索引...")
        df_weekly_aligned, borrowing_log = align_to_theoretical_index(
            df_weekly_full, theoretical_index, self.target_freq, self.enable_borrowing
        )

        logger.info("周度->周度(对齐) 完成. Shape: %s", df_weekly_aligned.shape)
        return df_weekly_aligned, borrowing_log

    @staticmethod
    def _is_dekad_date(date) -> bool:
        """判断是否是旬日（10日、20日、或月末）

        旬度数据的时间点定义：
        - 上旬：每月10日
        - 中旬：每月20日
        - 下旬：每月最后一天（28/29/30/31日，取决于月份）
        """
        day = date.day
        if day == 10 or day == 20:
            return True
        # 月末：当日是该月最后一天
        last_day_of_month = monthrange(date.year, date.month)[1]
        return day == last_day_of_month

    def convert_dekad_to_weekly(
        self,
        dekad_data_list: List[pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        将旬度数据转换为周度数据（基于理论索引）

        新逻辑：
        1. 合并并过滤旬日数据
        2. 将旬日映射到对应的周五
        3. 生成理论周度索引
        4. 对齐到理论索引（包含借调逻辑）

        旬日定义：
        - 上旬：每月10日
        - 中旬：每月20日
        - 下旬：每月最后一天（28/29/30/31日）

        Args:
            dekad_data_list: 旬度数据DataFrame列表
            start_date: 用户设置的开始日期（用于生成理论索引）
            end_date: 用户设置的结束日期（用于生成理论索引）

        Returns:
            Tuple[pd.DataFrame, Dict]: (转换后的周度数据, 借调日志字典)
        """
        borrowing_log = {}
        if not dekad_data_list:
            return pd.DataFrame(), borrowing_log

        logger.info("处理旬度数据...")

        # 首先确保每个DataFrame的索引是唯一的
        cleaned_dekad_list = []
        for i, df in enumerate(dekad_data_list):
            if df.index.duplicated().any():
                logger.warning("旬度数据 %d 有重复索引，正在清理...", i)
                df = df[~df.index.duplicated(keep='first')]
            cleaned_dekad_list.append(df)

        # 合并所有旬度数据
        df_dekad_full = pd.concat(cleaned_dekad_list, axis=1, join='outer', sort=True)
        df_dekad_full = self.cleaner.remove_duplicate_columns(df_dekad_full, "[旬度合并] ")

        if df_dekad_full.empty:
            logger.warning("合并后的旬度数据为空。")
            return pd.DataFrame(), borrowing_log

        # 过滤出旬日（10日、20日、月末）的数据
        logger.info("旬度数据原始索引范围: %s 到 %s", df_dekad_full.index.min(), df_dekad_full.index.max())
        is_dekad_day = df_dekad_full.index.map(self._is_dekad_date)

        if not is_dekad_day.any():
            raise ValueError("旬度数据中没有旬日（10日、20日、月末），数据格式不正确")

        df_dekad_filtered = df_dekad_full[is_dekad_day]
        logger.info("过滤后旬日数据行数: %d", len(df_dekad_filtered))

        # 将旬日映射到对应的周五
        logger.info("映射旬度数据到周五...")
        df_temp = df_dekad_filtered.copy()
        df_temp['target_friday'] = df_temp.index.map(get_dekad_friday)

        # 按目标周五分组，保留最后一个值
        df_dekad_mapped = df_temp.groupby('target_friday').last()
        df_dekad_mapped.index.name = 'Date'

        # 生成理论周度索引
        if start_date and end_date:
            theoretical_index = generate_theoretical_index(start_date, end_date, self.target_freq)
            logger.info("理论日期索引: %s 至 %s, 共 %d 个周五", theoretical_index[0].date(), theoretical_index[-1].date(), len(theoretical_index))
        else:
            data_start = df_dekad_mapped.index.min()
            data_end = df_dekad_mapped.index.max()
            theoretical_index = generate_theoretical_index(data_start, data_end, self.target_freq)
            logger.warning("未指定日期范围，使用数据范围")

        # 对齐到理论索引（包含借调逻辑）
        # 注意：旬度数据本来就稀疏（每月3个点），借调可能不太适用
        # 但仍然使用相同的逻辑以保持一致性
        logger.info("对齐旬度数据到理论索引...")
        df_dekad_aligned, borrowing_log = align_to_theoretical_index(
            df_dekad_mapped, theoretical_index, self.target_freq, self.enable_borrowing
        )

        logger.info("旬度->周度 完成. Shape: %s", df_dekad_aligned.shape)
        return df_dekad_aligned, borrowing_log

    def create_full_date_range(self, all_indices: List[pd.DatetimeIndex], start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DatetimeIndex:
        """
        创建完整的日期范围
        
        Args:
            all_indices: 所有数据的日期索引列表
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        Returns:
            pd.DatetimeIndex: 完整的日期范围
        """
        # 过滤出有效的索引
        valid_indices = [idx for idx in all_indices if idx is not None and not idx.empty]

        if not valid_indices:
            raise ValueError("所有收集到的索引都为空，无法创建日期范围")
        
        min_date_orig = min(idx.min() for idx in valid_indices)
        max_date_orig = max(idx.max() for idx in valid_indices)
        
        logger.info("所有原始数据中的最小/最大日期: %s / %s", min_date_orig.date(), max_date_orig.date())
        
        # 确定最终的开始/结束日期
        final_start_date = pd.to_datetime(start_date) if start_date else min_date_orig
        final_end_date = pd.to_datetime(end_date) if end_date else max_date_orig
        
        # 调整如果配置日期超出数据范围
        if start_date and pd.to_datetime(start_date) < min_date_orig:
            final_start_date = pd.to_datetime(start_date)
        if end_date and pd.to_datetime(end_date) > max_date_orig:
            final_end_date = pd.to_datetime(end_date)
        
        # 对齐最终开始/结束日期到周五频率
        min_date_fri = min_date_orig - pd.Timedelta(days=(min_date_orig.weekday() - 4 + 7) % 7)
        max_date_fri = max_date_orig - pd.Timedelta(days=(max_date_orig.weekday() - 4 + 7) % 7) + pd.Timedelta(weeks=0 if max_date_orig.weekday()==4 else 1)
        
        final_start_date_aligned = final_start_date - pd.Timedelta(days=(final_start_date.weekday() - 4 + 7) % 7)
        final_end_date_aligned = final_end_date - pd.Timedelta(days=(final_end_date.weekday() - 4 + 7) % 7)
        
        # 尊重实际数据边界
        actual_range_start = max(min_date_fri, final_start_date_aligned)
        actual_range_end = min(max_date_fri, final_end_date_aligned)
        
        if actual_range_start > actual_range_end:
            # 如果计算的开始日期晚于结束日期，使用原始数据范围
            logger.warning("计算出的实际开始日期 (%s) 晚于结束日期 (%s)，使用原始数据范围。", actual_range_start.date(), actual_range_end.date())
            actual_range_start = min_date_fri
            actual_range_end = max_date_fri
        
        full_date_range = pd.date_range(start=actual_range_start, end=actual_range_end, freq=self.target_freq)
        logger.info("最终确定的完整周度日期范围 (对齐到 %s): %s 到 %s", self.target_freq, full_date_range.min().date(), full_date_range.max().date())
        
        return full_date_range

    def align_by_type(
        self,
        df: pd.DataFrame,
        freq_type: str,
        data_start_date: Optional[str] = None,
        data_end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """根据频率类型对齐数据（统一入口）

        Args:
            df: 待对齐的DataFrame
            freq_type: 频率类型（'daily', 'weekly', 'dekad', 'monthly', 'quarterly', 'yearly'）
            data_start_date: 用户设置的开始日期（用于周度/旬度数据的理论索引）
            data_end_date: 用户设置的结束日期（用于周度/旬度数据的理论索引）

        Returns:
            Tuple[pd.DataFrame, Dict]: (对齐后的DataFrame, 借调日志字典)
        """
        borrowing_log = {}

        if freq_type == 'daily':
            return self.convert_daily_to_weekly([df]), borrowing_log
        elif freq_type == 'weekly':
            aligned_df, borrowing_log = self.align_weekly_data(
                [df], data_start_date, data_end_date
            )
            return aligned_df, borrowing_log
        elif freq_type == 'dekad':
            aligned_df, borrowing_log = self.convert_dekad_to_weekly(
                [df], data_start_date, data_end_date
            )
            return aligned_df, borrowing_log
        elif freq_type == 'monthly':
            return self.align_monthly_to_last_friday(df), borrowing_log
        elif freq_type == 'quarterly':
            return self.align_quarterly_to_friday(df), borrowing_log
        elif freq_type == 'yearly':
            return self.align_yearly_to_friday(df), borrowing_log
        else:
            raise ValueError(f"不支持的频率类型: {freq_type}")

# 导出的类和函数
__all__ = [
    'DataAligner',
    'generate_theoretical_index',
    'align_to_theoretical_index'
]


# ============================================================================
# 新的基于理论索引的对齐和借调逻辑
# ============================================================================

def generate_theoretical_index(start_date, end_date, freq='W-FRI'):
    """生成理论日期索引

    Args:
        start_date: 开始日期
        end_date: 结束日期
        freq: 目标频率，默认 'W-FRI'

    Returns:
        pd.DatetimeIndex: 理论日期索引（用户时间范围内的所有目标频率日期）
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # 生成完整的理论日期范围
    theoretical_index = pd.date_range(start=start_dt, end=end_dt, freq=freq)

    return theoretical_index


def _get_window_for_date(target_date, freq='W-FRI'):
    """获取目标日期对应的时间窗口

    对于周五频率，窗口是从上周六到本周五

    Args:
        target_date: 目标日期（周五）
        freq: 频率类型

    Returns:
        Tuple[pd.Timestamp, pd.Timestamp]: (窗口开始, 窗口结束)
    """
    if freq == 'W-FRI':
        # 窗口：上周六 到 本周五
        window_end = target_date
        window_start = target_date - pd.Timedelta(days=6)
        return window_start, window_end
    elif freq == 'ten_day':
        # 旬度窗口逻辑
        day = target_date.day
        if day == 10:
            window_start = target_date.replace(day=1)
            window_end = target_date
        elif day == 20:
            window_start = target_date.replace(day=11)
            window_end = target_date
        else:  # 月末
            window_start = target_date.replace(day=21)
            window_end = target_date
        return window_start, window_end
    else:
        raise ValueError(f"不支持的频率: {freq}")


def _align_column_to_theoretical_index(series_data, theoretical_index, freq='W-FRI', enable_borrowing=True, col_name=""):
    """将单列数据对齐到理论索引

    Args:
        series_data: 原始数据序列
        theoretical_index: 理论日期索引
        freq: 目标频率
        enable_borrowing: 是否启用借调
        col_name: 列名（用于日志）

    Returns:
        Tuple[pd.Series, List[Dict]]: (对齐后的序列, 借调日志)
    """
    result = pd.Series(index=theoretical_index, dtype=float)
    result[:] = np.nan
    borrowing_log = []

    # 第一步：将原始数据映射到对应的理论日期
    for i, target_date in enumerate(theoretical_index):
        window_start, window_end = _get_window_for_date(target_date, freq)

        # 找到窗口内的原始数据
        mask = (series_data.index >= window_start) & (series_data.index <= window_end)
        window_data = series_data[mask].dropna()

        if len(window_data) > 0:
            # 取窗口内最后一个值
            result.loc[target_date] = window_data.iloc[-1]

    # 第二步：借调逻辑（如果启用）
    if enable_borrowing:
        borrowing_count = 0
        for i, target_date in enumerate(theoretical_index[:-1]):  # 最后一个窗口无法借调
            if pd.isna(result.loc[target_date]):
                # 当前窗口没有数据，检查下一个窗口
                next_target = theoretical_index[i + 1]
                next_window_start, next_window_end = _get_window_for_date(next_target, freq)

                # 找到下一个窗口内的原始数据
                next_mask = (series_data.index >= next_window_start) & (series_data.index <= next_window_end)
                next_window_data = series_data[next_mask].dropna()

                # 借调条件：下一个窗口有2个或更多数据点
                if len(next_window_data) >= 2:
                    borrowed_value = next_window_data.iloc[0]
                    borrowed_time = next_window_data.index[0]

                    result.loc[target_date] = borrowed_value
                    borrowing_count += 1

                    borrowing_log.append({
                        'borrowed_from': borrowed_time,
                        'borrowed_to': target_date
                    })

        if borrowing_count > 0 and col_name:
            logger.info("[借调] %s: %d 次", col_name, borrowing_count)

    return result, borrowing_log


def align_to_theoretical_index(df_input, theoretical_index, freq='W-FRI', enable_borrowing=True):
    """将DataFrame对齐到理论索引

    这是新的核心对齐函数，基于理论索引进行：
    1. 将原始数据映射到对应的理论日期（取窗口内最后一个值）
    2. 如果启用借调，对空窗口尝试从下一个窗口借调

    Args:
        df_input: 原始数据DataFrame
        theoretical_index: 理论日期索引
        freq: 目标频率
        enable_borrowing: 是否启用借调

    Returns:
        Tuple[pd.DataFrame, Dict]: (对齐后的DataFrame, 借调日志字典)
    """
    if df_input.empty:
        return pd.DataFrame(index=theoretical_index), {}

    all_borrowing_log = {}
    result_columns = {}

    for col in df_input.columns:
        series_data = df_input[col]
        if series_data.dropna().empty:
            result_columns[col] = pd.Series(index=theoretical_index, dtype=float)
            continue

        aligned_series, borrowing_log = _align_column_to_theoretical_index(
            series_data, theoretical_index, freq, enable_borrowing, col_name=col
        )
        result_columns[col] = aligned_series

        if borrowing_log:
            all_borrowing_log[col] = borrowing_log

    result_df = pd.DataFrame(result_columns, index=theoretical_index)

    if all_borrowing_log:
        logger.info("借调完成: 共处理 %d 列", len(all_borrowing_log))

    return result_df, all_borrowing_log
