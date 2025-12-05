"""
数据对齐模块

负责将不同频率的数据对齐到目标频率（通常是周度）
包括：
- 目标变量对齐到最近周五
- 月度数据对齐到月末最后周五
- 日度数据转换为周度
- 周度数据对齐
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from calendar import monthrange

from dashboard.models.DFM.prep.modules.data_cleaner import DataCleaner

class DataAligner:
    """数据对齐器类"""

    def __init__(self, target_freq: str = 'W-FRI', enable_borrowing: bool = True):
        self.target_freq = target_freq
        self.enable_borrowing = enable_borrowing
        self.cleaner = DataCleaner()

        # 验证目标频率
        if not target_freq.upper().endswith('-FRI'):
            raise ValueError(f"当前目标对齐逻辑仅支持周五 (W-FRI)。提供的目标频率 '{target_freq}' 无效。")

    @staticmethod
    def _get_nearest_friday(dt):
        """
        计算给定日期的最近周五

        规则：
        - 周一到周五: 对齐到本周五
        - 周六日: 对齐到上周五

        Args:
            dt: 日期对象

        Returns:
            pd.Timestamp: 最近的周五日期
        """
        weekday = dt.weekday()
        if weekday <= 4:  # 周一到周五
            days_to_friday = 4 - weekday
        else:  # 周六日
            days_to_friday = -(weekday - 4)
        return dt + pd.Timedelta(days=days_to_friday)

    @staticmethod
    def _get_monthly_friday(dt):
        """
        计算给定日期所属月份内的周五（不跨月）

        规则：
        - 首先计算最近周五
        - 如果跨到下个月，返回当月最后一个周五
        - 如果跨到上个月，返回当月第一个周五

        Args:
            dt: 日期对象

        Returns:
            pd.Timestamp: 当月内的周五日期
        """
        year = dt.year
        month = dt.month
        weekday = dt.weekday()

        # 计算最近周五
        if weekday <= 4:  # 周一到周五
            days_to_friday = 4 - weekday
        else:  # 周六日
            days_to_friday = -(weekday - 4)

        target_friday = dt + pd.Timedelta(days=days_to_friday)

        # 检查是否跨月
        if target_friday.month != month:
            if target_friday.month > month or (target_friday.month == 1 and month == 12):
                # 跨到下个月 -> 当月最后一个周五
                last_day = monthrange(year, month)[1]
                last_date = pd.Timestamp(year, month, last_day)
                last_wd = last_date.weekday()
                if last_wd >= 4:  # 周五、周六、周日
                    days_back = last_wd - 4
                else:  # 周一到周四
                    days_back = last_wd + 3
                target_friday = last_date - pd.Timedelta(days=days_back)
            else:
                # 跨到上个月 -> 当月第一个周五
                first_date = pd.Timestamp(year, month, 1)
                first_wd = first_date.weekday()
                if first_wd <= 4:  # 周一到周五
                    days_forward = 4 - first_wd
                else:  # 周六日
                    days_forward = 11 - first_wd
                target_friday = first_date + pd.Timedelta(days=days_forward)

        return target_friday

    @staticmethod
    def _get_quarterly_friday(dt):
        """
        计算给定日期所属季度内的周五（不跨季）

        规则：
        - 首先计算最近周五
        - 如果跨到下个季度，返回当季最后一个周五
        - 如果跨到上个季度，返回当季第一个周五

        Args:
            dt: 日期对象

        Returns:
            pd.Timestamp: 当季内的周五日期
        """
        year = dt.year
        month = dt.month
        quarter = (month - 1) // 3 + 1  # 1, 2, 3, 4
        quarter_start_month = (quarter - 1) * 3 + 1  # 1, 4, 7, 10
        quarter_end_month = quarter * 3  # 3, 6, 9, 12

        weekday = dt.weekday()

        # 计算最近周五
        if weekday <= 4:
            days_to_friday = 4 - weekday
        else:
            days_to_friday = -(weekday - 4)

        target_friday = dt + pd.Timedelta(days=days_to_friday)
        target_quarter = (target_friday.month - 1) // 3 + 1

        # 检查是否跨季
        if target_friday.year != year or target_quarter != quarter:
            # 判断是跨到下季还是上季
            if (target_friday.year > year) or (target_friday.year == year and target_quarter > quarter):
                # 跨到下个季度 -> 当季最后一个周五
                last_day = monthrange(year, quarter_end_month)[1]
                last_date = pd.Timestamp(year, quarter_end_month, last_day)
                last_wd = last_date.weekday()
                if last_wd >= 4:
                    days_back = last_wd - 4
                else:
                    days_back = last_wd + 3
                target_friday = last_date - pd.Timedelta(days=days_back)
            else:
                # 跨到上个季度 -> 当季第一个周五
                first_date = pd.Timestamp(year, quarter_start_month, 1)
                first_wd = first_date.weekday()
                if first_wd <= 4:
                    days_forward = 4 - first_wd
                else:
                    days_forward = 11 - first_wd
                target_friday = first_date + pd.Timedelta(days=days_forward)

        return target_friday

    @staticmethod
    def _get_yearly_friday(dt):
        """
        计算给定日期所属年份内的周五（不跨年）

        规则：
        - 首先计算最近周五
        - 如果跨到下一年，返回当年最后一个周五
        - 如果跨到上一年，返回当年第一个周五

        Args:
            dt: 日期对象

        Returns:
            pd.Timestamp: 当年内的周五日期
        """
        year = dt.year
        weekday = dt.weekday()

        # 计算最近周五
        if weekday <= 4:
            days_to_friday = 4 - weekday
        else:
            days_to_friday = -(weekday - 4)

        target_friday = dt + pd.Timedelta(days=days_to_friday)

        # 检查是否跨年
        if target_friday.year != year:
            if target_friday.year > year:
                # 跨到下一年 -> 当年最后一个周五（12月31日前的最后周五）
                last_date = pd.Timestamp(year, 12, 31)
                last_wd = last_date.weekday()
                if last_wd >= 4:
                    days_back = last_wd - 4
                else:
                    days_back = last_wd + 3
                target_friday = last_date - pd.Timedelta(days=days_back)
            else:
                # 跨到上一年 -> 当年第一个周五
                first_date = pd.Timestamp(year, 1, 1)
                first_wd = first_date.weekday()
                if first_wd <= 4:
                    days_forward = 4 - first_wd
                else:
                    days_forward = 11 - first_wd
                target_friday = first_date + pd.Timedelta(days=days_forward)

        return target_friday

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
        
        print(f"  目标变量对齐到最近周五...")

        temp_target_df = pd.DataFrame({'value': target_values})

        temp_target_df['nearest_friday'] = temp_target_df.index.map(self._get_nearest_friday)
        
        # 处理同一个目标周五的重复：保留最新发布日期的数据
        # 我们先按原始发布日期索引排序，然后分组并取最后一个
        target_series_aligned = temp_target_df.sort_index(ascending=True).groupby('nearest_friday')['value'].last()
        target_series_aligned.index.name = 'Date'
        target_series_aligned.name = target_values.name

        print(f"    目标变量对齐完成。Shape: {target_series_aligned.shape}")

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
        
        print(f"  目标 Sheet 预测变量对齐到最近周五...")
        
        # 确保没有重复列
        predictors_df = self.cleaner.remove_duplicate_columns(predictors_df, "[目标Sheet预测变量对齐] ")

        temp_df = predictors_df.copy()
        temp_df['nearest_friday'] = temp_df.index.map(self._get_nearest_friday)
        
        # 处理同一个目标周五的重复：保留最新发布日期的数据
        aligned_predictors = temp_df.sort_index(ascending=True).groupby('nearest_friday').last()
        aligned_predictors.index.name = 'Date'
        
        print(f"    目标 Sheet 预测变量对齐完成。Shape: {aligned_predictors.shape}")

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

        print(f"  对齐{freq_name}数据到所属周期内的周五（不跨期）...")

        # 确保没有重复列
        data = self.cleaner.remove_duplicate_columns(data, f"[{log_prefix}] ")

        # 使用指定的对齐函数，默认使用 _get_nearest_friday
        if align_func is None:
            align_func = self._get_nearest_friday

        temp_df = data.copy()
        temp_df['nearest_friday'] = temp_df.index.map(align_func)

        # 处理同一个目标周五的重复：保留最新发布日期的数据
        aligned = temp_df.sort_index(ascending=True).groupby('nearest_friday').last()
        aligned.index.name = 'Date'

        print(f"    对齐到周五完成。Shape: {aligned.shape}")

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
            align_func=self._get_monthly_friday
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
            align_func=self._get_quarterly_friday
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
            align_func=self._get_yearly_friday
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

        print("  处理日度数据...")

        # 首先确保每个DataFrame的索引是唯一的
        cleaned_daily_list = []
        for i, df in enumerate(daily_data_list):
            # 移除重复的索引（保留第一个）
            if df.index.duplicated().any():
                print(f"    警告：日度数据 {i} 有重复索引，正在清理...")
                df = df[~df.index.duplicated(keep='first')]
            cleaned_daily_list.append(df)

        # 合并所有日度数据
        df_daily_full = pd.concat(cleaned_daily_list, axis=1, join='outer', sort=True)

        # 处理重复列
        df_daily_full = self.cleaner.remove_duplicate_columns(df_daily_full, "[日度合并] ")

        if df_daily_full.empty:
            print("    合并后的日度数据为空。")
            return pd.DataFrame()

        # 转换为周度（均值聚合）
        print("  转换日度数据为周度（均值聚合）...")
        # 零值处理已在processor全局预处理中完成
        df_daily_weekly = df_daily_full.resample(self.target_freq).mean()
        print(f"    日度->周度(均值) 完成. Shape: {df_daily_weekly.shape}")
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

        print("  处理周度数据...")

        # 首先确保每个DataFrame的索引是唯一的
        cleaned_weekly_list = []
        for i, df in enumerate(weekly_data_list):
            # 移除重复的索引（保留第一个）
            if df.index.duplicated().any():
                print(f"    警告：周度数据 {i} 有重复索引，正在清理...")
                df = df[~df.index.duplicated(keep='first')]
            cleaned_weekly_list.append(df)

        # 合并所有周度数据
        df_weekly_full = pd.concat(cleaned_weekly_list, axis=1, join='outer', sort=True)

        # 处理重复列
        df_weekly_full = self.cleaner.remove_duplicate_columns(df_weekly_full, "[周度合并] ")

        if df_weekly_full.empty:
            print("    合并后的周度数据为空。")
            return pd.DataFrame(), borrowing_log

        # 生成理论日期索引（用户时间范围内的所有周五）
        if start_date and end_date:
            theoretical_index = generate_theoretical_index(start_date, end_date, self.target_freq)
            print(f"    理论日期索引: {theoretical_index[0].date()} 至 {theoretical_index[-1].date()}, 共 {len(theoretical_index)} 个周五")
        else:
            # 如果没有指定日期范围，使用数据本身的范围
            data_start = df_weekly_full.index.min()
            data_end = df_weekly_full.index.max()
            theoretical_index = generate_theoretical_index(data_start, data_end, self.target_freq)
            print(f"    [警告] 未指定日期范围，使用数据范围: {theoretical_index[0].date()} 至 {theoretical_index[-1].date()}")

        # 对齐到理论索引（包含借调逻辑）
        print("  对齐周度数据到理论索引...")
        df_weekly_aligned, borrowing_log = align_to_theoretical_index(
            df_weekly_full, theoretical_index, self.target_freq, self.enable_borrowing
        )

        print(f"    周度->周度(对齐) 完成. Shape: {df_weekly_aligned.shape}")
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

        print("  处理旬度数据...")

        # 首先确保每个DataFrame的索引是唯一的
        cleaned_dekad_list = []
        for i, df in enumerate(dekad_data_list):
            if df.index.duplicated().any():
                print(f"    警告：旬度数据 {i} 有重复索引，正在清理...")
                df = df[~df.index.duplicated(keep='first')]
            cleaned_dekad_list.append(df)

        # 合并所有旬度数据
        df_dekad_full = pd.concat(cleaned_dekad_list, axis=1, join='outer', sort=True)
        df_dekad_full = self.cleaner.remove_duplicate_columns(df_dekad_full, "[旬度合并] ")

        if df_dekad_full.empty:
            print("    合并后的旬度数据为空。")
            return pd.DataFrame(), borrowing_log

        # 过滤出旬日（10日、20日、月末）的数据
        print(f"    旬度数据原始索引范围: {df_dekad_full.index.min()} 到 {df_dekad_full.index.max()}")
        is_dekad_day = df_dekad_full.index.map(self._is_dekad_date)

        if not is_dekad_day.any():
            raise ValueError("旬度数据中没有旬日（10日、20日、月末），数据格式不正确")

        df_dekad_filtered = df_dekad_full[is_dekad_day]
        print(f"    过滤后旬日数据行数: {len(df_dekad_filtered)}")

        # 将旬日映射到对应的周五
        print("  映射旬度数据到周五...")
        df_temp = df_dekad_filtered.copy()
        df_temp['target_friday'] = df_temp.index.map(self._get_dekad_friday)

        # 按目标周五分组，保留最后一个值
        df_dekad_mapped = df_temp.groupby('target_friday').last()
        df_dekad_mapped.index.name = 'Date'

        # 生成理论周度索引
        if start_date and end_date:
            theoretical_index = generate_theoretical_index(start_date, end_date, self.target_freq)
            print(f"    理论日期索引: {theoretical_index[0].date()} 至 {theoretical_index[-1].date()}, 共 {len(theoretical_index)} 个周五")
        else:
            data_start = df_dekad_mapped.index.min()
            data_end = df_dekad_mapped.index.max()
            theoretical_index = generate_theoretical_index(data_start, data_end, self.target_freq)
            print(f"    [警告] 未指定日期范围，使用数据范围")

        # 对齐到理论索引（包含借调逻辑）
        # 注意：旬度数据本来就稀疏（每月3个点），借调可能不太适用
        # 但仍然使用相同的逻辑以保持一致性
        print("  对齐旬度数据到理论索引...")
        df_dekad_aligned, borrowing_log = align_to_theoretical_index(
            df_dekad_mapped, theoretical_index, self.target_freq, self.enable_borrowing
        )

        print(f"    旬度->周度 完成. Shape: {df_dekad_aligned.shape}")
        return df_dekad_aligned, borrowing_log

    @staticmethod
    def _get_dekad_friday(date):
        """将旬日对齐到最近的周五（不跨月）

        规则：
        - 周一：上周五更近
        - 周二到周五：本周五更近
        - 周六日：上周五更近
        - 如果对齐后跨月，则调整到当月最后/第一个周五
        """
        year = date.year
        month = date.month
        weekday = date.weekday()

        if weekday == 0:  # 周一
            days_to_friday = -3
        elif weekday <= 4:  # 周二到周五
            days_to_friday = 4 - weekday
        else:  # 周六日
            days_to_friday = -(weekday - 4)

        target_friday = date + pd.Timedelta(days=days_to_friday)

        # 检查是否跨月
        if target_friday.month != month:
            if target_friday.month > month or (target_friday.month == 1 and month == 12):
                # 跨到下个月 -> 当月最后一个周五
                last_day = monthrange(year, month)[1]
                last_date = pd.Timestamp(year, month, last_day)
                last_wd = last_date.weekday()
                if last_wd >= 4:
                    days_back = last_wd - 4
                else:
                    days_back = last_wd + 3
                target_friday = last_date - pd.Timedelta(days=days_back)
            else:
                # 跨到上个月 -> 当月第一个周五
                first_date = pd.Timestamp(year, month, 1)
                first_wd = first_date.weekday()
                if first_wd <= 4:
                    days_forward = 4 - first_wd
                else:
                    days_forward = 11 - first_wd
                target_friday = first_date + pd.Timedelta(days=days_forward)

        return target_friday

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
        
        print(f"  所有原始数据中的最小/最大日期: {min_date_orig.date()} / {max_date_orig.date()}")
        
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
            print(f"  警告：计算出的实际开始日期 ({actual_range_start.date()}) 晚于结束日期 ({actual_range_end.date()})，使用原始数据范围。")
            actual_range_start = min_date_fri
            actual_range_end = max_date_fri
        
        full_date_range = pd.date_range(start=actual_range_start, end=actual_range_end, freq=self.target_freq)
        print(f"  最终确定的完整周度日期范围 (对齐到 {self.target_freq}): {full_date_range.min().date()} 到 {full_date_range.max().date()}")
        
        return full_date_range

# ============================================================================
# 借调逻辑（从Preview模块移植，支持周度和旬度）
# ============================================================================

def _generate_weekly_boundaries(min_date, max_date):
    """生成周度时间窗口边界

    Args:
        min_date: 最小日期
        max_date: 最大日期

    Returns:
        pd.DatetimeIndex: 周度边界时间点
    """
    start_of_first_week = min_date - pd.Timedelta(days=min_date.weekday() + 2)
    end_extended = max_date + pd.Timedelta(days=14)
    boundary_times = pd.date_range(start=start_of_first_week, end=end_extended, freq='W-FRI')
    # 过滤出与数据范围相关的边界
    boundary_times = boundary_times[(boundary_times >= min_date - pd.Timedelta(days=7)) &
                                   (boundary_times <= max_date + pd.Timedelta(days=7))]
    return boundary_times


def _generate_tenday_boundaries(min_date, max_date):
    """生成旬度时间窗口边界

    Args:
        min_date: 最小日期
        max_date: 最大日期

    Returns:
        pd.DatetimeIndex: 旬度边界时间点
    """
    from calendar import monthrange
    boundary_times = []

    # 从数据最小日期的前一个月开始，到最大日期的后一个月结束
    start_month = min_date.replace(day=1)
    if start_month.month == 1:
        start_month = start_month.replace(year=start_month.year - 1, month=12)
    else:
        start_month = start_month.replace(month=start_month.month - 1)

    end_month = max_date.replace(day=1)
    if end_month.month == 12:
        end_month = end_month.replace(year=end_month.year + 1, month=1)
    else:
        end_month = end_month.replace(month=end_month.month + 1)

    current_date = start_month
    while current_date <= end_month:
        for day in [10, 20]:
            try:
                boundary_date = current_date.replace(day=day)
                boundary_times.append(boundary_date)
            except ValueError:
                pass

        # 月末
        try:
            last_day = monthrange(current_date.year, current_date.month)[1]
            month_end = current_date.replace(day=last_day)
            boundary_times.append(month_end)
        except ValueError:
            pass

        # 移动到下个月
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)

    return pd.DatetimeIndex(boundary_times).sort_values()


def _get_window_bounds(i, boundary_times, target_freq, min_date):
    """获取时间窗口的开始和结束时间

    Args:
        i: 边界索引
        boundary_times: 边界时间序列
        target_freq: 频率类型
        min_date: 最小日期

    Returns:
        tuple: (current_start, current_end, next_start, next_end)
    """
    current_boundary = boundary_times[i]
    next_boundary = boundary_times[i+1]

    if target_freq == 'W-FRI':
        current_window_start = current_boundary - pd.Timedelta(days=6)
        current_window_end = current_boundary
        next_window_start = next_boundary - pd.Timedelta(days=6)
        next_window_end = next_boundary
    elif target_freq == 'ten_day':
        if i == 0:
            current_window_start = min_date
        else:
            current_window_start = boundary_times[i-1] + pd.Timedelta(days=1)
        current_window_end = current_boundary
        next_window_start = current_boundary + pd.Timedelta(days=1)
        next_window_end = next_boundary
    else:
        return None, None, None, None

    return current_window_start, current_window_end, next_window_start, next_window_end


def _precompute_windows(boundary_times, target_freq, min_date):
    """预计算所有窗口边界，避免循环内重复计算

    Args:
        boundary_times: 边界时间序列
        target_freq: 频率类型
        min_date: 最小日期

    Returns:
        list: 窗口边界字典列表
    """
    windows = []
    for i in range(len(boundary_times) - 1):
        window = _get_window_bounds(i, boundary_times, target_freq, min_date)
        if window[0] is not None:
            windows.append({
                'current_start': window[0],
                'current_end': window[1],
                'next_start': window[2],
                'next_end': window[3]
            })
    return windows


def _is_valid_borrow_target(series_data, target_time):
    """检查目标位置是否为有效的借调目标

    Args:
        series_data: 数据序列
        target_time: 目标时间

    Returns:
        bool: 是否可以借调到该位置
    """
    if target_time not in series_data.index:
        return False

    target_values = series_data.loc[target_time]
    if isinstance(target_values, pd.Series):
        # 如果是Series（重复索引），检查是否所有值都是NaN
        return target_values.isna().all()
    else:
        # 如果是标量
        return pd.isna(target_values)


def _precompute_window_data(series_data, windows):
    """预计算所有窗口的数据，避免重复切片（性能优化）

    Args:
        series_data: 列数据序列
        windows: 预计算的窗口边界列表

    Returns:
        list: 窗口数据列表，每项包含current、next、target_time、window_bounds
    """
    window_data_list = []

    for window in windows:
        current_mask = (series_data.index >= window['current_start']) & \
                      (series_data.index <= window['current_end'])
        next_mask = (series_data.index >= window['next_start']) & \
                   (series_data.index <= window['next_end'])

        window_data_list.append({
            'current': series_data[current_mask].dropna(),
            'next': series_data[next_mask].dropna(),
            'target_time': window['current_end'],
            'current_start': window['current_start'],
            'current_end': window['current_end']
        })

    return window_data_list


def _apply_borrowing_batch(series_data, window_data_list):
    """批量应用借调操作，减少索引操作（性能优化）

    Args:
        series_data: 列数据序列
        window_data_list: 预计算的窗口数据列表

    Returns:
        tuple: (更新后的序列, 借调次数, 借调详情列表)
    """
    updates = {}
    removes = []
    inserts = {}  # 新增：需要插入的新时间点
    borrowing_count = 0
    borrowing_log = []  # 记录借调详情

    for wd in window_data_list:
        current_data = wd['current']
        next_data = wd['next']
        target_time = wd['target_time']  # 这是周五边界

        # 借调条件：当前窗口无值 AND 下个窗口有2个或更多数据点
        if len(current_data) == 0 and len(next_data) >= 2:
            borrowed_value = next_data.iloc[0]
            borrowed_time = next_data.index[0]

            # 如果目标时间（周五）不在索引中，插入新的周五时间点
            if target_time not in series_data.index:
                # 直接插入到周五，不再找最接近的时间点
                inserts[target_time] = borrowed_value
                removes.append(borrowed_time)
                borrowing_count += 1
                borrowing_log.append({
                    'borrowed_from': borrowed_time,
                    'borrowed_to': target_time
                })
            elif _is_valid_borrow_target(series_data, target_time) and borrowed_time in series_data.index:
                # 目标时间在索引中且为NaN，直接更新
                updates[target_time] = borrowed_value
                removes.append(borrowed_time)
                borrowing_count += 1
                borrowing_log.append({
                    'borrowed_from': borrowed_time,
                    'borrowed_to': target_time
                })

    # 批量应用更新（一次性操作）
    if updates:
        series_data.update(pd.Series(updates))
    if removes:
        series_data.loc[removes] = np.nan

    # 插入新的时间点
    if inserts:
        insert_series = pd.Series(inserts)
        series_data = pd.concat([series_data, insert_series]).sort_index()
        # 处理可能的重复索引（取非NaN值）
        if series_data.index.duplicated().any():
            series_data = series_data.groupby(series_data.index).first()

    return series_data, borrowing_count, borrowing_log


def _apply_borrowing_to_column(series_data, windows, target_freq, col_name=""):
    """对单列应用借调逻辑（优化版：使用预计算和批量操作）

    Args:
        series_data: 列数据序列
        windows: 预计算的窗口边界列表
        target_freq: 频率类型
        col_name: 列名（用于日志）

    Returns:
        tuple: (应用借调后的序列, 借调日志列表)
    """
    series_data = series_data.copy()

    # 子任务1：预计算窗口数据（避免重复切片）
    window_data_list = _precompute_window_data(series_data, windows)

    # 子任务2：批量借调（减少索引操作）
    series_data, borrowing_count, borrowing_log = _apply_borrowing_batch(series_data, window_data_list)

    if borrowing_count > 0 and col_name:
        print(f"      [借调] {col_name}: {borrowing_count} 次")

    return series_data, borrowing_log


def _handle_duplicate_indices(df):
    """处理重复的时间索引

    Args:
        df: 输入DataFrame

    Returns:
        pd.DataFrame: 去重后的DataFrame
    """
    if not df.index.is_unique:
        duplicate_count = df.index.duplicated().sum()
        print(f"    [去重] 存在 {duplicate_count} 个重复时间戳，使用平均值合并")
        df = df.groupby(df.index).mean()
    return df


def apply_borrowing_logic(df_input, target_freq, start_date=None, end_date=None):
    """对时间序列数据应用"借调"逻辑（优化版）

    优化要点：
    1. 边界时间只生成一次（移到循环外）
    2. 预计算所有窗口边界
    3. 提取单列处理逻辑为独立函数

    借调逻辑：在原始数据的基础上，当某个时间窗口完全没有数据，但下一个时间窗口
    有2个或更多数据点时，将下一个窗口的第一个值复制到前一个窗口的合适位置。

    核心原则：
    1. 不进行频率转换，保持原始数据结构
    2. 总有效观测值数量严格不变（只是重新分布）
    3. 在原始时间序列中进行数据填充

    Args:
        df_input: 输入DataFrame，必须有DatetimeIndex
        target_freq: 目标频率标识 ('W-FRI' for weekly, 'ten_day' for ten-day)
        start_date: 用户设置的开始日期（可选，用于限制借调范围）
        end_date: 用户设置的结束日期（可选，用于限制借调范围）

    Returns:
        tuple: (应用借调逻辑后的DataFrame, 借调日志字典)
        借调日志格式: {变量名: [{'borrowed_from': datetime, 'borrowed_to': datetime}, ...]}
    """
    all_borrowing_log = {}  # 聚合所有列的借调日志

    if df_input.empty or not isinstance(df_input.index, pd.DatetimeIndex):
        return df_input, all_borrowing_log

    df_result = df_input.copy()

    # 如果指定了日期范围，只在该范围内进行借调
    if start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        min_date = max(df_result.index.min(), start_dt)
        max_date = min(df_result.index.max(), end_dt)
        print(f"    [借调] 使用用户设置的日期范围: {min_date.date()} 至 {max_date.date()}")
    else:
        min_date = df_result.index.min()
        max_date = df_result.index.max()

    try:
        if target_freq == 'W-FRI':
            boundary_times = _generate_weekly_boundaries(min_date, max_date)
        elif target_freq == 'ten_day':
            boundary_times = _generate_tenday_boundaries(min_date, max_date)
        else:
            print(f"    警告：不支持的频率 '{target_freq}'，跳过借调")
            return df_result, all_borrowing_log
    except Exception as e:
        print(f"    警告：生成边界时间失败: {e}")
        return df_result, all_borrowing_log

    if len(boundary_times) < 2:
        return df_result, all_borrowing_log

    # 优化点2：预计算窗口边界，避免循环内重复计算
    windows = _precompute_windows(boundary_times, target_freq, min_date)

    # 优化点3：对每列应用借调逻辑（使用独立函数）
    total_borrowing = 0
    all_new_indices = set()  # 收集所有新插入的时间点

    column_results = {}  # 存储每列的借调结果
    for col in df_result.columns:
        series_data = df_result[col]
        if series_data.dropna().empty:
            column_results[col] = series_data
            continue

        original_series = series_data.copy()
        result_series, borrowing_log = _apply_borrowing_to_column(
            series_data, windows, target_freq, col_name=col
        )
        column_results[col] = result_series

        # 收集新插入的时间点
        new_indices = set(result_series.index) - set(df_result.index)
        all_new_indices.update(new_indices)

        # 记录借调日志
        if borrowing_log:
            all_borrowing_log[col] = borrowing_log
            total_borrowing += 1

    # 如果有新插入的时间点，扩展 DataFrame 的索引
    if all_new_indices:
        new_index = df_result.index.union(pd.DatetimeIndex(list(all_new_indices))).sort_values()
        df_result = df_result.reindex(new_index)

    # 将借调结果更新回 DataFrame
    for col, result_series in column_results.items():
        df_result[col] = result_series

    if total_borrowing > 0:
        print(f"    借调完成: 共处理 {total_borrowing} 列")

    # 处理重复索引
    df_result = _handle_duplicate_indices(df_result)

    return df_result, all_borrowing_log


# 导出的类
__all__ = [
    'DataAligner',
    'apply_borrowing_logic',
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
            print(f"      [借调] {col_name}: {borrowing_count} 次")

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
        print(f"    借调完成: 共处理 {len(all_borrowing_log)} 列")

    return result_df, all_borrowing_log
