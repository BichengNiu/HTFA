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

from dashboard.models.DFM.prep.modules.data_cleaner import DataCleaner

class DataAligner:
    """数据对齐器类"""
    
    def __init__(self, target_freq: str = 'W-FRI'):
        self.target_freq = target_freq
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
        
        print(f"  目标变量对齐到最近周五...")
        
        temp_target_df = pd.DataFrame({'value': target_values})
        
        # 计算每个发布日期的最近周五
        # 周一到周五: 对齐到本周五 (days=0 to 4)
        # 周六日: 对齐到本周五 (days=-1, -2)
        # 注意: Python的weekday()是 周一=0, 周二=1, ..., 周五=4, 周六=5, 周日=6
        def get_nearest_friday(dt):
            weekday = dt.weekday()
            if weekday <= 4:  # 周一到周五
                days_to_friday = 4 - weekday
            else:  # 周六日
                days_to_friday = -(weekday - 4)
            return dt + pd.Timedelta(days=days_to_friday)

        temp_target_df['nearest_friday'] = temp_target_df.index.map(get_nearest_friday)
        
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
        
        # 应用与目标变量相同的最近周五逻辑
        def get_nearest_friday(dt):
            weekday = dt.weekday()
            if weekday <= 4:  # 周一到周五
                days_to_friday = 4 - weekday
            else:  # 周六日
                days_to_friday = -(weekday - 4)
            return dt + pd.Timedelta(days=days_to_friday)

        temp_df = predictors_df.copy()
        temp_df['nearest_friday'] = temp_df.index.map(get_nearest_friday)
        
        # 处理同一个目标周五的重复：保留最新发布日期的数据
        aligned_predictors = temp_df.sort_index(ascending=True).groupby('nearest_friday').last()
        aligned_predictors.index.name = 'Date'
        
        print(f"    目标 Sheet 预测变量对齐完成。Shape: {aligned_predictors.shape}")
        
        return aligned_predictors
    
    def align_monthly_to_last_friday(
        self,
        monthly_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        将月度数据对齐到发布日最近的周五（与周度数据对齐逻辑一致）

        Args:
            monthly_data: 月度数据DataFrame，索引为发布日期

        Returns:
            pd.DataFrame: 对齐后的数据
        """
        if monthly_data is None or monthly_data.empty:
            raise ValueError("月度数据为空，无法对齐到最近周五")

        print(f"  对齐月度预测变量到发布日最近的周五...")

        # 确保没有重复列
        monthly_data = self.cleaner.remove_duplicate_columns(monthly_data, "[月度对齐] ")

        # 使用与周度数据相同的最近周五对齐逻辑
        def get_nearest_friday(dt):
            weekday = dt.weekday()
            if weekday <= 4:  # 周一到周五
                days_to_friday = 4 - weekday
            else:  # 周六日
                days_to_friday = -(weekday - 4)
            return dt + pd.Timedelta(days=days_to_friday)

        temp_df = monthly_data.copy()
        temp_df['nearest_friday'] = temp_df.index.map(get_nearest_friday)

        # 处理同一个目标周五的重复：保留最新发布日期的数据
        monthly_aligned = temp_df.sort_index(ascending=True).groupby('nearest_friday').last()
        monthly_aligned.index.name = 'Date'

        print(f"    对齐到最近周五完成。Shape: {monthly_aligned.shape}")

        return monthly_aligned
    
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
        df_daily_weekly = df_daily_full.resample(self.target_freq).mean()
        print(f"    日度->周度(均值) 完成. Shape: {df_daily_weekly.shape}")
        return df_daily_weekly
    
    def align_weekly_data(
        self,
        weekly_data_list: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        对齐周度数据（使用最后值）

        Args:
            weekly_data_list: 周度数据DataFrame列表

        Returns:
            pd.DataFrame: 对齐后的周度数据
        """
        if not weekly_data_list:
            return pd.DataFrame()

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
            return pd.DataFrame()

        # 对齐到目标频率（周五）
        print("  对齐周度数据到周五...")

        # 关键修复：将0值替换为NaN，避免0值覆盖有效数据
        # 对于开工率、利用率等比率类指标，0表示有意义的停工状态
        # 但在周度对齐时，如果周三有非零数据，周五有0值
        # resample().last()会错误地用0覆盖有效数据
        # 因此先将0替换为NaN，让last()跳过NaN取到真实的非零值
        df_weekly_for_resample = df_weekly_full.replace(0, np.nan)

        df_weekly_aligned = df_weekly_for_resample.resample(self.target_freq).last()
        print(f"    周度->周度(对齐) 完成. Shape: {df_weekly_aligned.shape}")
        return df_weekly_aligned

    def convert_dekad_to_weekly(
        self,
        dekad_data_list: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        将旬度数据转换为周度数据（旬日：10、20、30日对应到所在周的周五）

        Args:
            dekad_data_list: 旬度数据DataFrame列表

        Returns:
            pd.DataFrame: 转换后的周度数据
        """
        if not dekad_data_list:
            return pd.DataFrame()

        print("  处理旬度数据...")

        # 首先确保每个DataFrame的索引是唯一的
        cleaned_dekad_list = []
        for i, df in enumerate(dekad_data_list):
            # 移除重复的索引（保留第一个）
            if df.index.duplicated().any():
                print(f"    警告：旬度数据 {i} 有重复索引，正在清理...")
                df = df[~df.index.duplicated(keep='first')]
            cleaned_dekad_list.append(df)

        # 合并所有旬度数据
        df_dekad_full = pd.concat(cleaned_dekad_list, axis=1, join='outer', sort=True)

        # 处理重复列
        df_dekad_full = self.cleaner.remove_duplicate_columns(df_dekad_full, "[旬度合并] ")

        if df_dekad_full.empty:
            print("    合并后的旬度数据为空。")
            return pd.DataFrame()

        # 过滤出旬日（10、20、30日）的数据
        print(f"    旬度数据原始索引范围: {df_dekad_full.index.min()} 到 {df_dekad_full.index.max()}")
        print(f"    旬度数据原始行数: {len(df_dekad_full)}")

        # 检查是否是旬日（10、20、30）
        dekad_days = df_dekad_full.index.day
        is_dekad_day = dekad_days.isin([10, 20, 30])

        if not is_dekad_day.any():
            raise ValueError("旬度数据中没有旬日（10、20、30日），数据格式不正确")

        df_dekad_filtered = df_dekad_full[is_dekad_day]
        print(f"    过滤后旬日数据行数: {len(df_dekad_filtered)}")

        # 将旬日映射到所在周的周五
        print("  映射旬度数据到所在周的周五...")
        df_temp = df_dekad_filtered.copy()

        # 计算每个日期对应的周五
        # 周一到周五: 对应到本周的周五
        # 周六日: 对应到本周的周五（而非下周）
        def get_week_friday(date):
            weekday = date.weekday()  # 周一=0, ..., 周五=4, 周六=5, 周日=6
            if weekday <= 4:  # 周一到周五
                days_to_friday = 4 - weekday
            else:  # 周六日
                days_to_friday = -(weekday - 4)
            return date + pd.Timedelta(days=days_to_friday)

        df_temp['target_friday'] = df_temp.index.map(get_week_friday)

        # 按目标周五分组，保留最后一个值（如果一周内有多个旬日）
        df_dekad_weekly = df_temp.groupby('target_friday').last()
        df_dekad_weekly.index.name = 'Date'

        # 对齐到目标频率
        df_dekad_weekly = df_dekad_weekly.resample(self.target_freq).last()

        print(f"    旬度->周度 完成. Shape: {df_dekad_weekly.shape}")
        print(f"    转换后周度数据索引范围: {df_dekad_weekly.index.min()} 到 {df_dekad_weekly.index.max()}")
        return df_dekad_weekly
    
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

# 导出的类
__all__ = [
    'DataAligner'
]
