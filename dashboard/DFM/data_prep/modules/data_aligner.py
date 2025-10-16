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

from dashboard.DFM.data_prep.modules.stationarity_processor import ensure_stationarity
from dashboard.DFM.data_prep.modules.data_cleaner import DataCleaner

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
            return pd.Series(dtype=float)
        
        print(f"  目标变量对齐到最近周五...")
        
        temp_target_df = pd.DataFrame({'value': target_values})
        
        # 计算每个发布日期的最近周五
        # 如果是周一、周二、周三 -> 去到即将到来的周五 (4 - weekday)
        # 如果是周四、周五、周六、周日 -> 去到之前的周五 (4 - weekday)
        # 注意: Python的weekday()是 周一=0, 周二=1, ..., 周五=4, 周六=5, 周日=6
        temp_target_df['nearest_friday'] = temp_target_df.index.map(
            lambda dt: dt + pd.Timedelta(days=4 - dt.weekday())
        )
        
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
            return pd.DataFrame()
        
        print(f"  目标 Sheet 预测变量对齐到最近周五...")
        
        # 确保没有重复列
        predictors_df = self.cleaner.remove_duplicate_columns(predictors_df, "[目标Sheet预测变量对齐] ")
        
        # 应用与目标变量相同的最近周五逻辑
        temp_df = predictors_df.copy()
        temp_df['nearest_friday'] = temp_df.index.map(
            lambda dt: dt + pd.Timedelta(days=4 - dt.weekday())
        )
        
        # 处理同一个目标周五的重复：保留最新发布日期的数据
        aligned_predictors = temp_df.sort_index(ascending=True).groupby('nearest_friday').last()
        aligned_predictors.index.name = 'Date'
        
        print(f"    目标 Sheet 预测变量对齐完成。Shape: {aligned_predictors.shape}")
        
        return aligned_predictors
    
    def align_monthly_to_last_friday(
        self, 
        monthly_data: pd.DataFrame,
        apply_stationarity: bool = True,
        skip_cols: Optional[set] = None
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        将月度数据对齐到月末最后周五
        
        Args:
            monthly_data: 月度数据DataFrame，索引为发布日期
            apply_stationarity: 是否应用平稳性检查
            skip_cols: 跳过平稳性检查的列
            
        Returns:
            Tuple[pd.DataFrame, Dict, Dict]: (对齐后的数据, 转换日志, 移除列信息)
        """
        if monthly_data is None or monthly_data.empty:
            return pd.DataFrame(), {}, {}
        
        print(f"  聚合月度预测变量到月末 (取当月最后有效值)...")
        
        # 确保没有重复列
        monthly_data = self.cleaner.remove_duplicate_columns(monthly_data, "[月度对齐] ")
        
        # 聚合到月末
        df_monthly_for_stat = monthly_data.copy()
        df_monthly_for_stat = df_monthly_for_stat.resample('M').last()
        
        print(f"    聚合到月末完成。Shape: {df_monthly_for_stat.shape}")
        
        # 移除连续NaN检查（这里可以根据需要添加）
        # df_monthly_for_stat = self.cleaner.handle_consecutive_nans(df_monthly_for_stat, threshold)
        
        # 平稳性检查
        transform_log = {}
        removed_cols_info = {}
        
        if apply_stationarity and not df_monthly_for_stat.empty:
            print(f"  对月度预测变量进行平稳性检查...")
            df_monthly_stationary, transform_log, removed_cols_info = ensure_stationarity(
                df_monthly_for_stat,
                skip_cols=skip_cols,
                adf_p_threshold=0.05
            )
            print(f"    月度预测变量平稳性处理完成。处理后 Shape: {df_monthly_stationary.shape}")
        else:
            df_monthly_stationary = df_monthly_for_stat
        
        # 对齐到当月最后周五
        if not df_monthly_stationary.empty:
            print("  对齐处理后的月度预测变量到当月最后周五...")
            
            # 计算月末的最后周五
            df_monthly_stationary['last_friday'] = df_monthly_stationary.index.map(
                lambda dt: dt - pd.Timedelta(days=(dt.weekday() - 4 + 7) % 7)  # 回到最后一个周五
            )
            
            monthly_aligned = df_monthly_stationary.set_index('last_friday', drop=True)
            monthly_aligned.index.name = 'Date'
            
            # 处理潜在的重复（如果多个月末映射到同一个最后周五）
            # 保留最新月份的数据
            monthly_aligned = monthly_aligned[~monthly_aligned.index.duplicated(keep='last')]
            monthly_aligned = monthly_aligned.sort_index()
            
            print(f"    对齐到最后周五完成。 Shape: {monthly_aligned.shape}")
            
            return monthly_aligned, transform_log, removed_cols_info
        else:
            print("  没有月度预测变量可供对齐。")
            return pd.DataFrame(), transform_log, removed_cols_info
    
    def convert_daily_to_weekly(self, daily_data_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        将日度数据转换为周度数据（使用均值）
        
        Args:
            daily_data_list: 日度数据DataFrame列表
            
        Returns:
            pd.DataFrame: 转换后的周度数据
        """
        if not daily_data_list:
            return pd.DataFrame()
        
        print("  处理日度数据 -> 周度 (均值)...")
        
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
        df_daily_full = self.cleaner.remove_duplicate_columns(df_daily_full, "[日度转周度] ")
        
        if not df_daily_full.empty:
            # 转换为周度
            df_daily_weekly = df_daily_full.resample(self.target_freq).mean()
            print(f"    日度->周度(均值) 完成. Shape: {df_daily_weekly.shape}")
            return df_daily_weekly
        else:
            print("    合并后的日度数据为空，无法进行周度转换。")
            return pd.DataFrame()
    
    def align_weekly_data(self, weekly_data_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        对齐周度数据（使用最后值）
        
        Args:
            weekly_data_list: 周度数据DataFrame列表
            
        Returns:
            pd.DataFrame: 对齐后的周度数据
        """
        if not weekly_data_list:
            return pd.DataFrame()
        
        print("  处理周度数据 -> 周度 (最后值)...")
        
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
        df_weekly_full = self.cleaner.remove_duplicate_columns(df_weekly_full, "[周度对齐] ")
        
        if not df_weekly_full.empty:
            # 对齐到目标频率
            df_weekly_aligned = df_weekly_full.resample(self.target_freq).last()
            print(f"    周度->周度(对齐) 完成. Shape: {df_weekly_aligned.shape}")
            return df_weekly_aligned
        else:
            print("    合并后的周度数据为空，无法进行周度转换。")
            return pd.DataFrame()
    
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
            # 提供默认的日期范围而不是抛出异常
            print("  警告：所有收集到的索引都为空，使用默认日期范围。")
            default_start = pd.to_datetime('2010-01-01')
            default_end = pd.to_datetime('2025-12-31')
            return pd.date_range(start=default_start, end=default_end, freq=self.target_freq)
        
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
