"""
数据清理模块

负责数据清理相关的功能，包括：
- 重复列处理
- 连续NaN值处理
- 零值处理
- 数据验证
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter

from dashboard.DFM.data_prep.utils.date_utils import filter_by_date_range


class DataCleaner:
    """数据清理器类"""
    
    def __init__(self):
        self.removed_variables_log = []
    
    def remove_duplicate_columns(self, df: pd.DataFrame, log_prefix: str = "") -> pd.DataFrame:
        """
        移除重复的列
        
        Args:
            df: 输入DataFrame
            log_prefix: 日志前缀
            
        Returns:
            pd.DataFrame: 移除重复列后的DataFrame
        """
        if df.empty:
            return df
            
        original_col_count = len(df.columns)
        duplicate_mask = df.columns.duplicated(keep='first')
        
        if duplicate_mask.any():
            removed_count = duplicate_mask.sum()
            removed_cols = df.columns[duplicate_mask].tolist()
            
            # 高效去除重复列
            df_cleaned = df.iloc[:, ~duplicate_mask]
            
            # 记录移除的列
            for col in removed_cols:
                self.removed_variables_log.append({
                    'Variable': col,
                    'Reason': f'{log_prefix}duplicate_column'
                })
            
            print(f"    {log_prefix}移除重复列: {original_col_count} → {len(df_cleaned.columns)} (减少了 {removed_count} 列)")
            return df_cleaned
        else:
            print(f"    {log_prefix}未发现重复列")
            return df
    
    def handle_consecutive_nans(
        self,
        df: pd.DataFrame,
        threshold: int,
        log_prefix: str = "",
        data_start_date: Optional[str] = None,
        data_end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        处理连续NaN值超过阈值的列

        Args:
            df: 输入DataFrame
            threshold: 连续NaN的阈值
            log_prefix: 日志前缀
            data_start_date: 用户选择的数据开始日期（可选）
            data_end_date: 用户选择的数据结束日期（可选）

        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        if df.empty or threshold <= 0:
            return df

        # 如果指定了时间范围，使用增强的筛选逻辑
        if data_start_date or data_end_date:
            return self._handle_consecutive_nans_with_time_range(
                df, threshold, log_prefix, data_start_date, data_end_date
            )

        # 否则使用原有逻辑
        return self._handle_consecutive_nans_original(df, threshold, log_prefix)

    def _handle_consecutive_nans_with_time_range(
        self,
        df: pd.DataFrame,
        threshold: int,
        log_prefix: str,
        data_start_date: Optional[str],
        data_end_date: Optional[str]
    ) -> pd.DataFrame:
        """基于用户选择时间范围的连续NaN处理"""

        # 应用时间范围筛选
        df_filtered = self._apply_time_range_filter(df, data_start_date, data_end_date)

        if df_filtered.empty:
            print(f"  {log_prefix}警告: 时间范围筛选后数据为空，使用原有逻辑")
            return self._handle_consecutive_nans_original(df, threshold, log_prefix)

        print(f"  {log_prefix}基于用户选择时间范围检查连续缺失值 (阈值 >= {threshold})...")
        print(f"  {log_prefix}时间范围: {data_start_date} 到 {data_end_date}")
        print(f"  {log_prefix}筛选后数据形状: {df_filtered.shape} (原始: {df.shape})")

        cols_to_remove = []

        for col in df_filtered.columns:
            series = df_filtered[col]
            # 使用新方法获取最大连续NaN及其时间段
            max_consecutive_nan, nan_start_date, nan_end_date = self._find_max_consecutive_nan_period(series)

            if max_consecutive_nan >= threshold:
                cols_to_remove.append(col)

                # 计算时间范围内的数据质量统计
                total_points = len(series)
                missing_points = series.isnull().sum()
                missing_ratio = missing_points / total_points * 100 if total_points > 0 else 0

                # 格式化缺失时间段
                if nan_start_date is not None and nan_end_date is not None:
                    # 转换为字符串格式
                    nan_period_str = f"{nan_start_date} to {nan_end_date}"
                    print(f"    {log_prefix}标记移除变量: '{col}' "
                          f"(最大连续NaN: {max_consecutive_nan} >= {threshold}, "
                          f"缺失时间段: {nan_period_str}, "
                          f"缺失率: {missing_ratio:.1f}%)")
                else:
                    nan_period_str = "未知"
                    print(f"    {log_prefix}标记移除变量: '{col}' "
                          f"(时间范围内最大连续 NaN: {max_consecutive_nan} >= {threshold}, "
                          f"缺失率: {missing_ratio:.1f}%)")

                self.removed_variables_log.append({
                    'Variable': col,
                    'Reason': f'{log_prefix}consecutive_nan_in_time_range',
                    'Details': {
                        'time_range': f"{data_start_date} to {data_end_date}",
                        'max_consecutive_nan': max_consecutive_nan,
                        'nan_start_date': str(nan_start_date) if nan_start_date else None,
                        'nan_end_date': str(nan_end_date) if nan_end_date else None,
                        'nan_period': nan_period_str,
                        'threshold': threshold,
                        'total_points_in_range': total_points,
                        'missing_points_in_range': missing_points,
                        'missing_ratio_in_range': missing_ratio
                    }
                })

        if cols_to_remove:
            print(f"    {log_prefix}正在移除 {len(cols_to_remove)} 个在选定时间范围内连续缺失值超标的变量...")
            df_cleaned = df.drop(columns=cols_to_remove)
            print(f"      {log_prefix}移除后 Shape: {df_cleaned.shape}")
            return df_cleaned
        else:
            print(f"    {log_prefix}所有变量在选定时间范围内的连续缺失值均低于阈值。")
            return df

    def _handle_consecutive_nans_original(
        self,
        df: pd.DataFrame,
        threshold: int,
        log_prefix: str
    ) -> pd.DataFrame:
        """原有的连续NaN处理逻辑"""
        print(f"  {log_prefix}开始检查连续缺失值 (阈值 >= {threshold})...")
        cols_to_remove = []

        for col in df.columns:
            series = df[col]
            first_valid_idx = series.first_valid_index()

            if first_valid_idx is None:
                continue  # 跳过全为NaN的列

            series_after_first_valid = series.loc[first_valid_idx:]
            is_na = series_after_first_valid.isna()
            na_blocks = is_na.ne(is_na.shift()).cumsum()[is_na]
            max_consecutive_nan = 0

            if not na_blocks.empty:
                try:
                    block_counts = na_blocks.value_counts()
                    if not block_counts.empty:
                        max_consecutive_nan = block_counts.max()
                except Exception as e_nan_count:
                     print(f"    {log_prefix}警告: 计算 '{col}' 的 NaN 块时出错: {e_nan_count}. 跳过此列检查.")
                     continue

            if max_consecutive_nan >= threshold:
                cols_to_remove.append(col)
                print(f"    {log_prefix}标记移除变量: '{col}' (最大连续 NaN: {max_consecutive_nan} >= {threshold})")
                self.removed_variables_log.append({
                    'Variable': col,
                    'Reason': f'{log_prefix}consecutive_nan'
                })

        if cols_to_remove:
            print(f"    {log_prefix}正在移除 {len(cols_to_remove)} 个连续缺失值超标的变量...")
            df_cleaned = df.drop(columns=cols_to_remove)
            print(f"      {log_prefix}移除后 Shape: {df_cleaned.shape}")
            return df_cleaned
        else:
            print(f"    {log_prefix}所有变量的连续缺失值均低于阈值。")
            return df

    def _apply_time_range_filter(
        self,
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """应用时间范围筛选"""
        return filter_by_date_range(df, start_date, end_date)

    def _calculate_max_consecutive_nans(self, series: pd.Series) -> int:
        """计算最大连续NaN数量"""
        if series.empty:
            return 0

        # 找到第一个有效值
        first_valid_idx = series.first_valid_index()
        if first_valid_idx is None:
            return len(series)  # 全为NaN

        # 从第一个有效值开始分析
        series_after_first_valid = series.loc[first_valid_idx:]
        is_na = series_after_first_valid.isna()

        if not is_na.any():
            return 0  # 没有NaN

        # 计算连续NaN块
        na_blocks = is_na.ne(is_na.shift()).cumsum()[is_na]
        max_consecutive_nan = 0

        if not na_blocks.empty:
            try:
                block_counts = na_blocks.value_counts()
                if not block_counts.empty:
                    max_consecutive_nan = block_counts.max()
            except Exception as e:
                print(f"    警告: 计算连续NaN时出错: {e}")
                return 0

        return max_consecutive_nan

    def _find_max_consecutive_nan_period(self, series: pd.Series) -> tuple:
        """
        找到最大连续NaN块的起止日期

        Returns:
            tuple: (max_consecutive_nan, start_date, end_date)
        """
        if series.empty:
            return 0, None, None

        # 找到第一个有效值
        first_valid_idx = series.first_valid_index()
        if first_valid_idx is None:
            # 全为NaN
            if len(series) > 0:
                return len(series), series.index[0], series.index[-1]
            return 0, None, None

        # 从第一个有效值开始分析
        series_after_first_valid = series.loc[first_valid_idx:]
        is_na = series_after_first_valid.isna()

        if not is_na.any():
            return 0, None, None  # 没有NaN

        # 计算连续NaN块
        na_blocks = is_na.ne(is_na.shift()).cumsum()[is_na]

        if na_blocks.empty:
            return 0, None, None

        try:
            # 找到最大连续NaN块的块ID
            block_counts = na_blocks.value_counts()
            if block_counts.empty:
                return 0, None, None

            max_consecutive_nan = block_counts.max()
            max_block_id = block_counts.idxmax()

            # 找到该块的所有索引
            max_block_indices = na_blocks[na_blocks == max_block_id].index

            if len(max_block_indices) > 0:
                start_date = max_block_indices[0]
                end_date = max_block_indices[-1]
                return max_consecutive_nan, start_date, end_date
            else:
                return max_consecutive_nan, None, None

        except Exception as e:
            print(f"    警告: 查找最大连续NaN时间段时出错: {e}")
            return 0, None, None

    def clean_zero_values(self, df: pd.DataFrame, log_prefix: str = "") -> pd.DataFrame:
        """
        将0值替换为NaN
        
        Args:
            df: 输入DataFrame
            log_prefix: 日志前缀
            
        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        if df.empty:
            return df
            
        df_cleaned = df.replace(0, np.nan)
        print(f"      {log_prefix}将 0 值替换为 NaN。")
        return df_cleaned
    
    def remove_unnamed_columns(self, df: pd.DataFrame, log_prefix: str = "") -> pd.DataFrame:
        """
        移除Unnamed列
        
        Args:
            df: 输入DataFrame
            log_prefix: 日志前缀
            
        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        if df.empty:
            return df
            
        unnamed_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('Unnamed:')]
        
        if unnamed_cols:
            print(f"      {log_prefix}发现并移除 Unnamed 列: {unnamed_cols}")
            df_cleaned = df.drop(columns=unnamed_cols)
            
            # 记录移除的列
            for col in unnamed_cols:
                self.removed_variables_log.append({
                    'Variable': col,
                    'Reason': f'{log_prefix}unnamed_column'
                })
            
            return df_cleaned
        
        return df
    
    def remove_all_nan_columns(self, df: pd.DataFrame, log_prefix: str = "") -> pd.DataFrame:
        """
        移除全为NaN的列

        Args:
            df: 输入DataFrame
            log_prefix: 日志前缀

        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        if df.empty:
            return df

        cols_before = set(df.columns)
        df_cleaned = df.dropna(axis=1, how='all')
        removed_cols = cols_before - set(df_cleaned.columns)

        if removed_cols:
            print(f"  {log_prefix}移除了 {len(removed_cols)} 个全 NaN 列: {list(removed_cols)[:10]}{'...' if len(removed_cols)>10 else ''}")

            # 添加调试信息，检查关键变量
            debug_vars = [
                '中国:日产量:尿素',
                '中国:开工率:聚丙烯下游:塑编',
                '中国:开工率:聚丙烯下游:双向拉伸聚丙烯薄膜',
                '中国:开工率:聚丙烯下游:共聚注塑',
                '中国:开工率:纯苯:下游行业'
            ]
            for var in debug_vars:
                if var in removed_cols:
                    print(f"    [DEBUG] 关键变量被移除: '{var}'")
                    if var in df.columns:
                        non_nan_count = df[var].notna().sum()
                        total_count = len(df[var])
                        print(f"      非NaN值数量: {non_nan_count}/{total_count}")
                        if non_nan_count > 0:
                            print(f"      前5个非NaN值索引: {df[var].dropna().head().index.tolist()}")

            # 记录移除的列
            for col in removed_cols:
                self.removed_variables_log.append({
                    'Variable': col,
                    'Reason': f'{log_prefix}all_nan'
                })

        return df_cleaned
    
    def remove_all_nan_rows(self, df: pd.DataFrame, log_prefix: str = "") -> pd.DataFrame:
        """
        移除全为NaN的行
        
        Args:
            df: 输入DataFrame
            log_prefix: 日志前缀
            
        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        if df.empty:
            return df
            
        original_rows = df.shape[0]
        df_cleaned = df.dropna(how='all')
        
        if df_cleaned.shape[0] < original_rows:
            print(f"    {log_prefix}移除全NaN行: {original_rows} → {df_cleaned.shape[0]} 行")
        
        return df_cleaned
    
    def get_removed_variables_log(self) -> List[Dict]:
        """获取移除变量的日志"""
        return self.removed_variables_log.copy()
    
    def clear_log(self):
        """清空日志"""
        self.removed_variables_log.clear()

# 便利函数
def clean_dataframe(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    remove_zeros: bool = True,
    remove_unnamed: bool = True,
    remove_all_nan_cols: bool = True,
    remove_all_nan_rows: bool = True,
    consecutive_nan_threshold: Optional[int] = None,
    data_start_date: Optional[str] = None,
    data_end_date: Optional[str] = None,
    log_prefix: str = ""
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    一站式数据清理函数

    Args:
        df: 输入DataFrame
        remove_duplicates: 是否移除重复列
        remove_zeros: 是否将0值替换为NaN
        remove_unnamed: 是否移除Unnamed列
        remove_all_nan_cols: 是否移除全NaN列
        remove_all_nan_rows: 是否移除全NaN行
        consecutive_nan_threshold: 连续NaN阈值，None表示不检查
        data_start_date: 用户选择的数据开始日期（可选）
        data_end_date: 用户选择的数据结束日期（可选）
        log_prefix: 日志前缀

    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (清理后的DataFrame, 移除变量日志)
    """
    cleaner = DataCleaner()
    result_df = df.copy()
    
    if remove_zeros:
        result_df = cleaner.clean_zero_values(result_df, log_prefix)
    
    if remove_unnamed:
        result_df = cleaner.remove_unnamed_columns(result_df, log_prefix)
    
    if remove_duplicates:
        result_df = cleaner.remove_duplicate_columns(result_df, log_prefix)
    
    if consecutive_nan_threshold is not None:
        result_df = cleaner.handle_consecutive_nans(
            result_df, consecutive_nan_threshold, log_prefix, data_start_date, data_end_date
        )
    
    if remove_all_nan_cols:
        result_df = cleaner.remove_all_nan_columns(result_df, log_prefix)
    
    if remove_all_nan_rows:
        result_df = cleaner.remove_all_nan_rows(result_df, log_prefix)
    
    return result_df, cleaner.get_removed_variables_log()

# 导出的类和函数
__all__ = [
    'DataCleaner',
    'clean_dataframe'
]
