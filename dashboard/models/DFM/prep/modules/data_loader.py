"""
数据加载模块

负责从Excel文件中加载不同类型的数据：
- 目标变量数据
- 日度/周度预测变量数据
- 月度预测变量数据
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict

from dashboard.models.DFM.prep.modules.format_detection import detect_sheet_format, parse_sheet_info
from dashboard.models.DFM.prep.modules.data_cleaner import DataCleaner
from dashboard.models.DFM.prep.utils.text_utils import normalize_text

class DataLoader:
    """数据加载器类"""

    def __init__(
        self,
        reference_industry_map: Optional[Dict[str, str]] = None,
        reference_frequency_map: Optional[Dict[str, str]] = None
    ):
        """
        初始化数据加载器

        Args:
            reference_industry_map: 从指标体系加载的行业映射（可选，用于校验）
            reference_frequency_map: 从指标体系加载的频率映射（用于数据分类）★新增★
        """
        self.cleaner = DataCleaner()
        self.var_industry_map = {}  # 从sheet名称推断的行业映射（保留用于校验）
        self.sheet_inferred_map = {}  # 明确记录sheet推断的行业
        self.reference_industry_map = reference_industry_map or {}
        self.reference_frequency_map = reference_frequency_map or {}
        self.raw_columns_across_all_sheets = set()
    
    def load_target_sheet(
        self, 
        excel_file, 
        sheet_name: str,
        target_variable_name: str,
        industry_name: str = "Macro"
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.DataFrame], Set[str]]:
        """
        加载目标表格数据
        
        Args:
            excel_file: Excel文件对象
            sheet_name: 表格名称
            target_variable_name: 目标变量名称
            industry_name: 行业名称
            
        Returns:
            Tuple: (发布日期, 目标变量值, 预测变量DataFrame, 目标表格列名集合)
        """
        print(f"      检测到目标 Sheet，行业: '{industry_name}'...")

        # 删除try-except包装，让异常直接抛出
        # 使用统一格式读取数据
        print(f"    [目标Sheet读取] 使用统一格式，第一行为列名，第一列为时间列")

        df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)

        # 清理数据
        df_raw = self.cleaner.clean_zero_values(df_raw, "[目标Sheet] ")

        if df_raw.shape[1] < 2:
            raise ValueError(f"目标 Sheet '{sheet_name}' 列数 < 2，数据格式不正确")

        # 提取日期和目标变量
        date_col_name = df_raw.columns[0]
        actual_target_variable_name = df_raw.columns[1]
        target_sheet_cols = {actual_target_variable_name}

        print(f"      确认目标变量 (B列): '{actual_target_variable_name}'")
        print(f"      解析发布日期 (A列: '{date_col_name}')...")

        publication_dates = pd.to_datetime(df_raw[date_col_name], errors='coerce')
        valid_date_mask = publication_dates.notna()

        # 数据质量检查
        valid_date_ratio = valid_date_mask.sum() / len(df_raw)
        if valid_date_ratio < 0.5:
            raise ValueError(f"日期解析质量不合格：仅 {valid_date_ratio:.1%} 的日期有效（要求≥50%）")

        if not valid_date_mask.any():
            raise ValueError(f"无法从列 '{date_col_name}' 解析任何有效日期")

        publication_dates = publication_dates[valid_date_mask]

        # 提取目标变量值
        print(f"      提取目标变量原始值...")
        target_values = pd.to_numeric(df_raw.loc[valid_date_mask, actual_target_variable_name], errors='coerce')
        target_values.index = publication_dates

        # 检查目标变量转换质量
        target_valid_ratio = target_values.notna().sum() / len(target_values)
        if target_valid_ratio < 0.5:
            raise ValueError(f"目标变量数值转换质量不合格：仅 {target_valid_ratio:.1%} 的值有效（要求≥50%）")

        # 更新映射（同时记录sheet推断的行业用于后续校验）
        norm_target_name = normalize_text(actual_target_variable_name)
        self.var_industry_map[norm_target_name] = industry_name
        self.sheet_inferred_map[norm_target_name] = industry_name

        # 提取月度预测变量 (C列及以后)
        target_sheet_predictors = pd.DataFrame()
        if df_raw.shape[1] > 2:
            print(f"      提取目标 Sheet 的月度预测变量 (C列及以后)...")

            # 移除Unnamed列
            df_raw = self.cleaner.remove_unnamed_columns(df_raw, "[目标Sheet预测变量] ")

            temp_monthly_predictors = {}
            for col_idx in range(2, df_raw.shape[1]):
                col_name = df_raw.columns[col_idx]
                target_sheet_cols.add(col_name)

                # 清理值
                cleaned_series = df_raw[col_name].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.strip()
                predictor_values = pd.to_numeric(cleaned_series, errors='coerce')

                # 检查预测变量转换质量
                if valid_date_mask.any():
                    pred_valid_ratio = predictor_values[valid_date_mask].notna().sum() / valid_date_mask.sum()
                    if pred_valid_ratio < 0.3:
                        print(f"        警告: 变量 '{col_name}' 数值转换质量较低 ({pred_valid_ratio:.1%})")

                # 创建按发布日期索引的序列
                temp_monthly_predictors[col_name] = pd.Series(
                    predictor_values[valid_date_mask].values,
                    index=publication_dates
                )

                # 更新映射和跟踪
                norm_pred_col = normalize_text(col_name)
                if norm_pred_col:
                    self.var_industry_map[norm_pred_col] = industry_name
                    self.sheet_inferred_map[norm_pred_col] = industry_name
                    self.raw_columns_across_all_sheets.add(norm_pred_col)

            target_sheet_predictors = pd.DataFrame(temp_monthly_predictors).sort_index()
            target_sheet_predictors = target_sheet_predictors.dropna(axis=1, how='all')

            print(f"      提取了 {target_sheet_predictors.shape[1]} 个有效的月度预测变量 (按发布日期索引)。")
        else:
            print(f"      目标 Sheet 仅含 A, B 列。")

        return publication_dates, target_values, target_sheet_predictors, target_sheet_cols
    
    def load_daily_weekly_sheet(
        self,
        excel_file,
        sheet_name: str,
        freq_type: str,
        industry_name: str
    ) -> Optional[pd.DataFrame]:
        """
        加载日度/周度数据表格

        Args:
            excel_file: Excel文件对象
            sheet_name: 表格名称
            freq_type: 频率类型 ('daily' 或 'weekly')
            industry_name: 行业名称

        Returns:
            Optional[pd.DataFrame]: 加载的数据，如果失败返回None
        """
        print(f"      检测到预测变量 Sheet ('{freq_type}', 行业: '{industry_name}')...")

        # 删除try-except包装，让异常直接抛出
        # 使用统一格式读取数据
        print(f"        [使用统一格式] 第一行为列名，第一列为时间列")

        # 统一格式读取：第一行是列名，第一列是时间
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=0, index_col=0, parse_dates=True)

        # 清理数据
        df = self.cleaner.clean_zero_values(df, f"[{freq_type}] ")
        df = self.cleaner.remove_unnamed_columns(df, f"[{freq_type}] ")

        # 强制索引转换为日期时间
        print(f"      尝试将 '{sheet_name}' 的索引转换为日期时间...")
        original_index_len = len(df.index)
        df.index = pd.to_datetime(df.index, errors='coerce')

        if df is None or df.empty:
            return None

        df = df.loc[df.index.notna()]  # 过滤索引转换失败的行
        filtered_index_len = len(df.index)

        if filtered_index_len < original_index_len:
            print(f"      警告: 在 '{sheet_name}' 中移除了 {original_index_len - filtered_index_len} 行，因为它们的索引无法解析为有效日期。")

        df = df.dropna(axis=1, how='all')
        if df.empty:
            return None

        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.empty or df_numeric.isnull().all().all():
            return None

        print(f"      Sheet '{sheet_name}' ({industry_name}, {freq_type}) 加载完成。 Shape: {df_numeric.shape}")

        # 更新映射
        for col in df_numeric.columns:
            norm_col = normalize_text(col)
            if norm_col:
                self.var_industry_map[norm_col] = industry_name
                self.sheet_inferred_map[norm_col] = industry_name
                self.raw_columns_across_all_sheets.add(norm_col)

        return df_numeric

    def load_dekad_sheet(
        self,
        excel_file,
        sheet_name: str,
        industry_name: str
    ) -> Optional[pd.DataFrame]:
        """
        加载旬度数据表格（10日、20日、30日为旬日）

        Args:
            excel_file: Excel文件对象
            sheet_name: 表格名称
            industry_name: 行业名称

        Returns:
            Optional[pd.DataFrame]: 加载的数据，如果失败返回None
        """
        print(f"      检测到旬度预测变量 Sheet (行业: '{industry_name}')...")

        # 删除try-except包装，让异常直接抛出
        # 使用统一格式读取数据
        print(f"        [使用统一格式] 第一行为列名，第一列为时间列")

        # 统一格式读取：第一行是列名，第一列是时间
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=0, index_col=0, parse_dates=True)

        # 清理数据
        df = self.cleaner.clean_zero_values(df, f"[旬度] ")
        df = self.cleaner.remove_unnamed_columns(df, f"[旬度] ")

        # 强制索引转换为日期时间
        print(f"      尝试将 '{sheet_name}' 的索引转换为日期时间...")
        original_index_len = len(df.index)
        df.index = pd.to_datetime(df.index, errors='coerce')

        if df is None or df.empty:
            return None

        df = df.loc[df.index.notna()]  # 过滤索引转换失败的行
        filtered_index_len = len(df.index)

        if filtered_index_len < original_index_len:
            print(f"      警告: 在 '{sheet_name}' 中移除了 {original_index_len - filtered_index_len} 行，因为它们的索引无法解析为有效日期。")

        df = df.dropna(axis=1, how='all')
        if df.empty:
            return None

        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.empty or df_numeric.isnull().all().all():
            return None

        print(f"      Sheet '{sheet_name}' ({industry_name}, dekad) 加载完成。 Shape: {df_numeric.shape}")

        # 更新映射
        for col in df_numeric.columns:
            norm_col = normalize_text(col)
            if norm_col:
                self.var_industry_map[norm_col] = industry_name
                self.sheet_inferred_map[norm_col] = industry_name
                self.raw_columns_across_all_sheets.add(norm_col)

        return df_numeric

    def load_monthly_predictor_sheet(
        self,
        excel_file,
        sheet_name: str,
        industry_name: str
    ) -> Optional[pd.DataFrame]:
        """
        加载月度预测变量表格

        Args:
            excel_file: Excel文件对象
            sheet_name: 表格名称
            industry_name: 行业名称

        Returns:
            Optional[pd.DataFrame]: 加载的数据，如果失败返回None
        """
        print(f"      检测到非目标月度预测 Sheet，行业: '{industry_name}'...")

        # 删除try-except包装，让异常直接抛出
        # 使用统一格式读取数据
        print(f"        [使用统一格式] 第一行为列名，第一列为时间列")

        df_raw_pred = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)

        # 清理数据
        df_raw_pred = self.cleaner.remove_unnamed_columns(df_raw_pred, "[月度预测] ")
        df_raw_pred = self.cleaner.clean_zero_values(df_raw_pred, "[月度预测] ")

        if df_raw_pred.shape[1] < 2:
            print(f"      错误: 月度预测 Sheet '{sheet_name}' 列数 < 2。跳过。")
            return None

        date_col_name_pred = df_raw_pred.columns[0]
        print(f"      解析发布日期 (A列: '{date_col_name_pred}')...")

        publication_dates_predictor = pd.to_datetime(df_raw_pred[date_col_name_pred], errors='coerce')
        valid_date_mask_pred = publication_dates_predictor.notna()

        if not valid_date_mask_pred.any():
            print(f"      错误: 无法从列 '{date_col_name_pred}' 解析任何有效日期。跳过此Sheet。")
            return None

        publication_dates_predictor = publication_dates_predictor[valid_date_mask_pred]

        print(f"      提取月度预测变量 (B列及以后)...")
        temp_monthly_predictors_sheet = {}

        for col_idx_pred in range(1, df_raw_pred.shape[1]):  # 从B列开始
            col_name_pred = df_raw_pred.columns[col_idx_pred]

            # 清理值
            cleaned_series_pred = df_raw_pred[col_name_pred].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.strip()
            predictor_values_pred = pd.to_numeric(cleaned_series_pred, errors='coerce')

            # 创建按发布日期索引的序列
            temp_monthly_predictors_sheet[col_name_pred] = pd.Series(
                predictor_values_pred[valid_date_mask_pred].values,
                index=publication_dates_predictor
            )

            # 更新映射和跟踪
            norm_pred_col_p = normalize_text(col_name_pred)
            if norm_pred_col_p:
                self.var_industry_map[norm_pred_col_p] = industry_name
                self.sheet_inferred_map[norm_pred_col_p] = industry_name
                self.raw_columns_across_all_sheets.add(norm_pred_col_p)

        df_monthly_pred_sheet = pd.DataFrame(temp_monthly_predictors_sheet).sort_index()
        df_monthly_pred_sheet = df_monthly_pred_sheet.dropna(axis=1, how='all')

        if not df_monthly_pred_sheet.empty:
            print(f"      提取了 {df_monthly_pred_sheet.shape[1]} 个有效的月度预测变量 (按发布日期索引)。")
            return df_monthly_pred_sheet
        else:
            print("      此 Sheet 未包含有效的月度预测变量数据。")
            return None

    def get_var_industry_map(self) -> Dict[str, str]:
        """获取变量行业映射"""
        return self.var_industry_map.copy()

    def get_raw_columns_set(self) -> Set[str]:
        """获取所有加载的列名集合（标准化后）"""
        return self.raw_columns_across_all_sheets.copy()

    def get_removed_variables_log(self) -> List[Dict]:
        """获取移除变量的日志"""
        return self.cleaner.get_removed_variables_log()

# 导出的类
__all__ = [
    'DataLoader'
]
