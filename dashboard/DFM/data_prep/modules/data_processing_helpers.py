"""
数据处理辅助函数模块

包含prepare_data主流程的各个子步骤函数
遵循单一职责原则，每个函数只做一件事
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass

from dashboard.DFM.data_prep.modules.data_loader import DataLoader
from dashboard.DFM.data_prep.modules.data_aligner import DataAligner
from dashboard.DFM.data_prep.modules.data_cleaner import clean_dataframe
from dashboard.DFM.data_prep.modules.format_detection import parse_sheet_info
from dashboard.DFM.data_prep.utils.text_utils import normalize_text


@dataclass
class LoadedData:
    """加载的数据集合"""
    raw_target_values: Optional[pd.Series]
    actual_target_variable_name: Optional[str]
    target_sheet_cols: Set[str]
    target_sheet_predictors: pd.DataFrame
    other_monthly_predictors: pd.DataFrame
    data_parts_by_freq: Dict[str, List[pd.DataFrame]]
    var_industry_map: Dict[str, str]
    raw_columns_set: Set[str]
    removed_variables_log: List[Dict]


@dataclass
class AlignedData:
    """对齐后的数据集合"""
    target_series: pd.Series
    target_sheet_predictors: pd.DataFrame
    monthly_predictors: pd.DataFrame
    daily_weekly_combined: pd.DataFrame
    all_indices: List[pd.DatetimeIndex]
    monthly_transform_log: Dict
    removed_variables_log: List[Dict]


def load_reference_variables(
    excel_file,
    reference_sheet_name: str,
    reference_column_name: str,
    available_sheets: List[str]
) -> Set[str]:
    """
    加载并标准化参考变量名

    Args:
        excel_file: Excel文件对象
        reference_sheet_name: 参考sheet名称
        reference_column_name: 参考列名称
        available_sheets: 可用的sheet列表

    Returns:
        Set[str]: 标准化后的参考变量集合
    """
    reference_predictor_variables = set()

    if reference_sheet_name not in available_sheets:
        print(f"  [Data Prep] 警告: 未找到参考 Sheet '{reference_sheet_name}'。")
        return reference_predictor_variables

    try:
        ref_df = pd.read_excel(excel_file, sheet_name=reference_sheet_name)
        ref_df.columns = ref_df.columns.str.strip()
        clean_reference_column_name = reference_column_name.strip()

        if clean_reference_column_name not in ref_df.columns:
            print(f"  [Data Prep] 警告: 在 '{reference_sheet_name}' 未找到参考列 '{clean_reference_column_name}'。")
            return reference_predictor_variables

        raw_reference_vars = (
            ref_df[clean_reference_column_name]
            .astype(str).str.strip().replace('nan', np.nan).dropna().unique()
        )
        raw_reference_vars = [v for v in raw_reference_vars if v]
        reference_predictor_variables = set(
            normalize_text(var)
            for var in raw_reference_vars
        )
        print(f"  [Data Prep] 从 '{reference_sheet_name}' 加载并规范化了 {len(reference_predictor_variables)} 个参考变量名。")

    except Exception as e_ref:
        print(f"  [Data Prep] 警告: 读取参考 Sheet '{reference_sheet_name}' 出错: {e_ref}。")

    return reference_predictor_variables


def load_all_sheets(
    excel_file,
    available_sheets: List[str],
    target_sheet_name: str,
    target_variable_name: str,
    reference_sheet_name: str,
    data_loader: DataLoader
) -> LoadedData:
    """
    遍历并加载所有数据sheets

    Args:
        excel_file: Excel文件对象
        available_sheets: 可用的sheet列表
        target_sheet_name: 目标sheet名称
        target_variable_name: 目标变量名
        reference_sheet_name: 参考sheet名称
        data_loader: 数据加载器实例

    Returns:
        LoadedData: 加载的数据集合
    """
    print("\n--- [Data Prep V3 ] 步骤 1: 加载数据 ---")

    # 初始化数据存储
    data_parts = defaultdict(list)
    publication_dates_from_target = None
    raw_target_values = None
    df_target_sheet_predictors_pubdate = pd.DataFrame()
    df_other_monthly_predictors_pubdate = pd.DataFrame()
    actual_target_variable_name = None
    target_sheet_cols = set()

    # 遍历sheets加载数据
    for sheet_name in available_sheets:
        print(f"    [Data Prep] 正在检查 Sheet: {sheet_name}...")
        is_target_sheet = (sheet_name == target_sheet_name)
        sheet_info = parse_sheet_info(sheet_name, target_sheet_name)
        freq_type = sheet_info['freq_type']
        industry_name = sheet_info['industry'] if sheet_info['industry'] else "Uncategorized"

        if is_target_sheet:
            # 处理目标表格
            pub_dates, target_vals, target_preds, target_cols = data_loader.load_target_sheet(
                excel_file, sheet_name, target_variable_name, industry_name
            )

            if pub_dates is not None and target_vals is not None:
                publication_dates_from_target = pub_dates
                raw_target_values = target_vals
                actual_target_variable_name = target_vals.name
                target_sheet_cols = target_cols

                if not target_preds.empty:
                    df_target_sheet_predictors_pubdate = target_preds

        elif freq_type in ['daily', 'weekly']:
            # 处理日度/周度数据
            df_loaded = data_loader.load_daily_weekly_sheet(excel_file, sheet_name, freq_type, industry_name)
            if df_loaded is not None:
                data_parts[freq_type].append(df_loaded)

        elif freq_type == 'monthly_predictor':
            # 处理其他月度预测变量
            df_monthly = data_loader.load_monthly_predictor_sheet(excel_file, sheet_name, industry_name)
            if df_monthly is not None:
                # 检查并处理重复索引
                if df_monthly.index.duplicated().any():
                    print(f"      警告: Sheet '{sheet_name}' 包含重复的日期索引，保留第一个值")
                    df_monthly = df_monthly[~df_monthly.index.duplicated(keep='first')]

                if df_other_monthly_predictors_pubdate.empty:
                    df_other_monthly_predictors_pubdate = df_monthly
                else:
                    # 确保现有数据也没有重复索引
                    if df_other_monthly_predictors_pubdate.index.duplicated().any():
                        print(f"      警告: 现有月度数据包含重复索引，正在清理")
                        df_other_monthly_predictors_pubdate = df_other_monthly_predictors_pubdate[~df_other_monthly_predictors_pubdate.index.duplicated(keep='first')]

                    df_other_monthly_predictors_pubdate = pd.concat(
                        [df_other_monthly_predictors_pubdate, df_monthly],
                        axis=1,
                        join='outer'
                    )
        else:
            if sheet_name != reference_sheet_name:
                print(f"      Sheet '{sheet_name}' 不符合要求或非目标 Sheet，已跳过。")
            continue

    # 汇总加载结果
    print(f"\n--- [Data Prep V3 ] 数据加载完成 ---")
    print(f"  目标变量: {actual_target_variable_name}")
    print(f"  目标Sheet预测变量: {df_target_sheet_predictors_pubdate.shape[1]} 个")
    print(f"  其他月度预测变量: {df_other_monthly_predictors_pubdate.shape[1]} 个")
    print(f"  日度数据表格: {len(data_parts['daily'])} 个")
    print(f"  周度数据表格: {len(data_parts['weekly'])} 个")

    return LoadedData(
        raw_target_values=raw_target_values,
        actual_target_variable_name=actual_target_variable_name,
        target_sheet_cols=target_sheet_cols,
        target_sheet_predictors=df_target_sheet_predictors_pubdate,
        other_monthly_predictors=df_other_monthly_predictors_pubdate,
        data_parts_by_freq=data_parts,
        var_industry_map=data_loader.get_var_industry_map(),
        raw_columns_set=data_loader.get_raw_columns_set(),
        removed_variables_log=data_loader.get_removed_variables_log()
    )


def align_all_frequencies(
    data_aligner: DataAligner,
    loaded_data: LoadedData,
    all_indices_for_range: List[pd.DatetimeIndex]
) -> AlignedData:
    """
    对齐所有不同频率的数据到目标频率

    Args:
        data_aligner: 数据对齐器实例
        loaded_data: 加载的数据
        all_indices_for_range: 用于计算日期范围的索引列表

    Returns:
        AlignedData: 对齐后的数据集合
    """
    print("\n--- [Data Prep V3 ] 步骤 2: 数据对齐 ---")

    removed_variables_log = []

    # 2a: 目标变量对齐到最近周五
    target_series_aligned = data_aligner.align_target_to_nearest_friday(loaded_data.raw_target_values)
    if not target_series_aligned.empty:
        all_indices_for_range.append(loaded_data.raw_target_values.index)

    # 2b: 目标Sheet预测变量对齐到最近周五
    target_sheet_predictors_aligned = data_aligner.align_target_sheet_predictors_to_nearest_friday(
        loaded_data.target_sheet_predictors
    )
    if not target_sheet_predictors_aligned.empty:
        all_indices_for_range.append(target_sheet_predictors_aligned.index)

    # 2c: 其他月度预测变量对齐到月末最后周五
    if not loaded_data.other_monthly_predictors.empty:
        all_indices_for_range.append(loaded_data.other_monthly_predictors.index)

    monthly_predictors_aligned, monthly_transform_log, monthly_removed_info = data_aligner.align_monthly_to_last_friday(
        loaded_data.other_monthly_predictors, apply_stationarity=True
    )

    # 记录月度处理的移除变量
    for reason, cols in monthly_removed_info.items():
        for col in cols:
            removed_variables_log.append({
                'Variable': col,
                'Reason': f'monthly_stationarity_{reason}'
            })

    # 2d: 日度数据转换为周度
    if loaded_data.data_parts_by_freq['daily']:
        for df in loaded_data.data_parts_by_freq['daily']:
            if not df.empty:
                all_indices_for_range.append(df.index)

    df_daily_weekly = data_aligner.convert_daily_to_weekly(loaded_data.data_parts_by_freq['daily'])

    # 2e: 周度数据对齐
    if loaded_data.data_parts_by_freq['weekly']:
        for df in loaded_data.data_parts_by_freq['weekly']:
            if not df.empty:
                all_indices_for_range.append(df.index)

    df_weekly_aligned = data_aligner.align_weekly_data(loaded_data.data_parts_by_freq['weekly'])

    return AlignedData(
        target_series=target_series_aligned,
        target_sheet_predictors=target_sheet_predictors_aligned,
        monthly_predictors=monthly_predictors_aligned,
        daily_weekly_combined=pd.DataFrame(),  # 将在下一步合并
        all_indices=all_indices_for_range,
        monthly_transform_log=monthly_transform_log,
        removed_variables_log=removed_variables_log
    ), df_daily_weekly, df_weekly_aligned


def combine_daily_weekly_data(
    df_daily_weekly: pd.DataFrame,
    df_weekly_aligned: pd.DataFrame,
    consecutive_nan_threshold: Optional[int],
    data_start_date: Optional[str],
    data_end_date: Optional[str]
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    合并日度和周度数据并进行清理

    Args:
        df_daily_weekly: 日度转周度后的数据
        df_weekly_aligned: 周度对齐后的数据
        consecutive_nan_threshold: 连续NaN阈值
        data_start_date: 数据开始日期
        data_end_date: 数据结束日期

    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (合并后的数据, 移除变量日志)
    """
    print("\n--- [Data Prep V3 ] 步骤 3: 合并日度/周度数据 ---")

    parts_to_combine = []
    if not df_daily_weekly.empty:
        parts_to_combine.append(df_daily_weekly)
    if not df_weekly_aligned.empty:
        parts_to_combine.append(df_weekly_aligned)

    if not parts_to_combine:
        return pd.DataFrame(), []

    df_combined = pd.concat(parts_to_combine, axis=1)
    df_combined, removed_log = clean_dataframe(
        df_combined,
        consecutive_nan_threshold=consecutive_nan_threshold,
        data_start_date=data_start_date,
        data_end_date=data_end_date,
        log_prefix="[日度/周度合并] "
    )

    print(f"    合并后日度/周度数据 Shape: {df_combined.shape}")

    return df_combined, removed_log


def collect_data_parts(
    target_series_aligned: pd.Series,
    target_sheet_predictors_aligned: pd.DataFrame,
    df_combined_dw_weekly: pd.DataFrame,
    monthly_predictors_aligned: pd.DataFrame,
    actual_target_variable_name: str
) -> List[pd.DataFrame]:
    """
    收集所有需要合并的数据部分，并去除重复的目标变量

    Args:
        target_series_aligned: 对齐后的目标序列
        target_sheet_predictors_aligned: 对齐后的目标Sheet预测变量
        df_combined_dw_weekly: 合并后的日度/周度数据
        monthly_predictors_aligned: 对齐后的月度预测变量
        actual_target_variable_name: 实际目标变量名

    Returns:
        List[pd.DataFrame]: 待合并的数据部分列表
    """
    all_final_weekly_parts = []
    target_variable_added = False

    # 添加目标变量
    if target_series_aligned is not None and not target_series_aligned.empty:
        target_series_aligned.name = actual_target_variable_name
        target_df = target_series_aligned.to_frame()
        all_final_weekly_parts.append(target_df)
        target_variable_added = True
        print(f"  添加目标变量 '{actual_target_variable_name}' (最近周五对齐)...")
        print(f"    目标变量DataFrame形状: {target_df.shape}, 列: {list(target_df.columns)}")

    # 添加目标Sheet预测变量（检查重复）
    if target_sheet_predictors_aligned is not None and not target_sheet_predictors_aligned.empty:
        if actual_target_variable_name in target_sheet_predictors_aligned.columns:
            print(f"  [WARNING] 警告：目标Sheet预测变量中包含目标变量 '{actual_target_variable_name}'")
            if target_variable_added:
                target_sheet_predictors_aligned = target_sheet_predictors_aligned.drop(
                    columns=[actual_target_variable_name]
                )

        if not target_sheet_predictors_aligned.empty:
            all_final_weekly_parts.append(target_sheet_predictors_aligned)
            print(f"  添加目标 Sheet 预测变量 ({target_sheet_predictors_aligned.shape[1]} 个)...")
            cols_preview = list(target_sheet_predictors_aligned.columns[:5])
            if len(target_sheet_predictors_aligned.columns) > 5:
                cols_preview.append(f"...还有{len(target_sheet_predictors_aligned.columns)-5}个")
            print(f"    列预览: {cols_preview}")

    # 添加日度/周度数据（检查重复）
    if df_combined_dw_weekly is not None and not df_combined_dw_weekly.empty:
        if actual_target_variable_name in df_combined_dw_weekly.columns:
            print(f"  [WARNING] 警告：日度/周度数据中包含目标变量 '{actual_target_variable_name}'")
            if target_variable_added:
                df_combined_dw_weekly = df_combined_dw_weekly.drop(columns=[actual_target_variable_name])

        if not df_combined_dw_weekly.empty:
            all_final_weekly_parts.append(df_combined_dw_weekly)
            print(f"  添加日度/周度预测变量 (Shape: {df_combined_dw_weekly.shape})...")
            cols_preview = list(df_combined_dw_weekly.columns[:5])
            if len(df_combined_dw_weekly.columns) > 5:
                cols_preview.append(f"...还有{len(df_combined_dw_weekly.columns)-5}个")
            print(f"    列预览: {cols_preview}")

    # 添加月度预测变量（检查重复）
    if monthly_predictors_aligned is not None and not monthly_predictors_aligned.empty:
        if actual_target_variable_name in monthly_predictors_aligned.columns:
            print(f"  [WARNING] 警告：月度预测变量中包含目标变量 '{actual_target_variable_name}'")
            if target_variable_added:
                monthly_predictors_aligned = monthly_predictors_aligned.drop(
                    columns=[actual_target_variable_name]
                )

        if not monthly_predictors_aligned.empty:
            all_final_weekly_parts.append(monthly_predictors_aligned)
            print(f"  添加其他月度预测变量 (Shape: {monthly_predictors_aligned.shape})...")
            cols_preview = list(monthly_predictors_aligned.columns[:5])
            if len(monthly_predictors_aligned.columns) > 5:
                cols_preview.append(f"...还有{len(monthly_predictors_aligned.columns)-5}个")
            print(f"    列预览: {cols_preview}")

    return all_final_weekly_parts


def merge_and_align_parts(
    data_parts: List[pd.DataFrame],
    full_date_range: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    将所有数据部分重新索引到完整日期范围，然后合并

    Args:
        data_parts: 待合并的数据部分列表
        full_date_range: 完整的日期范围

    Returns:
        pd.DataFrame: 合并后的数据
    """
    # 重新索引所有数据部分
    aligned_parts = []
    for i, part in enumerate(data_parts):
        if part is not None and not part.empty:
            print(f"  数据部分 {i+1} 重新索引前: Shape {part.shape}, 索引范围 {part.index.min()} 到 {part.index.max()}")
            aligned_part = part.reindex(full_date_range)
            aligned_parts.append(aligned_part)
            print(f"  数据部分 {i+1} 重新索引后: Shape {aligned_part.shape}, 索引范围 {aligned_part.index.min()} 到 {aligned_part.index.max()}")
        else:
            print(f"  数据部分 {i+1} 为空，跳过")

    # 检查并清理重复列
    print(f"\n  [调试] 检查合并前各部分的列情况:")
    for i, part in enumerate(aligned_parts):
        dup_cols = part.columns[part.columns.duplicated(keep=False)].tolist()
        if dup_cols:
            unique_dups = list(set(dup_cols))
            print(f"    部分{i+1}: {part.shape[1]}列，有重复: {unique_dups}")
        else:
            print(f"    部分{i+1}: {part.shape[1]}列，无重复")

    # 去除各部分内部的重复列
    cleaned_parts = []
    for i, part in enumerate(aligned_parts):
        if part.columns.duplicated().any():
            part_cleaned = part.loc[:, ~part.columns.duplicated(keep='first')]
            print(f"    部分{i+1}去重: {part.shape[1]} -> {part_cleaned.shape[1]}列")
            cleaned_parts.append(part_cleaned)
        else:
            cleaned_parts.append(part)

    # 合并所有部分
    combined_data = pd.concat(cleaned_parts, axis=1)
    print(f"  合并所有 {len(aligned_parts)} 个已对齐的周度数据部分. 合并后 Shape: {combined_data.shape}")

    # 检查合并后的重复列
    dup_after_concat = combined_data.columns[combined_data.columns.duplicated(keep=False)].tolist()
    if dup_after_concat:
        from collections import Counter
        col_counts = Counter(combined_data.columns)
        print(f"\n  [警告] pd.concat后产生重复列:")
        for col, count in col_counts.items():
            if count > 1:
                print(f"    '{col}': 出现{count}次")

    return combined_data


# 导出的类和函数
__all__ = [
    'LoadedData',
    'AlignedData',
    'load_reference_variables',
    'load_all_sheets',
    'align_all_frequencies',
    'combine_daily_weekly_data',
    'collect_data_parts',
    'merge_and_align_parts'
]
