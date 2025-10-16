"""
主数据处理器模块

协调各个模块的工作，实现完整的数据准备流程
这是重构后的主要接口，保持与原始prepare_data函数的兼容性
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from collections import defaultdict, Counter
from datetime import datetime, date

from dashboard.DFM.data_prep.modules.config_constants import ADF_P_THRESHOLD
from dashboard.DFM.data_prep.modules.format_detection import parse_sheet_info
from dashboard.DFM.data_prep.modules.mapping_manager import load_mappings, create_industry_map_from_data
from dashboard.DFM.data_prep.modules.stationarity_processor import ensure_stationarity, apply_stationarity_transforms
from dashboard.DFM.data_prep.modules.data_loader import DataLoader
from dashboard.DFM.data_prep.modules.data_aligner import DataAligner
from dashboard.DFM.data_prep.modules.data_cleaner import DataCleaner, clean_dataframe
from dashboard.DFM.data_prep.modules.final_processor import apply_final_stationarity_check
from dashboard.DFM.data_prep.modules.data_processing_helpers import (
    load_reference_variables,
    load_all_sheets,
    align_all_frequencies,
    combine_daily_weekly_data,
    collect_data_parts,
    merge_and_align_parts
)
from dashboard.DFM.data_prep.modules.ui_input_helpers import (
    map_ui_variables_to_columns,
    build_final_variable_list,
    filter_by_date_range as filter_data_by_date_range,
    apply_stationarity_check
)
from dashboard.DFM.data_prep.utils.date_utils import standardize_date
from dashboard.DFM.data_prep.utils.text_utils import normalize_text

def prepare_data(
    excel_path: str,
    target_freq: str,
    target_sheet_name: str,
    target_variable_name: str,
    consecutive_nan_threshold: Optional[int] = None,
    data_start_date: Optional[str] = None,
    data_end_date: Optional[str] = None,
    reference_sheet_name: str = '指标体系',
    reference_column_name: str = '指标名称'
) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[Dict], Optional[List[Dict]]]:
    """
    重构后的数据准备主函数
    
    加载数据，执行平稳性检查，对齐所有数据到目标频率，执行NaN检查和最终平稳性检查
    
    Args:
        excel_path: Excel文件路径
        target_freq: 目标频率（如'W-FRI'）
        target_sheet_name: 目标表格名称
        target_variable_name: 目标变量名称
        consecutive_nan_threshold: 连续NaN阈值
        data_start_date: 数据开始日期
        data_end_date: 数据结束日期
        reference_sheet_name: 参考表格名称
        reference_column_name: 参考列名称
        
    Returns:
        Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[Dict], Optional[List[Dict]]]:
            - 最终对齐的周度数据 (DataFrame)
            - 变量到行业的映射 (Dict)
            - 合并的转换日志 (Dict)
            - 详细的移除日志 (List[Dict])
    """
    print(f"\n--- [Data Prep V3 ] 开始加载和处理数据 (目标频率: {target_freq}) ---")

    # 验证目标频率
    if not target_freq.upper().endswith('-FRI'):
        print(f"错误: [Data Prep] 当前目标对齐逻辑仅支持周五 (W-FRI)。提供的目标频率 '{target_freq}' 无效。")
        return None, None, None, None
    
    print(f"  [Data Prep] 目标 Sheet: '{target_sheet_name}', 目标变量名(预期B列): '{target_variable_name}'")
    
    try:
        # 标准化日期参数
        data_start_date = standardize_date(data_start_date)
        data_end_date = standardize_date(data_end_date)

        # 初始化组件
        data_loader = DataLoader()
        data_aligner = DataAligner(target_freq)

        # 存储日志
        removed_variables_detailed_log = []
        all_indices_for_range = []

        # 加载Excel文件
        excel_file = pd.ExcelFile(excel_path)
        available_sheets = excel_file.sheet_names
        print(f"  [Data Prep] Excel 文件中可用的 Sheets: {available_sheets}")

        # 步骤1: 加载参考变量
        reference_predictor_variables = load_reference_variables(
            excel_file, reference_sheet_name, reference_column_name, available_sheets
        )

        # 步骤2: 加载所有数据sheets
        loaded_data = load_all_sheets(
            excel_file, available_sheets, target_sheet_name,
            target_variable_name, reference_sheet_name, data_loader
        )

        # 检查是否加载了必要数据
        if loaded_data.raw_target_values is None or loaded_data.raw_target_values.empty:
            print(f"错误：[Data Prep] 未能成功加载目标变量 '{target_variable_name}' 或其发布日期。")
            return None, None, None, None

        # 合并日志
        removed_variables_detailed_log.extend(loaded_data.removed_variables_log)

        # 步骤3: 对齐所有频率的数据
        aligned_data, df_daily_weekly, df_dekad_weekly, df_weekly_aligned = align_all_frequencies(
            data_aligner, loaded_data, all_indices_for_range
        )
        removed_variables_detailed_log.extend(aligned_data.removed_variables_log)

        # 步骤4: 合并日度/旬度/周度数据
        df_combined_dw_weekly, dw_removed_log = combine_daily_weekly_data(
            df_daily_weekly, df_dekad_weekly, df_weekly_aligned,
            consecutive_nan_threshold, data_start_date, data_end_date
        )
        removed_variables_detailed_log.extend(dw_removed_log)

        # 继续到最终合并步骤
        return _finalize_data_processing(
            aligned_data.target_series, aligned_data.target_sheet_predictors,
            aligned_data.monthly_predictors, df_combined_dw_weekly,
            loaded_data.actual_target_variable_name, loaded_data.target_sheet_cols,
            all_indices_for_range, data_start_date, data_end_date, target_freq,
            aligned_data.monthly_transform_log, removed_variables_detailed_log,
            loaded_data.var_industry_map, loaded_data.raw_columns_set,
            reference_predictor_variables
        )
        
    except FileNotFoundError:
        print(f"错误: [Data Prep] Excel 数据文件 {excel_path} 未找到。")
        return None, None, None, None
    except Exception as err:
        print(f"错误: [Data Prep] 数据准备过程中发生意外错误: {err}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def _finalize_data_processing(
    target_series_aligned, target_sheet_predictors_aligned, monthly_predictors_aligned,
    df_combined_dw_weekly, actual_target_variable_name, target_sheet_cols,
    all_indices_for_range, data_start_date, data_end_date, target_freq,
    monthly_transform_log, removed_variables_detailed_log, var_industry_map,
    raw_columns_across_all_sheets, reference_predictor_variables
):
    """最终数据处理和合并"""

    print("\n--- [Data Prep V3 ] 步骤 4: 最终合并和处理 ---")

    # 收集所有数据部分
    all_final_weekly_parts = collect_data_parts(
        target_series_aligned, target_sheet_predictors_aligned,
        df_combined_dw_weekly, monthly_predictors_aligned, actual_target_variable_name
    )

    if not all_final_weekly_parts:
        print("错误：[Data Prep] 没有成功处理的数据部分可以合并。无法继续。")
        return None, None, None, None

    # 创建完整日期范围
    try:
        data_aligner = DataAligner(target_freq)
        full_date_range = data_aligner.create_full_date_range(
            all_indices_for_range, data_start_date, data_end_date
        )
    except Exception as e:
        print(f"错误: 创建日期范围失败: {e}")
        return None, None, None, None

    # 合并并对齐所有数据部分
    combined_data_weekly_final = merge_and_align_parts(all_final_weekly_parts, full_date_range)

    if combined_data_weekly_final.empty:
        print("错误：[Data Prep] 没有有效的数据部分可以合并。")
        return None, None, None, None

    # 处理重复列
    data_cleaner = DataCleaner()
    combined_data_weekly_final = data_cleaner.remove_duplicate_columns(
        combined_data_weekly_final, "[最终合并] "
    )
    removed_variables_detailed_log.extend(data_cleaner.get_removed_variables_log())

    # 数据已经对齐到完整日期范围
    all_data_aligned_weekly = combined_data_weekly_final

    # 恢复频率信息（重新索引会丢失频率）
    if hasattr(full_date_range, 'freq') and full_date_range.freq is not None:
        all_data_aligned_weekly.index.freq = full_date_range.freq

    print(f"  最终周度数据对齐完成. Shape: {all_data_aligned_weekly.shape}")
    print(f"  最终索引频率: {all_data_aligned_weekly.index.freq}")

    # 添加调试信息：检查数据的实际时间范围和NaN情况
    print(f"  [DEBUG] 最终数据索引范围: {all_data_aligned_weekly.index.min()} 到 {all_data_aligned_weekly.index.max()}")
    print(f"  [DEBUG] 指定的data_start_date: {data_start_date}, data_end_date: {data_end_date}")
    non_nan_counts = all_data_aligned_weekly.notna().sum()
    print(f"  [DEBUG] 前5列的非NaN计数: {dict(non_nan_counts.head())}")

    # 移除全NaN列（但保留全NaN行以匹配原始版本行为）
    all_data_aligned_weekly, final_clean_log = clean_dataframe(
        all_data_aligned_weekly,
        remove_all_nan_cols=True,
        remove_all_nan_rows=False,  # 原始版本不移除全NaN行
        data_start_date=data_start_date,
        data_end_date=data_end_date,
        log_prefix="[最终清理] "
    )
    removed_variables_detailed_log.extend(final_clean_log)

    if all_data_aligned_weekly is None or all_data_aligned_weekly.empty:
        print("错误: [Data Prep] 最终合并和对齐后的数据为空。")
        print(f"  [DEBUG] 最终清理移除了 {len(final_clean_log)} 个变量")
        return None, None, None, None

    # 继续到平稳性检查
    return apply_final_stationarity_check(
        all_data_aligned_weekly, actual_target_variable_name, target_sheet_cols,
        monthly_transform_log, removed_variables_detailed_log, var_industry_map,
        raw_columns_across_all_sheets, reference_predictor_variables
    )

def prepare_data_from_ui_input(
    input_df: pd.DataFrame,
    target_variable: str,
    selected_variables: List[str] = None,
    training_start_date: str = None,
    validation_end_date: str = None,
    skip_stationarity_check: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    从UI输入准备训练数据

    Args:
        input_df: 输入数据DataFrame
        target_variable: 目标变量名
        selected_variables: 选择的变量列表
        training_start_date: 训练开始日期
        validation_end_date: 验证结束日期
        skip_stationarity_check: 是否跳过平稳性检查

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]: (处理后的数据, 转换详情, 移除变量日志)
    """
    print(f"\n--- [UI数据准备] 开始处理UI输入数据 ---")

    try:
        # 验证输入数据
        if input_df is None or input_df.empty:
            raise ValueError("输入数据为空")

        if target_variable not in input_df.columns:
            raise ValueError(f"目标变量 '{target_variable}' 不在数据中")

        # 步骤1: 映射UI变量到实际列名
        final_variables = [target_variable]
        if selected_variables:
            print(f"  [VIEW] [UI数据准备] 开始变量名映射:")
            print(f"    UI选择的变量: {selected_variables}")

            available_columns = list(input_df.columns)
            variable_mapping = map_ui_variables_to_columns(selected_variables, available_columns)

            final_variables = build_final_variable_list(target_variable, variable_mapping)

            print(f"    [SUCCESS] 成功映射的变量: {variable_mapping}")
            print(f"    [INFO] 最终变量列表: {final_variables}")

        # 步骤2: 筛选数据
        missing_vars = [var for var in final_variables if var not in input_df.columns]
        if missing_vars:
            raise ValueError(f"以下变量在数据中不存在: {missing_vars}")

        filtered_df = input_df[final_variables].copy()
        print(f"  [DATA] [UI数据准备] 筛选后数据形状: {filtered_df.shape}")

        # 步骤3: 日期范围筛选
        if training_start_date or validation_end_date:
            print(f"  [DATE] [UI数据准备] 应用日期范围筛选:")
            filtered_df, original_rows = filter_data_by_date_range(
                filtered_df, training_start_date, validation_end_date
            )
            print(f"    日期筛选: {original_rows} → {filtered_df.shape[0]} 行")

        # 步骤4: 平稳性检查和转换
        filtered_df, transform_details, removed_variables_log = apply_stationarity_check(
            filtered_df, target_variable, skip_stationarity_check
        )

        # 生成处理摘要
        print(f"\n  [INFO] [UI数据准备] 处理摘要:")
        print(f"    最终数据形状: {filtered_df.shape}")
        print(f"    目标变量: {target_variable}")
        print(f"    预测变量数量: {len(final_variables) - 1}")
        print(f"    转换的变量数量: {len(transform_details)}")

        if not filtered_df.empty:
            print(f"    数据时间范围: {filtered_df.index.min()} 到 {filtered_df.index.max()}")

        return filtered_df, transform_details, removed_variables_log

    except Exception as e:
        print(f"\n[ERROR] [UI数据准备] 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()

        # 返回空结果
        empty_df = pd.DataFrame()
        return empty_df, {}, {'error': str(e)}

# 导出的函数
__all__ = [
    'prepare_data',
    'prepare_data_from_ui_input'
]
