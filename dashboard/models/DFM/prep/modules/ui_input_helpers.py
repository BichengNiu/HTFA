"""
UI输入数据处理辅助函数模块

提取prepare_data_from_ui_input函数的辅助功能
"""

import pandas as pd
from typing import List, Dict, Tuple

from dashboard.models.DFM.prep.utils.text_utils import normalize_text


def map_ui_variables_to_columns(
    selected_variables: List[str],
    available_columns: List[str]
) -> Dict[str, str]:
    """
    将UI选择的变量名映射到实际的列名

    Args:
        selected_variables: UI选择的变量列表
        available_columns: 数据中可用的列名

    Returns:
        Dict[str, str]: UI变量名到实际列名的映射
    """
    variable_mapping = {}

    for ui_var in selected_variables:
        # 直接匹配
        if ui_var in available_columns:
            variable_mapping[ui_var] = ui_var
            continue

        # 标准化匹配
        ui_var_normalized = normalize_text(ui_var)
        found_match = False

        for col in available_columns:
            col_normalized = normalize_text(col)
            if ui_var_normalized == col_normalized:
                variable_mapping[ui_var] = col
                found_match = True
                break

        if not found_match:
            print(f"    [WARNING] 警告: UI变量 '{ui_var}' 在数据中未找到匹配列")

    return variable_mapping


def build_final_variable_list(
    target_variable: str,
    variable_mapping: Dict[str, str]
) -> List[str]:
    """
    构建最终的变量列表

    Args:
        target_variable: 目标变量名
        variable_mapping: 变量映射字典

    Returns:
        List[str]: 最终变量列表
    """
    final_variables = [target_variable]

    for ui_var, actual_col in variable_mapping.items():
        if actual_col not in final_variables:
            final_variables.append(actual_col)

    return final_variables


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: str = None,
    end_date: str = None
) -> Tuple[pd.DataFrame, int]:
    """
    按日期范围筛选数据

    Args:
        df: 输入数据
        start_date: 开始日期字符串
        end_date: 结束日期字符串

    Returns:
        Tuple[pd.DataFrame, int]: (筛选后的数据, 原始行数)
    """
    original_rows = df.shape[0]
    filtered_df = df.copy()

    if start_date:
        try:
            start_dt = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df.index >= start_dt]
            print(f"    应用开始日期后剩余: {filtered_df.shape[0]} 行")
        except Exception as e:
            print(f"    [WARNING] 开始日期解析失败: {e}")

    if end_date:
        try:
            end_dt = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df.index <= end_dt]
            print(f"    应用结束日期后剩余: {filtered_df.shape[0]} 行")
        except Exception as e:
            print(f"    [WARNING] 结束日期解析失败: {e}")

    return filtered_df, original_rows


def apply_stationarity_check(
    df: pd.DataFrame,
    target_variable: str,
    skip_stationarity: bool = False
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    应用平稳性检查和转换

    Args:
        df: 输入数据
        target_variable: 目标变量名
        skip_stationarity: 是否跳过平稳性检查

    Returns:
        Tuple: (转换后的数据, 转换详情, 移除变量日志)
    """
    from dashboard.models.DFM.prep.modules.stationarity_processor import ensure_stationarity

    transform_details = {}
    removed_variables_log = {}

    if skip_stationarity:
        print(f"  [UI数据准备] 跳过平稳性检查")
        return df, transform_details, removed_variables_log

    print(f"  [LOADING] [UI数据准备] 执行平稳性检查和转换...")

    try:
        transformed_df, transform_log, removed_info = ensure_stationarity(
            df,
            skip_cols={target_variable} if target_variable in df.columns else set()
        )

        # 整理转换详情
        for var, details in transform_log.items():
            if isinstance(details, dict) and 'status' in details:
                transform_details[var] = details
            else:
                transform_details[var] = {'status': str(details)}

        print(f"    [SUCCESS] 平稳性转换完成，处理了 {len(transform_details)} 个变量")
        return transformed_df, transform_details, removed_variables_log

    except Exception as e:
        print(f"    [WARNING] 平稳性检查失败: {e}")
        print(f"    将跳过平稳性转换，使用原始数据")
        return df, transform_details, removed_variables_log


# 导出的函数
__all__ = [
    'map_ui_variables_to_columns',
    'build_final_variable_list',
    'filter_by_date_range',
    'apply_stationarity_check'
]
