"""
最终处理器模块

包含数据准备的最后阶段处理功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import Counter
import streamlit as st

from dashboard.models.DFM.prep.modules.stationarity_processor import ensure_stationarity, apply_stationarity_transforms
from dashboard.models.DFM.prep.modules.mapping_manager import create_industry_map_from_data
from dashboard.models.DFM.prep.utils.text_utils import normalize_text

def apply_final_stationarity_check(
    all_data_aligned_weekly, actual_target_variable_name, target_sheet_cols,
    monthly_transform_log, removed_variables_detailed_log, var_industry_map,
    raw_columns_across_all_sheets, reference_predictor_variables, validation_result
):
    """
    应用最终的平稳性检查

    注意：var_industry_map现在是基于指标体系的映射（非sheet推断）
    """

    print("\n--- [Data Prep V3] 步骤 6: 最终平稳性检查 ---")

    # 定义需要跳过平稳性检查的列
    cols_to_skip_weekly_stationarity = set()

    # 添加月度变量（从月度转换日志中获取）
    if monthly_transform_log:
        monthly_cols = set(monthly_transform_log.keys())
        monthly_cols_in_final = monthly_cols.intersection(all_data_aligned_weekly.columns)
        if monthly_cols_in_final:
            cols_to_skip_weekly_stationarity.update(monthly_cols_in_final)
            print(f"    标记跳过月度来源变量: {len(monthly_cols_in_final)} 个")

    # 添加目标变量
    if actual_target_variable_name in all_data_aligned_weekly.columns:
        cols_to_skip_weekly_stationarity.add(actual_target_variable_name)
        print(f"    标记跳过目标变量: '{actual_target_variable_name}'")

    # 添加目标Sheet预测变量
    target_sheet_predictor_cols_in_final = target_sheet_cols.intersection(all_data_aligned_weekly.columns)
    if target_sheet_predictor_cols_in_final:
        cols_to_skip_weekly_stationarity.update(target_sheet_predictor_cols_in_final)
        print(f"    标记跳过目标 Sheet 预测变量: {len(target_sheet_predictor_cols_in_final)} 个")

    print(f"    总共将跳过 {len(cols_to_skip_weekly_stationarity)} 个变量进行周度平稳性检查。")

    # 标准化跳过列表
    skip_cols_normalized = {normalize_text(c) for c in cols_to_skip_weekly_stationarity}
    
    # 检查是否有预定义的平稳性规则
    use_config_stationarity = False
    config_stationarity_rules = {}

    import importlib
    from dashboard.models.DFM import config
    importlib.reload(config)

    if hasattr(config, 'PREDEFINED_STATIONARITY_TRANSFORMS') and isinstance(config.PREDEFINED_STATIONARITY_TRANSFORMS, dict):
        config_stationarity_rules_raw = {
            normalize_text(k): v
            for k, v in config.PREDEFINED_STATIONARITY_TRANSFORMS.items()
            if isinstance(v, dict) and 'status' in v
        }

        if config_stationarity_rules_raw:
            print(f"  检测到来自 config.py 的预定义平稳性转换规则 ({len(config_stationarity_rules_raw)} 条)。")

            # 过滤掉需要跳过的列的规则
            config_stationarity_rules = {
                k: v for k, v in config_stationarity_rules_raw.items()
                if k not in skip_cols_normalized
            }

            removed_rules_count = len(config_stationarity_rules_raw) - len(config_stationarity_rules)
            if removed_rules_count > 0:
                print(f"    已从预定义规则中移除 {removed_rules_count} 条，因为它们对应于需要跳过的变量。")

            if config_stationarity_rules:
                use_config_stationarity = True
            else:
                print("    过滤后，没有适用于日/周变量的预定义规则。将回退到 ADF 检验。")
        else:
            print("  config.py 中 PREDEFINED_STATIONARITY_TRANSFORMS 为空或格式无效，将执行 ADF 检验。")
    else:
        print("  config.py 中未定义 PREDEFINED_STATIONARITY_TRANSFORMS，将执行 ADF 检验。")
    
    # 应用平稳性转换
    final_data_stationary = all_data_aligned_weekly.copy()
    weekly_transform_log = {}
    
    if use_config_stationarity:
        print(f"  应用过滤后的预定义平稳性规则 ({len(config_stationarity_rules)} 条规则)...")

        # 标准化列名以匹配规则
        original_columns_map = {normalize_text(c): c for c in final_data_stationary.columns}
        final_data_stationary.columns = list(original_columns_map.keys())
        
        final_data_stationary = apply_stationarity_transforms(
            final_data_stationary,
            config_stationarity_rules
        )
        
        # 恢复原始列名
        final_data_stationary.columns = final_data_stationary.columns.map(original_columns_map)
        weekly_transform_log = {"status": "Applied filtered rules from config"}
    else:
        print("  通过 ADF 检验确定平稳性 (仅日/周来源变量)...")

        # 标准化列名
        original_columns_map = {normalize_text(c): c for c in final_data_stationary.columns}
        final_data_stationary.columns = list(original_columns_map.keys())

        final_data_stationary, weekly_transform_log, removed_cols_info_weekly = ensure_stationarity(
            final_data_stationary,
            skip_cols=skip_cols_normalized,
            adf_p_threshold=0.05
        )

        # 恢复原始列名
        final_data_stationary.columns = final_data_stationary.columns.map(original_columns_map)
        
        # 记录移除的变量
        for reason, cols_norm in removed_cols_info_weekly.items():
            for col_norm in cols_norm:
                original_col_name = original_columns_map.get(col_norm, col_norm)
                removed_variables_detailed_log.append({'Variable': original_col_name, 'Reason': f'weekly_stationarity_{reason}'})
    
    # 最终检查和总结
    return finalize_results(
        final_data_stationary, actual_target_variable_name, monthly_transform_log,
        weekly_transform_log, removed_variables_detailed_log, var_industry_map,
        raw_columns_across_all_sheets, reference_predictor_variables, validation_result
    )

def finalize_results(
    final_data_stationary, actual_target_variable_name, monthly_transform_log,
    weekly_transform_log, removed_variables_detailed_log, var_industry_map,
    raw_columns_across_all_sheets, reference_predictor_variables, validation_result
):
    """完成最终结果处理"""
    
    print("\n--- [Data Prep V3] 步骤 6: 完成与检查 ---")
    
    if final_data_stationary is None or final_data_stationary.empty:
        print("错误: [Data Prep] 最终数据在平稳性处理后为空。")
        return None, None, None, None
    
    print(f"  最终数据 Shape: {final_data_stationary.shape}")
    
    # 检查目标变量存在性
    target_exists = actual_target_variable_name in final_data_stationary.columns
    print(f"  目标变量 '{actual_target_variable_name}' 是否存在: {target_exists}")
    
    if not target_exists:
        # 检查标准化名称
        norm_target_name = normalize_text(actual_target_variable_name)
        temp_cols_lower = {normalize_text(c): c for c in final_data_stationary.columns}
        if norm_target_name in temp_cols_lower:
            print(f"  注意：目标变量以规范化名称 '{norm_target_name}' 存在。")
        else:
            print(f"  严重警告: 目标变量 '{actual_target_variable_name}' 在最终数据中不存在！")
    
    # 计算预测变量数量
    final_predictor_count = final_data_stationary.shape[1] - (1 if target_exists else 0)
    
    # 合并转换日志
    combined_transform_log = {
        "monthly_predictor_stationarity_checks": monthly_transform_log,
        "weekly_final_stationarity_checks": weekly_transform_log
    }
    
    # 创建最终的行业映射
    final_columns_in_data = set(final_data_stationary.columns)
    updated_var_industry_map = create_industry_map_from_data(
        final_columns_in_data, var_industry_map, "Unknown"
    )
    
    # 变量数量对比
    raw_predictor_count = len(raw_columns_across_all_sheets)
    reference_count = len(reference_predictor_variables)
    
    print(f"\n--- [Data Prep] 变量数量与指标体系对比 ---")
    print(f"  指标体系变量数 (规范化): {reference_count}")
    print(f"  原始加载预测变量数 (规范化, 不含目标): {raw_predictor_count}")
    print(f"  最终输出预测变量数: {final_predictor_count}")
    
    # 对比分析
    if reference_predictor_variables:
        final_output_predictors_norm = {
            normalize_text(col)
            for col in final_columns_in_data
            if col != actual_target_variable_name
        }
        
        missing_in_data = reference_predictor_variables - raw_columns_across_all_sheets
        if missing_in_data:
            norm_target_name = normalize_text(actual_target_variable_name)
            missing_to_print = [v for v in sorted(list(missing_in_data)) if v != norm_target_name]
            if missing_to_print:
                print(f"\n  [!] 以下 {len(missing_to_print)} 个变量在指标体系中，但未在任何数据 Sheets 中加载:")
                for i, var_norm in enumerate(missing_to_print[:10]):  # 只显示前10个
                    print(f"      {i+1}. {var_norm}")
                if len(missing_to_print) > 10:
                    print(f"      ... 还有 {len(missing_to_print) - 10} 个变量")
    
    # 转换日志摘要
    print(f"\n--- [Data Prep] 转换日志摘要 ---")
    if isinstance(monthly_transform_log, dict) and monthly_transform_log:
        monthly_statuses = Counter(log.get('status', 'unknown') for log in monthly_transform_log.values())
        print(f"  月度预测变量检查状态 (ADF): {dict(monthly_statuses)}")
    
    if isinstance(weekly_transform_log, dict):
        if weekly_transform_log.get("status") == "Applied filtered rules from config":
            print(f"  周度最终检查状态: 应用了来自 config 的过滤后预定义规则。")
        elif weekly_transform_log:
            weekly_statuses = Counter(log.get('status', 'unknown') for log in weekly_transform_log.values())
            filtered_weekly_statuses = {k:v for k,v in weekly_statuses.items() if k != 'skipped_by_request'}
            skipped_count = weekly_statuses.get('skipped_by_request', 0)
            print(f"  周度最终检查状态 (ADF, 仅日/周源): {dict(filtered_weekly_statuses)}")
            if skipped_count > 0:
                print(f"    (另有 {skipped_count} 个变量按计划被跳过)")
    
    print(f"\n--- [Data Prep V3] 数据准备完成 ---")
    print(f"  共记录了 {len(removed_variables_detailed_log)} 个移除事件。")
    
    # 自动加载映射数据（简化版本）
    try:
        print(f"[映射加载] 尝试保存映射数据到统一状态管理器...")
        st.session_state['data_prep.dfm_industry_map_obj'] = updated_var_industry_map
        print(f"[映射加载] 成功保存行业映射 {len(updated_var_industry_map)} 个")
    except Exception as e:
        print(f"[映射加载] 警告：保存映射数据失败: {e}")
    
    return final_data_stationary, updated_var_industry_map, combined_transform_log, removed_variables_detailed_log, validation_result

# 导出的函数
__all__ = [
    'apply_final_stationarity_check',
    'finalize_results'
]
