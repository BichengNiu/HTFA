"""
映射管理模块

负责加载和管理变量类型映射和行业映射
"""

import logging
import pandas as pd
import io
from typing import Dict, Tuple, Optional, Union

from dashboard.models.DFM.utils.text_utils import normalize_text

logger = logging.getLogger(__name__)

def load_mappings(
    excel_path: Union[str, io.BytesIO, pd.ExcelFile, object],
    sheet_name: str,
    indicator_col: str = '指标名称',
    type_col: str = '类型',
    industry_col: Optional[str] = '行业',
    single_stage_col: Optional[str] = '一次估计',
    first_stage_pred_col: Optional[str] = '一阶段预测',
    first_stage_target_col: Optional[str] = '一阶段目标',
    second_stage_target_col: Optional[str] = '二阶段目标'
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    从指定的Excel表格中加载变量类型和行业映射

    将指标名称（键）标准化为小写NFKC格式

    Args:
        excel_path: Excel文件路径、BytesIO对象、pd.ExcelFile对象或类文件对象
        sheet_name: 表格名称
        indicator_col: 指标列名，默认'指标名称'
        type_col: 类型列名，默认'类型'
        industry_col: 行业列名，默认'行业'，可选
        single_stage_col: 一次估计列名，默认'一次估计'，可选
        first_stage_pred_col: 一阶段预测列名，默认'一阶段预测'，可选
        first_stage_target_col: 一阶段目标列名，默认'一阶段目标'，可选
        second_stage_target_col: 二阶段目标列名，默认'二阶段目标'，可选

    Returns:
        Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
        (变量类型映射, 变量行业映射, 一次估计默认选择映射, 一阶段预测默认选择映射, 一阶段目标映射, 二阶段目标默认选择映射)
    """
    var_type_map = {}
    var_industry_map = {}
    var_dfm_single_stage_map = {}
    var_first_stage_pred_map = {}
    var_first_stage_target_map = {}
    var_second_stage_target_map = {}

    logger.info("[Mappings] Loading type/industry/estimation maps from:")
    logger.info("    Excel: %s", excel_path)
    logger.info("    Sheet: %s", sheet_name)
    logger.info("    Indicator Col: '%s', Type Col: '%s', Industry Col: '%s'", indicator_col, type_col, industry_col)
    logger.info("    Single Stage Col: '%s', First Stage Pred Col: '%s'", single_stage_col, first_stage_pred_col)
    logger.info("    First Stage Target Col: '%s', Second Stage Target Col: '%s'", first_stage_target_col, second_stage_target_col)

    try:
        # 处理不同类型的输入
        if isinstance(excel_path, pd.ExcelFile):
            # 已经是ExcelFile对象，直接使用
            excel_file_obj = excel_path
            logger.debug("[Mappings] 使用已有的ExcelFile对象")
        elif isinstance(excel_path, str):
            # 字符串路径
            excel_file_obj = pd.ExcelFile(excel_path)
        elif isinstance(excel_path, io.BytesIO):
            # BytesIO对象
            excel_file_obj = pd.ExcelFile(excel_path)
        elif hasattr(excel_path, 'getvalue'):
            # 类文件对象（如Streamlit UploadedFile或自定义UploadedFileFromBytes）
            file_bytes = excel_path.getvalue()
            excel_file_obj = pd.ExcelFile(io.BytesIO(file_bytes))
        else:
            raise TypeError(f"不支持的文件类型: {type(excel_path)}")

        if sheet_name not in excel_file_obj.sheet_names:
             raise FileNotFoundError(f"Sheet '{sheet_name}' not found in '{excel_path}'")

        indicator_sheet = pd.read_excel(excel_file_obj, sheet_name=sheet_name)

        # 标准化列名
        indicator_sheet.columns = indicator_sheet.columns.str.strip()
        indicator_col = indicator_col.strip()
        type_col = type_col.strip()
        if industry_col:
            industry_col = industry_col.strip()
        if single_stage_col:
            single_stage_col = single_stage_col.strip()
        if first_stage_pred_col:
            first_stage_pred_col = first_stage_pred_col.strip()
        if first_stage_target_col:
            first_stage_target_col = first_stage_target_col.strip()
        if second_stage_target_col:
            second_stage_target_col = second_stage_target_col.strip()

        # 检查必需列是否存在
        if indicator_col not in indicator_sheet.columns or type_col not in indicator_sheet.columns:
            raise ValueError(f"未找到必需的列 '{indicator_col}' 或 '{type_col}' 在 sheet '{sheet_name}'")

        # 创建类型映射
        var_type_map_temp = pd.Series(
            indicator_sheet[type_col].astype(str).str.strip().values,
            index=indicator_sheet[indicator_col].astype(str).str.strip()
        ).to_dict()
        
        # 标准化键并过滤NaN/空字符串
        var_type_map = {
            normalize_text(k): str(v).strip()
            for k, v in var_type_map_temp.items()
            if pd.notna(k) and str(k).strip().lower() not in ['', 'nan']
            and pd.notna(v) and str(v).strip().lower() not in ['', 'nan']
        }
        logger.info("[Mappings] Successfully created type map with %d entries.", len(var_type_map))

        # 创建行业映射（可选）
        if industry_col and industry_col in indicator_sheet.columns:
            industry_map_temp = pd.Series(
                indicator_sheet[industry_col].astype(str).str.strip().values,
                index=indicator_sheet[indicator_col].astype(str).str.strip() # Use indicator_col for index
            ).to_dict()

            # 标准化键并过滤NaN/空字符串
            var_industry_map = {
                normalize_text(k): str(v).strip()
                for k, v in industry_map_temp.items()
                if pd.notna(k) and str(k).strip().lower() not in ['', 'nan']
                and pd.notna(v) and str(v).strip().lower() not in ['', 'nan']
            }
            logger.info("[Mappings] Successfully created industry map with %d entries.", len(var_industry_map))
        elif industry_col:
            logger.warning("[Mappings] Industry column '%s' not found in sheet '%s'. Industry map will be empty.", industry_col, sheet_name)
        else:
            logger.debug("[Mappings] Industry column not specified. Industry map will be empty.")

        # 创建一次估计默认选择映射
        if single_stage_col and single_stage_col in indicator_sheet.columns:
            single_stage_map_temp = pd.Series(
                indicator_sheet[single_stage_col].astype(str).str.strip().values,
                index=indicator_sheet[indicator_col].astype(str).str.strip()
            ).to_dict()

            # 标准化键并只保留标记为"是"的条目
            var_dfm_single_stage_map = {
                normalize_text(k): str(v).strip()
                for k, v in single_stage_map_temp.items()
                if pd.notna(k) and str(k).strip().lower() not in ['', 'nan']
                and pd.notna(v) and str(v).strip() == '是'
            }
            logger.info("[Mappings] 从一次估计列加载了 %d 个标记为'是'的变量", len(var_dfm_single_stage_map))
        else:
            logger.warning("[Mappings] 一次估计列未找到")

        # 创建一阶段预测默认选择映射
        if first_stage_pred_col and first_stage_pred_col in indicator_sheet.columns:
            first_stage_pred_map_temp = pd.Series(
                indicator_sheet[first_stage_pred_col].astype(str).str.strip().values,
                index=indicator_sheet[indicator_col].astype(str).str.strip()
            ).to_dict()

            # 标准化键并只保留标记为"是"的条目
            var_first_stage_pred_map = {
                normalize_text(k): str(v).strip()
                for k, v in first_stage_pred_map_temp.items()
                if pd.notna(k) and str(k).strip().lower() not in ['', 'nan']
                and pd.notna(v) and str(v).strip() == '是'
            }
            logger.info("[Mappings] 从一阶段预测列加载了 %d 个标记为'是'的变量", len(var_first_stage_pred_map))
        else:
            logger.warning("[Mappings] 一阶段预测列未找到")

        # 创建一阶段目标映射
        if first_stage_target_col and first_stage_target_col in indicator_sheet.columns:
            first_stage_target_map_temp = pd.Series(
                indicator_sheet[first_stage_target_col].astype(str).str.strip().values,
                index=indicator_sheet[indicator_col].astype(str).str.strip()
            ).to_dict()

            # 标准化键并只保留标记为"是"的条目
            var_first_stage_target_map = {
                normalize_text(k): str(v).strip()
                for k, v in first_stage_target_map_temp.items()
                if pd.notna(k) and str(k).strip().lower() not in ['', 'nan']
                and pd.notna(v) and str(v).strip() == '是'
            }
            logger.info("[Mappings] 从一阶段目标列加载了 %d 个标记为'是'的变量", len(var_first_stage_target_map))
        else:
            logger.warning("[Mappings] 一阶段目标列未找到")

        # 创建二阶段目标默认选择映射
        if second_stage_target_col and second_stage_target_col in indicator_sheet.columns:
            second_stage_target_map_temp = pd.Series(
                indicator_sheet[second_stage_target_col].astype(str).str.strip().values,
                index=indicator_sheet[indicator_col].astype(str).str.strip()
            ).to_dict()

            # 标准化键并只保留标记为"是"的条目
            var_second_stage_target_map = {
                normalize_text(k): str(v).strip()
                for k, v in second_stage_target_map_temp.items()
                if pd.notna(k) and str(k).strip().lower() not in ['', 'nan']
                and pd.notna(v) and str(v).strip() == '是'
            }
            logger.info("[Mappings] 从二阶段目标列加载了 %d 个标记为'是'的变量", len(var_second_stage_target_map))
        else:
            logger.warning("[Mappings] 二阶段目标列未找到")

    except FileNotFoundError as e:
        logger.error("Error loading mappings: %s", e)
        # Return empty maps on file/sheet not found
    except ValueError as e:
        logger.error("Error processing mapping sheet: %s", e)
        # Return empty maps on column errors
    except Exception as e:
        logger.error("An unexpected error occurred while loading mappings: %s", e)
        # Return empty maps on other errors

    logger.info("[Mappings] Loading finished. Type map size: %d, Industry map size: %d, Single stage map size: %d, First stage pred map size: %d, First stage target map size: %d, Second stage target map size: %d",
                len(var_type_map), len(var_industry_map), len(var_dfm_single_stage_map), len(var_first_stage_pred_map), len(var_first_stage_target_map), len(var_second_stage_target_map))
    return var_type_map, var_industry_map, var_dfm_single_stage_map, var_first_stage_pred_map, var_first_stage_target_map, var_second_stage_target_map

def create_industry_map_from_data(
    final_columns: set,
    var_industry_map_loaded: Dict[str, str],
    default_industry: str = "Unknown"
) -> Dict[str, str]:
    """
    为最终数据中的列创建行业映射
    
    Args:
        final_columns: 最终数据中的列名集合
        var_industry_map_loaded: 从文件加载的行业映射
        default_industry: 默认行业名称
        
    Returns:
        Dict[str, str]: 更新后的行业映射
    """
    updated_var_industry_map = {}

    for col_original in final_columns:
        col_norm = normalize_text(col_original)
        if col_norm:
            # 从原始映射中获取行业，默认为"Unknown"
            industry = var_industry_map_loaded.get(col_norm, default_industry)
            updated_var_industry_map[col_norm] = industry

    return updated_var_industry_map

# 导出的函数
__all__ = [
    'load_mappings',
    'create_industry_map_from_data'
]
