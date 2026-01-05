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
    predictor_col: Optional[str] = '预测变量',
    target_variable_col: Optional[str] = '目标变量'
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    从指定的Excel表格中加载变量类型和行业映射

    将指标名称（键）标准化为小写NFKC格式

    Args:
        excel_path: Excel文件路径、BytesIO对象、pd.ExcelFile对象或类文件对象
        sheet_name: 表格名称
        indicator_col: 指标列名，默认'指标名称'
        type_col: 类型列名，默认'类型'
        industry_col: 行业列名，默认'行业'，可选
        predictor_col: 预测变量列名，默认'预测变量'，可选
        target_variable_col: 目标变量列名，默认'目标变量'，可选

    Returns:
        Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
        (变量类型映射, 变量行业映射, 预测变量映射, 目标变量映射)
    """
    var_type_map = {}
    var_industry_map = {}
    var_predictor_map = {}
    var_target_variable_map = {}

    logger.info("[Mappings] Loading type/industry/estimation maps from:")
    logger.info("    Excel: %s", excel_path)
    logger.info("    Sheet: %s", sheet_name)
    logger.info("    Indicator Col: '%s', Type Col: '%s', Industry Col: '%s'", indicator_col, type_col, industry_col)
    logger.info("    Predictor Col: '%s', Target Variable Col: '%s'", predictor_col, target_variable_col)

    try:
        # 处理不同类型的输入
        if isinstance(excel_path, pd.ExcelFile):
            excel_file_obj = excel_path
            logger.debug("[Mappings] 使用已有的ExcelFile对象")
        elif isinstance(excel_path, str):
            excel_file_obj = pd.ExcelFile(excel_path)
        elif isinstance(excel_path, io.BytesIO):
            excel_file_obj = pd.ExcelFile(excel_path)
        elif hasattr(excel_path, 'getvalue'):
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
        if predictor_col:
            predictor_col = predictor_col.strip()
        if target_variable_col:
            target_variable_col = target_variable_col.strip()

        # 检查必需列是否存在
        if indicator_col not in indicator_sheet.columns or type_col not in indicator_sheet.columns:
            raise ValueError(f"未找到必需的列 '{indicator_col}' 或 '{type_col}' 在 sheet '{sheet_name}'")

        # 创建类型映射
        var_type_map_temp = pd.Series(
            indicator_sheet[type_col].astype(str).str.strip().values,
            index=indicator_sheet[indicator_col].astype(str).str.strip()
        ).to_dict()

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
                index=indicator_sheet[indicator_col].astype(str).str.strip()
            ).to_dict()

            var_industry_map = {
                normalize_text(k): str(v).strip()
                for k, v in industry_map_temp.items()
                if pd.notna(k) and str(k).strip().lower() not in ['', 'nan']
                and pd.notna(v) and str(v).strip().lower() not in ['', 'nan']
            }
            logger.info("[Mappings] Successfully created industry map with %d entries.", len(var_industry_map))
        elif industry_col:
            logger.warning("[Mappings] Industry column '%s' not found in sheet '%s'. Industry map will be empty.", industry_col, sheet_name)

        # 创建预测变量映射
        if predictor_col and predictor_col in indicator_sheet.columns:
            predictor_map_temp = pd.Series(
                indicator_sheet[predictor_col].astype(str).str.strip().values,
                index=indicator_sheet[indicator_col].astype(str).str.strip()
            ).to_dict()

            var_predictor_map = {
                normalize_text(k): str(v).strip()
                for k, v in predictor_map_temp.items()
                if pd.notna(k) and str(k).strip().lower() not in ['', 'nan']
                and pd.notna(v) and str(v).strip() == '是'
            }
            logger.info("[Mappings] 从预测变量列加载了 %d 个标记为'是'的变量", len(var_predictor_map))
        else:
            logger.warning("[Mappings] 预测变量列未找到")

        # 创建目标变量映射
        if target_variable_col and target_variable_col in indicator_sheet.columns:
            target_variable_map_temp = pd.Series(
                indicator_sheet[target_variable_col].astype(str).str.strip().values,
                index=indicator_sheet[indicator_col].astype(str).str.strip()
            ).to_dict()

            var_target_variable_map = {
                normalize_text(k): str(v).strip()
                for k, v in target_variable_map_temp.items()
                if pd.notna(k) and str(k).strip().lower() not in ['', 'nan']
                and pd.notna(v) and str(v).strip() == '是'
            }
            logger.info("[Mappings] 从目标变量列加载了 %d 个标记为'是'的变量", len(var_target_variable_map))
        else:
            logger.warning("[Mappings] 目标变量列未找到")

    except FileNotFoundError as e:
        logger.error("Error loading mappings: %s", e)
    except ValueError as e:
        logger.error("Error processing mapping sheet: %s", e)
    except Exception as e:
        logger.error("An unexpected error occurred while loading mappings: %s", e)

    logger.info("[Mappings] Loading finished. Type map: %d, Industry map: %d, Predictor map: %d, Target variable map: %d",
                len(var_type_map), len(var_industry_map), len(var_predictor_map), len(var_target_variable_map))
    return var_type_map, var_industry_map, var_predictor_map, var_target_variable_map


def create_industry_map_from_data(
    final_columns: set,
    var_industry_map_loaded: Dict[str, str],
    default_industry: str = "Unknown"
) -> Dict[str, str]:
    """
    根据最终数据列和加载的行业映射创建完整的行业映射

    Args:
        final_columns: 最终数据的列名集合
        var_industry_map_loaded: 从指标体系加载的行业映射
        default_industry: 未找到映射时的默认行业

    Returns:
        Dict[str, str]: 完整的行业映射
    """
    final_industry_map = {}
    for col in final_columns:
        col_norm = normalize_text(col)
        if col_norm in var_industry_map_loaded:
            final_industry_map[col_norm] = var_industry_map_loaded[col_norm]
        else:
            final_industry_map[col_norm] = default_industry
    return final_industry_map
