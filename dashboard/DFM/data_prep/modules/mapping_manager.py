"""
映射管理模块

负责加载和管理变量类型映射和行业映射
"""

import pandas as pd
import io
from typing import Dict, Tuple, Optional, Union

from dashboard.DFM.data_prep.utils.text_utils import normalize_text

def load_mappings(
    excel_path: Union[str, io.BytesIO, object],
    sheet_name: str,
    indicator_col: str = '指标名称',
    type_col: str = '类型',
    industry_col: Optional[str] = '行业'  # Industry column is optional
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    从指定的Excel表格中加载变量类型和行业映射

    将指标名称（键）标准化为小写NFKC格式

    Args:
        excel_path: Excel文件路径、BytesIO对象或类文件对象
        sheet_name: 表格名称
        indicator_col: 指标列名，默认'指标名称'
        type_col: 类型列名，默认'类型'
        industry_col: 行业列名，默认'行业'，可选

    Returns:
        Tuple[Dict[str, str], Dict[str, str]]: (变量类型映射, 变量行业映射)
    """
    var_type_map = {}
    var_industry_map = {}

    print(f"\n--- [Mappings] Loading type/industry maps from: ")
    print(f"    Excel: {excel_path}")
    print(f"    Sheet: {sheet_name}")
    print(f"    Indicator Col: '{indicator_col}', Type Col: '{type_col}', Industry Col: '{industry_col}'")

    try:
        # 处理不同类型的输入
        if isinstance(excel_path, str):
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
        print(f"  [Mappings] Successfully created type map with {len(var_type_map)} entries.")

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
            print(f"  [Mappings] Successfully created industry map with {len(var_industry_map)} entries.")
        elif industry_col:
             print(f"  [Mappings] Warning: Industry column '{industry_col}' not found in sheet '{sheet_name}'. Industry map will be empty.")
        else:
             print(f"  [Mappings] Industry column not specified. Industry map will be empty.")

    except FileNotFoundError as e:
        print(f"Error loading mappings: {e}")
        # Return empty maps on file/sheet not found
    except ValueError as e:
        print(f"Error processing mapping sheet: {e}")
        # Return empty maps on column errors
    except Exception as e:
        print(f"An unexpected error occurred while loading mappings: {e}")
        # Return empty maps on other errors

    print(f"--- [Mappings] Loading finished. Type map size: {len(var_type_map)}, Industry map size: {len(var_industry_map)} ---")
    return var_type_map, var_industry_map

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
