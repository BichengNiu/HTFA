"""
Data Loader Utility
数据加载工具 - 统一的Excel数据加载函数（带缓存）

消除重复代码:
- 原 macro_operations.py:load_template_data (66-91行)
- 原 data_processor.py:load_macro_data (47-65行)
- 原 macro_operations.py:load_weights_data (93-127行)
- 原 data_processor.py:load_weights_data (16-44行)
- 原 macro_operations.py:load_overall_industrial_data (146-177行)

性能优化:
- 添加@st.cache_data装饰器，避免重复读取相同文件
- 缓存基于文件内容的哈希值，文件内容变化时自动更新
"""

import pandas as pd
from typing import Optional
import logging
import streamlit as st
from io import BytesIO

logger = logging.getLogger(__name__)

# 注：@st.cache_data装饰器会自动根据函数参数（包括uploaded_file的内容）生成缓存键
# 不需要手动计算MD5哈希，Streamlit会自动处理文件内容的哈希


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro_data(uploaded_file, sheet_name: str = '分行业工业增加值同比增速') -> Optional[pd.DataFrame]:
    """
    使用统一格式加载宏观工业数据：第一行是列名，第一列是时间列

    性能优化：添加缓存装饰器，避免重复读取相同文件（缓存1小时）
    数据清洗：将0值转换为NaN（新版本数据中0代表缺失值）

    Args:
        uploaded_file: Streamlit uploaded file object 或文件路径
        sheet_name: Excel工作表名称

    Returns:
        DataFrame: 包含宏观工业数据，第一列为时间索引；如果失败返回None
    """
    try:
        logger.info(f"读取{sheet_name}数据")

        # 统一格式读取：第一行是列名，第一列是时间
        df = pd.read_excel(
            uploaded_file,
            sheet_name=sheet_name,
            header=0,
            index_col=0,
            parse_dates=True
        )

        # 清理数据：删除全为空的行和列
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # 将0值转换为NaN（新版本数据中0代表缺失值）
        import numpy as np
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace(0, np.nan)

        logger.info(f"{sheet_name}数据形状: {df.shape}")
        logger.info(f"已将0值转换为NaN")

        return df

    except Exception as e:
        logger.error(f"读取{sheet_name}数据失败: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_weights_data() -> Optional[pd.DataFrame]:
    """
    加载权重数据：从内部CSV文件读取行业属性和权重

    内部数据文件：data/工业分行业属性及权重.csv
    包含列：指标名称、门类、出口依赖、上中下游、权重_2012、权重_2018、
           权重_2020、权重_2022、权重_2023、权重_2024、权重_2025

    性能优化：添加缓存装饰器，避免重复读取相同文件（缓存1小时）

    Returns:
        DataFrame: 包含权重数据；如果失败返回None
    """
    try:
        from dashboard.analysis.industrial.constants import INTERNAL_WEIGHTS_FILE_PATH
        from pathlib import Path

        # 获取项目根目录（data_loader.py的位置是dashboard/analysis/industrial/utils/）
        # utils -> industrial -> analysis -> dashboard -> 项目根目录
        project_root = Path(__file__).parent.parent.parent.parent.parent
        weights_file = project_root / INTERNAL_WEIGHTS_FILE_PATH

        logger.info(f"从内部文件读取权重数据: {weights_file}")

        if not weights_file.exists():
            logger.error(f"内部权重文件不存在: {weights_file}")
            return None

        # 读取CSV文件（使用utf-8-sig处理BOM）
        df_weights = pd.read_csv(weights_file, encoding='utf-8-sig')

        # 清理数据：删除全为空的行和列
        df_weights = df_weights.dropna(how='all').dropna(axis=1, how='all')

        logger.info(f"权重数据形状: {df_weights.shape}")
        logger.info(f"权重数据列名: {list(df_weights.columns)}")

        # 验证必要的列是否存在
        if '指标名称' in df_weights.columns:
            valid_indicators = df_weights['指标名称'].notna().sum()
            logger.info(f"找到 {valid_indicators} 个有效指标")
        else:
            logger.error("未找到'指标名称'列")
            return None

        return df_weights

    except Exception as e:
        logger.error(f"读取内部权重数据失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_overall_industrial_data(uploaded_file, sheet_name: str = '总体工业增加值同比增速') -> Optional[pd.DataFrame]:
    """
    使用统一格式加载总体工业增加值数据：第一行是列名，第一列是时间列

    特殊处理：
    1. 对总体工业增加值当月同比的1月和2月数据设为NaN
    2. 将0值转换为NaN（新版本数据中0代表缺失值）
    3. 兼容新旧两种列名格式

    性能优化：添加缓存装饰器，避免重复读取相同文件（缓存1小时）

    Args:
        uploaded_file: Streamlit uploaded file object 或文件路径
        sheet_name: Excel工作表名称

    Returns:
        DataFrame: 包含总体工业增加值数据；如果失败返回None
    """
    try:
        logger.info(f"读取{sheet_name}数据")

        # 统一格式读取：第一行是列名，第一列是时间
        df = pd.read_excel(
            uploaded_file,
            sheet_name=sheet_name,
            header=0,
            index_col=0,
            parse_dates=True
        )

        # 清理数据：删除全为空的行和列
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # 将0值转换为NaN（新版本数据中0代表缺失值）
        import numpy as np
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace(0, np.nan)

        logger.info(f"已将0值转换为NaN")

        # 特殊处理：对总体工业增加值当月同比的1月和2月数据设为NaN
        # 兼容新旧两种列名格式
        target_column_old = "规模以上工业增加值:当月同比"
        target_column_new = "中国:工业增加值:规模以上工业企业:当月同比"

        target_column = None
        if target_column_old in df.columns:
            target_column = target_column_old
        elif target_column_new in df.columns:
            target_column = target_column_new

        if target_column and hasattr(df.index, 'month'):
            jan_feb_mask = (df.index.month == 1) | (df.index.month == 2)
            df.loc[jan_feb_mask, target_column] = np.nan
            logger.info(f"已将{target_column}的1月和2月数据设为NaN")

        logger.info(f"{sheet_name}数据形状: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"读取{sheet_name}数据失败: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_profit_breakdown_data(uploaded_file, sheet_name: str = '分上中下游利润拆解') -> Optional[pd.DataFrame]:
    """
    使用统一格式读取分上中下游利润拆解数据：第一行是列名，第一列是时间列

    性能优化：添加缓存装饰器，避免重复读取相同文件（缓存1小时）

    Args:
        uploaded_file: 上传的Excel文件对象或文件路径
        sheet_name: Excel工作表名称

    Returns:
        DataFrame: 包含分上中下游利润拆解数据，如果读取失败则返回None
    """
    try:
        logger.info(f"读取{sheet_name}数据")

        # 处理不同类型的文件输入
        file_input = uploaded_file
        if hasattr(uploaded_file, 'getvalue'):
            file_input = BytesIO(uploaded_file.getvalue())
        elif hasattr(uploaded_file, 'path'):
            file_input = uploaded_file.path

        # 统一格式读取：第一行是列名，第一列是时间
        df = pd.read_excel(
            file_input,
            sheet_name=sheet_name,
            header=0,
            index_col=0,
            parse_dates=True
        )

        # 清理数据：删除全为空的行和列
        df = df.dropna(how='all').dropna(axis=1, how='all')

        logger.info(f"{sheet_name}数据形状: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"读取{sheet_name}数据失败: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_enterprise_profit_data(uploaded_file, sheet_name: str = '工业企业利润') -> Optional[pd.DataFrame]:
    """
    使用统一格式读取工业企业利润数据：第一行是列名，第一列是时间列

    性能优化：添加缓存装饰器，避免重复读取相同文件（缓存1小时）

    Args:
        uploaded_file: 上传的Excel文件对象或文件路径
        sheet_name: Excel工作表名称

    Returns:
        DataFrame: 包含企业利润拆解数据，如果读取失败则返回None
    """
    try:
        logger.info(f"读取{sheet_name}数据")

        # 处理不同类型的文件输入
        file_input = uploaded_file
        if hasattr(uploaded_file, 'getvalue'):
            file_input = BytesIO(uploaded_file.getvalue())
        elif hasattr(uploaded_file, 'path'):
            file_input = uploaded_file.path

        # 统一格式读取：第一行是列名，第一列是时间（不设置为索引）
        df = pd.read_excel(
            file_input,
            sheet_name=sheet_name,
            header=0
        )

        # 清理数据：删除全为空的行和列
        df = df.dropna(how='all').dropna(axis=1, how='all')

        logger.info(f"{sheet_name}数据形状: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"读取{sheet_name}数据失败: {e}")
        return None
