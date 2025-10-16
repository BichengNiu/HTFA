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

        logger.info(f"{sheet_name}数据形状: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"读取{sheet_name}数据失败: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_weights_data(uploaded_file, sheet_name: str = '工业增加值分行业指标权重') -> Optional[pd.DataFrame]:
    """
    加载权重数据：第一行是列名，第一列是指标名称（保留为普通列，不设为索引）

    性能优化：添加缓存装饰器，避免重复读取相同文件（缓存1小时）

    Args:
        uploaded_file: Streamlit uploaded file object 或文件路径
        sheet_name: Excel工作表名称

    Returns:
        DataFrame: 包含权重数据，包含列：指标名称、出口依赖、上中下游、权重_2012、权重_2018、权重_2020；
                  如果失败返回None
    """
    try:
        logger.info(f"读取{sheet_name}数据")

        # 读取权重数据：第一行是列名，第一列是指标名称（不作为索引）
        df_weights = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=0)

        # 清理数据：删除全为空的行和列
        df_weights = df_weights.dropna(how='all').dropna(axis=1, how='all')

        logger.info(f"{sheet_name}数据形状: {df_weights.shape}")
        logger.info(f"{sheet_name}数据列名: {list(df_weights.columns)}")

        # 验证必要的列是否存在
        if '指标名称' in df_weights.columns:
            valid_indicators = df_weights['指标名称'].notna().sum()
            logger.info(f"找到'指标名称'列，包含 {valid_indicators} 个有效指标")
        else:
            logger.error("未找到'指标名称'列")

        return df_weights

    except Exception as e:
        logger.error(f"读取{sheet_name}数据失败: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_overall_industrial_data(uploaded_file, sheet_name: str = '总体工业增加值同比增速') -> Optional[pd.DataFrame]:
    """
    使用统一格式加载总体工业增加值数据：第一行是列名，第一列是时间列

    特殊处理：对"规模以上工业增加值:当月同比"的1月和2月数据设为NaN

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

        # 特殊处理：对"规模以上工业增加值:当月同比"的1月和2月数据设为NaN
        target_column = "规模以上工业增加值:当月同比"
        if target_column in df.columns and hasattr(df.index, 'month'):
            jan_feb_mask = (df.index.month == 1) | (df.index.month == 2)
            import numpy as np
            df.loc[jan_feb_mask, target_column] = np.nan

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
def load_enterprise_profit_data(uploaded_file, sheet_name: str = '工业企业利润拆解') -> Optional[pd.DataFrame]:
    """
    使用统一格式读取工业企业利润拆解数据：第一行是列名，第一列是时间列

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
