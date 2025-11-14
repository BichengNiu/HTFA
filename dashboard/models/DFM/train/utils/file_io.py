"""
文件读取工具模块

提供统一的数据文件读取功能，消除重复代码
支持读取Excel多sheet文件
"""
from pathlib import Path
from typing import Union, Optional, Dict
import pandas as pd


def read_data_file(
    file_path: Union[str, Path],
    index_col: int = 0,
    parse_dates: bool = True,
    check_exists: bool = True,
    sheet_name: Optional[Union[str, int]] = None
) -> pd.DataFrame:
    """
    根据文件扩展名读取数据文件（CSV或Excel）

    Args:
        file_path: 数据文件路径
        index_col: 索引列位置，默认为0
        parse_dates: 是否解析日期，默认为True
        check_exists: 是否检查文件存在性，默认为True
        sheet_name: Excel文件的sheet名称或索引（仅对Excel有效）
                   - None: 默认读取'数据'sheet（向后兼容），如不存在则读取第一个sheet
                   - str: 读取指定名称的sheet
                   - int: 读取指定索引的sheet（0为第一个）

    Returns:
        读取的DataFrame

    Raises:
        FileNotFoundError: 当check_exists=True且文件不存在时
        ValueError: 当文件格式不支持时

    Examples:
        >>> data = read_data_file('data.csv')
        >>> data = read_data_file('data.xlsx', sheet_name='数据')
        >>> mapping = read_data_file('data.xlsx', sheet_name='映射')
        >>> data = read_data_file('data.xlsx', sheet_name=0)
    """
    file_path = Path(file_path)

    if check_exists and not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    file_suffix = file_path.suffix.lower()

    if file_suffix in ['.xlsx', '.xls']:
        # Excel文件：处理sheet_name参数
        if sheet_name is None:
            # 默认尝试读取'数据'sheet，如不存在则读取第一个sheet
            try:
                return pd.read_excel(
                    str(file_path),
                    sheet_name='数据',
                    index_col=index_col,
                    parse_dates=parse_dates
                )
            except ValueError:
                # '数据'sheet不存在，读取第一个sheet（索引0）
                return pd.read_excel(
                    str(file_path),
                    sheet_name=0,
                    index_col=index_col,
                    parse_dates=parse_dates
                )
        else:
            # 读取指定的sheet
            return pd.read_excel(
                str(file_path),
                sheet_name=sheet_name,
                index_col=index_col,
                parse_dates=parse_dates
            )
    elif file_suffix == '.csv':
        return pd.read_csv(
            str(file_path),
            index_col=index_col,
            parse_dates=parse_dates
        )
    else:
        raise ValueError(
            f"不支持的文件格式: {file_suffix}，仅支持 .xlsx, .xls, .csv"
        )


def read_excel_with_mapping(
    file_path: Union[str, Path],
    data_sheet_name: str = '数据',
    mapping_sheet_name: str = '映射',
    index_col: int = 0,
    parse_dates: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    读取包含数据和映射两个sheet的Excel文件

    Args:
        file_path: Excel文件路径
        data_sheet_name: 数据sheet名称，默认'数据'
        mapping_sheet_name: 映射sheet名称，默认'映射'
        index_col: 数据sheet的索引列位置，默认为0
        parse_dates: 是否解析日期，默认为True

    Returns:
        包含'data'和'mapping'两个键的字典

    Examples:
        >>> result = read_excel_with_mapping('prepared_data.xlsx')
        >>> data_df = result['data']
        >>> mapping_df = result['mapping']
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 读取数据sheet
    data_df = pd.read_excel(
        str(file_path),
        sheet_name=data_sheet_name,
        index_col=index_col,
        parse_dates=parse_dates
    )

    # 读取映射sheet
    mapping_df = pd.read_excel(
        str(file_path),
        sheet_name=mapping_sheet_name
    )

    return {
        'data': data_df,
        'mapping': mapping_df
    }


__all__ = ['read_data_file', 'read_excel_with_mapping']
