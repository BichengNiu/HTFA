"""
工业增加值拉动率计算模块

计算各分组和单个行业对总体工业增加值增速的拉动率。
拉动率公式：拉动率 = 增速 × 权重
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from dashboard.core.ui.utils.debug_helpers import debug_log


def prepare_weights_series_for_contribution(
    weights_mapping: Dict[str, Dict],
    time_index: pd.DatetimeIndex
) -> Dict[str, pd.Series]:
    """
    为每个指标准备权重时间序列

    Args:
        weights_mapping: 权重映射 {指标名: {出口依赖, 上中下游, weights_row}}
        time_index: 时间索引

    Returns:
        {指标名: 权重Series}
    """
    from .weight_calculator import get_weight_for_year

    weights_series_mapping = {}

    for indicator, info in weights_mapping.items():
        weights_row = info['weights_row']

        weights_list = []
        for timestamp in time_index:
            year = timestamp.year
            weight = get_weight_for_year(weights_row, year)
            weights_list.append(weight)

        weights_series_mapping[indicator] = pd.Series(
            weights_list,
            index=time_index,
            name=indicator
        )

    return weights_series_mapping


def calculate_individual_contributions(
    df_macro: pd.DataFrame,
    weights_mapping: Dict[str, Dict]
) -> pd.DataFrame:
    """
    计算单个行业对总体工业增加值的拉动率

    拉动率 = 行业增速 × 行业权重

    Args:
        df_macro: 宏观数据（各行业增速），已过滤2012年及以后
        weights_mapping: 权重映射

    Returns:
        DataFrame，列为各行业拉动率，索引为时间
    """
    debug_log("开始计算单个行业拉动率", "DEBUG")

    weights_series_mapping = prepare_weights_series_for_contribution(
        weights_mapping, df_macro.index
    )

    contribution_df = pd.DataFrame(index=df_macro.index)
    missing_weights = []

    for indicator in df_macro.columns:
        if indicator not in weights_series_mapping:
            missing_weights.append(indicator)
            continue

        growth_series = df_macro[indicator]
        weight_series = weights_series_mapping[indicator]

        contribution_series = growth_series * weight_series
        contribution_df[indicator] = contribution_series

    if missing_weights:
        debug_log(
            f"警告: {len(missing_weights)} 个指标缺少权重，这些行业的拉动率将被忽略",
            "WARNING"
        )
        debug_log(f"缺少权重的指标: {missing_weights}", "WARNING")

    debug_log(f"单个行业拉动率计算完成，共 {len(contribution_df.columns)} 个行业", "DEBUG")

    return contribution_df


def calculate_group_contributions(
    df_macro: pd.DataFrame,
    weights_mapping: Dict[str, Dict],
    groups: Dict[str, List[str]],
    prefix: str = "",
    df_overall_growth: Optional[pd.DataFrame] = None,
    industry_to_column_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    计算分组对总体工业增加值的拉动率

    分组拉动率 = Σ(组内各行业拉动率)

    注意：对于三大产业分组，如果提供了df_overall_growth和industry_to_column_map，
    将优先使用"总体增速 × 总权重"的方式计算拉动率，以避免细分行业数据不一致性导致的误差。

    Args:
        df_macro: 宏观数据（各行业增速），已过滤2012年及以后
        weights_mapping: 权重映射
        groups: 分组字典 {组名: [指标列表]}
        prefix: 列名前缀
        df_overall_growth: 总体增速数据（可选），用于三大产业拉动率计算
        industry_to_column_map: 产业名称到总体增速列名的映射（可选）

    Returns:
        DataFrame，列为各分组拉动率，索引为时间
    """
    debug_log(f"开始计算分组拉动率（前缀: {prefix}），共 {len(groups)} 组", "DEBUG")

    individual_contributions = calculate_individual_contributions(
        df_macro, weights_mapping
    )

    group_contribution_df = pd.DataFrame(index=df_macro.index)

    # 标记是否是三大产业分组
    is_three_industries = (prefix == "三大产业_")
    use_overall_growth = (
        is_three_industries
        and df_overall_growth is not None
        and industry_to_column_map is not None
    )

    for group_name, indicators in groups.items():
        valid_indicators = [
            ind for ind in indicators
            if ind in individual_contributions.columns
        ]

        if not valid_indicators:
            debug_log(f"警告: 分组 {group_name} 没有有效指标", "WARNING")
            continue

        # 对于三大产业，优先使用总体增速计算
        if use_overall_growth and group_name in industry_to_column_map:
            overall_col = industry_to_column_map[group_name]

            if overall_col in df_overall_growth.columns:
                # 计算该产业的总权重
                total_weight_series = pd.Series(0.0, index=df_macro.index)

                for indicator in valid_indicators:
                    if indicator in weights_mapping:
                        weights_row = weights_mapping[indicator]['weights_row']
                        weight_series = pd.Series(index=df_macro.index, dtype=float)

                        for timestamp in df_macro.index:
                            year = timestamp.year
                            from .weight_calculator import get_weight_for_year
                            weight = get_weight_for_year(weights_row, year)
                            weight_series.loc[timestamp] = weight

                        total_weight_series += weight_series

                # 对齐索引
                overall_growth_series = df_overall_growth[overall_col].reindex(df_macro.index)

                # 拉动率 = 总体增速 × 总权重
                group_contribution = overall_growth_series * total_weight_series

                debug_log(
                    f"分组 {group_name} 使用总体增速计算拉动率（{len(valid_indicators)} 个指标）",
                    "INFO"
                )
            else:
                # 如果总体增速列不存在，回退到细分行业加权和
                group_contribution = individual_contributions[valid_indicators].sum(axis=1)
                debug_log(
                    f"警告: 分组 {group_name} 总体增速列 '{overall_col}' 不存在，使用细分行业加权和",
                    "WARNING"
                )
        else:
            # 其他分组使用细分行业加权和
            group_contribution = individual_contributions[valid_indicators].sum(axis=1)

        column_name = f"{prefix}{group_name}"
        group_contribution_df[column_name] = group_contribution

        if not use_overall_growth or group_name not in industry_to_column_map:
            debug_log(
                f"分组 {group_name} 拉动率计算完成，包含 {len(valid_indicators)} 个指标",
                "DEBUG"
            )

    return group_contribution_df


def validate_contributions(
    contribution_df: pd.DataFrame,
    total_growth: pd.Series,
    tolerance: float = 0.5
) -> Tuple[bool, pd.Series, pd.Series]:
    """
    验证拉动率总和是否等于总体增速

    Args:
        contribution_df: 拉动率DataFrame（各组或各行业）
        total_growth: 总体工业增加值增速Series
        tolerance: 容忍误差（百分点）

    Returns:
        (验证通过, 拉动率总和Series, 差值Series)
    """
    debug_log("开始验证拉动率总和", "DEBUG")

    contribution_df_aligned = contribution_df.reindex(total_growth.index)
    contribution_sum = contribution_df_aligned.sum(axis=1)

    difference = contribution_sum - total_growth

    max_diff = difference.abs().max()
    mean_diff = difference.abs().mean()

    validation_passed = max_diff <= tolerance

    debug_log(
        f"拉动率验证结果: 最大误差={max_diff:.4f}百分点, "
        f"平均误差={mean_diff:.4f}百分点, "
        f"容忍阈值={tolerance}百分点",
        "INFO"
    )

    if not validation_passed:
        debug_log(
            f"警告: 拉动率总和与总体增速差异超过容忍阈值 "
            f"(最大误差 {max_diff:.4f} > {tolerance})",
            "WARNING"
        )

        max_diff_date = difference.abs().idxmax()
        debug_log(
            f"最大误差发生在: {max_diff_date}, "
            f"拉动率总和={contribution_sum.loc[max_diff_date]:.4f}, "
            f"总体增速={total_growth.loc[max_diff_date]:.4f}",
            "WARNING"
        )

    return validation_passed, contribution_sum, difference


def _find_column_by_keywords(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """
    根据关键词列表查找列名（兼容新旧格式）

    Args:
        df: DataFrame
        keywords: 关键词列表，按优先级排序

    Returns:
        找到的列名，如果未找到返回None
    """
    for col in df.columns:
        if all(keyword in col for keyword in keywords):
            return col
    return None


def calculate_all_contributions(
    df_macro: pd.DataFrame,
    df_weights: pd.DataFrame,
    total_growth_column: Optional[str] = None,
    df_overall_growth: Optional[pd.DataFrame] = None
) -> Dict[str, pd.DataFrame]:
    """
    统一计算所有拉动率（分组+单个行业）

    Args:
        df_macro: 宏观数据（各行业增速），已过滤2012年及以后
        df_weights: 权重数据
        total_growth_column: 总体增速列名（可选，自动检测）
        df_overall_growth: 总体增速数据（可选），包含三大产业的总体增速，用于提高三大产业拉动率准确性

    Returns:
        {
            'export_groups': 出口依赖分组拉动率,
            'stream_groups': 上中下游分组拉动率,
            'industry_groups': 三大产业分组拉动率,
            'individual': 单个行业拉动率,
            'total_growth': 总体增速Series,
            'validation': 验证结果
        }
    """
    from .weighted_calculation import build_weights_mapping, categorize_indicators

    debug_log("开始统一拉动率计算流程", "INFO")

    # 自动检测总体增速列名（兼容新旧格式）
    if total_growth_column is None:
        total_growth_column = _find_column_by_keywords(df_macro, ['工业增加值', '当月同比'])
        if total_growth_column is None:
            # 尝试旧格式
            if '规模以上工业增加值:当月同比' in df_macro.columns:
                total_growth_column = '规模以上工业增加值:当月同比'
            else:
                raise ValueError(
                    f"无法自动检测总体增速列。可用列: {list(df_macro.columns)}"
                )
        debug_log(f"自动检测到总体增速列: {total_growth_column}", "INFO")

    if total_growth_column not in df_macro.columns:
        raise ValueError(
            f"总体增速列 '{total_growth_column}' 不存在。"
            f"可用列: {list(df_macro.columns)}"
        )

    total_growth = df_macro[total_growth_column]

    target_columns = [col for col in df_macro.columns if col != total_growth_column]

    weights_mapping = build_weights_mapping(df_weights, target_columns)

    export_groups, stream_groups, industry_groups = categorize_indicators(weights_mapping)

    export_contribution = calculate_group_contributions(
        df_macro, weights_mapping, export_groups, '出口依赖_'
    )

    stream_contribution = calculate_group_contributions(
        df_macro, weights_mapping, stream_groups, '上中下游_'
    )

    # 三大产业分组拉动率：如果提供了总体增速数据，使用总体增速计算
    # 兼容新旧两种格式的列名映射
    industry_to_column_map = {}

    # 采矿业列名检测
    mining_col = _find_column_by_keywords(df_macro, ['采矿业', '当月同比'])
    if mining_col:
        industry_to_column_map['采矿业'] = mining_col

    # 制造业列名检测
    manufacturing_col = _find_column_by_keywords(df_macro, ['制造业', '当月同比'])
    if manufacturing_col:
        industry_to_column_map['制造业'] = manufacturing_col

    # 电力、燃气及水列名检测（兼容新旧两种表述）
    utilities_col = (_find_column_by_keywords(df_macro, ['电力', '当月同比']) or
                     _find_column_by_keywords(df_macro, ['燃气', '当月同比']))
    if utilities_col:
        industry_to_column_map['电力、热力、燃气及水生产和供应业'] = utilities_col

    debug_log(f"三大产业列名映射: {industry_to_column_map}", "INFO")

    industry_contribution = calculate_group_contributions(
        df_macro, weights_mapping, industry_groups, '三大产业_',
        df_overall_growth=df_overall_growth,
        industry_to_column_map=industry_to_column_map
    )

    individual_contribution = calculate_individual_contributions(
        df_macro, weights_mapping
    )

    validation_passed, contribution_sum, difference = validate_contributions(
        individual_contribution, total_growth
    )

    result = {
        'export_groups': export_contribution,
        'stream_groups': stream_contribution,
        'industry_groups': industry_contribution,
        'individual': individual_contribution,
        'total_growth': total_growth,
        'validation': {
            'passed': validation_passed,
            'contribution_sum': contribution_sum,
            'difference': difference
        }
    }

    debug_log(
        f"拉动率计算完成: "
        f"出口依赖 {len(export_contribution.columns)} 组, "
        f"上中下游 {len(stream_contribution.columns)} 组, "
        f"三大产业 {len(industry_contribution.columns)} 组, "
        f"单个行业 {len(individual_contribution.columns)} 个",
        "INFO"
    )

    return result
