# -*- coding: utf-8 -*-
"""
行业数据处理工具

用于二次估计法的行业数据提取、分组和验证
"""

import re
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class IndustryDataInfo:
    """行业数据信息"""
    industry_name: str
    target_column: str
    predictor_columns: List[str]
    data_points: int
    missing_rate: float


class IndustryDataProcessor:
    """
    行业数据处理器

    职责：
    1. 从prepared_data中提取分行业工业增加值列
    2. 根据var_industry_map构建行业→预测变量映射
    3. 验证各行业数据完整性
    """

    INDUSTRY_TARGET_PATTERN = r'工业增加值:.*?:([^:]+):当月同比'

    def __init__(self, prepared_data: pd.DataFrame, var_industry_map: Dict[str, str]):
        """
        初始化行业数据处理器

        Args:
            prepared_data: 数据准备阶段生成的完整数据表
            var_industry_map: 变量名→行业名映射字典
        """
        self.prepared_data = prepared_data
        self.var_industry_map = var_industry_map

        self._industry_list = None
        self._industry_targets = None
        self._industry_to_predictors = None

    def get_industry_list(self) -> List[str]:
        """
        从prepared_data列名中提取行业列表

        Returns:
            按字母顺序排序的行业名称列表

        Example:
            ['专用设备制造业', '农副食品加工业', '化学原料及化学制品制造业', ...]
        """
        if self._industry_list is not None:
            return self._industry_list

        industries = set()

        for col in self.prepared_data.columns:
            match = re.search(self.INDUSTRY_TARGET_PATTERN, col)
            if match:
                industry_name = match.group(1)
                industries.add(industry_name)

        self._industry_list = sorted(list(industries))
        return self._industry_list

    def get_industry_target_column(self, industry: str) -> Optional[str]:
        """
        获取指定行业的目标变量列名

        Args:
            industry: 行业名称

        Returns:
            目标变量列名，如果不存在返回None

        Example:
            >>> processor.get_industry_target_column('专用设备制造业')
            '中国:工业增加值:规模以上工业企业:专用设备制造业:当月同比'
        """
        if self._industry_targets is None:
            self._build_industry_targets()

        return self._industry_targets.get(industry)

    def get_industry_predictors(self, industry: str, exclude_target: bool = True) -> List[str]:
        """
        获取指定行业的预测变量列表

        Args:
            industry: 行业名称
            exclude_target: 是否排除该行业的目标变量（默认True）

        Returns:
            预测变量列名列表

        Example:
            >>> processor.get_industry_predictors('专用设备制造业')
            ['用电量:专用设备制造业', '中国:产能利用率:专用设备', ...]
        """
        if self._industry_to_predictors is None:
            self._build_industry_to_predictors_map()

        predictors = self._industry_to_predictors.get(industry, [])

        if exclude_target:
            target_col = self.get_industry_target_column(industry)
            if target_col:
                predictors = [p for p in predictors if p != target_col]

        return predictors

    def get_all_industry_info(self, min_predictors: int = 3) -> Dict[str, IndustryDataInfo]:
        """
        获取所有行业的数据信息

        Args:
            min_predictors: 最少预测变量数量要求（默认3个）

        Returns:
            行业名→数据信息字典
        """
        industry_list = self.get_industry_list()
        info_dict = {}

        for industry in industry_list:
            target_col = self.get_industry_target_column(industry)
            predictor_cols = self.get_industry_predictors(industry)

            if not target_col:
                continue

            if len(predictor_cols) < min_predictors:
                continue

            target_data = self.prepared_data[target_col]
            data_points = target_data.notna().sum()
            total_points = len(target_data)
            missing_rate = (total_points - data_points) / total_points if total_points > 0 else 1.0

            info_dict[industry] = IndustryDataInfo(
                industry_name=industry,
                target_column=target_col,
                predictor_columns=predictor_cols,
                data_points=data_points,
                missing_rate=missing_rate
            )

        return info_dict

    def validate_industry_data(
        self,
        industry: str,
        min_predictors: int = 1
    ) -> Tuple[bool, Optional[str]]:
        """
        验证指定行业的数据完整性

        Args:
            industry: 行业名称
            min_predictors: 最少预测变量数量

        Returns:
            (是否有效, 错误信息)
        """
        target_col = self.get_industry_target_column(industry)
        if not target_col:
            return False, f"未找到行业 {industry} 的目标变量列"

        if target_col not in self.prepared_data.columns:
            return False, f"目标变量列 {target_col} 不存在于数据中"

        predictors = self.get_industry_predictors(industry)
        if len(predictors) < min_predictors:
            return False, f"行业 {industry} 预测变量数量不足（需要至少{min_predictors}个，实际{len(predictors)}个）"

        missing_predictors = [p for p in predictors if p not in self.prepared_data.columns]
        if missing_predictors:
            return False, f"预测变量不存在于数据中: {missing_predictors[:3]}"

        target_data = self.prepared_data[target_col]
        data_points = target_data.notna().sum()

        if data_points == 0:
            return False, f"目标变量 {target_col} 无有效数据"

        return True, None

    def build_industry_training_data(
        self,
        industry: str,
        include_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        构建指定行业的训练数据

        Args:
            industry: 行业名称
            include_columns: 额外包含的列（可选）

        Returns:
            包含目标变量和预测变量的DataFrame
        """
        target_col = self.get_industry_target_column(industry)
        predictor_cols = self.get_industry_predictors(industry)

        columns_to_include = [target_col] + predictor_cols

        if include_columns:
            extra_cols = [c for c in include_columns if c in self.prepared_data.columns and c not in columns_to_include]
            columns_to_include.extend(extra_cols)

        return self.prepared_data[columns_to_include].copy()

    def _build_industry_targets(self):
        """构建行业→目标变量列名映射"""
        self._industry_targets = {}

        for col in self.prepared_data.columns:
            match = re.search(self.INDUSTRY_TARGET_PATTERN, col)
            if match:
                industry_name = match.group(1)
                self._industry_targets[industry_name] = col

    def _build_industry_to_predictors_map(self):
        """根据var_industry_map构建行业→预测变量映射"""
        self._industry_to_predictors = {}

        # 导入normalize_text函数，与行业映射文件保持一致的标准化方式
        from dashboard.models.DFM.prep.utils.text_utils import normalize_text

        # 构建标准化列名到原始列名的映射
        col_name_map = {
            normalize_text(col): col
            for col in self.prepared_data.columns
        }

        # 统计匹配情况
        total_vars = len(self.var_industry_map)
        matched_vars = 0
        unmatched_vars = []

        for var_name, industry_name in self.var_industry_map.items():
            if industry_name not in self._industry_to_predictors:
                self._industry_to_predictors[industry_name] = []

            # var_name已经是标准化的形式，直接查找对应的原始列名
            if var_name in col_name_map:
                original_col_name = col_name_map[var_name]
                self._industry_to_predictors[industry_name].append(original_col_name)
                matched_vars += 1
            else:
                unmatched_vars.append((var_name, industry_name))

        # 打印匹配统计
        print(f"[行业映射匹配] 总变量数: {total_vars}, 成功匹配: {matched_vars}, 未匹配: {len(unmatched_vars)}")

        if unmatched_vars and len(unmatched_vars) <= 10:
            print(f"[行业映射匹配] 未匹配变量示例: {unmatched_vars[:5]}")

        # 打印各行业变量统计
        print(f"[行业映射匹配] 各行业变量数量:")
        for industry, predictors in sorted(self._industry_to_predictors.items()):
            print(f"  - {industry}: {len(predictors)} 个变量")


def extract_industry_list(prepared_data: pd.DataFrame) -> List[str]:
    """
    从prepared_data列名中提取行业列表（快捷函数）

    Args:
        prepared_data: 数据准备阶段生成的完整数据表

    Returns:
        按字母顺序排序的行业名称列表
    """
    pattern = r'工业增加值:.*?:([^:]+):当月同比'
    industries = set()

    for col in prepared_data.columns:
        match = re.search(pattern, col)
        if match:
            industry_name = match.group(1)
            industries.add(industry_name)

    return sorted(list(industries))


def build_industry_to_vars_map(var_industry_map: Dict[str, str]) -> Dict[str, List[str]]:
    """
    将变量→行业映射转换为行业→变量列表映射（快捷函数）

    Args:
        var_industry_map: 变量名→行业名映射

    Returns:
        行业名→变量列表映射
    """
    industry_to_vars = {}

    for var_name, industry_name in var_industry_map.items():
        if industry_name not in industry_to_vars:
            industry_to_vars[industry_name] = []
        industry_to_vars[industry_name].append(var_name)

    return industry_to_vars
