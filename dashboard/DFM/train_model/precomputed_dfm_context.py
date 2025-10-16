# -*- coding: utf-8 -*-
"""
DFM预计算上下文模块
用于预计算DFM评估中的重复计算部分，提升变量选择过程中的性能
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dashboard.DFM.train_model.dfm_core import _validate_inputs, _prepare_data, _clean_and_validate_data, _mask_seasonal_data

logger = logging.getLogger(__name__)


class PrecomputedDFMContext:
    """
    DFM预计算上下文类
    
    在变量选择过程中，许多计算步骤（数据准备、清理、验证等）在不同变量组合间是重复的。
    此类预计算这些共同步骤，避免重复计算，提升性能。
    
    特性：
    1. 预计算数据准备和清理步骤
    2. 缓存处理后的数据状态
    3. 为不同变量组合提供优化的数据访问
    4. 统计性能提升情况
    """
    
    def __init__(
        self,
        full_data: pd.DataFrame,
        initial_variables: List[str],
        target_variable: str,
        params: Dict,
        validation_start: str,
        validation_end: str,
        target_freq: str,
        train_end_date: str,
        target_mean_original: float,
        target_std_original: float,
        max_iter: int,
        max_lags: int = 1
    ):
        """
        初始化预计算上下文
        
        Args:
            full_data: 完整的数据DataFrame
            initial_variables: 初始变量列表
            target_variable: 目标变量名称
            params: DFM参数字典
            validation_start: 验证期开始日期
            validation_end: 验证期结束日期
            target_freq: 目标频率
            train_end_date: 训练期结束日期
            target_mean_original: 原始目标变量均值
            target_std_original: 原始目标变量标准差
            max_iter: 最大迭代次数
            max_lags: 最大滞后阶数
        """
        self.full_data = full_data
        self.initial_variables = initial_variables
        self.target_variable = target_variable
        self.params = params
        self.validation_start = validation_start
        self.validation_end = validation_end
        self.target_freq = target_freq
        self.train_end_date = train_end_date
        self.target_mean_original = target_mean_original
        self.target_std_original = target_std_original
        self.max_iter = max_iter
        self.max_lags = max_lags
        
        # 预计算结果存储
        self.base_data_prepared = None
        self.base_variables_cleaned = None
        self.k_factors = params.get('k_factors', 1)
        
        # 性能统计
        self.stats = {
            'precompute_time': 0.0,
            'data_access_count': 0,
            'total_time_saved': 0.0,
            'creation_time': time.time()
        }
        
        # 执行预计算
        self._precompute_base_context()
        
        logger.info(f"PrecomputedDFMContext初始化完成，预计算耗时: {self.stats['precompute_time']:.3f}秒")
    
    def _precompute_base_context(self):
        """执行基础预计算步骤"""
        start_time = time.time()
        
        try:
            # 步骤1: 验证基础输入（使用完整变量列表）
            is_valid, error_msg, k_factors = _validate_inputs(
                self.initial_variables, 
                self.full_data, 
                self.target_variable, 
                self.params,
                self.validation_end, 
                self.target_mean_original, 
                self.target_std_original
            )
            
            if not is_valid:
                logger.error(f"预计算上下文验证失败: {error_msg}")
                return
            
            # 步骤2: 准备基础数据
            success, error_msg, base_data, base_variables = _prepare_data(
                self.initial_variables, 
                self.full_data, 
                self.validation_end
            )
            
            if not success:
                logger.error(f"预计算数据准备失败: {error_msg}")
                return
            
            # 步骤3: 执行基础数据清理（适用于所有变量）
            # 这里我们保存原始状态，让具体的变量组合进行最终清理
            self.base_data_prepared = base_data.copy()
            self.base_variables_cleaned = base_variables.copy()
            
            logger.info(f"预计算成功: 基础数据 {self.base_data_prepared.shape}, 变量数 {len(self.base_variables_cleaned)}")
            
        except Exception as e:
            logger.error(f"预计算过程中发生错误: {e}")
            self.base_data_prepared = None
            self.base_variables_cleaned = None
        
        self.stats['precompute_time'] = time.time() - start_time
    
    def get_prepared_data_for_variables(
        self, 
        variables: List[str]
    ) -> Tuple[bool, str, pd.DataFrame, List[str]]:
        """
        为特定变量组合获取预处理后的数据
        
        Args:
            variables: 目标变量组合
            
        Returns:
            Tuple[bool, str, pd.DataFrame, List[str]]: 
                (is_success, error_message, prepared_data, prepared_variables)
        """
        start_time = time.time()
        self.stats['data_access_count'] += 1
        
        # 检查预计算是否成功
        if self.base_data_prepared is None or self.base_variables_cleaned is None:
            return False, "预计算上下文未成功初始化", pd.DataFrame(), []
        
        try:
            # 检查请求的变量是否都在基础变量中
            missing_vars = [v for v in variables if v not in self.base_variables_cleaned]
            if missing_vars:
                return False, f"请求的变量不在预处理后的基础变量中: {missing_vars}", pd.DataFrame(), []
            
            # 选择指定变量的数据
            selected_data = self.base_data_prepared[variables].copy()
            
            # 对选定的变量组合进行最终清理和验证
            success, error_msg, cleaned_data, final_variables = _clean_and_validate_data(
                selected_data, variables, self.target_variable, self.k_factors
            )
            
            if not success:
                return False, error_msg, pd.DataFrame(), []
            
            # 应用季节性掩码
            success, error_msg, masked_data = _mask_seasonal_data(cleaned_data, self.target_variable)
            
            if not success:
                return False, error_msg, pd.DataFrame(), []
            
            # 记录性能提升
            elapsed = time.time() - start_time
            # 估算如果从头计算需要的时间（基于预计算时间）
            estimated_full_time = self.stats['precompute_time']
            time_saved = max(0, estimated_full_time - elapsed)
            self.stats['total_time_saved'] += time_saved
            
            logger.debug(f"优化数据获取完成: 变量数={len(variables)}, 耗时={elapsed:.3f}秒, 节省≈{time_saved:.3f}秒")
            
            return True, "", masked_data, final_variables
            
        except Exception as e:
            return False, f"处理变量组合时发生错误: {e}", pd.DataFrame(), []
    
    def is_context_valid(self) -> bool:
        """检查预计算上下文是否有效"""
        return (self.base_data_prepared is not None and 
                self.base_variables_cleaned is not None and
                not self.base_data_prepared.empty)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.stats.copy()
        stats['uptime_seconds'] = time.time() - stats['creation_time']
        stats['avg_time_saved_per_access'] = (
            self.stats['total_time_saved'] / max(1, self.stats['data_access_count'])
        )
        return stats
    
    def print_statistics(self):
        """打印性能统计信息"""
        stats = self.get_statistics()
        print("\n=== DFM预计算上下文统计 ===")
        print(f"预计算耗时: {stats['precompute_time']:.3f}秒")
        print(f"数据访问次数: {stats['data_access_count']}")
        print(f"总节省时间: {stats['total_time_saved']:.3f}秒")
        print(f"平均每次节省: {stats['avg_time_saved_per_access']:.3f}秒")
        print(f"上下文运行时间: {stats['uptime_seconds']:.1f}秒")
        print(f"上下文状态: {'有效' if self.is_context_valid() else '无效'}")
        print("=" * 27)