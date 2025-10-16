# -*- coding: utf-8 -*-
"""
PrecomputedDFMContext - 预计算上下文类

用于存储所有预计算的中间结果，消除冗余计算，显著提升DFM训练性能。
将原本需要重复75次的计算减少到仅计算1次。

主要优化：
- 数据清洗 (75次 → 1次)  
- 季节性掩码应用 (75次 → 1次)
- 标准化 (75次 → 1次)
- PCA初始化 (75次 → 1次)
"""

import os
import json
import pickle
import hashlib
import logging
import threading
import traceback
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# 导入现有的DFM核心模块
from dashboard.DFM.train_model.dfm_core import _clean_and_validate_data
from dashboard.DFM.train_model.analysis_utils import calculate_pca_variance

logger = logging.getLogger(__name__)


@dataclass
class StandardizationParameters:
    """标准化参数数据类
    
    存储训练集的均值和标准差，用于一致的数据标准化
    """
    means: pd.Series
    stds: pd.Series
    train_start_date: str
    train_end_date: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """验证标准化参数"""
        if len(self.means) != len(self.stds):
            raise ValueError(f"均值数量 ({len(self.means)}) 与标准差数量 ({len(self.stds)}) 不匹配")
        
        # 处理标准差为0的情况
        zero_std_vars = self.stds[self.stds == 0].index.tolist()
        if zero_std_vars:
            logger.warning(f"发现标准差为0的变量，将设为1.0: {zero_std_vars}")
            self.stds = self.stds.copy()
            self.stds[self.stds == 0] = 1.0
    
    def standardize(self, data: pd.DataFrame) -> pd.DataFrame:
        """使用预计算的参数标准化数据
        
        Args:
            data: 要标准化的数据
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        # 确保列对齐
        common_cols = data.columns.intersection(self.means.index)
        if len(common_cols) == 0:
            raise ValueError("数据与标准化参数没有共同变量")
        
        data_aligned = data[common_cols].copy()
        means_aligned = self.means[common_cols]
        stds_aligned = self.stds[common_cols]
        
        # 执行标准化
        standardized = (data_aligned - means_aligned) / stds_aligned
        
        # 处理可能的NaN
        standardized = standardized.fillna(0)
        
        return standardized


@dataclass  
class PCAInitializationResults:
    """PCA初始化结果数据类"""
    components: np.ndarray  # 主成分
    explained_variance_ratio: np.ndarray  # 解释方差比例
    explained_variance: np.ndarray  # 解释方差
    mean_values: np.ndarray  # PCA拟合时使用的均值
    n_components: int
    feature_names: List[str]  # 特征变量名
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """验证PCA结果"""
        if self.components.shape[0] != self.n_components:
            raise ValueError(f"主成分数量不匹配: {self.components.shape[0]} vs {self.n_components}")
        
        if len(self.feature_names) != self.components.shape[1]:
            raise ValueError(f"特征名数量 ({len(self.feature_names)}) 与主成分维度 ({self.components.shape[1]}) 不匹配")
    
    def get_initial_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """基于PCA结果获取初始因子
        
        Args:
            data: 标准化后的数据
            
        Returns:
            pd.DataFrame: 初始因子数据
        """
        # 确保列对齐
        common_cols = [col for col in self.feature_names if col in data.columns]
        if len(common_cols) == 0:
            raise ValueError("数据与PCA特征没有共同变量")
        
        data_aligned = data[common_cols].values
        
        # 中心化数据（使用训练时的均值）
        mean_aligned = self.mean_values[:len(common_cols)]
        data_centered = data_aligned - mean_aligned
        
        # 投影到主成分空间
        factors = data_centered @ self.components[:, :len(common_cols)].T
        
        # 创建DataFrame
        factor_names = [f'Factor_{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(factors, index=data.index, columns=factor_names)


class PrecomputedDFMContext:
    """预计算DFM上下文类
    
    存储所有预计算的中间结果，消除变量选择过程中的冗余计算
    """
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """初始化预计算上下文
        
        Args:
            cache_dir: 缓存目录，用于持久化存储预计算结果
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / ".dfm_precomputed_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 预计算结果存储
        self._cleaned_data: Optional[pd.DataFrame] = None
        self._cleaned_variables: Optional[List[str]] = None
        self._seasonal_mask: Optional[pd.DataFrame] = None
        self._standardization_params: Optional[StandardizationParameters] = None
        self._pca_results: Optional[PCAInitializationResults] = None
        self._train_validation_split: Optional[Dict[str, Any]] = None
        
        # 原始数据和配置的哈希值，用于验证缓存有效性
        self._data_hash: Optional[str] = None
        self._config_hash: Optional[str] = None
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 性能监控
        self._computation_times: Dict[str, float] = {}
        self._created_at = datetime.now().isoformat()
        
        logger.info(f"初始化PrecomputedDFMContext，缓存目录: {self.cache_dir}")
    
    def _generate_data_hash(self, data: pd.DataFrame) -> str:
        """生成数据哈希值
        
        Args:
            data: 输入数据
            
        Returns:
            str: 数据哈希值
        """
        try:
            data_bytes = pd.util.hash_pandas_object(data, index=True).values.tobytes()
            return hashlib.sha256(data_bytes).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"生成数据哈希失败，使用形状和列名: {e}")
            shape_str = f"{data.shape}_{','.join(data.columns)}"
            return hashlib.sha256(shape_str.encode()).hexdigest()[:16]
    
    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """生成配置哈希值
        
        Args:
            config: 配置字典
            
        Returns:
            str: 配置哈希值
        """
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def precompute_data_processing(
        self, 
        data: pd.DataFrame, 
        target_variable: str,
        k_factors: int,
        config: Dict[str, Any]
    ) -> bool:
        """预计算数据处理步骤
        
        Args:
            data: 原始数据
            target_variable: 目标变量名
            k_factors: 因子数量
            config: 处理配置
            
        Returns:
            bool: 是否成功
        """
        with self._lock:
            start_time = datetime.now()
            
            try:
                logger.info("开始预计算数据处理...")
                
                # 生成哈希值
                self._data_hash = self._generate_data_hash(data)
                self._config_hash = self._generate_config_hash(config)
                
                # 尝试从缓存加载
                if self._try_load_from_cache("data_processing"):
                    logger.info("从缓存加载数据处理结果")
                    return True
                
                # 执行数据清洗和验证
                success, error_msg, cleaned_data, cleaned_variables = _clean_and_validate_data(
                    data.copy(),
                    data.columns.tolist(),
                    target_variable,
                    k_factors
                )
                
                if not success:
                    logger.error(f"数据清洗失败: {error_msg}")
                    return False
                
                # 存储清洗结果
                self._cleaned_data = cleaned_data
                self._cleaned_variables = cleaned_variables
                
                # 应用季节性掩码（如果配置中指定）
                self._apply_seasonal_mask(config.get('seasonal_adjustment', {}))
                
                # 保存到缓存
                self._save_to_cache("data_processing")
                
                computation_time = (datetime.now() - start_time).total_seconds()
                self._computation_times["data_processing"] = computation_time
                
                logger.info(f"数据处理完成，耗时: {computation_time:.2f}秒")
                logger.info(f"清洗后数据形状: {cleaned_data.shape}")
                logger.info(f"变量数量: {len(cleaned_variables)}")
                
                return True
                
            except Exception as e:
                logger.error(f"预计算数据处理失败: {e}")
                logger.debug(traceback.format_exc())
                return False
    
    def _apply_seasonal_mask(self, seasonal_config: Dict[str, Any]):
        """应用季节性掩码
        
        Args:
            seasonal_config: 季节性调整配置
        """
        try:
            if not seasonal_config.get('enabled', False):
                logger.debug("季节性调整未启用")
                return
            
            # 创建季节性掩码
            mask_type = seasonal_config.get('mask_type', 'monthly')
            exclude_months = seasonal_config.get('exclude_months', [])
            
            if self._cleaned_data is None:
                return
            
            # 基于日期索引创建掩码
            if hasattr(self._cleaned_data.index, 'month'):
                if mask_type == 'monthly' and exclude_months:
                    seasonal_mask = ~self._cleaned_data.index.month.isin(exclude_months)
                else:
                    seasonal_mask = pd.Series(True, index=self._cleaned_data.index)
            else:
                logger.warning("数据索引非日期类型，跳过季节性掩码")
                seasonal_mask = pd.Series(True, index=self._cleaned_data.index)
            
            # 将掩码转换为DataFrame格式以便后续使用
            self._seasonal_mask = pd.DataFrame(
                seasonal_mask.values.reshape(-1, 1), 
                index=self._cleaned_data.index,
                columns=['seasonal_mask']
            )
            
            logger.debug(f"应用季节性掩码，过滤比例: {(~seasonal_mask).mean():.2%}")
            
        except Exception as e:
            logger.warning(f"应用季节性掩码失败: {e}")
            # 创建默认掩码（全部为True）
            if self._cleaned_data is not None:
                self._seasonal_mask = pd.DataFrame(
                    True, 
                    index=self._cleaned_data.index, 
                    columns=['seasonal_mask']
                )
    
    def precompute_standardization(
        self, 
        train_start: Optional[str] = None,
        train_end: Optional[str] = None
    ) -> bool:
        """预计算标准化参数
        
        Args:
            train_start: 训练开始日期
            train_end: 训练结束日期
            
        Returns:
            bool: 是否成功
        """
        with self._lock:
            start_time = datetime.now()
            
            try:
                logger.info("开始预计算标准化参数...")
                
                if self._cleaned_data is None:
                    logger.error("需要先执行数据处理")
                    return False
                
                # 尝试从缓存加载
                if self._try_load_from_cache("standardization"):
                    logger.info("从缓存加载标准化参数")
                    return True
                
                # 确定训练数据范围
                train_data = self._cleaned_data
                if train_start and train_end:
                    try:
                        train_mask = (
                            (train_data.index >= pd.to_datetime(train_start)) &
                            (train_data.index <= pd.to_datetime(train_end))
                        )
                        train_data = train_data[train_mask]
                        
                        if train_data.empty:
                            logger.warning("训练日期范围内无数据，使用全部数据")
                            train_data = self._cleaned_data
                    except Exception as e:
                        logger.warning(f"处理训练日期范围失败: {e}，使用全部数据")
                        train_data = self._cleaned_data
                
                # 应用季节性掩码（如果存在）
                if self._seasonal_mask is not None:
                    mask_values = self._seasonal_mask['seasonal_mask']
                    train_data = train_data[mask_values]
                
                # 计算均值和标准差
                means = train_data.mean()
                stds = train_data.std()
                
                # 验证计算结果
                if means.isnull().any():
                    logger.warning("发现均值为NaN的变量，将替换为0")
                    means = means.fillna(0)
                
                if stds.isnull().any():
                    logger.warning("发现标准差为NaN的变量，将替换为1")
                    stds = stds.fillna(1)
                
                # 创建标准化参数对象
                self._standardization_params = StandardizationParameters(
                    means=means,
                    stds=stds,
                    train_start_date=train_start or "全部",
                    train_end_date=train_end or "全部"
                )
                
                # 保存到缓存
                self._save_to_cache("standardization")
                
                computation_time = (datetime.now() - start_time).total_seconds()
                self._computation_times["standardization"] = computation_time
                
                logger.info(f"标准化参数计算完成，耗时: {computation_time:.2f}秒")
                logger.info(f"变量数量: {len(means)}")
                logger.info(f"训练数据范围: {train_start} 至 {train_end}")
                
                return True
                
            except Exception as e:
                logger.error(f"预计算标准化参数失败: {e}")
                logger.debug(traceback.format_exc())
                return False
    
    def precompute_pca_initialization(
        self, 
        n_components: int,
        impute_strategy: str = 'mean'
    ) -> bool:
        """预计算PCA初始化
        
        Args:
            n_components: 主成分数量
            impute_strategy: 缺失值填充策略
            
        Returns:
            bool: 是否成功
        """
        with self._lock:
            start_time = datetime.now()
            
            try:
                logger.info(f"开始预计算PCA初始化，主成分数: {n_components}...")
                
                if self._cleaned_data is None:
                    logger.error("需要先执行数据处理")
                    return False
                
                if self._standardization_params is None:
                    logger.error("需要先计算标准化参数")
                    return False
                
                # 尝试从缓存加载
                cache_key = f"pca_{n_components}_{impute_strategy}"
                if self._try_load_from_cache(cache_key):
                    logger.info("从缓存加载PCA结果")
                    return True
                
                # 标准化数据
                standardized_data = self._standardization_params.standardize(self._cleaned_data)
                
                # 应用季节性掩码
                if self._seasonal_mask is not None:
                    mask_values = self._seasonal_mask['seasonal_mask']
                    standardized_data = standardized_data[mask_values]
                
                # 处理缺失值
                if standardized_data.isnull().any().any():
                    logger.info(f"检测到缺失值，使用{impute_strategy}策略填充")
                    imputer = SimpleImputer(strategy=impute_strategy)
                    imputed_data = imputer.fit_transform(standardized_data)
                    standardized_data = pd.DataFrame(
                        imputed_data, 
                        index=standardized_data.index, 
                        columns=standardized_data.columns
                    )
                
                # 检查数据有效性
                if standardized_data.shape[1] < n_components:
                    logger.warning(f"变量数 ({standardized_data.shape[1]}) 小于主成分数 ({n_components})")
                    n_components = min(n_components, standardized_data.shape[1])
                
                if n_components <= 0:
                    logger.error("主成分数量必须大于0")
                    return False
                
                # 执行PCA
                pca = PCA(n_components=n_components)
                pca.fit(standardized_data)
                
                # 存储PCA结果
                self._pca_results = PCAInitializationResults(
                    components=pca.components_,
                    explained_variance_ratio=pca.explained_variance_ratio_,
                    explained_variance=pca.explained_variance_,
                    mean_values=pca.mean_,
                    n_components=n_components,
                    feature_names=standardized_data.columns.tolist()
                )
                
                # 保存到缓存
                self._save_to_cache(cache_key)
                
                computation_time = (datetime.now() - start_time).total_seconds()
                self._computation_times["pca_initialization"] = computation_time
                
                total_variance = self._pca_results.explained_variance_ratio.sum()
                logger.info(f"PCA初始化完成，耗时: {computation_time:.2f}秒")
                logger.info(f"前{n_components}个主成分解释方差: {total_variance:.2%}")
                
                return True
                
            except Exception as e:
                logger.error(f"预计算PCA初始化失败: {e}")
                logger.debug(traceback.format_exc())
                return False
    
    def get_subset_context(self, variable_list: List[str]) -> 'PrecomputedDFMContext':
        """获取特定变量子集的上下文
        
        Args:
            variable_list: 变量列表
            
        Returns:
            PrecomputedDFMContext: 子集上下文
        """
        with self._lock:
            try:
                # 创建新的上下文实例
                subset_context = PrecomputedDFMContext(cache_dir=None)  # 不使用磁盘缓存
                
                # 复制基本属性
                subset_context._data_hash = self._data_hash
                subset_context._config_hash = self._config_hash
                subset_context._created_at = self._created_at
                
                # 过滤数据到指定变量
                if self._cleaned_data is not None:
                    common_vars = [var for var in variable_list if var in self._cleaned_data.columns]
                    if common_vars:
                        subset_context._cleaned_data = self._cleaned_data[common_vars].copy()
                        subset_context._cleaned_variables = common_vars
                
                # 复制季节性掩码
                subset_context._seasonal_mask = self._seasonal_mask
                
                # 过滤标准化参数
                if self._standardization_params is not None:
                    common_vars = [var for var in variable_list if var in self._standardization_params.means.index]
                    if common_vars:
                        subset_context._standardization_params = StandardizationParameters(
                            means=self._standardization_params.means[common_vars],
                            stds=self._standardization_params.stds[common_vars],
                            train_start_date=self._standardization_params.train_start_date,
                            train_end_date=self._standardization_params.train_end_date
                        )
                
                # PCA结果需要重新计算，因为变量子集不同
                subset_context._pca_results = None
                
                # 复制训练验证分割信息
                subset_context._train_validation_split = self._train_validation_split
                
                logger.debug(f"创建变量子集上下文，变量数: {len(variable_list)}")
                
                return subset_context
                
            except Exception as e:
                logger.error(f"创建子集上下文失败: {e}")
                logger.debug(traceback.format_exc())
                return self  # 返回原始上下文作为后备
    
    def validate_context(self) -> Tuple[bool, List[str]]:
        """验证上下文有效性
        
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        with self._lock:
            errors = []
            
            try:
                # 检查基本数据
                if self._cleaned_data is None:
                    errors.append("清洗数据为空")
                elif self._cleaned_data.empty:
                    errors.append("清洗数据为空DataFrame")
                
                if not self._cleaned_variables:
                    errors.append("清洗变量列表为空")
                
                # 检查数据一致性
                if self._cleaned_data is not None and self._cleaned_variables:
                    if len(self._cleaned_variables) != self._cleaned_data.shape[1]:
                        errors.append("变量列表与数据列数不匹配")
                    
                    missing_vars = set(self._cleaned_variables) - set(self._cleaned_data.columns)
                    if missing_vars:
                        errors.append(f"变量列表中的变量在数据中缺失: {missing_vars}")
                
                # 检查标准化参数
                if self._standardization_params is not None:
                    if len(self._standardization_params.means) != len(self._standardization_params.stds):
                        errors.append("标准化参数中均值与标准差数量不匹配")
                    
                    if self._cleaned_variables:
                        std_vars = set(self._standardization_params.means.index)
                        clean_vars = set(self._cleaned_variables)
                        if not std_vars.issubset(clean_vars):
                            missing = std_vars - clean_vars
                            errors.append(f"标准化参数包含清洗数据中不存在的变量: {missing}")
                
                # 检查PCA结果
                if self._pca_results is not None:
                    if self._pca_results.n_components <= 0:
                        errors.append("PCA主成分数量无效")
                    
                    if len(self._pca_results.feature_names) == 0:
                        errors.append("PCA特征名列表为空")
                
                # 检查哈希值
                if not self._data_hash:
                    errors.append("数据哈希值缺失")
                
                if not self._config_hash:
                    errors.append("配置哈希值缺失")
                
                is_valid = len(errors) == 0
                
                if is_valid:
                    logger.debug("上下文验证通过")
                else:
                    logger.warning(f"上下文验证失败，错误数: {len(errors)}")
                    for error in errors:
                        logger.warning(f"  - {error}")
                
                return is_valid, errors
                
            except Exception as e:
                error_msg = f"验证过程异常: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                return False, errors
    
    def _try_load_from_cache(self, cache_key: str) -> bool:
        """尝试从缓存加载结果
        
        Args:
            cache_key: 缓存键
            
        Returns:
            bool: 是否成功加载
        """
        try:
            if not self._data_hash or not self._config_hash:
                return False
            
            cache_file = self.cache_dir / f"{cache_key}_{self._data_hash}_{self._config_hash}.pkl"
            
            if not cache_file.exists():
                return False
            
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # 恢复缓存的数据
            if cache_key == "data_processing":
                self._cleaned_data = cached_data['cleaned_data']
                self._cleaned_variables = cached_data['cleaned_variables']
                self._seasonal_mask = cached_data['seasonal_mask']
            elif cache_key == "standardization":
                self._standardization_params = cached_data['standardization_params']
            elif cache_key.startswith("pca_"):
                self._pca_results = cached_data['pca_results']
            
            logger.debug(f"从缓存加载成功: {cache_key}")
            return True
            
        except Exception as e:
            logger.debug(f"从缓存加载失败: {cache_key}, {e}")
            return False
    
    def _save_to_cache(self, cache_key: str):
        """保存结果到缓存
        
        Args:
            cache_key: 缓存键
        """
        try:
            if not self._data_hash or not self._config_hash:
                return
            
            cache_file = self.cache_dir / f"{cache_key}_{self._data_hash}_{self._config_hash}.pkl"
            
            # 准备缓存数据
            if cache_key == "data_processing":
                cache_data = {
                    'cleaned_data': self._cleaned_data,
                    'cleaned_variables': self._cleaned_variables,
                    'seasonal_mask': self._seasonal_mask
                }
            elif cache_key == "standardization":
                cache_data = {
                    'standardization_params': self._standardization_params
                }
            elif cache_key.startswith("pca_"):
                cache_data = {
                    'pca_results': self._pca_results
                }
            else:
                return
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.debug(f"保存到缓存: {cache_key}")
            
        except Exception as e:
            logger.warning(f"保存缓存失败: {cache_key}, {e}")
    
    def clear_cache(self):
        """清除所有缓存文件"""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            for cache_file in cache_files:
                cache_file.unlink()
            
            logger.info(f"清除缓存文件: {len(cache_files)} 个")
            
        except Exception as e:
            logger.warning(f"清除缓存失败: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息
        
        Returns:
            Dict: 性能统计
        """
        with self._lock:
            total_computation_time = sum(self._computation_times.values())
            
            stats = {
                'created_at': self._created_at,
                'total_computation_time': total_computation_time,
                'computation_times': self._computation_times.copy(),
                'data_hash': self._data_hash,
                'config_hash': self._config_hash,
                'cache_dir': str(self.cache_dir),
                'context_status': {
                    'has_cleaned_data': self._cleaned_data is not None,
                    'has_seasonal_mask': self._seasonal_mask is not None,
                    'has_standardization_params': self._standardization_params is not None,
                    'has_pca_results': self._pca_results is not None,
                    'cleaned_data_shape': self._cleaned_data.shape if self._cleaned_data is not None else None,
                    'num_variables': len(self._cleaned_variables) if self._cleaned_variables else 0
                }
            }
            
            return stats
    
    # 属性访问器
    @property
    def cleaned_data(self) -> Optional[pd.DataFrame]:
        """获取清洗后的数据"""
        return self._cleaned_data
    
    @property
    def cleaned_variables(self) -> Optional[List[str]]:
        """获取清洗后的变量列表"""
        return self._cleaned_variables
    
    @property
    def seasonal_mask(self) -> Optional[pd.DataFrame]:
        """获取季节性掩码"""
        return self._seasonal_mask
    
    @property
    def standardization_params(self) -> Optional[StandardizationParameters]:
        """获取标准化参数"""
        return self._standardization_params
    
    @property
    def pca_results(self) -> Optional[PCAInitializationResults]:
        """获取PCA结果"""
        return self._pca_results
    
    def get_standardized_data(self, data: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """获取标准化数据
        
        Args:
            data: 要标准化的数据，如果为None则使用清洗后的数据
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        if self._standardization_params is None:
            logger.warning("标准化参数未计算")
            return None
        
        target_data = data if data is not None else self._cleaned_data
        if target_data is None:
            logger.warning("无数据可标准化")
            return None
        
        try:
            return self._standardization_params.standardize(target_data)
        except Exception as e:
            logger.error(f"标准化数据失败: {e}")
            return None
    
    def get_initial_factors(self, data: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """获取初始因子
        
        Args:
            data: 标准化后的数据，如果为None则使用内部标准化数据
            
        Returns:
            pd.DataFrame: 初始因子数据
        """
        if self._pca_results is None:
            logger.warning("PCA结果未计算")
            return None
        
        if data is None:
            data = self.get_standardized_data()
        
        if data is None:
            return None
        
        try:
            return self._pca_results.get_initial_factors(data)
        except Exception as e:
            logger.error(f"获取初始因子失败: {e}")
            return None


# 工厂函数和辅助函数

def create_precomputed_context(
    data: pd.DataFrame,
    target_variable: str,
    k_factors: int,
    config: Dict[str, Any],
    cache_dir: Optional[Union[str, Path]] = None,
    n_pca_components: Optional[int] = None
) -> PrecomputedDFMContext:
    """创建并完成所有预计算的DFM上下文
    
    Args:
        data: 原始数据
        target_variable: 目标变量名
        k_factors: 因子数量
        config: 配置字典
        cache_dir: 缓存目录
        n_pca_components: PCA主成分数，默认等于k_factors
        
    Returns:
        PrecomputedDFMContext: 完成预计算的上下文
    """
    start_time = datetime.now()
    logger.info("开始创建预计算DFM上下文...")
    
    # 创建上下文
    context = PrecomputedDFMContext(cache_dir=cache_dir)
    
    # 执行所有预计算步骤
    steps = [
        ("数据处理", lambda: context.precompute_data_processing(data, target_variable, k_factors, config)),
        ("标准化参数", lambda: context.precompute_standardization(
            config.get('train_start'), config.get('train_end')
        )),
        ("PCA初始化", lambda: context.precompute_pca_initialization(
            n_pca_components or k_factors, config.get('impute_strategy', 'mean')
        ))
    ]
    
    failed_steps = []
    for step_name, step_func in steps:
        try:
            if not step_func():
                failed_steps.append(step_name)
                logger.error(f"预计算步骤失败: {step_name}")
        except Exception as e:
            failed_steps.append(step_name)
            logger.error(f"预计算步骤异常: {step_name}, {e}")
    
    # 验证结果
    is_valid, errors = context.validate_context()
    if not is_valid:
        logger.warning("上下文验证失败:")
        for error in errors:
            logger.warning(f"  - {error}")
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    if failed_steps:
        logger.warning(f"预计算完成但有失败步骤: {failed_steps}, 总耗时: {total_time:.2f}秒")
    else:
        logger.info(f"预计算DFM上下文创建成功，总耗时: {total_time:.2f}秒")
    
    # 输出性能统计
    stats = context.get_performance_stats()
    logger.info("预计算性能统计:")
    for step, time_cost in stats['computation_times'].items():
        logger.info(f"  - {step}: {time_cost:.2f}秒")
    
    return context


def validate_precomputed_context(context: PrecomputedDFMContext) -> bool:
    """验证预计算上下文是否可用
    
    Args:
        context: 预计算上下文
        
    Returns:
        bool: 是否可用
    """
    if not isinstance(context, PrecomputedDFMContext):
        logger.error("输入不是PrecomputedDFMContext实例")
        return False
    
    is_valid, errors = context.validate_context()
    
    if not is_valid:
        logger.error("预计算上下文验证失败:")
        for error in errors:
            logger.error(f"  - {error}")
    
    return is_valid