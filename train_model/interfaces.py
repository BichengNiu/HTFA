# -*- coding: utf-8 -*-
"""
DFM训练模块抽象接口定义

提供系统核心组件的抽象接口，确保松耦合和高内聚
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path


class IDataProcessor(ABC):
    """数据处理器接口
    
    定义数据加载、预处理、频率对齐等操作的标准接口
    """
    
    @abstractmethod
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """加载数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            pd.DataFrame: 加载的数据框
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """数据预处理
        
        Args:
            data: 原始数据
            config: 预处理配置
            
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        pass
    
    @abstractmethod
    def align_frequencies(self, 
                         data: pd.DataFrame, 
                         target_freq: str,
                         method: str = 'mean') -> pd.DataFrame:
        """频率对齐
        
        Args:
            data: 输入数据
            target_freq: 目标频率 ('M', 'Q', 'Y' 等)
            method: 对齐方法 ('mean', 'last', 'sum' 等)
            
        Returns:
            pd.DataFrame: 频率对齐后的数据
        """
        pass
    
    @abstractmethod
    def handle_missing_values(self, 
                            data: pd.DataFrame, 
                            method: str = 'interpolate') -> pd.DataFrame:
        """处理缺失值
        
        Args:
            data: 输入数据
            method: 处理方法 ('interpolate', 'forward_fill', 'drop' 等)
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        pass
    
    @abstractmethod
    def get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取数据信息
        
        Args:
            data: 输入数据
            
        Returns:
            Dict: 包含数据形状、缺失值、数据类型等信息
        """
        pass


class IVariableSelector(ABC):
    """变量选择器接口
    
    定义变量选择、评估等操作的标准接口
    """
    
    @abstractmethod
    def select_variables(self, 
                        data: pd.DataFrame, 
                        target_variable: str,
                        config: Dict[str, Any]) -> List[str]:
        """选择变量
        
        Args:
            data: 输入数据
            target_variable: 目标变量名
            config: 选择配置（包含方法、参数等）
            
        Returns:
            List[str]: 选中的变量列表
        """
        pass
    
    @abstractmethod
    def evaluate_selection(self, 
                          data: pd.DataFrame,
                          selected_vars: List[str],
                          target_variable: str) -> Dict[str, float]:
        """评估变量选择结果
        
        Args:
            data: 输入数据
            selected_vars: 选中的变量
            target_variable: 目标变量
            
        Returns:
            Dict: 评估指标（如 AIC, BIC, RMSE 等）
        """
        pass
    
    @abstractmethod
    def get_selection_criteria(self) -> List[str]:
        """获取支持的选择标准
        
        Returns:
            List[str]: 支持的选择标准列表
        """
        pass
    
    @abstractmethod
    def get_variable_importance(self, 
                               data: pd.DataFrame,
                               variables: List[str],
                               target_variable: str) -> Dict[str, float]:
        """获取变量重要性
        
        Args:
            data: 输入数据
            variables: 变量列表
            target_variable: 目标变量
            
        Returns:
            Dict: 变量名到重要性分数的映射
        """
        pass


class IModel(ABC):
    """模型接口
    
    定义模型训练、预测、保存等操作的标准接口
    """
    
    @abstractmethod
    def fit(self, 
            data: pd.DataFrame,
            target_variable: str,
            config: Dict[str, Any]) -> None:
        """训练模型
        
        Args:
            data: 训练数据
            target_variable: 目标变量名
            config: 模型配置
        """
        pass
    
    @abstractmethod
    def predict(self, 
                data: pd.DataFrame,
                periods: Optional[int] = None) -> pd.Series:
        """预测
        
        Args:
            data: 输入数据
            periods: 预测期数（如果为None，则进行样本内预测）
            
        Returns:
            pd.Series: 预测结果
        """
        pass
    
    @abstractmethod
    def evaluate(self, 
                 y_true: pd.Series,
                 y_pred: pd.Series) -> Dict[str, float]:
        """评估模型
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            Dict: 评估指标
        """
        pass
    
    @abstractmethod
    def save_model(self, file_path: Union[str, Path]) -> None:
        """保存模型
        
        Args:
            file_path: 保存路径
        """
        pass
    
    @abstractmethod
    def load_model(self, file_path: Union[str, Path]) -> None:
        """加载模型
        
        Args:
            file_path: 模型文件路径
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            Dict: 模型参数、状态等信息
        """
        pass
    
    @abstractmethod
    def get_factors(self) -> Optional[pd.DataFrame]:
        """获取因子（针对因子模型）
        
        Returns:
            pd.DataFrame: 因子数据，如果不适用则返回None
        """
        pass
    
    @abstractmethod
    def get_loadings(self) -> Optional[pd.DataFrame]:
        """获取因子载荷（针对因子模型）
        
        Returns:
            pd.DataFrame: 因子载荷矩阵，如果不适用则返回None
        """
        pass


class IEvaluator(ABC):
    """评估器接口
    
    定义模型评估、报告生成等操作的标准接口
    """
    
    @abstractmethod
    def calculate_metrics(self, 
                         y_true: pd.Series,
                         y_pred: pd.Series,
                         metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            metrics: 要计算的指标列表（如果为None，计算所有支持的指标）
            
        Returns:
            Dict: 指标名到值的映射
        """
        pass
    
    @abstractmethod
    def generate_report(self, 
                       model_results: Dict[str, Any],
                       output_path: Optional[Union[str, Path]] = None) -> str:
        """生成评估报告
        
        Args:
            model_results: 模型结果字典
            output_path: 输出路径（如果为None，返回报告字符串）
            
        Returns:
            str: 报告内容或保存路径
        """
        pass
    
    @abstractmethod
    def compare_models(self, 
                      models: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """比较多个模型
        
        Args:
            models: 模型名到结果的映射
            
        Returns:
            pd.DataFrame: 模型比较表
        """
        pass
    
    @abstractmethod
    def plot_results(self, 
                    y_true: pd.Series,
                    y_pred: pd.Series,
                    save_path: Optional[Union[str, Path]] = None) -> None:
        """绘制结果图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            save_path: 保存路径（如果为None，显示图形）
        """
        pass
    
    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """获取支持的评估指标
        
        Returns:
            List[str]: 支持的指标列表
        """
        pass


class IDataPipeline(ABC):
    """数据流水线接口
    
    定义数据处理流水线的标准接口
    """
    
    @abstractmethod
    def add_step(self, name: str, processor: Any, **kwargs) -> None:
        """添加处理步骤
        
        Args:
            name: 步骤名称
            processor: 处理器实例或函数
            **kwargs: 额外参数
        """
        pass
    
    @abstractmethod
    def remove_step(self, name: str) -> None:
        """移除处理步骤
        
        Args:
            name: 步骤名称
        """
        pass
    
    @abstractmethod
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """执行流水线
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        pass
    
    @abstractmethod
    def get_steps(self) -> List[Tuple[str, Any]]:
        """获取所有步骤
        
        Returns:
            List: 步骤列表
        """
        pass
    
    @abstractmethod
    def clear_cache(self) -> None:
        """清除缓存"""
        pass
    
    @abstractmethod
    def save_pipeline(self, file_path: Union[str, Path]) -> None:
        """保存流水线配置
        
        Args:
            file_path: 保存路径
        """
        pass
    
    @abstractmethod
    def load_pipeline(self, file_path: Union[str, Path]) -> None:
        """加载流水线配置
        
        Args:
            file_path: 配置文件路径
        """
        pass


class IStateManager(ABC):
    """状态管理器接口
    
    定义状态管理的标准接口，用于与统一状态管理器集成
    """
    
    @abstractmethod
    def get_state(self, key: str, default: Any = None) -> Any:
        """获取状态
        
        Args:
            key: 状态键
            default: 默认值
            
        Returns:
            Any: 状态值
        """
        pass
    
    @abstractmethod
    def set_state(self, key: str, value: Any) -> None:
        """设置状态
        
        Args:
            key: 状态键
            value: 状态值
        """
        pass
    
    @abstractmethod
    def clear_state(self, pattern: Optional[str] = None) -> None:
        """清除状态
        
        Args:
            pattern: 键模式（如果为None，清除所有）
        """
        pass
    
    @abstractmethod
    def get_all_states(self) -> Dict[str, Any]:
        """获取所有状态
        
        Returns:
            Dict: 所有状态的字典
        """
        pass