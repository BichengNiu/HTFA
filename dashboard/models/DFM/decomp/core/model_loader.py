# -*- coding: utf-8 -*-
"""
模型文件加载器

专注于从joblib和pickle文件中提取已计算的nowcast值和相关数据，
为影响分析提供基础数据。
"""

import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import io
from datetime import datetime

from ..utils.validators import validate_model_data
from ..utils.exceptions import ModelLoadError, ValidationError, DataFormatError


class SavedNowcastData:
    """保存的nowcast数据结构"""

    def __init__(self):
        self.nowcast_series: Optional[pd.Series] = None
        self.kalman_gains_history: Optional[List[np.ndarray]] = None  # 修改：保存每个时刻的K_t矩阵列表
        self.factor_loadings: Optional[np.ndarray] = None  # H矩阵（观测矩阵/因子载荷）- 预测变量
        self.target_factor_loading: Optional[np.ndarray] = None  # 新增：目标变量的因子载荷向量
        self.factor_series: Optional[pd.DataFrame] = None
        self.target_variable: Optional[str] = None
        self.target_variable_index: Optional[int] = None  # 新增：目标变量在观测向量中的索引
        self.variable_index_map: Optional[Dict[str, int]] = None  # 新增：变量名到索引的映射
        self.var_industry_map: Optional[Dict[str, str]] = None  # 新增：变量名到行业分类的映射
        self.model_parameters: Optional[Dict[str, Any]] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.data_period: Optional[Tuple[str, str]] = None
        self.convergence_info: Optional[Dict[str, Any]] = None
        self.prepared_data: Optional[pd.DataFrame] = None  # 新增：complete_aligned_table（包含所有历史观测数据）


class ModelLoader:
    """
    模型文件加载器

    专门用于加载包含已计算nowcast值的DFM模型文件和元数据。
    核心功能是数据提取而非重新计算。
    """

    def __init__(self):
        self._model: Optional[Any] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._nowcast_data: Optional[SavedNowcastData] = None
        self._load_time: Optional[datetime] = None

    def load_model(self, model_content: bytes) -> Any:
        """
        从字节内容加载DFM模型文件

        Args:
            model_content: joblib模型文件的字节内容

        Returns:
            加载的DFM模型对象

        Raises:
            ModelLoadError: 加载失败时抛出
        """
        try:
            # 使用BytesIO避免写入临时文件
            model_stream = io.BytesIO(model_content)
            self._model = joblib.load(model_stream)
            self._load_time = datetime.now()

            # 验证模型对象
            if not hasattr(self._model, '__dict__'):
                raise ModelLoadError("加载的对象不是有效的模型实例")

            print(f"[ModelLoader] 成功加载DFM模型，类型: {type(self._model)}")
            return self._model

        except Exception as e:
            raise ModelLoadError(
                f"加载DFM模型文件失败: {str(e)}",
                original_error=e
            )

    def load_metadata(self, metadata_content: bytes) -> Dict[str, Any]:
        """
        从字节内容加载模型元数据

        Args:
            metadata_content: pickle元数据文件的字节内容

        Returns:
            元数据字典

        Raises:
            ModelLoadError: 加载失败时抛出
        """
        try:
            # 使用BytesIO避免写入临时文件
            metadata_stream = io.BytesIO(metadata_content)
            self._metadata = pickle.load(metadata_stream)

            # 验证元数据格式
            if not isinstance(self._metadata, dict):
                raise ModelLoadError("元数据文件格式无效，应为字典")

            print(f"[ModelLoader] 成功加载元数据，包含 {len(self._metadata)} 个键")
            return self._metadata

        except Exception as e:
            raise ModelLoadError(
                f"加载元数据文件失败: {str(e)}",
                original_error=e
            )

    def extract_saved_nowcast(self) -> SavedNowcastData:
        """
        从加载的模型和元数据中提取nowcast相关数据

        Returns:
            SavedNowcastData: 提取的nowcast数据结构

        Raises:
            ModelLoadError: 数据提取失败时抛出
            ValidationError: 数据验证失败时抛出
        """
        if self._model is None or self._metadata is None:
            raise ModelLoadError("请先加载模型和元数据文件")

        try:
            nowcast_data = SavedNowcastData()

            # 1. 提取nowcast时间序列
            nowcast_data.nowcast_series = self._extract_nowcast_series()

            # 2. 提取卡尔曼增益历史（修改：使用新字段名）
            nowcast_data.kalman_gains_history = self._extract_kalman_gains()

            # 3. 提取因子载荷矩阵（H矩阵）
            nowcast_data.factor_loadings = self._extract_factor_loadings()

            # 4. 提取因子时间序列
            nowcast_data.factor_series = self._extract_factor_series()

            # 4.5 提取目标变量的因子载荷
            nowcast_data.target_factor_loading = self._extract_target_factor_loading()

            # 5. 提取目标变量信息
            nowcast_data.target_variable = self._extract_target_variable()

            # 6. 提取目标变量索引和变量映射（新增）
            nowcast_data.target_variable_index, nowcast_data.variable_index_map = self._extract_variable_mapping()

            # 7. 提取模型参数
            nowcast_data.model_parameters = self._extract_model_parameters()

            # 8. 提取数据时间范围
            nowcast_data.data_period = self._extract_data_period()

            # 9. 提取收敛信息
            nowcast_data.convergence_info = self._extract_convergence_info()

            # 10. 提取历史观测数据表（complete_aligned_table）
            nowcast_data.prepared_data = self._extract_prepared_data()

            # 11. 提取行业分类映射
            nowcast_data.var_industry_map = self._extract_industry_map()

            # 12. 保存原始元数据
            nowcast_data.metadata = self._metadata.copy()

            # 验证提取的数据
            self._validate_extracted_data(nowcast_data)

            self._nowcast_data = nowcast_data
            print(f"[ModelLoader] 成功提取nowcast数据")
            return nowcast_data

        except Exception as e:
            raise ModelLoadError(f"提取nowcast数据失败: {str(e)}")

    def validate_model_compatibility(self, model: Any, metadata: Dict[str, Any]) -> bool:
        """
        验证模型和元数据的兼容性

        Args:
            model: DFM模型对象
            metadata: 元数据字典

        Returns:
            bool: 是否兼容

        Raises:
            ValidationError: 验证失败时抛出
        """
        try:
            is_valid, errors = validate_model_data(model, metadata)

            if not is_valid:
                error_msg = "模型兼容性验证失败:\n" + "\n".join(f"  - {error}" for error in errors)
                raise ValidationError(error_msg)

            # 额外的兼容性检查
            # 检查因子数量一致性
            if hasattr(model, 'H') and 'factor_loadings_df' in metadata:
                model_factors = model.H.shape[1] if hasattr(model.H, 'shape') else 0
                metadata_factors = metadata['factor_loadings_df'].shape[1] if hasattr(metadata['factor_loadings_df'], 'shape') else 0

                if model_factors != metadata_factors:
                    raise ValidationError(
                        f"因子数量不匹配: 模型={model_factors}, 元数据={metadata_factors}"
                    )

            print(f"[ModelLoader] 模型兼容性验证通过")
            return True

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"兼容性验证过程异常: {str(e)}")

    def _extract_nowcast_series(self) -> pd.Series:
        """提取nowcast时间序列"""
        if 'complete_aligned_table' not in self._metadata:
            raise DataFormatError("元数据中缺少complete_aligned_table")

        nowcast_table = self._metadata['complete_aligned_table']

        if not isinstance(nowcast_table, pd.DataFrame):
            raise DataFormatError("nowcast数据格式无效")

        if 'Nowcast (Original Scale)' not in nowcast_table.columns:
            raise DataFormatError("nowcast数据中缺少目标列")

        nowcast_series = nowcast_table['Nowcast (Original Scale)']

        if nowcast_series.empty:
            raise DataFormatError("nowcast时间序列为空")

        print(f"[ModelLoader] 提取nowcast序列: {len(nowcast_series)} 个数据点")
        return nowcast_series

    def _extract_kalman_gains(self) -> Optional[List[np.ndarray]]:
        """提取卡尔曼增益历史"""
        # 优先从元数据中提取卡尔曼增益历史（方案C）
        if 'kalman_gains_history' in self._metadata:
            kalman_gains_history = self._metadata['kalman_gains_history']
            if kalman_gains_history is not None and len(kalman_gains_history) > 0:
                print(f"[ModelLoader] 提取卡尔曼增益历史: {len(kalman_gains_history)} 个时间步")
                # 验证格式
                first_non_none = next((k for k in kalman_gains_history if k is not None), None)
                if first_non_none is not None:
                    print(f"[ModelLoader] K_t矩阵形状: {first_non_none.shape}")
                return kalman_gains_history

        # 如果元数据中没有，尝试从模型对象中获取
        if hasattr(self._model, 'kalman_gains_history'):
            kalman_gains_history = self._model.kalman_gains_history
            if kalman_gains_history is not None:
                print(f"[ModelLoader] 从模型对象提取卡尔曼增益历史: {len(kalman_gains_history)} 个时间步")
                return kalman_gains_history

        # 向后兼容：如果都没有，返回None并记录警告
        print("[ModelLoader] 警告: 未找到卡尔曼增益历史，新闻分解功能将不可用")
        print("[ModelLoader] 提示: 请使用新版本的训练模块重新训练模型以支持影响分解功能")
        return None

    def _extract_factor_loadings(self) -> np.ndarray:
        """提取因子载荷矩阵"""
        if 'factor_loadings_df' not in self._metadata:
            raise DataFormatError("元数据中缺少因子载荷数据")

        factor_loadings = self._metadata['factor_loadings_df']

        if not isinstance(factor_loadings, pd.DataFrame):
            raise DataFormatError("因子载荷数据格式无效")

        if factor_loadings.empty:
            raise DataFormatError("因子载荷矩阵为空")

        print(f"[ModelLoader] 提取因子载荷矩阵: {factor_loadings.shape}")
        return factor_loadings.values

    def _extract_target_factor_loading(self) -> Optional[np.ndarray]:
        """提取目标变量的因子载荷向量"""
        try:
            # 优先从元数据中提取
            if 'target_factor_loading' in self._metadata:
                target_loading = self._metadata['target_factor_loading']
                if target_loading is not None and isinstance(target_loading, np.ndarray):
                    print(f"[ModelLoader] 提取目标变量因子载荷: 形状={target_loading.shape}")
                    return target_loading
                else:
                    print("[ModelLoader] 警告: target_factor_loading字段存在但值无效")

            # 降级方案：尝试从模型结果中提取
            if self._model and hasattr(self._model, 'target_factor_loading'):
                target_loading = self._model.target_factor_loading
                if target_loading is not None:
                    print(f"[ModelLoader] 从模型对象提取目标变量因子载荷: 形状={target_loading.shape}")
                    return target_loading

            print("[ModelLoader] 警告: 未找到目标变量因子载荷（新闻分析功能受限）")
            return None

        except Exception as e:
            print(f"[ModelLoader] 目标变量因子载荷提取失败: {str(e)}")
            return None

    def _extract_factor_series(self) -> pd.DataFrame:
        """提取因子时间序列"""
        if 'factor_series' not in self._metadata:
            raise DataFormatError("元数据中缺少因子序列数据")

        factor_series = self._metadata['factor_series']

        if not isinstance(factor_series, pd.DataFrame):
            raise DataFormatError("因子序列数据格式无效")

        if factor_series.empty:
            raise DataFormatError("因子序列为空")

        print(f"[ModelLoader] 提取因子序列: {factor_series.shape}")
        return factor_series

    def _extract_target_variable(self) -> str:
        """提取目标变量信息"""
        if 'target_variable' not in self._metadata:
            raise DataFormatError("元数据中缺少目标变量信息")

        target_variable = self._metadata['target_variable']

        if not isinstance(target_variable, str):
            raise DataFormatError("目标变量信息格式无效")

        print(f"[ModelLoader] 提取目标变量: {target_variable}")
        return target_variable

    def _extract_model_parameters(self) -> Dict[str, Any]:
        """提取模型参数"""
        parameters = {}

        # 从模型中提取参数
        if hasattr(self._model, 'A'):
            parameters['state_transition'] = self._model.A
        if hasattr(self._model, 'Q'):
            parameters['state_noise'] = self._model.Q
        if hasattr(self._model, 'R'):
            parameters['observation_noise'] = self._model.R

        # 从元数据中提取参数
        if 'best_params' in self._metadata:
            parameters.update(self._metadata['best_params'])

        print(f"[ModelLoader] 提取模型参数: {len(parameters)} 个")
        return parameters

    def _extract_data_period(self) -> Tuple[str, str]:
        """提取数据时间范围"""
        if self._nowcast_data and self._nowcast_data.nowcast_series is not None:
            start_date = self._nowcast_data.nowcast_series.index.min()
            end_date = self._nowcast_data.nowcast_series.index.max()
            return str(start_date), str(end_date)

        # 从元数据中获取
        start_date = self._metadata.get('training_start_date')
        end_date = self._metadata.get('validation_end_date')

        if not start_date or not end_date:
            raise DataFormatError("无法确定数据时间范围")

        return start_date, end_date

    def _extract_convergence_info(self) -> Dict[str, Any]:
        """提取收敛信息"""
        convergence_info = {}

        if hasattr(self._model, 'converged'):
            convergence_info['converged'] = bool(self._model.converged)

        if hasattr(self._model, 'iterations'):
            convergence_info['iterations'] = int(self._model.iterations)

        if hasattr(self._model, 'log_likelihood'):
            convergence_info['log_likelihood'] = float(self._model.log_likelihood)

        # 从元数据中获取性能指标
        performance_keys = ['is_rmse', 'oos_rmse', 'is_hit_rate', 'oos_hit_rate']
        for key in performance_keys:
            if key in self._metadata:
                convergence_info[key] = float(self._metadata[key])

        print(f"[ModelLoader] 提取收敛信息: {len(convergence_info)} 项")
        return convergence_info

    def _extract_prepared_data(self) -> Optional[pd.DataFrame]:
        """提取历史观测数据表（prepared_data）"""
        try:
            # 优先使用prepared_data字段（包含所有观测变量）
            if 'prepared_data' in self._metadata:
                prepared_data = self._metadata['prepared_data']

                if prepared_data is None:
                    print("[ModelLoader] 警告: prepared_data字段存在但值为None")
                    return None

                if not isinstance(prepared_data, pd.DataFrame):
                    print("[ModelLoader] 警告: prepared_data不是DataFrame类型")
                    return None

                if prepared_data.empty:
                    print("[ModelLoader] 警告: prepared_data为空")
                    return None

                print(f"[ModelLoader] 提取历史观测数据表: 形状={prepared_data.shape}, 列数={len(prepared_data.columns)}")
                print(f"[ModelLoader] 数据时间范围: {prepared_data.index[0]} 到 {prepared_data.index[-1]}")

                return prepared_data

            # 降级方案：如果没有prepared_data，尝试使用complete_aligned_table（仅包含2列）
            elif 'complete_aligned_table' in self._metadata:
                print("[ModelLoader] 警告: 元数据中缺少prepared_data字段，使用complete_aligned_table（功能受限）")
                aligned_table = self._metadata['complete_aligned_table']

                if not isinstance(aligned_table, pd.DataFrame):
                    print("[ModelLoader] 警告: complete_aligned_table不是DataFrame类型")
                    return None

                if aligned_table.empty:
                    print("[ModelLoader] 警告: complete_aligned_table为空")
                    return None

                print(f"[ModelLoader] 提取complete_aligned_table: 形状={aligned_table.shape}, 列数={len(aligned_table.columns)}")
                return aligned_table

            else:
                print("[ModelLoader] 警告: 元数据中缺少prepared_data和complete_aligned_table")
                return None

        except Exception as e:
            print(f"[ModelLoader] 历史数据表提取失败: {str(e)}")
            return None

    def _extract_variable_mapping(self) -> Tuple[Optional[int], Optional[Dict[str, int]]]:
        """提取变量索引映射和目标变量索引

        注意：variable_index_map应该只包含预测变量（不含目标变量），
        因为K_t矩阵的列数对应预测变量数。
        """
        try:
            # 从元数据中获取变量列表
            variable_list = []

            # 优先从factor_loadings_df获取（已排除目标变量，与K_t矩阵对齐）
            if 'factor_loadings_df' in self._metadata:
                factor_loadings_df = self._metadata['factor_loadings_df']
                if hasattr(factor_loadings_df, 'index') and len(factor_loadings_df.index) > 0:
                    variable_list = list(factor_loadings_df.index)
                    print(f"[ModelLoader] 从factor_loadings_df提取{len(variable_list)}个预测变量")

            # 降级方案：从best_variables获取（需要排除目标变量）
            if not variable_list and 'best_variables' in self._metadata:
                target_variable = self._metadata.get('target_variable')
                all_variables = self._metadata['best_variables']
                # 排除目标变量
                variable_list = [v for v in all_variables if v != target_variable]
                print(f"[ModelLoader] 从best_variables提取{len(variable_list)}个预测变量（已排除目标变量）")

            # 次优方案：从var_industry_map获取
            if not variable_list and 'var_industry_map' in self._metadata:
                target_variable = self._metadata.get('target_variable')
                all_variables = list(self._metadata['var_industry_map'].keys())
                variable_list = [v for v in all_variables if v != target_variable]
                print(f"[ModelLoader] 从var_industry_map提取{len(variable_list)}个预测变量")

            if not variable_list:
                print("[ModelLoader] 警告: 无法提取变量列表，变量映射将不可用")
                return None, None

            # 构建变量名到索引的映射（索引对应K_t矩阵的列索引）
            variable_index_map = {var_name: idx for idx, var_name in enumerate(variable_list)}

            # 获取目标变量信息
            target_variable = self._metadata.get('target_variable')
            # 注意：target_variable_index在预测变量列表中不存在（已排除）
            # 如果后续需要目标变量索引，应该在完整变量列表中查找
            target_variable_index = None  # 在预测变量映射中不存在

            if target_variable:
                print(f"[ModelLoader] 目标变量: {target_variable} (不在预测变量映射中)")
            else:
                print(f"[ModelLoader] 警告: 未找到目标变量信息")

            print(f"[ModelLoader] 提取变量映射: {len(variable_index_map)} 个预测变量")
            return target_variable_index, variable_index_map

        except Exception as e:
            print(f"[ModelLoader] 变量映射提取失败: {str(e)}")
            return None, None

    def _extract_industry_map(self) -> Optional[Dict[str, str]]:
        """
        从元数据中提取行业分类映射

        Returns:
            变量名到行业的映射字典，如果不存在则返回None
        """
        try:
            # 从元数据中获取var_industry_map
            if 'var_industry_map' in self._metadata:
                var_industry_map = self._metadata['var_industry_map']

                if not isinstance(var_industry_map, dict):
                    print("[ModelLoader] 警告: var_industry_map格式无效，应为字典类型")
                    return None

                if not var_industry_map:
                    print("[ModelLoader] 警告: var_industry_map为空字典")
                    return {}

                print(f"[ModelLoader] 提取行业分类映射: {len(var_industry_map)} 个变量")

                # 打印前几个映射示例
                sample_items = list(var_industry_map.items())[:3]
                print(f"[ModelLoader] 示例映射: {sample_items}")

                return var_industry_map
            else:
                print("[ModelLoader] 警告: 元数据中缺少var_industry_map字段")
                return None

        except Exception as e:
            print(f"[ModelLoader] 行业分类映射提取失败: {str(e)}")
            return None

    def _validate_extracted_data(self, nowcast_data: SavedNowcastData) -> None:
        """验证提取的数据完整性"""
        required_components = [
            (nowcast_data.nowcast_series, "nowcast时间序列"),
            (nowcast_data.target_variable, "目标变量信息")
        ]

        missing_components = []
        for component, name in required_components:
            if component is None:
                missing_components.append(name)

        if missing_components:
            raise ValidationError(f"缺少必要组件: {', '.join(missing_components)}")

        # 检查卡尔曼增益历史（可选，向后兼容）
        if nowcast_data.kalman_gains_history is None:
            print("[ModelLoader] 警告: 卡尔曼增益历史缺失，影响分解功能将不可用")
        else:
            # 验证卡尔曼增益历史的格式
            first_non_none = next((k for k in nowcast_data.kalman_gains_history if k is not None), None)
            if first_non_none is not None:
                if nowcast_data.factor_loadings is not None:
                    # 检查形状一致性：K_t形状应为(n_factors, n_variables)
                    # H形状应为(n_variables, n_factors)
                    if first_non_none.shape[0] != nowcast_data.factor_loadings.shape[1]:
                        print(f"[ModelLoader] 警告: K_t因子维度({first_non_none.shape[0]})与H因子维度({nowcast_data.factor_loadings.shape[1]})不一致")
                    if first_non_none.shape[1] != nowcast_data.factor_loadings.shape[0]:
                        print(f"[ModelLoader] 警告: K_t变量维度({first_non_none.shape[1]})与H变量维度({nowcast_data.factor_loadings.shape[0]})不一致")

        print("[ModelLoader] 数据完整性验证通过")

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取加载的模型信息摘要

        Returns:
            模型信息字典
        """
        if self._model is None or self._metadata is None:
            return {"status": "未加载"}

        info = {
            "status": "已加载",
            "load_time": self._load_time.isoformat() if self._load_time else None,
            "model_type": str(type(self._model)),
            "metadata_keys": len(self._metadata) if self._metadata else 0,
        }

        if self._nowcast_data:
            # 从factor_loadings获取因子数和变量数（H矩阵形状为n_variables × n_factors）
            factor_count = None
            variable_count = None
            if self._nowcast_data.factor_loadings is not None:
                variable_count = self._nowcast_data.factor_loadings.shape[0]
                factor_count = self._nowcast_data.factor_loadings.shape[1]

            info.update({
                "nowcast_data_points": len(self._nowcast_data.nowcast_series) if self._nowcast_data.nowcast_series is not None else 0,
                "target_variable": self._nowcast_data.target_variable,
                "target_variable_index": self._nowcast_data.target_variable_index,
                "data_period": self._nowcast_data.data_period,
                "factor_count": factor_count,
                "variable_count": variable_count,
                "has_kalman_gains": self._nowcast_data.kalman_gains_history is not None,
                "kalman_gains_timesteps": len(self._nowcast_data.kalman_gains_history) if self._nowcast_data.kalman_gains_history else 0,
            })

        return info