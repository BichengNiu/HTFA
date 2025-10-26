# -*- coding: utf-8 -*-
"""
训练结果导出器（合并版）

负责将TrainingResult导出为文件：
- 模型文件（.joblib）
- 元数据文件（.pkl）

整合了原metadata_builder和utils的功能
"""

import os
import tempfile
import pickle
from datetime import datetime
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import joblib
from dashboard.models.DFM.train.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingResultExporter:
    """训练结果文件导出器（合并版）"""

    def export_all(
        self,
        result,  # TrainingResult
        config,  # TrainingConfig
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        导出所有结果文件

        Args:
            result: 训练结果
            config: 训练配置
            output_dir: 输出目录（None=创建临时目录）

        Returns:
            文件路径字典 {
                'final_model_joblib': 模型文件路径,
                'metadata': 元数据文件路径
            }
        """
        logger.info("开始导出训练结果文件")

        # 创建输出目录
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='dfm_results_')
            logger.info(f"使用临时目录: {output_dir}")
        else:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"使用指定目录: {output_dir}")

        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 导出各个文件
        file_paths = {}

        try:
            model_path = os.path.join(output_dir, f'final_dfm_model_{timestamp}.joblib')
            self._export_model(result, model_path)
            file_paths['final_model_joblib'] = model_path
            logger.info(f"模型文件已导出: {os.path.basename(model_path)}")
        except Exception as e:
            logger.error(f"导出模型文件失败: {e}", exc_info=True)
            file_paths['final_model_joblib'] = None

        try:
            metadata_path = os.path.join(output_dir, f'final_dfm_metadata_{timestamp}.pkl')
            self._export_metadata(result, config, metadata_path, timestamp)
            file_paths['metadata'] = metadata_path
            logger.info(f"元数据文件已导出: {os.path.basename(metadata_path)}")
        except Exception as e:
            logger.error(f"导出元数据文件失败: {e}", exc_info=True)
            file_paths['metadata'] = None

        # 验证文件
        for file_type, path in file_paths.items():
            if path and os.path.exists(path):
                size = os.path.getsize(path)
                logger.debug(f"{file_type}: {path} ({size} bytes)")
            else:
                logger.warning(f"{file_type}: 文件不存在或导出失败")

        logger.info(f"文件导出完成,共 {len([p for p in file_paths.values() if p])} 个文件")
        return file_paths

    def _export_model(self, result, path: str) -> None:
        """导出模型文件"""
        if result.model_result is None:
            raise ValueError("训练结果中没有模型对象")

        joblib.dump(result.model_result, path, compress=3)

        if not os.path.exists(path):
            raise IOError(f"模型文件保存失败: {path}")

        file_size = os.path.getsize(path) / (1024 * 1024)
        logger.debug(f"模型文件大小: {file_size:.2f} MB")

    def _export_metadata(self, result, config, path: str, timestamp: str) -> None:
        """导出元数据文件"""
        metadata = self._build_metadata(result, config, timestamp)
        self._validate_metadata(metadata)

        with open(path, 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        if not os.path.exists(path):
            raise IOError(f"元数据文件保存失败: {path}")

        file_size = os.path.getsize(path) / (1024 * 1024)
        logger.debug(f"元数据文件大小: {file_size:.2f} MB")

    # ========== 元数据构建方法 ==========

    def _build_metadata(self, result, config, timestamp: str) -> Dict[str, Any]:
        """构建元数据字典（与results模块格式一致）"""
        logger.debug("开始构建元数据")

        # 计算目标变量统计参数
        target_mean, target_std = self._calculate_target_statistics(config)

        metadata = {
            # 基本信息
            'timestamp': timestamp,
            'target_variable': config.target_variable,
            'best_variables': result.selected_variables,

            # 模型参数
            'best_params': {
                'k_factors': result.k_factors,
                'variable_selection_method': '后向逐步' if config.enable_variable_selection else '全选',
                'tuning_objective': 'RMSE' if config.enable_variable_selection else 'N/A',
            },

            # 日期
            'training_start_date': self._calculate_training_start(config),
            'train_end_date': getattr(config, 'train_end', ''),
            'validation_start_date': self._calculate_validation_start(config),
            'validation_end_date': getattr(config, 'validation_end', ''),

            # 标准化参数
            'target_mean_original': target_mean,
            'target_std_original': target_std,

            # 训练统计
            'total_runtime_seconds': float(getattr(result, 'training_time', 0.0)),
            'var_industry_map': {var: '综合' for var in result.selected_variables},
        }

        # 评估指标（使用results模块需要的字段名）
        if result.metrics:
            metadata.update({
                'revised_is_hr': float(result.metrics.is_hit_rate),
                'revised_oos_hr': float(result.metrics.oos_hit_rate),
                'revised_is_rmse': float(result.metrics.is_rmse),
                'revised_oos_rmse': float(result.metrics.oos_rmse),
                'revised_is_mae': float(result.metrics.is_mae),
                'revised_oos_mae': float(result.metrics.oos_mae),
            })

        # 因子载荷DataFrame
        metadata['factor_loadings_df'] = self._extract_factor_loadings(result, config)

        # 因子序列DataFrame
        if result.model_result and hasattr(result.model_result, 'factors'):
            factors_data = result.model_result.factors
            if isinstance(factors_data, np.ndarray):
                factor_names = [f'Factor_{i+1}' for i in range(factors_data.shape[1])]
                metadata['factor_series'] = pd.DataFrame(factors_data, columns=factor_names)
            else:
                metadata['factor_series'] = factors_data.copy() if isinstance(factors_data, pd.DataFrame) else None
        else:
            metadata['factor_series'] = None

        # PCA结果DataFrame
        if result.pca_analysis:
            metadata['pca_results_df'] = self._convert_pca_to_dataframe(result.pca_analysis, result.k_factors)
        else:
            metadata['pca_results_df'] = None

        # 对齐表格（核心数据）
        metadata['complete_aligned_table'] = self._generate_aligned_table(result, config, metadata)

        # R²分析结果（可选）
        metadata['industry_r2_results'] = None
        metadata['factor_industry_r2_results'] = None

        logger.info(f"元数据构建完成,包含 {len(metadata)} 个字段")
        return metadata

    def _calculate_target_statistics(self, config) -> tuple[float, float]:
        """计算目标变量的均值和标准差"""
        try:
            if not hasattr(config, 'data_path') or not config.data_path:
                return 0.0, 1.0

            # 读取数据文件
            data = self._read_data_file(config.data_path)

            if config.target_variable not in data.columns:
                logger.warning(f"目标变量 {config.target_variable} 不在数据中")
                return 0.0, 1.0

            target_series = data[config.target_variable].dropna()
            if len(target_series) == 0:
                return 0.0, 1.0

            target_mean = float(target_series.mean())
            target_std = float(target_series.std())

            logger.debug(f"目标变量统计: mean={target_mean:.4f}, std={target_std:.4f}")
            return target_mean, target_std

        except Exception as e:
            logger.warning(f"计算目标变量统计参数失败: {e}")
            return 0.0, 1.0

    def _calculate_training_start(self, config) -> str:
        """计算训练开始日期"""
        try:
            # 优先使用配置的训练开始日期
            if hasattr(config, 'training_start') and config.training_start:
                return config.training_start

            # 如果有数据路径，尝试从数据中推断
            if hasattr(config, 'data_path') and config.data_path:
                try:
                    data = self._read_data_file(config.data_path)
                    if isinstance(data.index, pd.DatetimeIndex) and len(data) > 0:
                        return data.index.min().strftime('%Y-%m-%d')
                except Exception:
                    pass

            return ''
        except Exception as e:
            logger.warning(f"计算训练开始日期失败: {e}")
            return ''

    def _calculate_validation_start(self, config) -> str:
        """计算验证开始日期"""
        try:
            if hasattr(config, 'validation_start') and config.validation_start:
                return config.validation_start

            train_end_value = getattr(config, 'train_end', None)
            if train_end_value:
                train_end = pd.to_datetime(train_end_value)
                validation_start = train_end + pd.DateOffset(weeks=1)
                return validation_start.strftime('%Y-%m-%d')

            return ''
        except Exception as e:
            logger.warning(f"计算验证开始日期失败: {e}")
            return ''

    def _validate_metadata(self, metadata: Dict) -> None:
        """验证元数据包含所有必需字段"""
        required_fields = [
            'timestamp', 'target_variable', 'best_variables',
            'best_params', 'train_end_date', 'validation_end_date',
        ]

        missing_fields = [f for f in required_fields if f not in metadata]

        if missing_fields:
            raise ValueError(f"元数据缺少必需字段: {missing_fields}")

        logger.debug(f"元数据验证通过,包含 {len(metadata)} 个字段")

    # ========== 工具方法 ==========

    def _read_data_file(self, file_path: str) -> pd.DataFrame:
        """
        根据文件扩展名读取数据文件

        Args:
            file_path: 数据文件路径

        Returns:
            pd.DataFrame: 读取的数据
        """
        file_path = str(file_path)
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, index_col=0)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path, index_col=0)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

    def _extract_factor_loadings(self, result, config=None) -> pd.DataFrame:
        """提取因子载荷矩阵（H矩阵）"""
        try:
            if not result.model_result:
                return pd.DataFrame()

            H = result.model_result.H

            if H is None:
                return pd.DataFrame()

            if isinstance(H, np.ndarray):
                factor_names = [f'Factor_{i+1}' for i in range(H.shape[1])]

                # 从selected_variables中排除目标变量，因为H只包含预测变量
                var_names = result.selected_variables
                if config and hasattr(config, 'target_variable'):
                    var_names = [v for v in var_names if v != config.target_variable]

                # 严格检查变量名列表长度与H的行数是否匹配
                if len(var_names) != H.shape[0]:
                    raise ValueError(
                        f"变量名数量({len(var_names)})与H矩阵行数({H.shape[0]})不匹配。"
                        f"selected_variables: {result.selected_variables}, "
                        f"target_variable: {config.target_variable if config else 'N/A'}"
                    )

                return pd.DataFrame(H, columns=factor_names, index=var_names)
            elif isinstance(H, pd.DataFrame):
                return H.copy()

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"提取因子载荷失败: {e}")
            return pd.DataFrame()

    def _convert_pca_to_dataframe(self, pca_analysis: Dict, n_components: int) -> pd.DataFrame:
        """
        将PCA分析结果转换为DataFrame格式

        Args:
            pca_analysis: PCA分析结果字典
            n_components: 主成分数量

        Returns:
            DataFrame包含列: 主成分(PC), 解释方差(%), 累计解释方差(%), 特征值
        """
        try:
            if not pca_analysis:
                return pd.DataFrame()

            # 从pca_analysis中提取数据
            explained_variance = pca_analysis.get('explained_variance', [])
            explained_variance_ratio = pca_analysis.get('explained_variance_ratio', [])

            if not explained_variance or not explained_variance_ratio:
                logger.warning("PCA分析结果缺少必要数据")
                return pd.DataFrame()

            # 转换为百分比
            explained_variance_ratio_pct = [v * 100 for v in explained_variance_ratio[:n_components]]
            cumulative_explained_variance_pct = np.cumsum(explained_variance_ratio_pct)

            # 构建DataFrame（格式与train_model完全一致）
            pca_results_df = pd.DataFrame({
                '主成分 (PC)': [f'PC{i+1}' for i in range(n_components)],
                '解释方差 (%)': explained_variance_ratio_pct,
                '累计解释方差 (%)': cumulative_explained_variance_pct,
                '特征值 (Eigenvalue)': explained_variance[:n_components]
            })

            logger.debug(f"PCA结果转换完成，包含 {len(pca_results_df)} 个主成分")
            return pca_results_df

        except Exception as e:
            logger.error(f"转换PCA结果为DataFrame失败: {e}")
            return pd.DataFrame()

    def _generate_aligned_table(self, result, config, metadata: Dict) -> Optional[pd.DataFrame]:
        """
        生成complete_aligned_table

        对齐规则：
        1. 每个月最后一个周五的Nowcast值
        2. 对应下一个月的实际值

        Args:
            result: 训练结果
            config: 训练配置
            metadata: 元数据字典

        Returns:
            DataFrame包含两列: 'Nowcast (Original Scale)' 和目标变量名
        """
        try:
            logger.debug("开始生成complete_aligned_table...")

            # 1. 获取Nowcast数据（原始尺度）
            nowcast_data = None
            if result.model_result and hasattr(result.model_result, 'forecast_oos'):
                forecast_oos = result.model_result.forecast_oos
                if forecast_oos is not None:
                    # 反标准化到原始尺度
                    target_mean = metadata.get('target_mean_original', 0.0)
                    target_std = metadata.get('target_std_original', 1.0)

                    if isinstance(forecast_oos, pd.Series):
                        nowcast_data = forecast_oos * target_std + target_mean
                    elif isinstance(forecast_oos, np.ndarray):
                        nowcast_data = pd.Series(forecast_oos * target_std + target_mean)

            if nowcast_data is None or len(nowcast_data) == 0:
                logger.warning("无法获取Nowcast数据，complete_aligned_table将为空")
                return pd.DataFrame(columns=['Nowcast (Original Scale)', config.target_variable])

            # 2. 获取目标变量实际值
            target_data = None
            if hasattr(config, 'data_path') and config.data_path:
                try:
                    data = self._read_data_file(config.data_path)
                    if config.target_variable in data.columns:
                        target_data = data[config.target_variable].dropna()
                except Exception as e:
                    logger.warning(f"从数据文件读取目标变量失败: {e}")

            if target_data is None or len(target_data) == 0:
                logger.warning("无法获取目标变量实际值")
                # 创建基本对齐表格（仅包含Nowcast）
                return pd.DataFrame({
                    'Nowcast (Original Scale)': nowcast_data,
                    config.target_variable: np.nan
                })

            # 3. 执行对齐逻辑（参考老代码）
            aligned_table = self._align_nowcast_target(
                nowcast_data,
                target_data,
                config.target_variable
            )

            logger.info(f"complete_aligned_table生成完成，包含 {len(aligned_table)} 行数据")
            return aligned_table

        except Exception as e:
            logger.error(f"生成complete_aligned_table失败: {e}", exc_info=True)
            return pd.DataFrame(columns=['Nowcast (Original Scale)', config.target_variable])

    def _align_nowcast_target(
        self,
        nowcast_weekly: pd.Series,
        target_orig: pd.Series,
        target_variable_name: str
    ) -> pd.DataFrame:
        """
        对齐周度Nowcast和月度Target

        对齐规则：
        1. 选取每个月最后一个周五的Nowcast值
        2. 将该Nowcast值与下一个月的Target值进行匹配

        Args:
            nowcast_weekly: 周度Nowcast序列
            target_orig: 目标变量序列
            target_variable_name: 目标变量名称

        Returns:
            对齐后的DataFrame
        """
        try:
            # 确保索引是DatetimeIndex
            if not isinstance(nowcast_weekly.index, pd.DatetimeIndex):
                nowcast_weekly.index = pd.to_datetime(nowcast_weekly.index)
            if not isinstance(target_orig.index, pd.DatetimeIndex):
                target_orig.index = pd.to_datetime(target_orig.index)

            # 筛选出所有周五
            fridays_index = nowcast_weekly[nowcast_weekly.index.dayofweek == 4].index
            if fridays_index.empty:
                logger.warning("Nowcast序列中未找到任何周五")
                return pd.DataFrame(columns=['Nowcast (Original Scale)', target_variable_name])

            # 使用所有周五的Nowcast值
            nowcast_all_fridays = nowcast_weekly.loc[fridays_index].copy()
            nowcast_all_fridays.name = 'Nowcast (Original Scale)'

            # 计算每月最后一个周五
            last_fridays = fridays_index.to_series().groupby(fridays_index.to_period('M')).max()

            # 准备Target数据
            target_df = target_orig.dropna().to_frame(target_variable_name)
            target_df['TargetPeriod'] = target_df.index.to_period('M')

            # 创建完整的对齐DataFrame
            final_aligned_table = nowcast_all_fridays.to_frame()
            final_aligned_table[target_variable_name] = np.nan

            # 对齐逻辑：每个月最后一个周五对应下个月的真实值
            aligned_count = 0
            for target_date, target_value in target_df[target_variable_name].items():
                target_period = target_df.loc[target_date, 'TargetPeriod']

                # 找到上个月的最后一个周五
                prev_month = target_period - 1

                if prev_month in last_fridays.index:
                    last_friday_date = last_fridays.loc[prev_month]

                    # 将真实值分配到对应的周五行
                    if last_friday_date in final_aligned_table.index:
                        final_aligned_table.loc[last_friday_date, target_variable_name] = target_value
                        aligned_count += 1

            logger.debug(f"成功对齐 {aligned_count} 个数据点")
            return final_aligned_table

        except Exception as e:
            logger.error(f"对齐Nowcast和Target失败: {e}")
            return pd.DataFrame(columns=['Nowcast (Original Scale)', target_variable_name])


__all__ = ['TrainingResultExporter']
