"""
DFM Metadata Accessor
DFM元数据访问器 - 封装重复的元数据访问模式
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """模型评估指标数据类"""
    mae: Optional[float] = None
    rmse: Optional[float] = None
    hit_rate: Optional[float] = None


@dataclass
class TrainingInfo:
    """训练信息数据类"""
    training_start: str = 'N/A'
    training_end: str = 'N/A'
    validation_start: str = 'N/A'
    validation_end: str = 'N/A'
    target_variable: str = ''
    estimation_method: str = 'N/A'
    n_variables: Any = 'N/A'
    n_factors: Any = 'N/A'


class DFMMetadataAccessor:
    """
    DFM元数据访问器

    封装所有元数据访问，消除重复的metadata.get()调用模式。
    提供类型安全和默认值处理。
    """

    def __init__(self, metadata: Dict[str, Any]):
        """
        初始化元数据访问器

        Args:
            metadata: 从pkl文件加载的元数据字典
        """
        self._metadata = metadata
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def training_info(self) -> TrainingInfo:
        """获取训练信息"""
        best_variables = self._metadata.get('best_variables', [])
        if isinstance(best_variables, list) and len(best_variables) > 0:
            n_vars = len(best_variables)
        else:
            factor_loadings = self._metadata.get('factor_loadings_df')
            if factor_loadings is not None and hasattr(factor_loadings, 'index'):
                n_vars = len(factor_loadings.index)
            else:
                n_vars = 'N/A'

        return TrainingInfo(
            training_start=self._get_date_str('training_start_date'),
            training_end=self._get_date_str('train_end_date'),
            validation_start=self._get_date_str('validation_start_date'),
            validation_end=self._get_date_str('validation_end_date'),
            target_variable=self._metadata.get('target_variable', '规模以上工业增加值:当月同比'),
            estimation_method=self._metadata.get('estimation_method', 'N/A'),
            n_variables=n_vars,
            n_factors=self._metadata.get('k_factors', 'N/A')
        )

    @property
    def training_metrics(self) -> ModelMetrics:
        """获取训练期指标"""
        return ModelMetrics(
            mae=self._metadata.get('is_mae_original'),
            rmse=self._metadata.get('is_rmse_original'),
            hit_rate=self._metadata.get('is_hit_rate_original')
        )

    @property
    def validation_metrics(self) -> ModelMetrics:
        """获取验证期指标"""
        return ModelMetrics(
            mae=self._metadata.get('oos_mae_original'),
            rmse=self._metadata.get('oos_rmse_original'),
            hit_rate=self._metadata.get('oos_hit_rate_original')
        )

    @property
    def observation_metrics(self) -> ModelMetrics:
        """获取观察期指标"""
        return ModelMetrics(
            mae=self._metadata.get('obs_mae_original'),
            rmse=self._metadata.get('obs_rmse_original'),
            hit_rate=self._metadata.get('obs_hit_rate_original')
        )

    @property
    def has_observation_metrics(self) -> bool:
        """检查是否有观察期指标"""
        metrics = self.observation_metrics
        return any([
            metrics.rmse is not None,
            metrics.mae is not None,
            metrics.hit_rate is not None
        ])

    @property
    def complete_aligned_table(self) -> Optional[pd.DataFrame]:
        """获取完整对齐表"""
        return self._metadata.get('complete_aligned_table')

    @property
    def nowcast_comparison(self) -> Optional[pd.DataFrame]:
        """获取Nowcast对比数据"""
        return self._metadata.get('nowcast_comparison')

    @property
    def factor_loadings_df(self) -> Optional[pd.DataFrame]:
        """获取因子载荷矩阵"""
        return self._metadata.get('factor_loadings_df')

    @property
    def smoothed_factors(self) -> Optional[pd.DataFrame]:
        """获取平滑因子"""
        return self._metadata.get('smoothed_factors')

    @property
    def industry_r2(self) -> Optional[pd.Series]:
        """获取行业R²"""
        return self._metadata.get('industry_r2')

    @property
    def factor_industry_r2(self) -> Optional[Dict]:
        """获取因子-行业R²"""
        return self._metadata.get('factor_industry_r2')

    @property
    def pca_results_df(self) -> Optional[pd.DataFrame]:
        """获取PCA结果"""
        pca_df = self._metadata.get('pca_results_df')
        if pca_df is not None and isinstance(pca_df, pd.DataFrame) and not pca_df.empty:
            return pca_df
        return None

    def get(self, key: str, default: Any = None) -> Any:
        """
        通用获取方法（向后兼容）

        Args:
            key: 元数据键
            default: 默认值

        Returns:
            元数据值或默认值
        """
        return self._metadata.get(key, default)

    def _get_date_str(self, key: str) -> str:
        """
        获取日期字符串

        Args:
            key: 日期键

        Returns:
            格式化的日期字符串或'N/A'
        """
        date_val = self._metadata.get(key)
        if date_val is None:
            return 'N/A'

        if isinstance(date_val, str):
            return date_val

        if isinstance(date_val, pd.Timestamp):
            return date_val.strftime('%Y-%m-%d')

        try:
            return str(date_val)
        except:
            return 'N/A'

    @staticmethod
    def format_metric(val: Any, is_percent: bool = False, precision: int = 2) -> str:
        """
        格式化指标值

        Args:
            val: 原始值
            is_percent: 是否为百分比
            precision: 小数精度

        Returns:
            格式化字符串
        """
        if isinstance(val, (int, float)) and pd.notna(val):
            if is_percent:
                return f"{val:.{precision}f}%"
            return f"{val:.{precision}f}"
        return 'N/A' if val == 'N/A' or pd.isna(val) else str(val)
