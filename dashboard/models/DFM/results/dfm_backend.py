import pandas as pd
import joblib
import pickle
import io
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.cluster import hierarchy as sch

logger = logging.getLogger(__name__)


def load_dfm_results_from_uploads(loaded_model_object, loaded_metadata_object):
    """
    Receives already loaded DFM model and metadata objects.
    The actual loading (joblib.load, pickle.load) is expected to have happened
    before calling this function (e.g., in the UI layer with caching).
    """
    model = loaded_model_object
    metadata = loaded_metadata_object
    load_errors = []

    if model is None:
        logger.warning("接收到的 DFM 模型对象为 None")
    else:
        logger.info("成功接收 DFM 模型对象")

    if metadata is None:
        error_msg = "接收到的 DFM 元数据对象为 None"
        logger.error(error_msg)
        load_errors.append(error_msg)
        return model, metadata, load_errors
    else:
        logger.info("成功接收 DFM 元数据对象")

    logger.info("检查并加载评估指标")

    # 检查元数据中是否包含训练模块计算的标准指标
    standard_metric_keys = ['is_rmse', 'oos_rmse', 'is_mae', 'oos_mae', 'is_hit_rate', 'oos_hit_rate']
    has_standard_metrics = all(key in metadata for key in standard_metric_keys)

    # 添加调试日志：输出元数据中实际包含的键
    logger.info(f"元数据中包含的指标相关键: {[k for k in metadata.keys() if any(x in k for x in ['rmse', 'mae', 'hit_rate', 'hr'])]}")

    if has_standard_metrics:
        logger.info("发现训练模块计算的标准指标，直接使用...")
        logger.info(f"已加载标准指标: IS胜率={metadata['is_hit_rate']:.2f}%, OOS胜率={metadata['oos_hit_rate']:.2f}%")
        logger.info(f"                 IS_RMSE={metadata['is_rmse']:.4f}, OOS_RMSE={metadata['oos_rmse']:.4f}")
        logger.info(f"                 IS_MAE={metadata['is_mae']:.4f}, OOS_MAE={metadata['oos_mae']:.4f}")
    else:
        error_msg = "元数据中缺少必要的性能指标，请使用最新版本的训练模块重新训练模型"
        logger.error(error_msg)
        load_errors.append(error_msg)

    # 检查模型对象是否有效
    if model is None:
        error_msg = "模型对象为空，请上传有效的模型文件"
        logger.error(error_msg)
        load_errors.append(error_msg)

    return model, metadata, load_errors


def perform_loadings_clustering(loadings_df: pd.DataFrame, cluster_vars: bool = True, normalize: bool = True):
    """
    对因子载荷矩阵进行变量聚类计算。

    Args:
        loadings_df: 包含因子载荷的 DataFrame (原始形式：变量为行，因子为列)
        cluster_vars: 是否对变量进行聚类排序
        normalize: 是否对载荷进行行标准化（每个变量标准化到单位长度），默认True

    Returns:
        tuple: (clustered_loadings_df, variable_order, clustering_success)
            - clustered_loadings_df: 聚类后的载荷矩阵（如normalize=True则已标准化）
            - variable_order: 聚类后的变量顺序列表
            - clustering_success: 聚类是否成功的布尔值
    """
    if not isinstance(loadings_df, pd.DataFrame) or loadings_df.empty:
        logger.warning("无法进行聚类：提供的载荷数据无效。")
        return loadings_df, loadings_df.index.tolist() if not loadings_df.empty else [], False

    data_for_clustering = loadings_df.copy()  # 变量是行

    # 标准化：将每个变量的载荷向量标准化到单位长度
    if normalize:
        logger.info("对因子载荷进行行标准化（单位长度）...")
        norms = np.linalg.norm(data_for_clustering.values, axis=1, keepdims=True)
        # 避免除零
        norms = np.where(norms > 1e-10, norms, 1.0)
        data_for_clustering = pd.DataFrame(
            data_for_clustering.values / norms,
            index=data_for_clustering.index,
            columns=data_for_clustering.columns
        )
        logger.info("载荷标准化完成")

    variable_names_original = data_for_clustering.index.tolist()
    clustering_success = False

    if not cluster_vars:
        logger.info("跳过变量聚类，使用原始顺序。")
        return data_for_clustering, variable_names_original, False

    # 对变量进行聚类 (如果变量多于1个)
    if data_for_clustering.shape[0] > 1:
        try:
            linked = sch.linkage(data_for_clustering.values, method='ward', metric='euclidean')
            dendro = sch.dendrogram(linked, no_plot=True)
            clustered_indices = dendro['leaves']
            data_for_clustering = data_for_clustering.iloc[clustered_indices, :]
            variable_order = data_for_clustering.index.tolist()  # 聚类成功后更新
            clustering_success = True
            logger.info("因子载荷变量聚类成功。")
        except Exception as e_cluster:
            logger.warning(f"因子载荷变量聚类失败: {e_cluster}. 将按原始顺序显示变量。")
            variable_order = variable_names_original
            data_for_clustering = loadings_df.copy()  # 恢复原始数据
    else:
        logger.info("只有一个变量，跳过聚类。")
        variable_order = variable_names_original

    return data_for_clustering, variable_order, clustering_success 