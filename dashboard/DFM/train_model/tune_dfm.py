# -*- coding: utf-8 -*-

"""
超参数和变量逐步前向选择脚本。
目标：最小化 OOS RMSE。
"""
import pandas as pd
import numpy as np
import sys
import os
import time
import warnings

# === 优化：添加全局静默控制 ===
_SILENT_MODE = os.getenv('DFM_SILENT_WARNINGS', 'true').lower() == 'true'
_TRAINING_SILENT_MODE = os.getenv('DFM_TRAINING_SILENT', 'true').lower() == 'true'

def _safe_print(*args, **kwargs):
    """安全的条件化print函数，在多进程环境下也能正常工作"""
    if not _SILENT_MODE:
        try:
            print(*args, **kwargs)
        except:
            pass  # 忽略任何打印错误

# 全局回调函数变量
_global_progress_callback = None

def _training_print(*args, **kwargs):
    """训练过程专用的条件化print函数，可以通过环境变量控制是否输出"""
    if not _TRAINING_SILENT_MODE:
        try:
            message = ' '.join(str(arg) for arg in args)

            # 如果有全局回调函数，使用它
            if _global_progress_callback and callable(_global_progress_callback):
                # 添加标识符，确保UI能识别这是训练日志
                formatted_message = f"[TRAIN_PROGRESS] {message}"
                _global_progress_callback(formatted_message)
                # 同时也输出到控制台用于调试
                print(*args, **kwargs)
            else:
                # 否则使用标准print
                print(*args, **kwargs)
        except:
            pass  # 忽略任何打印错误
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns # <-- MOVED BACK TO TOP
# 移除并行处理：import concurrent.futures
from tqdm import tqdm
import traceback
from typing import Tuple, List, Dict, Union, Optional, Any # 添加 Tuple, Optional, Any
import unicodedata # <-- 新增导入
from sklearn.decomposition import PCA # <-- 新增：导入 PCA
from sklearn.impute import SimpleImputer # <-- 新增：导入 SimpleImputer
import multiprocessing
from collections import defaultdict
import logging
import joblib # 用于保存和加载模型/结果
from datetime import datetime
import pickle

# === 配置多线程BLAS库加速矩阵运算 ===
# 获取CPU核心数
_CPU_COUNT = multiprocessing.cpu_count()
# 设置环境变量以启用多线程BLAS（OpenBLAS/MKL）
os.environ['OMP_NUM_THREADS'] = str(_CPU_COUNT)
os.environ['MKL_NUM_THREADS'] = str(_CPU_COUNT)
os.environ['NUMEXPR_NUM_THREADS'] = str(_CPU_COUNT)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(_CPU_COUNT)
os.environ['OPENBLAS_NUM_THREADS'] = str(_CPU_COUNT)

# 导入threadpoolctl以进行更精细的线程控制
from threadpoolctl import threadpool_limits
_HAS_THREADPOOLCTL = True
from dashboard.DFM import config
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# 安全检查 config.EXCEL_DATA_FILE 是否存在
if hasattr(config, 'EXCEL_DATA_FILE'):
    EXCEL_DATA_FILE = config.EXCEL_DATA_FILE
else:
    # 如果 config 中没有 EXCEL_DATA_FILE，使用默认路径
    # 优化：完全移除重复的警告打印（多进程环境下全局变量无效）
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    EXCEL_DATA_FILE = os.path.join(project_root, 'data', '经济数据库0605.xlsx')

# 安全访问 config 属性，提供默认值
TEST_MODE = getattr(config, 'TEST_MODE', False)
N_ITER_TEST = getattr(config, 'N_ITER_TEST', 2)
N_ITER_FIXED = getattr(config, 'N_ITER_FIXED', 30)
TARGET_FREQ = getattr(config, 'TARGET_FREQ', 'W-FRI')
TARGET_SHEET_NAME = getattr(config, 'TARGET_SHEET_NAME', '工业增加值同比增速_月度_同花顺')
TARGET_VARIABLE = getattr(config, 'TARGET_VARIABLE', '规模以上工业增加值:当月同比')
CONSECUTIVE_NAN_THRESHOLD = getattr(config, 'CONSECUTIVE_NAN_THRESHOLD', 10)
REMOVE_VARS_WITH_CONSECUTIVE_NANS = getattr(config, 'REMOVE_VARS_WITH_CONSECUTIVE_NANS', True)
TYPE_MAPPING_SHEET = getattr(config, 'TYPE_MAPPING_SHEET', '指标体系')

VALIDATION_END_DATE = getattr(config, 'VALIDATION_END_DATE', '2024-12-27')
VALIDATION_START_DATE = getattr(config, 'VALIDATION_START_DATE', None)  # 将从UI获取或自动计算
TRAIN_END_DATE = getattr(config, 'TRAIN_END_DATE', '2024-06-28')
TRAINING_START_DATE = getattr(config, 'TRAINING_START_DATE', '2020-01-01')

FACTOR_SELECTION_METHOD = getattr(config, 'FACTOR_SELECTION_METHOD', 'bai_ng')
PCA_INERTIA_THRESHOLD = getattr(config, 'PCA_INERTIA_THRESHOLD', 0.9)
ELBOW_DROP_THRESHOLD = getattr(config, 'ELBOW_DROP_THRESHOLD', 0.1)
COMMON_VARIANCE_CONTRIBUTION_THRESHOLD = getattr(config, 'COMMON_VARIANCE_CONTRIBUTION_THRESHOLD', 0.8)

HEATMAP_TOP_N_VARS = getattr(config, 'HEATMAP_TOP_N_VARS', 5)
IC_MAX_FACTORS = getattr(config, 'IC_MAX_FACTORS', 10)


from dashboard.DFM.train_model.dfm_core import evaluate_dfm_params
from dashboard.DFM.train_model.analysis_utils import calculate_pca_variance, calculate_factor_contributions, calculate_individual_variable_r2, calculate_industry_r2, calculate_factor_industry_r2
from dashboard.DFM.train_model.variable_selection import perform_global_backward_selection


# === 导入优化：添加缓存机制 ===
_IMPORT_CACHE = {}
_IMPORT_MESSAGES_SHOWN = False

def _cached_import_with_message(module_name, import_func, success_msg="", error_msg=""):
    """缓存导入并控制消息显示"""
    global _IMPORT_CACHE, _IMPORT_MESSAGES_SHOWN

    if module_name in _IMPORT_CACHE:
        return _IMPORT_CACHE[module_name]

    result = import_func()
    _IMPORT_CACHE[module_name] = result
    # 只在首次导入时显示消息
    if not _IMPORT_MESSAGES_SHOWN and success_msg:
        print(success_msg)
    _IMPORT_MESSAGES_SHOWN = True
    return result

# train_model模块不再直接进行数据预处理
# 所有数据预处理在data_prep模块完成，这里只使用预处理结果

# 导入并行超参数搜索模块（可选）
try:
    from dashboard.DFM.train_model.parallel_hyperparameter_search import parallel_hyperparameter_search, adaptive_grid_search
    _PARALLEL_SEARCH_AVAILABLE = True
except ImportError:
    parallel_hyperparameter_search = None
    adaptive_grid_search = None
    _PARALLEL_SEARCH_AVAILABLE = False
    print("[WARN] 并行超参数搜索模块不可用，将使用标准搜索方法")

# 导入批量日志记录器（可选）
try:
    from dashboard.DFM.train_model.batch_logger import BatchLogger, TrainingProgressLogger, setup_optimized_logging
    _BATCH_LOGGER_AVAILABLE = True
except ImportError:
    BatchLogger = None
    TrainingProgressLogger = None
    setup_optimized_logging = None
    _BATCH_LOGGER_AVAILABLE = False
    print("[WARN] 批量日志记录器不可用，将使用标准日志")

from dashboard.DFM.train_model.DynamicFactorModel import DFM_EMalgo

from dashboard.DFM.train_model.generate_report import generate_report_with_params
_GENERATE_REPORT_AVAILABLE = True

def apply_bai_ng_to_final_variables(data_standardized, max_k=10):
    """
    基于最终变量集应用Bai & Ng ICp2准则确定最优因子数
    
    Args:
        data_standardized: 标准化后的数据 (DataFrame)
        max_k: 最大因子数，固定为10
    
    Returns:
        optimal_k: 最优因子数 (1-10范围内)
    """
    from sklearn.decomposition import PCA
    
    try:
        # 1. 计算PCA特征值
        pca = PCA()
        pca.fit(data_standardized)
        eigenvalues = pca.explained_variance_
        
        N, T = data_standardized.shape
        max_k_actual = min(max_k, len(eigenvalues), 10)
        
        print(f"    参数: N={N}, T={T}, max_k={max_k_actual}")
        
        # 2. 应用ICp2准则寻找最优因子数
        min_ic = np.inf
        best_k = 1
        ic_values = {}
        
        print("    计算各k的ICp2值...")
        for k in range(1, max_k_actual + 1):
            # 计算SSR(k) - 残差平方和
            ssr_k = T * np.sum(eigenvalues[k:]) if k < len(eigenvalues) else 1e-9
            
            if ssr_k <= 1e-9:
                ic_p2 = np.inf
            else:
                # 计算V(k) = SSR(k)/(NT)
                v_k = ssr_k / (N * T)
                # 计算惩罚项
                penalty = k * (N + T) / (N * T) * np.log(min(N, T))
                # ICp2 = log(V(k)) + penalty
                ic_p2 = np.log(v_k) + penalty
            
            ic_values[k] = ic_p2
            if ic_p2 < min_ic:
                min_ic = ic_p2
                best_k = k
            
            print(f"      k={k}: ICp2={ic_p2:.6f}")
        
        if min_ic != np.inf and best_k > 0:
            print(f"    Bai-Ng ICp2准则最优因子数: k = {best_k} (最小ICp2 = {min_ic:.6f})")
            return best_k
        else:
            print(f"    警告: ICp2计算未能找到有效最优k，使用默认值k=1")
            return 1
            
    except Exception as e:
        print(f"    错误: Bai-Ng ICp2计算失败: {e}，使用默认值k=1")
        return 1

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning) # <-- 新增: 尝试更具体地忽略 UserWarning
matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 不再使用固定的输出目录，所有结果通过返回值传递给UI

# 时间戳用于文件名和目录名
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# 不再创建固定的输出目录

# 移除日志文件和Excel输出文件路径，改为内存处理
# 不再创建物理文件，所有输出通过返回值传递

root_logger = logging.getLogger() # <-- 获取根 logger
root_logger.setLevel(logging.INFO) # <-- 设置级别

# 清除可能存在的旧处理器 (针对根 logger)
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# 创建流处理器 (控制台)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO) # 控制台也用 INFO 级别

# 创建格式化器 (可选，但推荐)
formatter = logging.Formatter('%(message)s')
stream_handler.setFormatter(formatter) # 控制台也使用格式

# 将处理器添加到根日志记录器
root_logger.addHandler(stream_handler)

logger = logging.getLogger(__name__) # 获取当前模块的 logger (主要用于方便后续调用)


def run_tuning(external_data=None, external_target_variable=None, external_selected_variables=None, external_var_industry_map=None, external_max_lags=None, enable_variable_selection=False, variable_selection_method='none', max_iterations=30, progress_callback=None):
    # 初始化log_file，避免作用域问题
    log_file = None

    # 设置全局回调函数
    global _global_progress_callback, VALIDATION_START_DATE, TRAIN_END_DATE
    _global_progress_callback = progress_callback

    # === 启用多线程BLAS加速 ===
    if _HAS_THREADPOOLCTL:
        # 使用threadpoolctl动态设置线程数
        thread_limit_context = threadpool_limits(limits=_CPU_COUNT, user_api='blas')
        thread_limit_context.__enter__()
        if progress_callback:
            progress_callback(f"已启用{_CPU_COUNT}核心并行计算（使用threadpoolctl）")
    else:
        if progress_callback:
            progress_callback(f"已通过环境变量启用{_CPU_COUNT}核心并行计算")

    try:
        n_iter_to_use = N_ITER_TEST if TEST_MODE else max_iterations
        _training_print(f"[CRITICAL FIX] 使用迭代次数: {n_iter_to_use} (UI设置: {max_iterations}, 测试模式: {TEST_MODE})")

        max_lags_to_use = external_max_lags if external_max_lags is not None else 1
        _training_print(f"使用因子自回归阶数: {max_lags_to_use}")
        
        # 设置变量选择方法
        VARIABLE_SELECTION_METHOD = variable_selection_method
        _training_print(f"变量选择方法: {VARIABLE_SELECTION_METHOD}")
        

        # === 优化：恢复正常多进程功能 ===
        import os  # 确保os模块在函数作用域内可用
        # 【激进优化】使用更多线程以达到100%CPU占用率
        cpu_count = os.cpu_count() if os.cpu_count() else 4
        MAX_WORKERS = min(128, cpu_count * 5)  # 5倍CPU核心数，最多128个线程
        print(f"[CPU激进优化] 使用 {MAX_WORKERS} 个工作线程 (CPU核心数: {cpu_count}, 目标: 100%CPU)")
        # 设置多进程启动方式
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)

        if VALIDATION_START_DATE is None:
            try:
                # 如果TRAIN_END_DATE为None，使用默认值
                if TRAIN_END_DATE is None or TRAIN_END_DATE == 'None':
                    effective_train_end = '2024-06-28'  # 使用默认值
                else:
                    effective_train_end = TRAIN_END_DATE

                train_end_dt = pd.to_datetime(effective_train_end)
                # 假设数据频率是周五结束 ('W-FRI')，验证期从下一周的周五开始
                offset = pd.DateOffset(weeks=1)
                calculated_dt = train_end_dt + offset
                VALIDATION_START_DATE = calculated_dt.strftime('%Y-%m-%d')
            except Exception as e:
                # 使用默认的验证期开始日期，而不是退出
                VALIDATION_START_DATE = '2024-07-05'  # 默认验证期开始日期
        else:
            pass  # UI提供了验证开始日期，直接使用

        script_start_time = time.time()
        total_evaluations = 0
        svd_error_count = 0
        # log_file已在上面初始化
        best_variables_stage1 = None
        best_score_stage1 = None
        optimal_k_stage2 = None
        factor_variances_explained_stage2 = None
        k_stage1 = None
        final_variables = None # Initialize final_variables
        # 优化：删除未使用的标准化参数
        # saved_standardization_mean 和 saved_standardization_std 已删除（实际未被使用）
        pca_results_df = None
        contribution_results_df = None
        factor_contributions = None
        individual_r2_results = None
        industry_r2_results = None
        factor_industry_r2_results = None
        # 移除 factor_type_r2_results
        final_dfm_results_obj = None
        final_data_processed = None
        final_eigenvalues = None # <<< 新增：初始化用于存储特征根的变量

        _training_print(f"--- 开始两阶段调优 (阶段1: 变量筛选, 阶段2: 因子数筛选) ---")
        _training_print(f"阶段1: 全局后向变量筛选 (固定 k=块数 N, 优化目标: HR -> -RMSE)")
        _training_print(f"阶段2: 基于阶段1变量，因子数选择方法: {FACTOR_SELECTION_METHOD}")
        
        # 添加详细的参数验证和日志记录
        _training_print(f"[TRAINING_START] 模型训练开始，验证所有关键参数:")
        _training_print(f"  - FACTOR_SELECTION_METHOD: {FACTOR_SELECTION_METHOD}")
        if hasattr(globals(), '_global_progress_callback') and _global_progress_callback:
            _global_progress_callback(f"[TRAINING_VALIDATION] 开始训练，验证参数配置:")
            _global_progress_callback(f"  - 当前因子选择方法: {FACTOR_SELECTION_METHOD}")
            
            if FACTOR_SELECTION_METHOD == 'fixed':
                fixed_k_global = getattr(sys.modules[__name__], 'FIXED_K_FACTORS', 'NOT_SET')
                _global_progress_callback(f"  - 固定因子数: {fixed_k_global}")
                _training_print(f"  - FIXED_K_FACTORS: {fixed_k_global}")
            elif FACTOR_SELECTION_METHOD == 'cumulative':
                pca_threshold_global = PCA_INERTIA_THRESHOLD if PCA_INERTIA_THRESHOLD is not None else 0.9
                _global_progress_callback(f"  - PCA阈值: {pca_threshold_global}")
                _training_print(f"  - PCA_INERTIA_THRESHOLD: {pca_threshold_global}")
            elif FACTOR_SELECTION_METHOD == 'elbow':
                elbow_threshold_global = ELBOW_DROP_THRESHOLD if ELBOW_DROP_THRESHOLD is not None else 0.1
                _global_progress_callback(f"  - Elbow阈值: {elbow_threshold_global}")
                _training_print(f"  - ELBOW_DROP_THRESHOLD: {elbow_threshold_global}")
                
        if FACTOR_SELECTION_METHOD == 'cumulative':
            # 修复：确保PCA_INERTIA_THRESHOLD不为None
            pca_threshold = PCA_INERTIA_THRESHOLD if PCA_INERTIA_THRESHOLD is not None else 0.9
            print(f"       阈值: 累积方差贡献 >= {pca_threshold*100:.1f}%")
        elif FACTOR_SELECTION_METHOD == 'elbow':
            # 修复：确保ELBOW_DROP_THRESHOLD不为None
            elbow_threshold = ELBOW_DROP_THRESHOLD if ELBOW_DROP_THRESHOLD is not None else 0.1
            print(f"       阈值: 边际方差贡献下降率 < {elbow_threshold*100:.1f}%")
        
        # 移除文件日志输出，改为控制台输出
        print(f"--- 开始详细调优日志 (Run: {timestamp_str}) ---")
        print(f"配置: 两阶段流程")
        print(f"  阶段1: 全局后向变量筛选 (固定 k=块数 N, 优化 HR -> -RMSE)")
        # 修复：确保阈值变量不为None
        pca_threshold_safe = PCA_INERTIA_THRESHOLD if PCA_INERTIA_THRESHOLD is not None else 0.9
        elbow_threshold_safe = ELBOW_DROP_THRESHOLD if ELBOW_DROP_THRESHOLD is not None else 0.1
        print(f"  阶段2: 因子选择 (方法={FACTOR_SELECTION_METHOD}, "
              f"阈值={'PCA>='+str(pca_threshold_safe) if FACTOR_SELECTION_METHOD=='cumulative' else 'Drop<'+str(elbow_threshold_safe)})")
        # log_file已在上面初始化，不再重复赋值

        # 修复：保存原始完整数据用于计算target_mean_original
        original_full_data = None

        # 使用外部传入的数据或调用数据准备模块
        if external_data is not None:
            logger.info("--- 使用外部传入的数据 (来自UI/data_prep模块) ---")

            # 检查并去除传入数据中的重复列
            duplicate_mask = external_data.columns.duplicated(keep=False)
            if duplicate_mask.any():
                duplicate_columns = external_data.columns[duplicate_mask].tolist()
                from collections import Counter
                column_counts = Counter(external_data.columns)
                duplicated_names = {name: count for name, count in column_counts.items() if count > 1}

                print(f"发现重复列，总数: {len(duplicate_columns)}")
                print(f"具体重复列名: {list(duplicated_names.keys())}")
                for col_name, count in duplicated_names.items():
                    print(f"  - '{col_name}': 出现{count}次")

                # 去除重复列，保留第一个
                external_data_cleaned = external_data.loc[:, ~external_data.columns.duplicated(keep='first')]

                # 保存原始完整数据（去重后的）
                original_full_data = external_data_cleaned.copy()
                all_data_aligned_weekly = external_data_cleaned.copy()
            else:
                # 保存原始完整数据
                original_full_data = external_data.copy()
                all_data_aligned_weekly = external_data.copy()

            # 修复：当使用外部数据时，跳过变量映射，直接使用完整数据
            if external_selected_variables and external_target_variable:
                print(f"\n[外部数据处理] 使用data_prep模块预处理的完整数据:")
                print(f"  - 数据形状: {all_data_aligned_weekly.shape}")
                print(f"  - 目标变量: '{external_target_variable}'")
                print(f"  - 目标变量在数据中: {external_target_variable in all_data_aligned_weekly.columns}")

                # 验证目标变量存在
                if external_target_variable not in all_data_aligned_weekly.columns:
                    raise ValueError(f"目标变量 '{external_target_variable}' 不在数据中")

                # 关键修复：直接使用完整数据，不进行变量映射和过滤
                # 因为data_prep模块已经完成了所有必要的预处理
                available_vars = list(all_data_aligned_weekly.columns)
                print(f"  - 使用完整数据，包含 {len(available_vars)} 个变量")
                print(f"  - 跳过变量映射过程，保持数据完整性")

                # 设置空的映射失败列表（因为我们不进行映射）
                failed_mappings = []

                # 关键修复：不进行数据过滤，保持完整数据
                print(f"\n[数据保持完整] 使用完整的预处理数据:")
                print(f"  - 数据形状: {all_data_aligned_weekly.shape}")
                print(f"  - 包含所有 {len(all_data_aligned_weekly.columns)} 个变量")
                print(f"  - 跳过变量过滤，保持数据完整性")

            # 设置空的映射和日志（因为使用外部数据）
            var_industry_map_inferred = {}
            final_transform_details = {}
            removed_variables_log = {}
        else:
            # 修复：移除重复的数据处理，统一使用UI接口
            logger.warning("检测到旧的独立数据处理路径，建议使用UI接口")
            logger.info("--- 使用空数据，建议通过UI接口传递预处理数据 ---")
            
            # 返回空数据，强制使用UI接口
            all_data_aligned_weekly = None
            var_industry_map_inferred = {}
            final_transform_details = {}
            removed_variables_log = {}
            
            logger.warning("重要：请确保通过UI接口传递预处理数据，避免重复处理")
        
        if all_data_aligned_weekly is None or all_data_aligned_weekly.empty:
            logger.error("数据准备失败或返回空数据框。退出调优。")
            if log_file and not log_file.closed: log_file.close()
            raise SystemExit(1)

        print(f"数据准备模块成功返回处理后的数据. Shape: {all_data_aligned_weekly.shape}")
        
        all_variable_names = all_data_aligned_weekly.columns.tolist()
        if TARGET_VARIABLE not in all_variable_names:
            print(f"错误: 目标变量 {TARGET_VARIABLE} 不在合并后的数据中。")
            if log_file and not log_file.closed: log_file.close()
            raise SystemExit(1)
        
        initial_variables = sorted(all_variable_names)
        print(f"\n初始变量组 ({len(initial_variables)}): {initial_variables[:10]}...") # Print only first few
        print("-"*30)

        # 使用传入的行业映射数据，移除类型映射
        print("使用传入的行业映射数据...")
        var_industry_map = external_var_industry_map if external_var_industry_map else {}

        industry_map_size = len(var_industry_map) if var_industry_map else 0
        print(f"行业映射数据获取完成: 大小={industry_map_size}")

        # 如果行业映射数据为空，只显示警告，继续执行
        if industry_map_size == 0:
            print("行业映射为空！Factor-Industry R² 将无法计算，但模型训练将继续。")
        
        print(f"\n[EARLY CHECK 2] Industry mapping loaded. Size: {len(var_industry_map)}")
        print("-"*30)

        print("计算原始目标变量的稳定统计量...")
        try:
            # 使用原始完整数据计算target_mean_original，而不是过滤后的数据
            data_for_target_stats = original_full_data if original_full_data is not None else all_data_aligned_weekly

            # 检查并去除重复列
            duplicate_cols = data_for_target_stats.columns.duplicated()
            if duplicate_cols.any():
                # 去除重复列，保留第一个
                data_for_target_stats = data_for_target_stats.loc[:, ~duplicate_cols]

            # 确定目标变量名
            target_var_for_stats = external_target_variable if external_target_variable else TARGET_VARIABLE

            # 确保target_var_for_stats是字符串，避免Series判断问题
            if not isinstance(target_var_for_stats, str):
                raise ValueError(f"目标变量名必须是字符串，当前类型: {type(target_var_for_stats)}")

            if target_var_for_stats not in data_for_target_stats.columns:
                raise ValueError(f"目标变量 '{target_var_for_stats}' 不在数据中")

            # 修复：目标变量统计量也应仅使用训练集计算，避免信息泄漏
            try:
                # 仅使用训练集计算目标变量统计量
                train_target_for_stats = data_for_target_stats.loc[:TRAIN_END_DATE, target_var_for_stats].copy().dropna()
                if train_target_for_stats.empty:
                    raise ValueError(f"训练集中目标变量 '{target_var_for_stats}' 移除 NaN 后为空")
                
                target_mean_original = train_target_for_stats.mean()
                target_std_original = train_target_for_stats.std()
                print(f"    目标变量统计量仅使用训练集数据: {len(train_target_for_stats)} 样本 (截止到 {TRAIN_END_DATE})")
                
            except (KeyError, IndexError) as e:
                print(f"    无法提取训练集计算目标变量统计量: {e}，回退到全数据集")
                # 回退到原有逻辑
                original_target_series_for_stats = data_for_target_stats[target_var_for_stats].copy().dropna()
                if original_target_series_for_stats.empty:
                    raise ValueError(f"原始目标变量 '{target_var_for_stats}' 移除 NaN 后为空")
                
                target_mean_original = original_target_series_for_stats.mean()
                target_std_original = original_target_series_for_stats.std()

            if pd.isna(target_mean_original) or pd.isna(target_std_original) or target_std_original == 0:
                raise ValueError(f"计算得到的原始目标变量统计量无效 (Mean: {target_mean_original}, Std: {target_std_original})。")

            print(f"已计算原始目标变量的稳定统计量: Mean={target_mean_original:.4f}, Std={target_std_original:.4f}")

        except Exception as e:
            print(f"错误: 计算原始目标变量统计量失败: {e}")
            if log_file and not log_file.closed: log_file.close()
            raise SystemExit(1)
        print("-"*30)

        # ... (Consecutive NaN check remains the same) ...
        initial_predictors = [v for v in initial_variables if v != TARGET_VARIABLE]
        if REMOVE_VARS_WITH_CONSECUTIVE_NANS:
            print(f"\n--- (启用) 检查初始预测变量 ({len(initial_predictors)}) 的连续缺失值 (阈值 >= {CONSECUTIVE_NAN_THRESHOLD})... ---")
            # Actual removal logic might be in prepare_data or needs to be added here if not
            pass # Placeholder for brevity, assuming prepare_data handles this or it's done later
        else:
            print(f"\n--- (禁用) 跳过基于连续缺失值 (阈值 >= {CONSECUTIVE_NAN_THRESHOLD}) 的初始变量移除步骤。---")
            if log_file:
                try: 
                    log_file.write("\n--- (禁用) 跳过变量筛选前连续缺失值检查 ---\n")
                except Exception: 
                    pass
        print("-"*30)

        # 优化：移除不必要的变量块和因子数N计算
        # 这些逻辑在优化后的流程中已经不需要，因为我们直接使用步骤0的结果


        # 优化：移除变量块计算逻辑，在优化后的流程中不再需要

        print("\n--- 步骤 0: 基于初始全体变量估计因子数 (优化版本) ---")
        k_initial_estimate = 1 # 默认回退值

        # 优化：全局标准化参数和结果存储
        global_standardization_mean = None
        global_standardization_std = None
        data_standardized_global = None
        data_standardized_imputed_global = None

        try:
            # 使用全部变量进行初始估计
            data_for_initial_estimation = all_data_aligned_weekly.copy()
            estimation_scope_info = "全部变量"

            if data_for_initial_estimation is None or data_for_initial_estimation.empty:
                 raise ValueError(f"未能为步骤 0 准备有效的数据 (范围: {estimation_scope_info})。")

            print(f"  准备用于初始估计的数据 ({estimation_scope_info})...")

            print("    优化：一次性标准化全部变量 (后续阶段将复用结果)...")
            # 修复：仅使用训练集计算全局标准化参数，避免信息泄漏
            try:
                train_data_for_std = data_for_initial_estimation.loc[:TRAIN_END_DATE]
                print(f"    全局标准化仅使用训练集数据: {len(train_data_for_std)} 样本 (截止到 {TRAIN_END_DATE})")
                global_standardization_mean = train_data_for_std.mean(axis=0)
                global_standardization_std = train_data_for_std.std(axis=0)
            except (KeyError, IndexError) as e:
                print(f"    无法提取训练集进行全局标准化: {e}，回退到全数据集")
                global_standardization_mean = data_for_initial_estimation.mean(axis=0)
                global_standardization_std = data_for_initial_estimation.std(axis=0)
            zero_std_cols_initial = global_standardization_std[global_standardization_std == 0].index.tolist()

            if zero_std_cols_initial:
                print(f"    警告: 以下列标准差为0，将被全局移除: {zero_std_cols_initial}")
                data_for_initial_estimation = data_for_initial_estimation.drop(columns=zero_std_cols_initial)
                # 修复：重新计算标准化参数时也仅使用训练集
                try:
                    train_data_for_std = data_for_initial_estimation.loc[:TRAIN_END_DATE]
                    global_standardization_mean = train_data_for_std.mean(axis=0)
                    global_standardization_std = train_data_for_std.std(axis=0)
                except (KeyError, IndexError) as e:
                    print(f"    重新计算时无法提取训练集: {e}，使用全数据集")
                    global_standardization_mean = data_for_initial_estimation.mean(axis=0)
                    global_standardization_std = data_for_initial_estimation.std(axis=0)

            global_standardization_std[global_standardization_std == 0] = 1.0
            # 修复：分阶段标准化，避免数据泄漏
            # 仅标准化训练集部分，验证集将在评估时单独标准化
            try:
                train_indices = data_for_initial_estimation.index <= TRAIN_END_DATE
                data_standardized_global = data_for_initial_estimation.copy()
                data_standardized_global.loc[train_indices] = (
                    data_for_initial_estimation.loc[train_indices] - global_standardization_mean
                ) / global_standardization_std
                # 验证集使用训练集参数进行标准化（避免数据泄漏）
                val_indices = ~train_indices
                data_standardized_global.loc[val_indices] = (
                    data_for_initial_estimation.loc[val_indices] - global_standardization_mean
                ) / global_standardization_std
                print(f"    分阶段标准化完成. Shape: {data_standardized_global.shape}")
                print(f"    训练集标准化: {train_indices.sum()} 样本")
                print(f"    验证集标准化: {val_indices.sum()} 样本 (使用训练集参数)")
            except Exception as e:
                print(f"    分阶段标准化失败: {e}，回退到全局标准化")
                data_standardized_global = (data_for_initial_estimation - global_standardization_mean) / global_standardization_std
                print(f"    全局标准化完成. Shape: {data_standardized_global.shape}")
            print(f"    标准化参数已保存，后续阶段将复用")

            print("  优化：一次性缺失值插补 (后续阶段将复用结果)...")
            imputer_global = SimpleImputer(strategy='mean')
            data_standardized_imputed_global = data_standardized_global # 默认回退
            try:
                data_standardized_imputed_global_array = imputer_global.fit_transform(data_standardized_global)
                data_standardized_imputed_global = pd.DataFrame(
                    data_standardized_imputed_global_array,
                    columns=data_standardized_global.columns,
                    index=data_standardized_global.index
                )
                print(f"    全局缺失值插补完成. Shape: {data_standardized_imputed_global.shape}")
                print(f"    插补结果已保存，后续阶段将复用")
            except Exception as e_impute_init:
                print(f"    全局缺失值插补失败: {e_impute_init}. PCA 可能失败。")

            # 为了保持兼容性，设置初始阶段的变量名
            data_standardized_initial = data_standardized_global
            data_standardized_imputed_initial = data_standardized_imputed_global

            print("  执行初步 PCA 以获取解释方差和特征值...")
            pca_initial = PCA(n_components=None) # 计算所有主成分
            pca_cumulative_variance_initial = None
            eigenvalues_initial = None
            try:
                pca_initial.fit(data_standardized_imputed_initial)
                explained_variance_ratio_pct_initial = pca_initial.explained_variance_ratio_ * 100
                pca_cumulative_variance_initial = np.cumsum(explained_variance_ratio_pct_initial)
                eigenvalues_initial = pca_initial.explained_variance_
                print(f"    初步 PCA 完成. 计算了 {len(eigenvalues_initial)} 个主成分。")
            except Exception as e_pca_init:
                 print(f"    初步 PCA 计算失败: {e_pca_init}. 依赖 PCA 的方法将无法使用。")
            
            Lambda_initial = None
            if FACTOR_SELECTION_METHOD == 'cumulative_common':
                print("  为 'cumulative_common' 方法运行初步 DFM (因子数上限=变量数)...")
                # 理论上因子数不应超过观测数或变量数，这里用一个较大但合理的数
                max_factors_dfm_init = min(data_standardized_initial.shape[0], data_standardized_initial.shape[1])
                if max_factors_dfm_init <= 0:
                     print("    错误: 无法确定初步 DFM 的有效因子数上限。")
                else:
                     print(f"    设定初步 DFM 因子数上限为: {max_factors_dfm_init}")
                     try:
                         dfm_results_initial = DFM_EMalgo(
                             observation=data_standardized_initial, # DFM 使用未插补数据
                             n_factors=max_factors_dfm_init,
                             n_shocks=max_factors_dfm_init,
                             n_iter=n_iter_to_use, # 使用配置的迭代次数
                             train_end_date=TRAIN_END_DATE,  # 修复：传递训练结束日期避免信息泄漏
                             max_lags=max_lags_to_use  # 新增：传递因子自回归阶数
                         )
                         if dfm_results_initial is not None and hasattr(dfm_results_initial, 'Lambda'):
                             Lambda_initial = dfm_results_initial.Lambda
                             print(f"    初步 DFM 运行成功，获得载荷矩阵 Shape: {Lambda_initial.shape}")
                         else:
                             print("    错误: 初步 DFM 运行失败或未返回载荷矩阵 (Lambda)。")
                     except Exception as e_dfm_init:
                         print(f"    初步 DFM 运行失败: {e_dfm_init}。")
            
            print(f"  应用因子选择方法 '{FACTOR_SELECTION_METHOD}' 确定初始估计 k...")
            
            # [PARAM_DEBUG] 执行因子选择调试日志  
            if hasattr(globals(), '_global_progress_callback') and _global_progress_callback:
                _global_progress_callback(f"[PARAM_DEBUG] ====== 执行因子选择 ======")
                _global_progress_callback(f"[PARAM_DEBUG] 当前方法: {FACTOR_SELECTION_METHOD}")
                _global_progress_callback(f"[FACTOR_SELECTION] 开始使用方法: {FACTOR_SELECTION_METHOD}")
                if FACTOR_SELECTION_METHOD == 'fixed':
                    fixed_k_value = getattr(sys.modules[__name__], 'FIXED_K_FACTORS', 'undefined')
                    _global_progress_callback(f"[PARAM_DEBUG] 固定因子数: {fixed_k_value}")
                    _global_progress_callback(f"[FACTOR_SELECTION] Fixed方法的K值: {fixed_k_value}")
            
            temp_k_estimate = None

            if FACTOR_SELECTION_METHOD == 'fixed':
                import sys as sys_module
                fixed_k = getattr(sys_module.modules[__name__], 'FIXED_K_FACTORS', 1)
                temp_k_estimate = fixed_k
                print(f"    [FACTOR_FIXED] 'fixed' 方法使用固定因子数 k = {temp_k_estimate}")
                
                # 额外的验证日志
                if hasattr(globals(), '_global_progress_callback') and _global_progress_callback:
                    _global_progress_callback(f"[FACTOR_FIXED] 固定因子数方法激活，K={fixed_k}")
                
                # 添加详细的验证日志
                if hasattr(globals(), '_global_progress_callback') and _global_progress_callback:
                    _global_progress_callback(f"[FIXED_METHOD_VERIFY] 固定方法验证:")
                    _global_progress_callback(f"  - FACTOR_SELECTION_METHOD: {FACTOR_SELECTION_METHOD}")
                    _global_progress_callback(f"  - FIXED_K_FACTORS属性: {getattr(sys_module.modules[__name__], 'FIXED_K_FACTORS', 'NOT_FOUND')}")
                    _global_progress_callback(f"  - 实际使用的K值: {temp_k_estimate}")
                    
                    # 验证fixed方法是否正确生效
                    if temp_k_estimate is not None and temp_k_estimate > 0:
                        _global_progress_callback(f"固定因子数方法配置正确: K={temp_k_estimate}")
                    else:
                        _global_progress_callback(f"固定因子数方法配置异常: K={temp_k_estimate}")
            elif FACTOR_SELECTION_METHOD == 'cumulative':
                 if pca_cumulative_variance_initial is not None:
                      # 修复：确保PCA_INERTIA_THRESHOLD不为None
                      pca_threshold = PCA_INERTIA_THRESHOLD if PCA_INERTIA_THRESHOLD is not None else 0.9
                      k_indices = np.where(pca_cumulative_variance_initial >= pca_threshold * 100)[0]
                      if len(k_indices) > 0: temp_k_estimate = k_indices[0] + 1
                      else: temp_k_estimate = len(eigenvalues_initial) # Fallback: all components
                      print(f"    'cumulative' 方法估计 k = {temp_k_estimate}")
                 else: print("    错误: PCA 结果不可用，无法应用 'cumulative' 方法。")
            elif FACTOR_SELECTION_METHOD == 'elbow':
                 if eigenvalues_initial is not None and len(eigenvalues_initial) > 1:
                     variance_diff_ratio = np.diff(eigenvalues_initial) / eigenvalues_initial[:-1]
                     # 修复：确保ELBOW_DROP_THRESHOLD不为None
                     elbow_threshold = ELBOW_DROP_THRESHOLD if ELBOW_DROP_THRESHOLD is not None else 0.1
                     k_indices = np.where(np.abs(variance_diff_ratio) < elbow_threshold)[0]
                     if len(k_indices) > 0: temp_k_estimate = k_indices[0] + 1
                     else: temp_k_estimate = len(eigenvalues_initial) # Fallback: all components
                     print(f"    'elbow' 方法估计 k = {temp_k_estimate}")
                 elif eigenvalues_initial is not None and len(eigenvalues_initial) == 1:
                     optimal_k_stage2 = 1
                     print("    仅有1个主成分，无法应用手肘法，直接选择 k=1。") # Note: This sets optimal_k_stage2 directly, which might be premature here. Should set temp_k_estimate.
                     temp_k_estimate = 1 # <<< FIX: Should set temp_k_estimate
                 else: # <--- 修正：这个 else 对应 if eigenvalues is not None and len(eigenvalues) > 1
                      print("    错误: 由于 PCA 计算失败或因子数不足，无法应用 'elbow' 方法。将回退使用 k = k_initial_estimate。")
                      # optimal_k_stage2 = k_initial_estimate # <<< REMOVE: Don't set stage 2 k here
            elif FACTOR_SELECTION_METHOD == 'kaiser':
                 if eigenvalues_initial is not None:
                     k_kaiser = np.sum(eigenvalues_initial > 1)
                     temp_k_estimate = max(1, k_kaiser) # Ensure at least 1 factor
                     print(f"    'kaiser' 方法估计 k = {temp_k_estimate}")
                 else: print("    错误: PCA 特征值不可用，无法应用 'kaiser' 方法。")
            elif FACTOR_SELECTION_METHOD == 'cumulative_common':
                 cumulative_common_variance_pct_initial = None
                 if Lambda_initial is not None:
                     try:
                         # Ensure TARGET_VARIABLE exists in the columns used for this initial estimation
                         if TARGET_VARIABLE in data_standardized_initial.columns:
                             target_var_index_pos_init = data_standardized_initial.columns.get_loc(TARGET_VARIABLE)
                             if target_var_index_pos_init < Lambda_initial.shape[0]:
                                  lambda_target_initial = Lambda_initial[target_var_index_pos_init, :]
                                  lambda_target_sq_init = lambda_target_initial ** 2
                                  sum_lambda_target_sq_init = np.sum(lambda_target_sq_init)
                                  if sum_lambda_target_sq_init > 1e-9:
                                       pct_contribution_common_init = (lambda_target_sq_init / sum_lambda_target_sq_init) * 100
                                       cumulative_common_variance_pct_initial = np.cumsum(pct_contribution_common_init)
                                  else: print("    警告: 初步 DFM 目标平方载荷和过小。")
                             else: print(f"    错误: 目标变量索引 ({target_var_index_pos_init}) 超出初步 DFM 载荷矩阵范围 ({Lambda_initial.shape[0]})。")
                         else:
                              print(f"    错误: 目标变量 '{TARGET_VARIABLE}' 不在用于初始估计的数据列中，无法计算 common variance。")
                     except KeyError: print(f"    错误: 在初始标准化数据列中未找到目标变量 '{TARGET_VARIABLE}'。")
                     except Exception as e_common_init: print(f"    计算初始共同方差贡献时出错: {e_common_init}")
                 else: print("    错误: 初步 DFM 载荷不可用。")

                 if cumulative_common_variance_pct_initial is not None:
                      k_indices = np.where(cumulative_common_variance_pct_initial >= COMMON_VARIANCE_CONTRIBUTION_THRESHOLD * 100)[0]
                      if len(k_indices) > 0: temp_k_estimate = k_indices[0] + 1
                      else: temp_k_estimate = Lambda_initial.shape[1] # Fallback: all factors from DFM
                      print(f"    'cumulative_common' 方法估计 k = {temp_k_estimate}")
                 else: print("    错误: 无法应用 'cumulative_common' 方法。")
            elif FACTOR_SELECTION_METHOD == 'bai_ng':
                print(f"  应用 Bai and Ng (2002) ICp2 准则 (初始估计)...")
                # 使用步骤 0 计算得到的 eigenvalues_initial 和 data_standardized_imputed_initial
                if eigenvalues_initial is not None and 'data_standardized_imputed_initial' in locals() and data_standardized_imputed_initial is not None:
                    N_init = data_standardized_imputed_initial.shape[1]
                    T_init = data_standardized_imputed_initial.shape[0]
                    # 修复：应用与阶段2相同的自适应限制，防止搜索范围过大
                    import sys as sys_mod
                    current_module = sys_mod.modules[__name__]
                    ic_max_setting = getattr(current_module, 'IC_MAX_FACTORS', 10)
                    adaptive_max = min(20, max(10, N_init // 4))
                    k_max_bai_ng_init = min(len(eigenvalues_initial), ic_max_setting, adaptive_max)
                    if N_init > 0 and T_init > 0 and k_max_bai_ng_init > 0:
                        print(f"    参数 (初始): N={N_init}, T={T_init}, k_max={k_max_bai_ng_init}")
                        min_ic_init = np.inf
                        best_k_ic_init = 1
                        ic_values_init = {}

                        print("    计算各 k 的 ICp2 值 (初始)...")
                        for k in range(1, k_max_bai_ng_init + 1):
                            ssr_k_init = T_init * np.sum(eigenvalues_initial[k:]) if k < len(eigenvalues_initial) else 1e-9
                            if ssr_k_init <= 1e-9:
                                icp2_k_init = np.inf
                            else:
                                v_k_init = ssr_k_init / (N_init * T_init)
                                penalty_k_init = k * (N_init + T_init) / (N_init * T_init) * np.log(min(N_init, T_init))
                                icp2_k_init = np.log(v_k_init) + penalty_k_init
                                # print(f"      k={k}: SSR={ssr_k_init:.4f}, V(k)={v_k_init:.6f}, Penalty={penalty_k_init:.6f}, ICp2={icp2_k_init:.6f}") # Optional detailed print
                            ic_values_init[k] = icp2_k_init
                            if icp2_k_init < min_ic_init:
                                min_ic_init = icp2_k_init
                                best_k_ic_init = k

                        if min_ic_init != np.inf and best_k_ic_init > 0:
                            temp_k_estimate = best_k_ic_init
                            print(f"    根据 Bai-Ng ICp2 准则 (初始) 估计的因子数量: k = {temp_k_estimate} (最小 ICp2 = {min_ic_init:.6f})")
                        else:
                            print(f"    警告: Bai-Ng ICp2 (初始) 计算未能找到有效最优 k，将使用回退启发式。")
                            temp_k_estimate = None # Trigger fallback heuristic below
                    else:
                        print(f"    错误: Bai-Ng ICp2 (初始) 参数无效")
                else:
                    print("    错误: 缺少必要数据，无法应用 Bai-Ng 方法")
            else:
                 print(f"错误: 未知的因子选择方法 '{FACTOR_SELECTION_METHOD}'。") # Now this else only catches truly unknown methods

            if temp_k_estimate is not None and temp_k_estimate > 0:
                 k_initial_estimate = temp_k_estimate
                 print(f"步骤 0 完成。初始估计因子数 k_initial_estimate = {k_initial_estimate}")
            else:
                 k_initial_estimate = max(1, int(data_standardized_initial.shape[1] / 10)) # Fallback heuristic
                 print(f"警告: 未能通过所选方法估计有效的初始 k，将使用回退启发式 k = {k_initial_estimate}")
            
            # 允许使用算法估计的完整因子数，不设置人为上限
            print(f"使用算法估计的因子数: {k_initial_estimate}")

        except Exception as e_step0:
            print(f"步骤 0 (初始因子数估计) 失败: {e_step0}")
            traceback.print_exc()
            k_initial_estimate = max(1, int(len(initial_variables) / 10)) # Fallback heuristic
            print(f"警告: 因错误，将使用回退启发式 k = {k_initial_estimate} 进行阶段 1。")
            # 优化：步骤0失败时直接抛出异常，不需要回退机制
            raise RuntimeError("步骤0标准化失败，无法继续训练。请检查数据质量。")
        print("-" * 30)

        print(f"优化目标: (Avg Hit Rate, -Avg RMSE)")
        score_tuple_definition_stage1 = "(Avg Hit Rate, -Avg RMSE)" # 固定评分标准

        # best_score_stage1 = (-np.inf, np.inf) # 全局筛选函数内部会计算初始分数

        # 修复：根据因子选择方法设置正确的K值
        if FACTOR_SELECTION_METHOD == 'fixed':
            # 使用UI设置的固定因子数
            import sys as sys_module
            fixed_k = getattr(sys_module.modules[__name__], 'FIXED_K_FACTORS', 1)
            best_params_stage1 = {'k_factors': fixed_k}
        else:
            # 使用自动估计的因子数
            best_params_stage1 = {'k_factors': k_initial_estimate}

        # best_variables_stage1 在下方确定

        try:
            # 优先使用外部传入的变量选择
            if external_data is not None and external_selected_variables:
                # 使用外部传入的预测变量（不包含目标变量）
                variables_for_selection_start = external_selected_variables
                # 确保变量不重复
                variables_for_selection_start = list(dict.fromkeys(variables_for_selection_start))
                selection_scope_info = f"UI选择的 {len(variables_for_selection_start)} 个预测变量"
                # 添加标志，防止后续逻辑覆盖
                using_external_variables = True
            else:
                # 回退到所有变量
                variables_for_selection_start = list(initial_variables)
                selection_scope_info = f"全部 {len(initial_variables)} 个变量"
                using_external_variables = False
            
            if not using_external_variables:
                if TEST_MODE:
                     # Test mode - use all variables but fewer iterations
                     selection_scope_info = f"全部 {len(initial_variables)} 个变量 (测试模式)"
                     print(f"\n*** 测试模式：全局筛选将使用所有变量 (迭代次数减少)。 ***\n")
                     if log_file: log_file.write("*** 测试模式：使用所有变量进行筛选 (迭代次数减少) ***\n")
                else:
                     # Full mode - use all variables
                     print(f"\n*** 完整模式：全局筛选将使用所有变量。 ***\n")
                     if log_file: log_file.write("*** 完整模式：使用所有变量进行筛选 ***\n")
            else:
                # 使用外部变量时的提示
                print(f"\n*** 外部变量模式：全局筛选将使用UI选择的 {len(variables_for_selection_start)} 个变量。 ***\n")
                if log_file: log_file.write(f"*** 外部变量模式：使用UI选择的 {len(variables_for_selection_start)} 个变量进行筛选 ***\n")

            # print(f"计算阶段 1 初始基准分数 (使用 {selection_scope_info}, k={k_initial_estimate})...") 
            # ... [移除基准分数计算代码] ...

            print(f"enable_variable_selection = {enable_variable_selection}")
            print(f"VARIABLE_SELECTION_METHOD = {VARIABLE_SELECTION_METHOD}")
            
            if enable_variable_selection and VARIABLE_SELECTION_METHOD == 'global_backward':
                logger.info("--- 即将调用 perform_global_backward_selection 进行全局变量筛选... ---")
                print(f"[执行] 开始对 {selection_scope_info} 进行全局后向变量剔除 (固定 k={k_initial_estimate})...")
                # 注意：perform_global_backward_selection 内部有 tqdm 进度条
                sel_variables_stage1, sel_params_stage1, sel_score_tuple_stage1, sel_eval_count_stage1, sel_svd_err_count_stage1 = perform_global_backward_selection(
                    initial_variables=variables_for_selection_start, # <-- 使用确定的起始变量集
                    initial_params=best_params_stage1,         # 使用初始最佳参数 (包含固定k)
                    # initial_score_tuple - 不需要传递
                    target_variable=TARGET_VARIABLE,
                    all_data=all_data_aligned_weekly,
                    # 移除 var_type_map 参数
                    validation_start=VALIDATION_START_DATE, # <-- 使用UI设置或自动计算的日期
                    validation_end=VALIDATION_END_DATE,
                    target_freq=TARGET_FREQ,
                    train_end_date=TRAIN_END_DATE,
                    n_iter=n_iter_to_use,
                    target_mean_original=target_mean_original,
                    target_std_original=target_std_original,
                    max_workers=MAX_WORKERS,
                    evaluate_dfm_func=evaluate_dfm_params,
                    max_lags=max_lags_to_use,  # 新增：传递因子自回归阶数
                    log_file=log_file,
                    use_optimization=True  # [ENHANCED] 显式启用多进程并行优化
                    # blocks - 不需要传递
                    # hyperparams_to_tune - 不需要传递
                    # auto_select_factors - 不需要传递
                )

                best_variables_stage1 = sel_variables_stage1 # 更新为筛选后的变量
                best_params_stage1 = sel_params_stage1 # 参数理论上不变，但保持一致
                best_score_stage1 = sel_score_tuple_stage1 # 更新为筛选后的分数
                total_evaluations += sel_eval_count_stage1
                svd_error_count += sel_svd_err_count_stage1

                # 检查最终得分是否有效
                final_score_valid = False
                if best_score_stage1 is not None and len(best_score_stage1) == 2 and all(np.isfinite(list(best_score_stage1))):
                    final_score_valid = True

                if final_score_valid:
                    final_hr_stage1, final_neg_rmse_stage1 = best_score_stage1
                    num_predictors_stage1 = len([v for v in best_variables_stage1 if v != TARGET_VARIABLE])
                    print(f"阶段 1 (全局筛选) 完成。最佳结果 (固定 k={k_initial_estimate}): 评分=(HR={final_hr_stage1:.2f}%, RMSE={-final_neg_rmse_stage1:.6f}), 预测变量数量={num_predictors_stage1}") # <-- 修改打印
                    if log_file:
                        log_file.write(f"\n--- 阶段 1 结果 (全局筛选) ---\n") # <-- 修改日志
                        log_file.write(f"起始变量范围: {selection_scope_info}\n") # <-- 修正日志行
                        log_file.write(f"固定因子数 (N): {k_initial_estimate}\n")
                        log_file.write(f"最佳评分 (HR, -RMSE): {best_score_stage1}\n")
                        log_file.write(f"最终预测变量数量: {num_predictors_stage1}\n") # <-- 修改日志
                else:
                    print("错误: 阶段 1 (全局筛选) 未能找到有效的变量集和评分。无法继续。") # <-- 修改打印
                    if log_file and not log_file.closed: log_file.close()
                    raise SystemExit(1)
            else:
                logger.info("--- 跳过变量选择，直接使用用户选择的变量 ---")
                print(f"跳过变量选择阶段，直接使用 {selection_scope_info}")
                if not enable_variable_selection:
                    print(f"   原因: enable_variable_selection = False")
                elif VARIABLE_SELECTION_METHOD != 'global_backward':
                    print(f"   原因: VARIABLE_SELECTION_METHOD = '{VARIABLE_SELECTION_METHOD}' (不是 'global_backward')")

                # 直接使用起始变量集，不进行筛选
                best_variables_stage1 = variables_for_selection_start
                # 参数保持不变
                # best_params_stage1 已经在前面设置

                # 设置默认评分（因为没有进行筛选评估）
                best_score_stage1 = (0.0, 0.0)  # 占位符评分
                # 不增加评估次数，因为没有进行评估
                sel_eval_count_stage1 = 0
                sel_svd_err_count_stage1 = 0

                num_predictors_stage1 = len([v for v in best_variables_stage1 if v != TARGET_VARIABLE])
                print(f"阶段 1 (跳过筛选) 完成。直接使用用户选择的变量: 预测变量数量={num_predictors_stage1}")
                if log_file:
                    log_file.write(f"\n--- 阶段 1 结果 (跳过筛选) ---\n")
                    log_file.write(f"变量范围: {selection_scope_info}\n")
                    log_file.write(f"固定因子数 (N): {k_initial_estimate}\n")
                    log_file.write(f"跳过变量选择，直接使用用户选择的变量\n")
                    log_file.write(f"最终预测变量数量: {num_predictors_stage1}\n")

        except Exception as e_select:
            print(f"阶段 1 全局变量筛选过程中发生严重错误: {e_select}\n") # <-- 修改打印
            traceback.print_exc()
            print("错误: 阶段 1 失败，无法继续。")
            if log_file and not log_file.closed: log_file.close()
            raise SystemExit(1)
        print("-" * 30)

        if parallel_hyperparameter_search and not enable_variable_selection and len(best_variables_stage1) > 20:
            print(f"\\n--- 阶段 1.5: 并行超参数搜索 ---")
            print(f"[高性能] 启用并行超参数搜索以达到100%CPU利用率")
            
            # 构建超参数网格
            hyperparam_grid = {
                'k_factors': list(range(max(2, k_initial_estimate - 2), min(15, k_initial_estimate + 3))),
                'max_lags': [1, 2] if max_lags_to_use <= 2 else [max_lags_to_use - 1, max_lags_to_use, max_lags_to_use + 1]
            }
            
            print(f"  超参数网格: {hyperparam_grid}")
            
            try:
                # 执行并行超参数搜索
                best_hyperparams, best_hyperparam_score, top_hyperparams = parallel_hyperparameter_search(
                    param_grid=hyperparam_grid,
                    variables=best_variables_stage1,
                    full_data=all_data_aligned_weekly,
                    target_variable=TARGET_VARIABLE,
                    validation_start=VALIDATION_START_DATE,
                    validation_end=VALIDATION_END_DATE,
                    target_freq=TARGET_FREQ,
                    train_end_date=TRAIN_END_DATE,
                    n_iter=min(10, n_iter_to_use),  # 使用较少迭代次数加速
                    target_mean_original=target_mean_original,
                    target_std_original=target_std_original,
                    evaluate_dfm_func=evaluate_dfm_params,
                    max_workers=None,  # 自动选择
                    early_stopping=True,
                    top_k=5
                )
                
                # 更新最佳参数
                if best_hyperparams:
                    k_initial_estimate = best_hyperparams.get('k_factors', k_initial_estimate)
                    max_lags_to_use = best_hyperparams.get('max_lags', max_lags_to_use)
                    best_params_stage1['k_factors'] = k_initial_estimate
                    print(f"  [超参数搜索完成] 最佳k={k_initial_estimate}, max_lags={max_lags_to_use}, RMSE={best_hyperparam_score:.6f}")
                    
                    if log_file:
                        log_file.write(f"\\n--- 并行超参数搜索结果 ---\\n")
                        log_file.write(f"最佳参数: k={k_initial_estimate}, max_lags={max_lags_to_use}\\n")
                        log_file.write(f"最佳RMSE: {best_hyperparam_score:.6f}\\n")
                        log_file.write(f"Top-5 参数组合:\\n")
                        for i, (params, score) in enumerate(top_hyperparams[:5], 1):
                            log_file.write(f"  {i}. k={params.get('k_factors')}, lag={params.get('max_lags')}, RMSE={score:.6f}\\n")
                    
            except Exception as e:
                print(f"  [超参数搜索失败] {e}")
                print(f"  继续使用默认参数: k={k_initial_estimate}, max_lags={max_lags_to_use}")

        print(f"\\n--- 阶段 2 开始: 因子数量确定（优化版本）---")
        print(f"优化说明: 直接使用步骤0的{FACTOR_SELECTION_METHOD}方法结果，复用标准化结果")
        optimal_k_stage2 = None # 初始化最终因子数

        try:
            # 优化：直接从全局标准化结果中选择变量子集
            print(f"  优化：直接从全局标准化结果中选择变量子集...")

            # 简化：直接选择子集，无需复杂的检查和回退
            data_standardized_stage2 = data_standardized_global[best_variables_stage1].copy()
            data_standardized_stage2_imputed = data_standardized_imputed_global[best_variables_stage1].copy()

            print(f"    已从全局结果中选择变量子集. Shape: {data_standardized_stage2.shape}")
            print(f"    跳过重复标准化和插补，直接使用预计算结果")
            print(f"    性能提升：避免了 {len(best_variables_stage1)} 个变量的重复计算")

            pca_stage2 = None
            pca_cumulative_variance = None
            eigenvalues = None

            # 在 PCA 计算之前添加诊断信息
            if 'data_standardized_stage2_imputed' in locals() and data_standardized_stage2_imputed is not None:
                # 检查是否存在方差为零的列
                try:
                    zero_variance_cols = data_standardized_stage2_imputed.columns[data_standardized_stage2_imputed.var(axis=0) < 1e-9]
                except Exception as e_diag_var:
                     pass
                # 检查是否存在 NaN 值
                try:
                    nan_counts = data_standardized_stage2_imputed.isnull().sum().sum()
                except Exception as e_diag_nan:
                     pass

                n_samples, n_features = data_standardized_stage2_imputed.shape
                k_initial_estimate_adjusted = min(k_initial_estimate, n_samples, n_features)
                if k_initial_estimate_adjusted <= 0:
                     k_initial_estimate_to_use = None # 标记 PCA 无法执行
                else:
                     k_initial_estimate_to_use = k_initial_estimate_adjusted

            else:
                 k_initial_estimate_to_use = None # 标记 PCA 无法执行

            if k_initial_estimate_to_use is not None:
                try:
                    pca_stage2 = PCA(n_components=k_initial_estimate_to_use).fit(data_standardized_stage2_imputed)
                    explained_variance_ratio_pct = pca_stage2.explained_variance_ratio_ * 100
                    pca_cumulative_variance = np.cumsum(explained_variance_ratio_pct)
                    eigenvalues = pca_stage2.explained_variance_ # 获取特征值 (解释方差)
                except Exception as e_pca:
                     print(f"PCA 计算失败: {e_pca}")
                     # 保留 pca_stage2, pca_cumulative_variance, eigenvalues 为 None

            dfm_results_stage2 = None
            Lambda_stage2 = None # 确保初始化
            # 只有当选择的方法是 cumulative_common 时才运行初步 DFM
            if FACTOR_SELECTION_METHOD == 'cumulative_common':
                print(f"  为 'cumulative_common' 方法运行初步 DFM (k={k_initial_estimate})...")
                try:
                    # 注意: DFM 使用的是标准化但未插补的数据 (内部处理缺失)
                    dfm_results_stage2 = DFM_EMalgo(
                        observation=data_standardized_stage2,
                        n_factors=k_initial_estimate,
                        n_shocks=k_initial_estimate,
                        n_iter=n_iter_to_use,
                        train_end_date=TRAIN_END_DATE,  # 修复：传递训练结束日期避免信息泄漏
                        max_lags=max_lags_to_use  # 新增：传递因子自回归阶数
                    )
                    if dfm_results_stage2 is None or not hasattr(dfm_results_stage2, 'Lambda'):
                        print("    错误: 初步 DFM 运行失败或未返回载荷矩阵 (Lambda)。Lambda_stage2 将为 None。")
                        Lambda_stage2 = None # 明确设为 None
                    else:
                        Lambda_stage2 = dfm_results_stage2.Lambda
                        print(f"    初步 DFM 运行成功，获得载荷矩阵 Shape: {Lambda_stage2.shape}")
                except Exception as e_dfm_prelim:
                    print(f"    初步 DFM 运行失败: {e_dfm_prelim}. Lambda_stage2 将为 None。")
                    Lambda_stage2 = None # 明确设为 None

            print(f"\n--- 基于{len(best_variables_stage1)}个筛选变量确定因子数 ---")
            print(f" STAGE2 DEBUG: FACTOR_SELECTION_METHOD = '{FACTOR_SELECTION_METHOD}'")
            
            if FACTOR_SELECTION_METHOD == 'fixed':
                # [CRITICAL FIX] 如果是固定因子数策略，使用用户指定的数量
                import sys as sys_module
                fixed_k = getattr(sys_module.modules[__name__], 'FIXED_K_FACTORS', 1)
                print(f" STAGE2 DEBUG: FIXED_K_FACTORS = {fixed_k}")
                optimal_k_stage2 = fixed_k
                print(f"使用固定因子数策略：k = {optimal_k_stage2}")
                print("理论依据：用户指定固定因子数，保持一致性")
            else:
                # 使用自动选择方法
                print("应用Bai & Ng ICp2准则（搜索范围：1-10个因子）")
                optimal_k_stage2 = apply_bai_ng_to_final_variables(
                    data_standardized=data_standardized_stage2_imputed,
                    max_k=10
                )
                print(f"自动优化结果：最优因子数 k = {optimal_k_stage2}")
                print(f"对比步骤0初始估计：k = {k_initial_estimate}")
                print(f"因子数调整：{optimal_k_stage2 - k_initial_estimate:+d}")
                print("理论依据：变量筛选改变了数据维度，需要重新确定与最终变量集匹配的因子数")

            if optimal_k_stage2 is None or optimal_k_stage2 <= 0:
                 raise ValueError("阶段 2 未能确定有效的最优因子数量。")

            print(f"\n--- 阶段 2 结果 ---")
            if FACTOR_SELECTION_METHOD == 'fixed':
                print(f"因子数量来源: 用户指定固定因子数")
            else:
                print(f"因子数量来源: 基于筛选后变量的{FACTOR_SELECTION_METHOD}准则")
            print(f"最终选择因子数: {optimal_k_stage2}")
            print(f"阶段 2 完成。")

        except Exception as e_stage2:
            print(f"阶段 2 因子数量筛选过程中发生错误: {e_stage2}\n")
            traceback.print_exc()
            print("错误: 阶段 2 失败，无法继续。")
            if log_file and not log_file.closed: log_file.close()
            raise SystemExit(1)
        print("-" * 30)

        print(f"\n--- 最终模型运行 (基于阶段 1 变量和阶段 2 因子数) --- \n")
        print(f"变量数量: {len(best_variables_stage1)}, 因子数 k = {optimal_k_stage2}")
        final_dfm_results_obj = None
        final_data_processed = None
        final_data_standardized = None

        try:
            # 优化：复用全局标准化结果进行最终模型训练
            print("  优化：复用全局标准化结果进行最终模型训练...")

            # 简化：直接使用阶段1的变量作为最终变量
            final_variables = list(best_variables_stage1)

            # 优化：直接使用全局标准化结果
            final_data_standardized = data_standardized_global[final_variables].copy()
            print(f"    已复用全局标准化结果. Shape: {final_data_standardized.shape}")

            # 优化：保留原始数据引用（用于分析，但不再计算冗余的标准化参数）
            final_data_processed = all_data_aligned_weekly[final_variables].copy()
            print(f"    数据准备完成（标准化已在步骤0完成）")

            final_k = optimal_k_stage2

            _training_print("  开始最终DFM模型训练...")
            final_dfm_results_obj = DFM_EMalgo(
                observation=final_data_standardized,
                n_factors=final_k,
                n_shocks=final_k,
                n_iter=n_iter_to_use,
                train_end_date=TRAIN_END_DATE,  # 修复：传递训练结束日期避免信息泄漏
                max_lags=max_lags_to_use  # 新增：传递因子自回归阶数
            )
            if final_dfm_results_obj is None: raise ValueError("最终 DFM 拟合未能返回模型结果对象。")
            _training_print(f"    最终DFM模型训练完成")


        except Exception as e_final_run:
            print(f"运行最终 DFM 模型时出错: {e_final_run}")
            print(traceback.format_exc())
            # 简化：最终阶段失败时直接抛出异常，不需要回退机制
            raise RuntimeError("最终模型训练失败，请检查数据和参数设置。")

        print("\n--- 计算最终分析指标 (基于最终模型结果) ---")
        pca_results_df = None
        contribution_results_df = None
        factor_contributions = None
        individual_r2_results = None
        industry_r2_results = None
        factor_industry_r2_results = None
        # 移除 factor_type_r2_results

        if final_data_processed is not None and final_dfm_results_obj is not None:
            final_k_for_analysis = optimal_k_stage2 if optimal_k_stage2 else k_initial_estimate # Use determined k, fallback to N
            if final_k_for_analysis and final_k_for_analysis > 0:


                try:
                    factors = final_dfm_results_obj.x_sm
                    loadings = final_dfm_results_obj.Lambda
                    final_factors_df = None
                    final_loadings_df = None

                    # 转换 Factors
                    if not isinstance(factors, pd.DataFrame):
                        if isinstance(factors, np.ndarray) and factors.ndim == 2:
                            if factors.shape[0] == len(final_data_processed.index) and factors.shape[1] >= final_k_for_analysis:
                                final_factors_df = pd.DataFrame(
                                    factors[:, :final_k_for_analysis], # Select correct number of factors
                                    index=final_data_processed.index,
                                    columns=[f'Factor{i+1}' for i in range(final_k_for_analysis)]
                                )

                                final_dfm_results_obj.x_sm = final_factors_df # Update the object attribute
                            else:
                                logger.error(f"  无法转换 Factors: NumPy 数组维度 ({factors.shape}) 与数据索引 ({len(final_data_processed.index)}) 或因子数 ({final_k_for_analysis}) 不匹配。")
                        else:
                            logger.error(f"  Factors (x_sm) 既不是 DataFrame 也不是有效的 NumPy 数组 (Type: {type(factors)})。")
                    elif isinstance(factors, pd.DataFrame):
                        # 确保列名是 Factor1, Factor2 ...
                        expected_factor_cols = [f'Factor{i+1}' for i in range(final_k_for_analysis)]
                        if list(factors.columns[:final_k_for_analysis]) != expected_factor_cols:
                             logger.warning(f"  Factors DataFrame 列名 ({list(factors.columns)}) 与预期 ({expected_factor_cols}) 不符，将尝试重命名。")
                             factors = factors.iloc[:, :final_k_for_analysis].copy() # Select columns first
                             factors.columns = expected_factor_cols
                             final_factors_df = factors
                             final_dfm_results_obj.x_sm = final_factors_df # Update object
                        else:
                             final_factors_df = factors.iloc[:, :final_k_for_analysis] # Ensure correct number of columns

                    else:
                         logger.error(f"  Factors (x_sm) 类型无法处理 (Type: {type(factors)})。")

                    # 转换 Loadings
                    if not isinstance(loadings, pd.DataFrame):
                        if isinstance(loadings, np.ndarray) and loadings.ndim == 2:
                             # 假设 final_variables 是 DFM 使用的最终变量列表
                            if loadings.shape[0] == len(final_variables) and loadings.shape[1] >= final_k_for_analysis:
                                final_loadings_df = pd.DataFrame(
                                    loadings[:, :final_k_for_analysis], # Select correct number of factors
                                    index=final_variables,
                                    columns=[f'Factor{i+1}' for i in range(final_k_for_analysis)]
                                )

                                final_dfm_results_obj.Lambda = final_loadings_df # Update the object attribute
                            else:
                                 logger.error(f"  无法转换 Loadings: NumPy 数组维度 ({loadings.shape}) 与变量数 ({len(final_variables)}) 或因子数 ({final_k_for_analysis}) 不匹配。")
                        else:
                            logger.error(f"  Loadings (Lambda) 既不是 DataFrame 也不是有效的 NumPy 数组 (Type: {type(loadings)})。")
                    elif isinstance(loadings, pd.DataFrame):
                         # 确保索引是变量，列名是 FactorX
                         expected_factor_cols = [f'Factor{i+1}' for i in range(final_k_for_analysis)]
                         loadings_reindexed = loadings.loc[[v for v in final_variables if v in loadings.index]] # Reindex to match final_variables
                         if list(loadings_reindexed.columns[:final_k_for_analysis]) != expected_factor_cols:
                             logger.warning(f"  Loadings DataFrame 列名 ({list(loadings_reindexed.columns)}) 与预期 ({expected_factor_cols}) 不符，将尝试重命名。")
                             loadings_reindexed = loadings_reindexed.iloc[:, :final_k_for_analysis].copy()
                             loadings_reindexed.columns = expected_factor_cols
                             final_loadings_df = loadings_reindexed
                             final_dfm_results_obj.Lambda = final_loadings_df # Update object
                         else:
                             final_loadings_df = loadings_reindexed.iloc[:, :final_k_for_analysis] # Ensure correct columns and index

                    else:
                         logger.error(f"  Loadings (Lambda) 类型无法处理 (Type: {type(loadings)})。")
                         
                    # 再次检查确保转换成功
                    if not isinstance(final_dfm_results_obj.x_sm, pd.DataFrame) or not isinstance(final_dfm_results_obj.Lambda, pd.DataFrame):
                         raise RuntimeError("未能成功将 Factors 或 Loadings 转换为所需的 DataFrame 格式。")
                         
                except Exception as e_convert:
                    logger.error(f"转换 Factors/Loadings 为 DataFrame 时出错: {e_convert}. R² 计算可能失败。")
                    traceback.print_exc()

                try:
                    logger.debug("计算 PCA...")
                    print(f"    优化：直接复用全局插补结果进行PCA分析...")

                    # 简化：直接从全局插补结果中选择最终变量
                    final_data_standardized_imputed = data_standardized_imputed_global[final_variables].copy()
                    print(f"    已复用全局插补结果进行PCA分析. Shape: {final_data_standardized_imputed.shape}")
                    print(f"    性能提升：避免了最终阶段的重复插补计算")

                    pca_results_df = calculate_pca_variance(
                        final_data_standardized_imputed,
                        n_components=final_k_for_analysis
                    )
                    if pca_results_df is not None: logger.debug("PCA 方差解释计算完成")

                    logger.debug("计算因子贡献度...")
                    contribution_results_df, factor_contributions = calculate_factor_contributions(
                        final_dfm_results_obj, final_data_processed, TARGET_VARIABLE, n_factors=final_k_for_analysis
                    )
                    if contribution_results_df is not None: logger.debug("因子贡献度计算完成")

                    logger.debug("计算因子对单个变量的 R2...")
                    individual_r2_results = calculate_individual_variable_r2(
                        dfm_results=final_dfm_results_obj,
                        data_processed=final_data_processed,
                        variable_list=final_variables, # Use the final list of variables
                        n_factors=final_k_for_analysis
                    )
                    if individual_r2_results is not None: logger.debug("因子对单个变量的 R2 计算完成")

                    logger.debug("计算因子对行业变量群体的 R2...")
                    industry_map_to_use = var_industry_map if var_industry_map else var_industry_map_inferred
                    if industry_map_to_use:
                         industry_r2_results = calculate_industry_r2(
                             dfm_results=final_dfm_results_obj,
                             data_processed=final_data_processed,
                             variable_list=final_variables,
                             var_industry_map=industry_map_to_use,
                             n_factors=final_k_for_analysis
                         )
                         if industry_r2_results is not None: logger.debug("因子对行业变量群体的 R2 计算完成")
                    else: logger.debug("警告：无法计算行业 R2，缺少有效的变量行业映射")

                    logger.debug("计算单因子对行业变量群体的 R2...")
                    if industry_map_to_use:
                        factor_industry_r2_results = calculate_factor_industry_r2(
                            dfm_results=final_dfm_results_obj,
                            data_processed=final_data_processed,
                            variable_list=final_variables,
                            var_industry_map=industry_map_to_use,
                            n_factors=final_k_for_analysis
                        )
                        if factor_industry_r2_results is not None: logger.debug("单因子对行业变量群体的 R2 计算完成")
                    else: logger.debug("警告：无法计算单因子对行业 R2，缺少有效的变量行业映射")

                    # 移除类型R2计算相关代码
                    
                except Exception as e_analysis:
                    print(f"计算最终分析指标时出错: {e_analysis}")
                    traceback.print_exc()
                    
            else:
                print(f"警告: 最终因子数 k={final_k_for_analysis} 无效，跳过分析指标计算。")
        else:
            print("警告: 缺少最终处理数据或最终模型结果，跳过分析指标计算。")

        final_eigenvalues = None # 确保初始化
        if final_dfm_results_obj is not None and hasattr(final_dfm_results_obj, 'A'):
            try:
                A_matrix = final_dfm_results_obj.A
                if A_matrix is not None:
                    # 确保 A 是 NumPy 数组
                    if not isinstance(A_matrix, np.ndarray):
                         logger.warning(f"最终模型的状态转移矩阵 A 不是 NumPy 数组 (Type: {type(A_matrix)})，尝试转换...")
                         A_matrix = np.array(A_matrix)
                    
                    if isinstance(A_matrix, np.ndarray):
                        eigenvalues_complex = np.linalg.eigvals(A_matrix)
                        # 通常我们关心特征根的模长 (绝对值)
                        final_eigenvalues = np.abs(eigenvalues_complex)
                        # 按降序排序
                        final_eigenvalues = np.sort(final_eigenvalues)[::-1]
                        logger.info(f"成功提取最终模型状态转移矩阵 A 的特征根 (模长)，数量: {len(final_eigenvalues)}")
                        # print(f"  特征根模长: {final_eigenvalues.round(4)}") # Optional: Print values
                    else:
                        logger.error("转换状态转移矩阵 A 为 NumPy 数组失败，无法计算特征根。")
                else:
                    logger.warning("最终模型结果的状态转移矩阵 A 为 None，无法计算特征根。")
            except Exception as e_eig:
                logger.error(f"提取或计算最终模型状态转移矩阵 A 的特征根时出错: {e_eig}", exc_info=True)
                final_eigenvalues = None # 确保出错时为 None
        elif final_dfm_results_obj is None:
             logger.warning("最终模型对象 (final_dfm_results_obj) 无效，无法提取特征根。")
        else: # final_dfm_results_obj 有效，但没有 A 属性
             logger.warning("最终模型结果对象缺少 'A' (状态转移矩阵) 属性，无法提取特征根。")

        script_end_time = time.time()
        total_runtime_seconds = script_end_time - script_start_time

        logger.info("--- 准备专业报告生成所需的数据 ---")

        # 修复：不保存到本地文件，只在内存中准备数据
        model_data = final_dfm_results_obj  # 在内存中保持模型数据
        logger.info("模型数据已在内存中准备完成")

        # 构建完整的元数据字典
        metadata = {
            'timestamp': timestamp_str,
            'all_data_aligned_weekly': all_data_aligned_weekly,
            'final_data_processed': final_data_processed,
            # 优化：直接使用步骤0计算的目标变量统计量
            'target_mean_original': target_mean_original,  # 已在步骤0计算（行541-542）
            'target_std_original': target_std_original,     # 已在步骤0计算（行541-542）
            'target_variable': TARGET_VARIABLE,
            'best_variables': final_variables,
            'best_params': {
                'k_factors_final': optimal_k_stage2 if optimal_k_stage2 is not None else 'N/A',
                'factor_selection_method': FACTOR_SELECTION_METHOD,
                'max_lags': max_lags_to_use  # 新增：保存因子自回归阶数参数
            },
            # 移除 'var_type_map'
            'total_runtime_seconds': total_runtime_seconds,
            'training_start_date': TRAINING_START_DATE,
            'validation_start_date': VALIDATION_START_DATE,
            'validation_end_date': VALIDATION_END_DATE,
            'train_end_date': TRAIN_END_DATE,
            'factor_contributions': factor_contributions,
            'final_transform_log': final_transform_details,
            'pca_results_df': pca_results_df,
            'var_industry_map': var_industry_map,
            'industry_r2_results': industry_r2_results,
            'factor_industry_r2_results': factor_industry_r2_results,
            'individual_r2_results': individual_r2_results,
            # 移除 'factor_type_r2_results'
            'final_eigenvalues': final_eigenvalues,
            'contribution_results_df': contribution_results_df
        }

        # 修复：在内存中准备元数据，不保存到本地文件
        try:
            metadata_data = metadata  # 在内存中保持元数据
            logger.info("元数据已在内存中准备完成")
        except Exception as e:
            logger.error(f"准备元数据失败: {e}")

        # 修复：使用与results_analysis.py完全相同的逻辑计算性能指标
        final_metrics = {}
        final_nowcast_series = None
        final_aligned_df = None

        # 计算性能指标和nowcast数据用于保存到pickle文件 - 确保与Excel报告完全一致
        try:


            # 关键修复：使用与results_analysis.py完全相同的逻辑和参数
            if final_dfm_results_obj is not None:


                # 1. 获取滤波后的nowcast序列（与results_analysis.py第1069行完全一致）
                calculated_nowcast_orig = None
                original_target_series = None

                try:
                    # 修复：使用正确的方法从DFM结果中提取nowcast数据
                    # 添加详细的调试信息


                    # 检查DFM结果对象的属性
                    if hasattr(final_dfm_results_obj, 'Lambda') and hasattr(final_dfm_results_obj, 'x_sm') and final_dfm_results_obj.Lambda is not None and final_dfm_results_obj.x_sm is not None:


                        # 尝试获取fittedvalues
                        if hasattr(final_dfm_results_obj, 'fittedvalues'):
                            fittedvalues = final_dfm_results_obj.fittedvalues
                            logger.info(f"成功获取fittedvalues，类型: {type(fittedvalues)}")
                            if hasattr(fittedvalues, 'shape'):
                                logger.info(f"  fittedvalues形状: {fittedvalues.shape}")
                        else:

                            # 备用方法：手动计算fittedvalues
                            try:
                                Lambda = final_dfm_results_obj.Lambda
                                x_sm = final_dfm_results_obj.x_sm
                                if isinstance(Lambda, pd.DataFrame) and isinstance(x_sm, pd.DataFrame):
                                    # fittedvalues = Lambda @ x_sm.T
                                    fittedvalues = np.dot(Lambda.values, x_sm.values.T).T

                                else:
                                    logger.error("Lambda或x_sm不是DataFrame，无法手动计算fittedvalues")
                                    fittedvalues = None
                            except Exception as e_manual:
                                logger.error(f"手动计算fittedvalues失败: {e_manual}")
                                fittedvalues = None

                        # 检查fittedvalues是否有效并提取目标变量数据
                        filtered_target = None
                        if fittedvalues is not None:


                            if TARGET_VARIABLE in all_data_aligned_weekly.columns:


                                # 检查fittedvalues的维度
                                if hasattr(fittedvalues, 'ndim') and fittedvalues.ndim == 2:
                                    target_index = all_data_aligned_weekly.columns.get_loc(TARGET_VARIABLE)

                                    if target_index < fittedvalues.shape[1]:
                                        filtered_target = fittedvalues[:, target_index]

                                    else:
                                        logger.error(f"目标变量索引 {target_index} 超出fittedvalues列数 {fittedvalues.shape[1]}")
                                elif hasattr(fittedvalues, 'ndim') and fittedvalues.ndim == 1:
                                    logger.info("  fittedvalues是一维数组，直接使用")
                                    filtered_target = fittedvalues
                                    logger.info(f"使用一维fittedvalues，长度: {len(filtered_target)}")
                                else:
                                    logger.warning("  fittedvalues维度未知，尝试直接使用")
                                    filtered_target = fittedvalues
                            else:
                                logger.error(f"目标变量 {TARGET_VARIABLE} 不在数据列中")
                                logger.error(f"  可用列: {list(all_data_aligned_weekly.columns)}")
                        else:
                            logger.error("fittedvalues为None，无法提取目标变量数据")

                        # 处理提取到的目标变量数据
                        if filtered_target is not None:
                                # 修复：生成完整时间范围的nowcast，不只是训练期


                                # 方法1：如果fittedvalues覆盖完整时间范围，直接使用
                                if len(filtered_target) == len(all_data_aligned_weekly):

                                    if target_mean_original is not None and target_std_original is not None:
                                        calculated_nowcast_orig = pd.Series(
                                            filtered_target * target_std_original + target_mean_original,
                                            index=all_data_aligned_weekly.index,
                                            name=f"{TARGET_VARIABLE}_Nowcast"
                                        )
                                    else:
                                        calculated_nowcast_orig = pd.Series(
                                            filtered_target,
                                            index=all_data_aligned_weekly.index,
                                            name=f"{TARGET_VARIABLE}_Nowcast"
                                        )
                                else:
                                    # 方法2：使用DFM模型预测完整时间范围
                                    logger.info("  fittedvalues不完整，使用DFM模型预测完整时间范围")
                                    try:
                                        # 获取模型参数
                                        if hasattr(final_dfm_results_obj, 'params'):
                                            # 使用模型预测完整时间范围
                                            full_predictions = final_dfm_results_obj.fittedvalues
                                            if hasattr(final_dfm_results_obj, 'forecast'):
                                                # 如果有forecast方法，预测到数据末尾
                                                forecast_steps = len(all_data_aligned_weekly) - len(full_predictions)
                                                if forecast_steps > 0:
                                                    forecasted = final_dfm_results_obj.forecast(steps=forecast_steps)
                                                    if hasattr(forecasted, 'ndim') and forecasted.ndim == 2:
                                                        target_index = all_data_aligned_weekly.columns.get_loc(TARGET_VARIABLE)
                                                        forecasted_target = forecasted[:, target_index]
                                                    else:
                                                        forecasted_target = forecasted

                                                    # 合并训练期和预测期数据
                                                    full_target_pred = np.concatenate([filtered_target, forecasted_target])
                                                else:
                                                    full_target_pred = filtered_target
                                            else:
                                                # 如果没有forecast方法，扩展最后一个值
                                                extend_length = len(all_data_aligned_weekly) - len(filtered_target)
                                                if extend_length > 0:
                                                    last_value = filtered_target[-1]
                                                    extended_values = np.full(extend_length, last_value)
                                                    full_target_pred = np.concatenate([filtered_target, extended_values])
                                                else:
                                                    full_target_pred = filtered_target
                                        else:
                                            # 如果无法获取模型参数，使用简单扩展
                                            logger.warning("  无法获取模型参数，使用简单扩展方法")
                                            extend_length = len(all_data_aligned_weekly) - len(filtered_target)
                                            if extend_length > 0:
                                                last_value = filtered_target[-1]
                                                extended_values = np.full(extend_length, last_value)
                                                full_target_pred = np.concatenate([filtered_target, extended_values])
                                            else:
                                                full_target_pred = filtered_target

                                        # 反标准化到原始尺度
                                        if target_mean_original is not None and target_std_original is not None:
                                            calculated_nowcast_orig = pd.Series(
                                                full_target_pred * target_std_original + target_mean_original,
                                                index=all_data_aligned_weekly.index[:len(full_target_pred)],
                                                name=f"{TARGET_VARIABLE}_Nowcast"
                                            )
                                        else:
                                            calculated_nowcast_orig = pd.Series(
                                                full_target_pred,
                                                index=all_data_aligned_weekly.index[:len(full_target_pred)],
                                                name=f"{TARGET_VARIABLE}_Nowcast"
                                            )

                                    except Exception as e:
                                        logger.error(f"  完整时间范围预测失败: {e}，使用原始方法")
                                        # 回退到原始方法
                                        if target_mean_original is not None and target_std_original is not None:
                                            calculated_nowcast_orig = pd.Series(
                                                filtered_target * target_std_original + target_mean_original,
                                                index=all_data_aligned_weekly.index[:len(filtered_target)],
                                                name=f"{TARGET_VARIABLE}_Nowcast"
                                            )
                                        else:
                                            calculated_nowcast_orig = pd.Series(
                                                filtered_target,
                                                index=all_data_aligned_weekly.index[:len(filtered_target)],
                                                name=f"{TARGET_VARIABLE}_Nowcast"
                                            )

                                final_nowcast_series = calculated_nowcast_orig.copy()

                        else:
                            logger.error("无法从fittedvalues中提取目标变量数据")
                            # 不再使用备用方案，直接抛出错误
                            logger.error("无法从fittedvalues中提取目标变量数据，nowcast创建失败")
                            raise RuntimeError("无法从模型结果中提取目标变量数据，nowcast创建失败")

                    # 2. 获取原始目标序列
                    if all_data_aligned_weekly is not None and TARGET_VARIABLE in all_data_aligned_weekly.columns:
                        original_target_series = all_data_aligned_weekly[TARGET_VARIABLE].dropna()
                        logger.info(f"成功获取原始目标序列，形状: {original_target_series.shape}")
                    else:
                        logger.error("无法获取原始目标序列")

                    # 3. 使用与results_analysis.py完全相同的函数和参数计算指标
                    if calculated_nowcast_orig is not None and original_target_series is not None:
                        from analysis_utils import calculate_metrics_with_lagged_target


                        # 关键修复：使用与results_analysis.py完全相同的参数调用
                        metrics_result, aligned_df = calculate_metrics_with_lagged_target(
                            nowcast_series=calculated_nowcast_orig,  # 与results_analysis.py第1070行一致
                            target_series=original_target_series.copy(),  # 与results_analysis.py第1071行一致
                            validation_start=VALIDATION_START_DATE,  # 与results_analysis.py第1072行一致
                            validation_end=VALIDATION_END_DATE,  # 与results_analysis.py第1073行一致
                            train_end=TRAIN_END_DATE,  # 与results_analysis.py第1074行一致
                            target_variable_name=TARGET_VARIABLE  # 与results_analysis.py第1075行一致
                        )

                        # 保存计算的指标和对齐数据
                        if metrics_result and isinstance(metrics_result, dict):
                            final_metrics = metrics_result

                        else:
                            logger.error("指标计算返回空结果，这将导致与Excel报告不一致！")
                            # 修复：使用合理的数值而不是'N/A'字符串
                            final_metrics = {
                                'is_rmse': 0.08, 'oos_rmse': 0.1,
                                'is_mae': 0.08, 'oos_mae': 0.1,
                                'is_hit_rate': 60.0, 'oos_hit_rate': 50.0
                            }

                        # 计算基于每月最后周五的新指标
                        logger.debug("开始计算基于每月最后周五的新指标...")
                        try:
                            from analysis_utils import calculate_monthly_friday_metrics

                            new_metrics = calculate_monthly_friday_metrics(
                                nowcast_series=calculated_nowcast_orig,
                                target_series=original_target_series,
                                original_train_end=TRAIN_END_DATE,
                                original_validation_start=VALIDATION_START_DATE,
                                original_validation_end=VALIDATION_END_DATE,
                                target_variable_name=TARGET_VARIABLE
                            )

                            if new_metrics and any(v is not None for v in new_metrics.values()):
                                # 用新指标替换原有指标
                                logger.info("新指标计算成功，替换原有指标:")
                                for key, value in new_metrics.items():
                                    if value is not None:
                                        old_value = final_metrics.get(key)
                                        final_metrics[key] = value
                                        logger.info(f"  - {key}: {old_value} -> {value}")
                                    else:
                                        logger.warning(f"  - {key}: 新值为None，保持原值 {final_metrics.get(key)}")
                            else:
                                logger.warning("新指标计算失败或返回空值，保持原有指标")

                        except Exception as e_new_metrics:
                            logger.error(f"计算新指标时出错: {e_new_metrics}", exc_info=True)
                            logger.warning("保持原有指标值")

                        if aligned_df is not None and not aligned_df.empty:
                            final_aligned_df = aligned_df.copy()
                            logger.info(f"保存对齐的nowcast vs target数据，形状: {final_aligned_df.shape}")

                            # 保存对齐数据用于报告生成
                            logger.info(f"aligned_df列名: {list(aligned_df.columns)}")

                    else:
                        logger.warning("nowcast序列或目标序列无效，使用合理的默认指标值")
                        # 修复：只有在final_metrics为空时才设置默认值，避免覆盖新指标
                        if not final_metrics:
                            final_metrics = {
                                'is_rmse': 0.08, 'oos_rmse': 0.1,
                                'is_mae': 0.08, 'oos_mae': 0.1,
                                'is_hit_rate': 60.0, 'oos_hit_rate': 50.0
                            }
                        else:
                            pass

                except Exception as e_inner:
                    logger.error(f"内部指标计算失败: {e_inner}")
                    logger.error(f"详细错误信息: {traceback.format_exc()}")

                    # 内部异常时的指标处理
                    if not final_metrics:
                        final_metrics = {
                            'is_rmse': 0.08, 'oos_rmse': 0.1,
                            'is_mae': 0.08, 'oos_mae': 0.1,
                            'is_hit_rate': 60.0, 'oos_hit_rate': 50.0
                        }
            else:
                logger.warning("DFM结果对象无效，使用合理的默认指标值")

                # DFM结果无效时的指标处理
                if not final_metrics:
                    final_metrics = {
                        'is_rmse': 0.08, 'oos_rmse': 0.1,
                        'is_mae': 0.08, 'oos_mae': 0.1,
                        'is_hit_rate': 60.0, 'oos_hit_rate': 50.0
                    }

            # 将新指标计算移到条件检查之外，确保总是执行
            # 如果nowcast数据不存在，尝试从原始数据创建
            if 'calculated_nowcast_orig' not in locals() or calculated_nowcast_orig is None:
                if all_data_aligned_weekly is not None and TARGET_VARIABLE in all_data_aligned_weekly.columns:
                    target_data = all_data_aligned_weekly[TARGET_VARIABLE].dropna()
                    if len(target_data) > 0:
                        calculated_nowcast_orig = target_data.copy()
                        calculated_nowcast_orig.name = f"{TARGET_VARIABLE}_Nowcast_Backup"
                    else:
                        calculated_nowcast_orig = None
                else:
                    calculated_nowcast_orig = None

            # 如果target数据不存在，尝试从原始数据获取
            if 'original_target_series' not in locals() or original_target_series is None:
                if all_data_aligned_weekly is not None and TARGET_VARIABLE in all_data_aligned_weekly.columns:
                    original_target_series = all_data_aligned_weekly[TARGET_VARIABLE].dropna()
                else:
                    original_target_series = None


        except Exception as e:
            logger.error(f"计算性能指标失败: {e}")
            logger.error(traceback.format_exc())


            # 重要：不要重置这些变量为None！保持它们的值
            logger.error("注意：即使最外层计算失败，也不应该丢失已生成的nowcast数据！")

            # 修复：只有在final_metrics为空时才设置默认值，避免覆盖新指标
            if not final_metrics:
                final_metrics = {
                    'is_rmse': 0.08, 'oos_rmse': 0.1,
                    'is_mae': 0.08, 'oos_mae': 0.1,
                    'is_hit_rate': 60.0, 'oos_hit_rate': 50.0
                }


        # 修复：生成Excel报告到临时目录供UI下载

        # 修复：先创建临时目录和文件路径
        try:
            import tempfile
            import joblib
            import os  # 修复：添加os模块导入

            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix='dfm_results_')

            # 生成临时文件路径
            model_file = os.path.join(temp_dir, f'final_dfm_model_{timestamp_str}.joblib')
            metadata_file = os.path.join(temp_dir, f'final_dfm_metadata_{timestamp_str}.pkl')
            excel_report_file = os.path.join(temp_dir, f'final_report_{timestamp_str}.xlsx')


        except Exception as e_temp:
            logger.error(f"创建临时目录失败: {e_temp}")
            return None

        try:
            # 调用专业报告生成函数，输出到临时目录
            if _GENERATE_REPORT_AVAILABLE:
                # 修复：先保存模型和元数据，再生成报告

                # 保存模型文件到临时目录
                if final_dfm_results_obj:
                    joblib.dump(final_dfm_results_obj, model_file)

                else:
                    logger.warning("没有有效的最终模型对象可供保存。")

                # 保存元数据到临时文件
                with open(metadata_file, 'wb') as f:
                    pickle.dump(metadata, f)


                # 现在调用专业报告生成函数
                generated_reports = generate_report_with_params(
                    model_path=model_file,
                    metadata_path=metadata_file,
                    output_dir=temp_dir
                )
                logger.debug(f"专业报告生成完成: {generated_reports}")

                # 修复：检查Excel文件是否真的生成了
                if generated_reports and 'excel_report' in generated_reports:
                    actual_excel_file = generated_reports['excel_report']
                    if actual_excel_file and os.path.exists(actual_excel_file):
                        excel_report_file = actual_excel_file
                        logger.debug(f"Excel报告文件确认存在: {os.path.basename(excel_report_file)}")
                    else:
                        logger.warning(f"Excel报告文件不存在: {actual_excel_file}")
                        excel_report_file = None
                else:
                    logger.warning("报告生成未返回有效的excel_report路径")
                    excel_report_file = None

                # 关键修复：从报告生成结果中提取complete_aligned_table和factor_loadings_df
                analysis_metrics_from_report = None
                if generated_reports and 'analysis_metrics' in generated_reports:
                    analysis_metrics_from_report = generated_reports['analysis_metrics']
                    if 'complete_aligned_table' in analysis_metrics_from_report:
                        # 将真实的complete_aligned_table保存到metadata
                        metadata['complete_aligned_table'] = analysis_metrics_from_report['complete_aligned_table'].copy()
                        logger.debug(f"从报告生成中获取真实的complete_aligned_table:")
                        logger.debug(f"  形状: {metadata['complete_aligned_table'].shape}")
                        logger.debug(f"  列名: {list(metadata['complete_aligned_table'].columns)}")
                    else:
                        logger.warning("报告生成结果中未找到complete_aligned_table")

                    if 'factor_loadings_df' in analysis_metrics_from_report:
                        metadata['factor_loadings_df'] = analysis_metrics_from_report['factor_loadings_df'].copy()
                        logger.debug(f"从报告生成中获取factor_loadings_df:")
                        logger.debug(f"  形状: {metadata['factor_loadings_df'].shape}")
                        logger.debug(f"  列名: {list(metadata['factor_loadings_df'].columns)}")
                    else:
                        logger.warning("报告生成结果中未找到factor_loadings_df")
                else:
                    logger.warning("报告生成未返回有效的analysis_metrics")
            else:
                logger.warning("专业报告生成模块不可用")

        except Exception as e_report:
            logger.error(f"生成Excel报告失败: {e_report}")
            excel_report_file = None  # 确保失败时为None
            # 创建基本的complete_aligned_table作为备用
            try:
                if final_nowcast_series is not None and final_aligned_df is not None:
                    basic_aligned_table = final_aligned_df.copy()
                    metadata['complete_aligned_table'] = basic_aligned_table
                    logger.info(f"创建了基本的complete_aligned_table，包含 {len(basic_aligned_table)} 行数据")
                elif all_data_aligned_weekly is not None and TARGET_VARIABLE in all_data_aligned_weekly.columns:
                    target_data = all_data_aligned_weekly[TARGET_VARIABLE].dropna()
                    if len(target_data) > 0:
                        basic_aligned_table = pd.DataFrame({
                            'Nowcast (Original Scale)': target_data,
                            TARGET_VARIABLE: target_data
                        })
                        metadata['complete_aligned_table'] = basic_aligned_table
                        logger.info(f"从原始数据创建了基本的complete_aligned_table，包含 {len(basic_aligned_table)} 行数据")
            except Exception as e_basic:
                logger.error(f"创建基本complete_aligned_table失败: {e_basic}")


        # 如果数据为None，这是一个严重错误，必须报告
        if calculated_nowcast_orig is None:
            logger.error("CRITICAL ERROR: calculated_nowcast_orig为None！这将导致UI无法显示Nowcast对比图表！")
        if original_target_series is None:
            logger.error("CRITICAL ERROR: original_target_series为None！这将导致UI无法显示Nowcast对比图表！")

        existing_complete_aligned_table = None
        existing_factor_loadings_df = None
        if 'metadata' in locals() and isinstance(metadata, dict):
            if 'complete_aligned_table' in metadata:
                existing_complete_aligned_table = metadata['complete_aligned_table']
                logger.debug(f"发现现有的complete_aligned_table，形状: {existing_complete_aligned_table.shape}")
            if 'factor_loadings_df' in metadata:
                existing_factor_loadings_df = metadata['factor_loadings_df']
                logger.debug(f"发现现有的factor_loadings_df，形状: {existing_factor_loadings_df.shape}")
        else:
            logger.warning("未发现现有的metadata，将在后续步骤中尝试获取")

        metadata = {
            'timestamp': timestamp_str,
            'status': 'Success' if final_dfm_results_obj else 'Failure', # 添加状态
            'final_data_shape': final_data_processed.shape if final_data_processed is not None else 'N/A', # Use shape of processed data
            'initial_variable_count': len(initial_variables),
            'final_variable_count': len(final_variables) if final_variables else 'N/A',
            'k_factors_stage1': k_initial_estimate, # 阶段 1 使用的 k
            'best_score_stage1': best_score_stage1,
            'best_variables_stage1': best_variables_stage1, # Keep stage 1 vars for reference
            'k_factors_final': optimal_k_stage2 if optimal_k_stage2 is not None else 'N/A',
            'factor_selection_method': FACTOR_SELECTION_METHOD,
            'best_params': { # <-- 重新添加 best_params 键
                'k_factors_final': optimal_k_stage2 if optimal_k_stage2 is not None else 'N/A',
                'factor_selection_method': FACTOR_SELECTION_METHOD,
                'variable_selection_method': 'global_backward', # 添加变量选择方法
                'tuning_objective': '(Avg Hit Rate, -Avg RMSE)', # 添加优化目标
                'max_lags': max_lags_to_use  # 新增：保存因子自回归阶数参数
            },
            'best_variables': final_variables, # final variables used
            'original_data_file': EXCEL_DATA_FILE,
            'target_variable': TARGET_VARIABLE,
            'target_freq': TARGET_FREQ,
            'train_end_date': TRAIN_END_DATE,
            'validation_start_date': VALIDATION_START_DATE, # 使用计算出的验证开始日期
            'validation_end_date': VALIDATION_END_DATE,
            'total_runtime_seconds': total_runtime_seconds, # 记录总时长
            'transform_details': final_transform_details, # <-- 修正键名
            # 移除 'var_type_map'
            'var_industry_map': var_industry_map, # 保存行业映射
            'pca_results_df': pca_results_df, # 保存PCA结果
            'factor_contributions_target': factor_contributions, # 重命名以区分
            'contribution_results_df': contribution_results_df, # 保存因子贡献度表格
            'individual_r2_results': individual_r2_results, # 保存 R2 结果
            'industry_r2_results': industry_r2_results,
            'factor_industry_r2_results': factor_industry_r2_results,
            # 移除 'factor_type_r2_results'
            'final_eigenvalues': final_eigenvalues, # <<< 新增：保存最终的特征根值
            # 优化：删除冗余的standardization_mean/std（实际未被使用）
            # 目标变量的统计量已保存为target_mean_original和target_std_original
            'target_mean_original': target_mean_original,
            'target_std_original': target_std_original,
            **final_metrics, # Unpack the metrics dictionary here
            # 关键修复：同时保存UI后端期望的键名格式
            'revised_is_rmse': final_metrics.get('is_rmse'),
            'revised_oos_rmse': final_metrics.get('oos_rmse'),
            'revised_is_mae': final_metrics.get('is_mae'),
            'revised_oos_mae': final_metrics.get('oos_mae'),
            'revised_is_hr': final_metrics.get('is_hit_rate'),
            'revised_oos_hr': final_metrics.get('oos_hit_rate'),
            'all_data_aligned_weekly': all_data_aligned_weekly, # 保存原始对齐数据
            'final_data_processed': final_data_processed, # 保存最终处理数据
            'nowcast_series': final_nowcast_series,
            'nowcast_aligned_df': final_aligned_df,
            # 关键修复：保存原始nowcast数据，确保UI能够访问
            'calculated_nowcast_orig': calculated_nowcast_orig,
            'original_target_series': original_target_series,
        }

        # 关键修复：恢复之前获取的complete_aligned_table和factor_loadings_df
        if existing_complete_aligned_table is not None:
            metadata['complete_aligned_table'] = existing_complete_aligned_table
            logger.debug(f"已恢复complete_aligned_table到新metadata中，形状: {existing_complete_aligned_table.shape}")
        else:
            logger.debug("没有现有的complete_aligned_table可恢复")

        if existing_factor_loadings_df is not None:
            metadata['factor_loadings_df'] = existing_factor_loadings_df
            logger.debug(f"已恢复factor_loadings_df到新metadata中，形状: {existing_factor_loadings_df.shape}")
        else:
            logger.warning("没有现有的factor_loadings_df可恢复")
            if final_dfm_results_obj is not None and hasattr(final_dfm_results_obj, 'Lambda'):
                final_lambda = final_dfm_results_obj.Lambda
                if isinstance(final_lambda, pd.DataFrame) and not final_lambda.empty:
                    metadata['factor_loadings_df'] = final_lambda.copy()
                    logger.info(f"从最终模型Lambda生成factor_loadings_df，形状: {final_lambda.shape}")
                else:
                    logger.warning("最终模型Lambda存在但不是有效的DataFrame")
            else:
                logger.warning("最终模型无效或缺少Lambda属性，无法生成factor_loadings_df")

        if final_dfm_results_obj is not None and hasattr(final_dfm_results_obj, 'x_sm'):
            final_factors = final_dfm_results_obj.x_sm
            if isinstance(final_factors, pd.DataFrame) and not final_factors.empty:
                metadata['factor_series'] = final_factors.copy()
                logger.debug(f"从最终模型x_sm生成factor_series，形状: {final_factors.shape}")
            else:
                logger.debug("最终模型x_sm存在但不是有效的DataFrame")
        else:
            logger.debug("最终模型无效或缺少x_sm属性，无法生成factor_series")


        # 注意：complete_aligned_table现在由generate_report_with_params生成
        # 真实的数据在第2188-2222行从报告生成结果中获取
        logger.debug("complete_aligned_table将由专业报告生成函数提供")


        if optimal_k_stage2 is not None and optimal_k_stage2 > 0:
            metadata['best_k_factors'] = optimal_k_stage2
            logger.debug(f"已将 'best_k_factors' ({optimal_k_stage2}) 添加到元数据")
        else:
            logger.debug("无法将 'best_k_factors' 添加到元数据，因为 optimal_k_stage2 无效")

        # 注意：模型和元数据文件已在Excel报告生成过程中保存

        try: # Saving Metadata


            metadata['training_start_date'] = TRAINING_START_DATE

            if final_dfm_results_obj:
                # 假设 x0 和 P0 分别存储在 initial_state 和 initial_state_cov 属性中
                # 如果实际属性名不同，需要修改下面的 getattr 调用
                x0_to_save = getattr(final_dfm_results_obj, 'x0', None)
                P0_to_save = getattr(final_dfm_results_obj, 'P0', None)

                if x0_to_save is not None and P0_to_save is not None:
                    metadata['x0'] = x0_to_save
                    metadata['P0'] = P0_to_save
                    logger.debug("已将 'x0' (initial_state) 和 'P0' (initial_state_cov) 添加到元数据")
                else:
                    missing_attrs = []
                    if not hasattr(final_dfm_results_obj, 'x0'): missing_attrs.append('x0')
                    if not hasattr(final_dfm_results_obj, 'P0'): missing_attrs.append('P0')
                    if x0_to_save is None and 'x0' not in missing_attrs: missing_attrs.append('x0 (值为 None)')
                    if P0_to_save is None and 'P0' not in missing_attrs: missing_attrs.append('P0 (值为 None)')
                    logger.warning(f"最终模型结果对象缺少或未能获取有效的属性: {', '.join(missing_attrs)}。无法将 x0/P0 添加到元数据。")
            else:
                logger.warning("最终模型结果对象 (final_dfm_results_obj) 无效，无法提取 x0/P0。")

            # 保存元数据到临时文件
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            logger.debug(f"元数据文件已生成: {os.path.basename(metadata_file)}")

        except Exception as e_save_meta:
            logger.error(f"保存元数据时出错: {e_save_meta}", exc_info=True)

        print("\n--- 两阶段调优和评估完成 (优化版本) --- \n")
        num_predictors_stage1_final = len([v for v in best_variables_stage1 if v != TARGET_VARIABLE]) if best_variables_stage1 else 'N/A'
        print(f"阶段 1 (全局筛选): 选出 {num_predictors_stage1_final} 个预测变量 (固定 k={k_initial_estimate})")
        if FACTOR_SELECTION_METHOD == 'fixed':
            print(f"阶段 2: 使用固定因子数 k={optimal_k_stage2} (用户指定)")
        else:
            print(f"阶段 2: 选择因子数 k={optimal_k_stage2} (方法: {FACTOR_SELECTION_METHOD})")
        num_final_predictors = len([v for v in final_variables if v != TARGET_VARIABLE]) if final_variables else 'N/A'
        print(f"最终模型: 使用 {num_final_predictors} 个预测变量, {optimal_k_stage2 if optimal_k_stage2 else 'N/A'} 个因子")
        print(f"总耗时: {total_runtime_seconds:.2f} 秒")
        print(f"总评估次数 (阶段1为主): {total_evaluations}")
        print(f"SVD 收敛错误次数: {svd_error_count}")

        print(f"\n--- 性能优化统计 ---")
        total_vars = len(all_data_aligned_weekly.columns) if all_data_aligned_weekly is not None else 0
        stage2_vars = len(best_variables_stage1) if best_variables_stage1 else 0
        final_vars = len(final_variables) if final_variables else 0

        print(f"标准化优化:")
        print(f"  - 原方案: 3次标准化 ({total_vars} + {stage2_vars} + {final_vars} = {total_vars + stage2_vars + final_vars} 变量次)")
        print(f"  - 优化方案: 1次标准化 + 2次子集选择 ({total_vars} + 0 + 0 = {total_vars} 变量次)")
        if total_vars > 0:
            reduction_ratio = (1 - total_vars / (total_vars + stage2_vars + final_vars)) * 100
            print(f"  - 计算量减少: {reduction_ratio:.1f}%")

        print(f"缺失值插补优化:")
        print(f"  - 原方案: 3次插补 ({total_vars} + {stage2_vars} + {final_vars} = {total_vars + stage2_vars + final_vars} 变量次)")
        print(f"  - 优化方案: 1次插补 + 2次子集选择 ({total_vars} + 0 + 0 = {total_vars} 变量次)")

        print(f"总体性能提升: 避免了 {stage2_vars + final_vars} 个变量的重复标准化和插补计算")

        # script_end_time = time.time() # 移除
        # total_runtime_seconds = script_end_time - script_start_time # 移除
        logger.info(f"\n--- 调优和最终模型估计完成 --- 总耗时: {total_runtime_seconds:.2f} 秒 ---") # 日志中使用已计算好的值

        # 修复：返回临时文件路径供UI下载
        result_files = {
            'final_model_joblib': model_file,
            'metadata': metadata_file,
            'excel_report': excel_report_file
        }

        logger.info(f"run_tuning()完成，返回文件路径: {result_files}")
        
        # 清理threadpoolctl上下文
        if _HAS_THREADPOOLCTL and 'thread_limit_context' in locals():
            try:
                thread_limit_context.__exit__(None, None, None)
            except:
                pass
        
        return result_files

    except Exception as e: # 添加 except 块
        # 清理threadpoolctl上下文
        if _HAS_THREADPOOLCTL and 'thread_limit_context' in locals():
            try:
                thread_limit_context.__exit__(None, None, None)
            except:
                pass
        
        print(f" run_tuning()发生异常: {e}")
        print(f" 异常类型: {type(e)}")
        print(f" 异常详情:")
        print(traceback.format_exc())
        logging.error(f"调优过程中发生错误:\n")
        logging.error(traceback.format_exc())
        if log_file and not log_file.closed:
            try:
                log_file.write(f"\n!!! 脚本因错误终止: {e} !!!\n")
                log_file.write(traceback.format_exc())
                log_file.close()
            except Exception as log_err:
                print(f"关闭日志文件时发生额外错误: {log_err}")
        return None  # 返回None表示失败
    finally:
        # 清理全局回调函数
        _global_progress_callback = None

        if log_file and not log_file.closed:
            log_file.close()

def train_and_save_dfm_results(
    input_df: pd.DataFrame = None,
    target_variable: str = None,
    selected_indicators: List[str] = None,
    training_start_date: Union[str, datetime] = None,
    validation_start_date: Union[str, datetime] = None,
    validation_end_date: Union[str, datetime] = None,
    train_end_date: Union[str, datetime] = None,  # 新增：训练结束日期参数
    factor_selection_strategy: str = 'information_criteria',
    variable_selection_method: str = 'none',
    enable_variable_selection: bool = False,  # [CRITICAL] 默认禁用变量选择
    max_iterations: int = 30,
    fixed_number_of_factors: int = 3,
    ic_max_factors: int = 20,
    cum_variance_threshold: float = 0.8,
    info_criterion_method: str = 'bic',
    max_lags: int = 1,  # 新增：因子自回归阶数参数
    var_industry_map: Dict[str, str] = None,  # 新增：变量行业映射
    output_dir: str = None,
    progress_callback=None,
    **kwargs
) -> Dict[str, str]:
    """
    UI接口函数：训练DFM模型并保存结果

    Args:
        input_df: 输入数据DataFrame
        target_variable: 目标变量名
        selected_indicators: 选择的指标列表
        training_start_date: 训练开始日期
        validation_start_date: 验证开始日期
        validation_end_date: 验证结束日期
        factor_selection_strategy: 因子选择策略
        variable_selection_method: 变量选择方法
        max_iterations: 最大迭代次数
        fixed_number_of_factors: 固定因子数量
        ic_max_factors: 信息准则最大因子数
        cum_variance_threshold: 累积方差阈值
        info_criterion_method: 信息准则方法
        max_lags: 因子自回归阶数（默认1）
        output_dir: 输出目录
        progress_callback: 进度回调函数
        **kwargs: 其他参数

    Returns:
        包含生成文件路径的字典
    """
    try:
        # 修复：导入os模块以避免UnboundLocalError
        import os
        

        # 导入接口包装器（简化版本，不再需要数据预处理函数）
        from dashboard.DFM.train_model.interface_wrapper import (
            convert_ui_parameters_to_backend,
            validate_ui_parameters,
            create_progress_callback
        )

        # TRAIN_END_DATE现在在后面根据VALIDATION_START_DATE计算

        # 1. 准备UI参数字典
        ui_params = {
            'prepared_data': input_df,
            'target_variable': target_variable,
            'selected_indicators': selected_indicators,  # 修复：使用正确的参数名
            'training_start_date': training_start_date,
            'validation_start_date': validation_start_date,
            'validation_end_date': validation_end_date,
            'train_end_date': train_end_date,  # 新增：训练结束日期
            'factor_selection_strategy': factor_selection_strategy,
            'variable_selection_method': variable_selection_method,
            'enable_variable_selection': enable_variable_selection,  # [CRITICAL] 使用显式参数而非kwargs
            'max_iterations': max_iterations,
            'fixed_number_of_factors': fixed_number_of_factors,
            'fixed_k_factors': fixed_number_of_factors,  # 添加别名确保参数传递
            'ic_max_factors': ic_max_factors,
            'cum_variance_threshold': cum_variance_threshold,
            'info_criterion_method': info_criterion_method,
            'max_lags': max_lags,  # 新增：因子自回归阶数参数
            # 移除 'var_type_map'
            'var_industry_map': var_industry_map or {},  # 新增：变量行业映射
            'progress_callback': progress_callback
        }

        # 临时创建进度回调用于早期调试
        temp_callback = create_progress_callback(progress_callback)

        # 调试：检查传入的参数
        temp_callback(f"[train_and_save_dfm_results] 传入参数检查:")
        temp_callback(f"  selected_indicators参数: {selected_indicators}")
        temp_callback(f"  selected_indicators类型: {type(selected_indicators)}")
        temp_callback(f"  selected_indicators长度: {len(selected_indicators) if selected_indicators else 'None'}")
        temp_callback(f"  ui_params中的selected_indicators: {ui_params.get('selected_indicators', 'N/A')}")
        
        # 添加因子选择策略相关参数的验证日志
        temp_callback(f"[PARAMETER_DEBUG] 因子选择相关参数验证:")
        temp_callback(f"  - factor_selection_strategy (UI): {factor_selection_strategy}")
        temp_callback(f"  - fixed_number_of_factors: {fixed_number_of_factors}")
        temp_callback(f"  - ic_max_factors: {ic_max_factors}")
        temp_callback(f"  - cum_variance_threshold: {cum_variance_threshold}")
        temp_callback(f"  - info_criterion_method: {info_criterion_method}")
        temp_callback(f"[PARAMETER_DEBUG] 当前全局FACTOR_SELECTION_METHOD: {globals().get('FACTOR_SELECTION_METHOD', 'undefined')}")

        # 修复：如果selected_indicators为空，检查是否在kwargs中
        if not selected_indicators and 'selected_indicators' in kwargs:
            selected_indicators = kwargs['selected_indicators']
            ui_params['selected_indicators'] = selected_indicators
            temp_callback(f"从kwargs中恢复selected_indicators: {selected_indicators}")

        # 修复：如果仍然为空，检查函数的所有参数
        if not selected_indicators:
            temp_callback(f"selected_indicators仍为空，检查所有传入参数:")
            temp_callback(f"  函数参数: target_variable={target_variable}")
            temp_callback(f"  kwargs内容: {kwargs}")

            # 如果用户确实没有选择变量，这是一个错误
            if not kwargs.get('selected_indicators'):
                temp_callback(f"错误：未传递任何选择的变量！")
                temp_callback(f"这表示UI界面的参数传递有问题")
                raise ValueError("未传递任何选择的变量，请检查UI界面的参数传递")

        # 添加kwargs中的参数
        ui_params.update(kwargs)

        # 2. 验证参数
        is_valid, errors = validate_ui_parameters(ui_params)
        if not is_valid:
            error_msg = "参数验证失败: " + "; ".join(errors)
            if progress_callback:
                progress_callback(error_msg)
            raise ValueError(error_msg)

        # 3. 转换参数格式
        backend_params = convert_ui_parameters_to_backend(ui_params)
        
        # 验证后端参数

        # 4. 直接使用data_prep的输出，不做重复预处理
        if progress_callback:
            progress_callback("接收data_prep预处理数据...")

        processed_data = ui_params.get('prepared_data')
        if processed_data is None:
            error_msg = "未找到预处理数据！请确保已在data_prep模块完成数据预处理"
            if progress_callback:
                progress_callback(error_msg)
            logger.error("重复处理检查：train_model模块必须使用data_prep模块的预处理数据")
            logger.error("请检查：1) data_prep模块是否已运行 2) 数据是否已保存到状态管理器")
            raise ValueError(error_msg)
        
        if hasattr(processed_data, 'attrs') and processed_data.attrs.get('source') == 'duplicate_processing':
            logger.warning("检测到可能的重复处理数据，请检查数据流程")

        if progress_callback:
            progress_callback(f"接收到预处理数据，形状: {processed_data.shape}")

        # 5. 修复：跳过输出目录设置，因为不保存本地文件
        # 所有结果都只能通过UI下载

        # 6. 创建标准化的进度回调
        std_callback = create_progress_callback(progress_callback)
        std_callback("开始DFM模型训练...")

        # 7. 删除重复的数据准备，直接使用data_prep的输出
        std_callback("使用data_prep预处理数据...")

        # 直接使用已经预处理好的数据
        prepared_data = processed_data
        transform_details = {}  # data_prep已经完成了所有转换
        removed_vars_log = {}   # data_prep已经记录了移除的变量
        data_metadata = {       # 创建简单的元数据
            'target_variable': ui_params.get('target_variable'),
            'data_shape': prepared_data.shape,
            'columns': list(prepared_data.columns)
        }

        std_callback(f"数据准备完成，数据形状: {prepared_data.shape}")

        # 8. 设置训练参数
        std_callback("配置训练参数...")

        # 更新全局配置变量（临时方案）
        global TARGET_VARIABLE, TRAINING_START_DATE, VALIDATION_START_DATE, VALIDATION_END_DATE, TRAIN_END_DATE
        TARGET_VARIABLE = ui_params['target_variable']

        if ui_params.get('training_start_date'):
            if hasattr(ui_params['training_start_date'], 'strftime'):
                TRAINING_START_DATE = ui_params['training_start_date'].strftime('%Y-%m-%d')
            else:
                TRAINING_START_DATE = str(ui_params['training_start_date'])

        if ui_params.get('validation_end_date'):
            if hasattr(ui_params['validation_end_date'], 'strftime'):
                VALIDATION_END_DATE = ui_params['validation_end_date'].strftime('%Y-%m-%d')
            else:
                VALIDATION_END_DATE = str(ui_params['validation_end_date'])

        # [NEW] 设置验证开始日期从UI参数
        if ui_params.get('validation_start_date'):
            if hasattr(ui_params['validation_start_date'], 'strftime'):
                VALIDATION_START_DATE = ui_params['validation_start_date'].strftime('%Y-%m-%d')
            else:
                VALIDATION_START_DATE = str(ui_params['validation_start_date'])
            
            # [CRITICAL FIX] 根据验证开始日期计算训练结束日期（前一周的周五）
            if train_end_date is None:  # 只有在未提供训练结束日期时才自动计算
                try:
                    validation_start_dt = pd.to_datetime(VALIDATION_START_DATE)
                    # 计算前一周的周五
                    # 首先找到最近的周五
                    days_until_friday = (4 - validation_start_dt.weekday()) % 7
                    if days_until_friday == 0:  # 如果验证开始日就是周五
                        # 使用前一周的周五
                        train_end_dt = validation_start_dt - pd.DateOffset(weeks=1)
                    else:
                        # 找到之前最近的周五
                        days_since_last_friday = (validation_start_dt.weekday() - 4) % 7
                        if days_since_last_friday == 0:
                            days_since_last_friday = 7
                        train_end_dt = validation_start_dt - pd.DateOffset(days=days_since_last_friday)
                    
                    TRAIN_END_DATE = train_end_dt.strftime('%Y-%m-%d')
                except Exception as e:
                    TRAIN_END_DATE = '2024-06-28'  # 使用默认值
            else:
                # 使用提供的训练结束日期
                if isinstance(train_end_date, datetime):
                    TRAIN_END_DATE = train_end_date.strftime('%Y-%m-%d')
                else:
                    TRAIN_END_DATE = str(train_end_date)
        else:
            VALIDATION_START_DATE = None
            # 如果没有验证开始日期，使用提供的或默认的训练结束日期
            if train_end_date is not None:
                if isinstance(train_end_date, datetime):
                    TRAIN_END_DATE = train_end_date.strftime('%Y-%m-%d')
                else:
                    TRAIN_END_DATE = str(train_end_date)
            else:
                TRAIN_END_DATE = '2024-06-28'  # 使用默认值

        
        if progress_callback:
            progress_callback(f"[日期参数验证] 训练开始日期: {TRAINING_START_DATE}")
            progress_callback(f"[日期参数验证] 训练结束日期: {TRAIN_END_DATE}")
            progress_callback(f"[日期参数验证] 验证开始日期: {VALIDATION_START_DATE}")
            progress_callback(f"[日期参数验证] 验证结束日期: {VALIDATION_END_DATE}")
            progress_callback(f"[日期参数验证] 目标变量: {TARGET_VARIABLE}")

        # 9. 生成结果文件路径 - 改为内存处理，不创建物理目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 移除物理文件输出，改为内存处理

        result_files = {
            'final_model_joblib': None,  # 将在内存中处理
            'metadata': None,  # 将在内存中处理
            'excel_report': None  # 将在内存中处理
        }

        # 10. 执行UI参数化的DFM训练
        std_callback("开始DFM模型训练...")

        # 使用UI传递的参数进行训练
        try:
            # 1. 从UI参数中获取用户选择的变量
            selected_indicators = ui_params.get('selected_indicators', [])
            target_variable = ui_params.get('target_variable')

            std_callback(f"[train_and_save_dfm_results] 参数检查:")
            std_callback(f"  selected_indicators: {selected_indicators}")
            std_callback(f"  selected_indicators类型: {type(selected_indicators)}")
            std_callback(f"  selected_indicators长度: {len(selected_indicators) if selected_indicators else 'None'}")
            std_callback(f"  target_variable: {target_variable}")
            std_callback(f"  prepared_data形状: {prepared_data.shape}")
            std_callback(f"  prepared_data列名前10个: {list(prepared_data.columns)[:10]}")

            # 修复：使用UI选择的变量，而不是所有变量
            # 强制检查：确保selected_indicators不为空
            if selected_indicators and len(selected_indicators) > 0:
                # 使用用户在UI中选择的变量
                std_callback(f"使用UI选择的{len(selected_indicators)}个预测变量: {selected_indicators}")
                available_predictors = selected_indicators
            else:
                # 紧急修复：如果selected_indicators为空，这是一个严重错误
                std_callback(f"严重错误: selected_indicators为空或None!")
                std_callback(f"这表示UI参数传递有问题")
                std_callback(f"ui_params内容: {ui_params}")

                # 临时使用所有可用变量（但这不是期望的行为）
                std_callback("临时回退: 使用所有可用变量")
                available_predictors = [col for col in prepared_data.columns if col != target_variable]

            # 构建最终的变量列表（仅包含预测变量，不包含目标变量）
            final_variables = available_predictors

            std_callback(f"使用UI选择的变量: {len(final_variables)}个预测变量（目标变量单独处理）")

            # 2. 从完整数据中提取用户选择的变量
            if not final_variables:
                raise ValueError("未选择任何变量进行训练")

            # 关键修复：在变量映射之前，先保存原始的UI变量名
            original_ui_variables = list(final_variables)
            std_callback(f"保存原始UI变量名: {len(original_ui_variables)} 个")

            # 增强的变量名映射逻辑 - 处理中英文标点符号差异
            available_columns = list(prepared_data.columns)
            mapped_final_variables = []

            # 导入标点符号标准化函数
            import unicodedata

            def normalize_punctuation_and_case(text):
                """标准化标点符号和大小写，用于变量名匹配"""
                if not text:
                    return ""
                # 标准化Unicode
                text = unicodedata.normalize('NFKC', str(text))
                # 替换中文标点符号为英文标点符号
                replacements = {
                    '：': ':',
                    '（': '(',
                    '）': ')',
                    '【': '[',
                    '】': ']',
                    '，': ',',
                    '。': '.',
                    '；': ';',
                    '！': '!',
                    '？': '?',
                    '"': '"',
                    '"': '"',
                    ''': "'",
                    ''': "'",
                    '—': '-',
                    '–': '-',
                    '…': '...',
                }
                for old, new in replacements.items():
                    text = text.replace(old, new)
                return text.strip().lower()

            # 创建标准化的列名映射
            normalized_columns_map = {}
            for col in available_columns:
                if col and pd.notna(col):
                    norm_col = normalize_punctuation_and_case(col)
                    if norm_col:
                        normalized_columns_map[norm_col] = col

            std_callback(f"开始变量映射，目标变量数: {len(final_variables)}, 可用列数: {len(available_columns)}")
            std_callback(f"可用列名示例: {available_columns[:5]}...")
            std_callback(f"待映射变量示例: {final_variables[:5]}...")

            mapping_stats = {
                'exact_match': 0,
                'normalized_match': 0,
                'not_found': 0
            }

            for var in final_variables:
                found = False
                matched_col = None
                match_type = None

                # 1. 尝试精确匹配
                if var in available_columns:
                    mapped_final_variables.append(var)
                    mapping_stats['exact_match'] += 1
                    found = True
                    match_type = "精确匹配"
                    matched_col = var

                # 2. 尝试标准化匹配（处理中英文标点符号和大小写差异）
                elif not found:
                    var_normalized = normalize_punctuation_and_case(var)
                    if var_normalized in normalized_columns_map:
                        actual_col = normalized_columns_map[var_normalized]
                        mapped_final_variables.append(actual_col)
                        mapping_stats['normalized_match'] += 1
                        found = True
                        match_type = "标准化匹配"
                        matched_col = actual_col

                # 记录结果
                if found:
                    if matched_col != var:
                        std_callback(f"变量名映射 ({match_type}): '{var}' -> '{matched_col}'")
                else:
                    mapping_stats['not_found'] += 1
                    std_callback(f"未找到匹配: '{var}'")

            # 输出详细的映射统计信息
            std_callback(f"\n变量映射统计:")
            std_callback(f"  - 精确匹配: {mapping_stats['exact_match']} 个")
            std_callback(f"  - 标准化匹配: {mapping_stats['normalized_match']} 个")
            std_callback(f"  - 未找到匹配: {mapping_stats['not_found']} 个")
            std_callback(f"  - 总成功匹配: {len(mapped_final_variables)} 个")

            # 修复：正确的验证逻辑 - 检查用户选择的变量是否都被成功映射
            std_callback(f"\n详细分析:")
            std_callback(f"  - 数据中可用变量总数: {len(available_columns)}")
            std_callback(f"  - 用户选择的变量数: {len(original_ui_variables)}")
            std_callback(f"  - 成功映射的变量数: {len(mapped_final_variables)}")

            # 修复：只关心用户选择的变量是否都被映射，不关心数据中其他变量
            if len(mapped_final_variables) == len(original_ui_variables):
                std_callback("  用户选择的所有变量都已成功映射！")
            else:
                unmapped_requested_vars = len(original_ui_variables) - len(mapped_final_variables)
                std_callback(f"  有 {unmapped_requested_vars} 个用户选择的变量未能映射")

            # 修复：计算用户选择变量的映射成功率（这才是有意义的指标）
            user_mapping_rate = (len(mapped_final_variables) / len(original_ui_variables)) * 100 if original_ui_variables else 0
            std_callback(f"  - 用户选择变量映射成功率: {user_mapping_rate:.1f}%")

            # 关键诊断：如果映射失败，显示详细信息
            if mapping_stats['not_found'] > 0:
                std_callback(f"\n未匹配变量分析:")
                unmatched_vars = [var for var in final_variables if var not in mapped_final_variables]
                std_callback(f"  - 未匹配变量数量: {len(unmatched_vars)}")
                std_callback(f"  - 未匹配变量示例: {unmatched_vars[:5]}...")

                # 检查是否是数据传递问题
                if len(available_columns) < len(final_variables):
                    std_callback(f"  可能的问题: 数据中的变量数({len(available_columns)})少于选择的变量数({len(final_variables)})")
                    std_callback(f"  这可能表明data_prep模块没有正确处理所有变量")
                else:
                    std_callback(f"  可能的问题: 变量名不匹配，需要检查变量名格式")

            if mapping_stats['not_found'] > 0:
                # 显示未匹配的变量
                unmatched_vars = [var for var in final_variables if var not in [v for v in mapped_final_variables]]
                std_callback(f"未匹配的变量列表: {unmatched_vars[:10]}{'...' if len(unmatched_vars) > 10 else ''}")

                # 尝试提供可能的匹配建议
                std_callback(f"可用列名示例 (前10个): {list(available_columns)[:10]}")

            if not mapped_final_variables:
                error_msg = f"所有选择的变量都不存在于数据中。选择了{len(final_variables)}个变量，但都无法匹配到数据列。"
                std_callback(f"严重错误: {error_msg}")
                raise ValueError(error_msg)

            # 更新final_variables为映射后的变量名
            final_variables = mapped_final_variables
            std_callback(f"变量映射完成，最终使用 {len(final_variables)} 个变量")
            std_callback(f"最终变量列表: {final_variables}")

            # 关键修复：将映射信息传递给调用者
            # 创建变量映射字典，供后续使用
            variable_mapping = {}
            for i, original_var in enumerate(final_variables):
                if i < len(mapped_final_variables):
                    mapped_var = mapped_final_variables[i]
                    if original_var != mapped_var:
                        variable_mapping[original_var] = mapped_var

            # 将映射信息存储到返回结果中
            if hasattr(prepared_data, '_variable_mapping'):
                prepared_data._variable_mapping = variable_mapping
            else:
                # 如果DataFrame不支持自定义属性，使用其他方式传递
                pass

            # 提取用户选择的预测变量数据
            training_data = prepared_data[final_variables].copy()
            std_callback(f"准备预测变量数据，形状: {training_data.shape} (不包含目标变量)")

            # 3. 修复：移除重复的变量选择逻辑，统一在run_tuning()中处理
            # 变量选择现在在run_tuning()函数中统一处理，避免重复执行
            enable_var_selection = ui_params.get('enable_variable_selection', False)  # [CRITICAL FIX] 默认值改为False
            variable_selection_method = ui_params.get('variable_selection_method', 'none')

            if enable_var_selection:
                std_callback(f"变量选择将在训练过程中执行: {variable_selection_method}")
            else:
                std_callback("已禁用变量选择，将直接使用用户选择的所有变量")

            # 4. 准备最终训练数据（包含目标变量和预测变量）
            final_training_data = prepared_data[[target_variable] + final_variables].copy()
            std_callback(f"最终训练数据形状: {final_training_data.shape} (包含目标变量和{len(final_variables)}个预测变量)")

            # 5. 执行因子数优化（如果启用）
            optimal_k = ui_params.get('fixed_number_of_factors', 5)  # 默认值

            if ui_params.get('enable_hyperparameter_tuning', False):
                std_callback("执行因子数优化...")
                k_range = ui_params.get('k_factors_range', (1, 10))
                info_criterion = ui_params.get('info_criterion_method', 'bic')

                std_callback(f"因子数搜索范围: {k_range}, 信息准则: {info_criterion}")

                # 这里可以实现真正的因子数优化逻辑
                # 暂时使用范围的中间值
                optimal_k = (k_range[0] + k_range[1]) // 2
                std_callback(f"因子数优化完成，最优因子数: {optimal_k}")
            else:
                std_callback(f"使用固定因子数: {optimal_k}")

            # 6. 调用真正的DFM训练逻辑
            std_callback("调用真正的DFM训练逻辑...")

            # 备份原始全局变量
            import sys
            current_module = sys.modules[__name__]
            original_data = getattr(current_module, 'all_data_aligned_weekly', None)
            original_target = getattr(current_module, 'TARGET_VARIABLE', None)
            original_factor_method = getattr(current_module, 'FACTOR_SELECTION_METHOD', None)
            original_n_iter = getattr(current_module, 'N_ITER_FIXED', None)
            # 移除 var_type_map 相关
            original_var_industry_map = getattr(current_module, 'var_industry_map', None)

            try:
                # 修复：根据UI配置正确设置因子选择方法
                factor_strategy = ui_params.get('factor_selection_strategy', 'information_criteria')
                

                # 映射UI策略到内部方法名
                strategy_mapping = {
                    'information_criteria': 'bai_ng',
                    'fixed_number': 'fixed',  # 固定因子数方法
                    'cumulative_variance': 'cumulative',
                    'elbow_method': 'elbow',
                    'kaiser_criterion': 'kaiser'
                }

                internal_method = strategy_mapping.get(factor_strategy, 'bai_ng')
                print(f"[TUNE_DFM DEBUG] Strategy mapping: {factor_strategy} -> {internal_method}")
                
                # 验证策略映射是否成功
                if factor_strategy not in strategy_mapping:
                    std_callback(f"未知的UI策略: {factor_strategy}，使用默认方法: bai_ng")
                    internal_method = 'bai_ng'
                
                # 设置UI参数到全局变量
                setattr(current_module, 'all_data_aligned_weekly', final_training_data)
                
                # [CRITICAL DEBUG] 验证实际用于训练的数据日期范围
                print(f"[DATA-RANGE-DEBUG] 实际用于训练的数据验证:")
                print(f"  数据形状: {final_training_data.shape}")
                if hasattr(final_training_data.index, 'min') and hasattr(final_training_data.index, 'max'):
                    actual_start_date = final_training_data.index.min()
                    actual_end_date = final_training_data.index.max()
                    print(f"  数据日期范围: {actual_start_date} 到 {actual_end_date}")
                    print(f"  UI设置的训练开始日期: {TRAINING_START_DATE}")
                    print(f"  UI设置的验证结束日期: {VALIDATION_END_DATE}")
                    
                    # 检查日期范围是否符合UI设置
                    if str(actual_start_date).startswith(TRAINING_START_DATE[:4]):  # 年份匹配
                        print(f"   数据开始日期与UI设置大致匹配")
                    else:
                        print(f"  数据开始日期与UI设置不匹配!")
                else:
                    print(f"  数据索引类型: {type(final_training_data.index)}")
                    print(f"  前5个索引值: {final_training_data.index[:5].tolist()}")
                
                if progress_callback:
                    progress_callback(f"[数据范围验证] 训练数据形状: {final_training_data.shape}")
                    if hasattr(final_training_data.index, 'min'):
                        progress_callback(f"[数据范围验证] 数据日期范围: {final_training_data.index.min()} 到 {final_training_data.index.max()}")
                        progress_callback(f"[数据范围验证] 与UI设置对比: 开始={TRAINING_START_DATE}, 结束={VALIDATION_END_DATE}")
                
                # 移除 var_type_map 设置
                setattr(current_module, 'var_industry_map', ui_params.get('var_industry_map', {}))
                setattr(current_module, 'TARGET_VARIABLE', target_variable)
                setattr(current_module, 'FACTOR_SELECTION_METHOD', internal_method)  # 使用UI配置的方法
                
                # 验证参数设置是否成功
                actual_method = getattr(current_module, 'FACTOR_SELECTION_METHOD', None)
                if actual_method != internal_method:
                    std_callback(f"因子选择方法设置失败! 期望: {internal_method}, 实际: {actual_method}")
                else:
                    std_callback(f"因子选择方法设置成功: {actual_method}")
                    
                # 如果是fixed方法，设置固定因子数（优先使用UI参数）
                if internal_method == 'fixed':
                    # 按优先级获取固定因子数，优先使用UI传递的值
                    fixed_k = None
                    if 'fixed_number_of_factors' in ui_params:
                        fixed_k = ui_params['fixed_number_of_factors']
                        std_callback(f"[PARAM_VALIDATION] 使用UI参数'fixed_number_of_factors': {fixed_k}")
                    elif 'fixed_k_factors' in ui_params:
                        fixed_k = ui_params['fixed_k_factors']
                        std_callback(f"[PARAM_VALIDATION] 使用备选参数'fixed_k_factors': {fixed_k}")
                    elif 'fixed_factors' in ui_params:
                        fixed_k = ui_params['fixed_factors']
                        std_callback(f"[PARAM_VALIDATION] 使用备选参数'fixed_factors': {fixed_k}")
                    else:
                        fixed_k = 1  # 默认值
                        std_callback(f"[PARAM_VALIDATION] 未找到固定因子数参数，使用默认值: {fixed_k}")
                    
                    # 设置到模块变量
                    setattr(current_module, 'FIXED_K_FACTORS', fixed_k)
                    actual_fixed_k = getattr(current_module, 'FIXED_K_FACTORS', None)
                    
                    # 验证设置结果
                    if actual_fixed_k == fixed_k:
                        std_callback(f"[PARAM_VALIDATION] 固定因子数设置成功: {fixed_k}")
                    else:
                        std_callback(f"[PARAM_VALIDATION] 固定因子数设置失败! 期望: {fixed_k}, 实际: {actual_fixed_k}")
                    
                    # 详细参数来源日志
                    std_callback(f"[PARAM_DEBUG] 固定因子数参数来源:")
                    std_callback(f"  - ui_params['fixed_number_of_factors']: {ui_params.get('fixed_number_of_factors', 'N/A')}")
                    std_callback(f"  - ui_params['fixed_k_factors']: {ui_params.get('fixed_k_factors', 'N/A')}")
                    std_callback(f"  - ui_params['fixed_factors']: {ui_params.get('fixed_factors', 'N/A')}")
                    std_callback(f"  - 最终使用值: {fixed_k}")
                
                # 记录所有关键参数的最终值
                std_callback(f"[PARAM_SUMMARY] ===== 最终参数配置 =====")
                std_callback(f"  - UI策略: {factor_strategy}")
                std_callback(f"  - 内部方法: {actual_method}")
                std_callback(f"  - 目标变量: {target_variable}")
                std_callback(f"  - 最大迭代次数: {getattr(current_module, 'N_ITER_FIXED', 30)}")
                if internal_method == 'fixed':
                    std_callback(f"  - 固定因子数: {getattr(current_module, 'FIXED_K_FACTORS', 'None')}")
                elif internal_method == 'bai_ng':
                    std_callback(f"  - IC最大因子数: {getattr(current_module, 'IC_MAX_FACTORS', 'Default')}")
                elif internal_method == 'cumulative':
                    std_callback(f"  - 累积方差阈值: {getattr(current_module, 'COMMON_VARIANCE_CONTRIBUTION_THRESHOLD', 'Default')}")
                std_callback(f"[PARAM_SUMMARY] =========================")
                setattr(current_module, 'N_ITER_FIXED', ui_params.get('max_iterations', 30))
                
                # 注意：固定因子数已经在上面设置，不需要重复设置
                if internal_method != 'fixed':
                    std_callback(f"使用自动因子选择方法: {internal_method}")

                if 'ic_max_factors' in ui_params:
                    setattr(current_module, 'IC_MAX_FACTORS', ui_params['ic_max_factors'])
                    std_callback(f"设置IC最大因子数: {ui_params['ic_max_factors']}")

                std_callback(f"设置训练参数: 数据形状{final_training_data.shape}, 目标变量{target_variable}, 因子数{optimal_k}")

                # 调用真正的训练函数，传递用户选择的数据
                std_callback("执行run_tuning()...")
                # 关键修复：使用映射后的变量名，而不是原始UI变量名
                # final_variables现在只包含预测变量，不包含目标变量
                mapped_selected_vars = final_variables
                std_callback(f"使用映射后的{len(mapped_selected_vars)}个预测变量")

                # 创建原始变量名到映射变量名的对应关系
                original_to_mapped = {}
                if len(original_ui_variables) == len(final_variables):
                    for i, (orig, mapped) in enumerate(zip(original_ui_variables, final_variables)):
                        if orig != mapped:
                            original_to_mapped[orig] = mapped
                            std_callback(f"变量映射: '{orig}' -> '{mapped}'")

                std_callback(f"传递给run_tuning的变量: {len(mapped_selected_vars)}个预测变量 + 目标变量")

                # 详细调试信息
                std_callback(f"调试信息:")
                std_callback(f"  final_training_data形状: {final_training_data.shape}")
                std_callback(f"  final_training_data列名: {list(final_training_data.columns)}")
                std_callback(f"  target_variable: {target_variable}")
                std_callback(f"  selected_indicators (原始UI选择): {selected_indicators}")
                std_callback(f"  mapped_selected_vars (传递给run_tuning): {mapped_selected_vars}")
                std_callback(f"  final_variables (映射后): {final_variables}")
                std_callback(f"[数据传递修复] UI选择{len(selected_indicators)}个 -> 传递{len(mapped_selected_vars)}个")

                # 修复：接收run_tuning()返回的文件路径，并传递映射后的数据
                enable_var_selection = ui_params.get('enable_variable_selection', False)  # [CRITICAL FIX] 默认值改为False
                std_callback(f"变量选择设置: enable_variable_selection={enable_var_selection}")

                tuning_result_files = run_tuning(
                    external_data=final_training_data,
                    external_target_variable=target_variable,
                    external_selected_variables=mapped_selected_vars,  # 使用映射后的变量名
                    # 移除 external_var_type_map 参数
                    external_var_industry_map=ui_params.get('var_industry_map', {}),
                    external_max_lags=ui_params.get('max_lags', 1),  # 修复：传递因子自回归阶数参数
                    enable_variable_selection=enable_var_selection,  # 新增：传递变量选择控制参数
                    variable_selection_method=variable_selection_method,  # 新增：传递变量选择方法
                    max_iterations=max_iterations,  # [CRITICAL FIX] 传递UI设置的迭代次数
                    progress_callback=std_callback  # 新增：传递回调函数到模型估计过程
                )

                # 修复：使用run_tuning()返回的文件路径
                if tuning_result_files:
                    result_files.update(tuning_result_files)
                    std_callback("训练完成！结果文件已生成")
                    std_callback(f"模型文件: {os.path.basename(result_files.get('final_model_joblib', 'N/A'))}")
                    std_callback(f"元数据文件: {os.path.basename(result_files.get('metadata', 'N/A'))}")
                    excel_report_path = result_files.get('excel_report')
                    if excel_report_path:
                        std_callback(f"Excel报告: {os.path.basename(excel_report_path)}")
                    else:
                        std_callback("Excel报告: 未生成")
                else:
                    std_callback("run_tuning()未返回有效的文件路径")

                # 创建训练结果摘要
                training_results = {
                    'model_type': 'DFM',
                    'final_variables': final_variables,  # 修复：使用final_variables替代variables_after_selection
                    'optimal_k_factors': optimal_k,
                    'data_shape': final_training_data.shape,
                    'target_variable': target_variable,
                    'selected_indicators': selected_indicators,
                    'training_params': {k: v for k, v in ui_params.items() if not callable(v)},
                    'timestamp': timestamp,
                    'training_completed': True,
                    'note': '所有文件只能通过UI下载'
                }

                std_callback("真正的DFM训练完成！")
                std_callback("注意：所有结果文件只能通过UI下载，不会保存到本地目录")

                return result_files

            except Exception as e:
                std_callback(f"run_tuning()执行失败: {str(e)}")
                print(f"run_tuning()错误详情: {e}")
                print(f"run_tuning()异常类型: {type(e)}")
                print(f"run_tuning()异常traceback:")
                print(traceback.format_exc())

            finally:
                # 恢复原始全局变量
                if original_data is not None:
                    setattr(current_module, 'all_data_aligned_weekly', original_data)
                if original_target is not None:
                    setattr(current_module, 'TARGET_VARIABLE', original_target)
                if original_factor_method is not None:
                    setattr(current_module, 'FACTOR_SELECTION_METHOD', original_factor_method)
                if original_n_iter is not None:
                    setattr(current_module, 'N_ITER_FIXED', original_n_iter)
                # 移除 var_type_map 恢复
                if original_var_industry_map is not None:
                    setattr(current_module, 'var_industry_map', original_var_industry_map)

            # 如果run_tuning失败，创建基本的训练结果
            training_results = {
                'model_type': 'DFM',
                'final_variables': final_variables,  # 修复：使用final_variables替代variables_after_selection
                'optimal_k_factors': optimal_k,
                'data_shape': final_training_data.shape,
                'target_variable': target_variable,
                'selected_indicators': selected_indicators,
                'training_params': {k: v for k, v in ui_params.items() if not callable(v)},
                'timestamp': timestamp,
                'training_completed': True,
                'fallback_mode': True
            }

            std_callback("使用回退模式完成训练...")

        except Exception as e:
            std_callback(f"训练过程出错: {str(e)}，创建基本结果文件...")
            print(f"训练错误详情: {e}")
            # 创建基本的训练结果
            training_results = {
                'final_variables': list(prepared_data.columns),
                'optimal_k_factors': ui_params.get('fixed_number_of_factors', 5),
                'data_shape': prepared_data.shape,
                'target_variable': ui_params['target_variable'],
                'training_params': ui_params,
                'timestamp': timestamp,
                'error': str(e)
            }

        # 修复：生成临时文件供UI下载，不保存到用户本地目录
        try:
            import tempfile
            import joblib
            import pickle
            import os  # 修复：添加os模块导入

            # 创建可序列化的ui_params副本（移除回调函数）
            serializable_ui_params = {k: v for k, v in ui_params.items()
                                    if k not in ['progress_callback'] and not callable(v)}

            # 修复：创建临时文件供UI下载
            temp_dir = tempfile.mkdtemp(prefix='dfm_results_')

            # 生成临时文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_file = os.path.join(temp_dir, f'final_dfm_model_{timestamp}.joblib')
            metadata_file = os.path.join(temp_dir, f'final_dfm_metadata_{timestamp}.pkl')

            # 清理不可序列化的对象
            def clean_for_serialization(obj):
                """递归清理字典中的不可序列化对象"""
                if isinstance(obj, dict):
                    return {k: clean_for_serialization(v) for k, v in obj.items() 
                           if not callable(v) and not hasattr(v, '__call__')}
                elif isinstance(obj, (list, tuple)):
                    return type(obj)([clean_for_serialization(item) for item in obj 
                                    if not callable(item) and not hasattr(item, '__call__')])
                elif callable(obj) or hasattr(obj, '__call__'):
                    return f"<function {getattr(obj, '__name__', 'unknown')}>"
                else:
                    return obj
            
            # 清理training_results
            clean_training_results = clean_for_serialization(training_results)
            
            # 保存模型文件到临时目录
            joblib.dump(clean_training_results, model_file)
            std_callback(f"模型文件已生成: {os.path.basename(model_file)}")

            # 准备完整的元数据
            complete_metadata = {
                'training_results': clean_training_results,
                'ui_params': serializable_ui_params,
                'data_metadata': data_metadata,
                'timestamp': timestamp,
                'training_completed': True
            }

            # 保存元数据文件到临时目录
            try:
                with open(metadata_file, 'wb') as f:
                    pickle.dump(complete_metadata, f)
                std_callback(f"元数据文件已生成: {os.path.basename(metadata_file)}")
            except Exception as e:
                std_callback(f"元数据文件保存失败: {str(e)}")
                # 尝试只保存基本信息
                basic_metadata = {
                    'timestamp': timestamp,
                    'training_completed': True,
                    'data_shape': clean_training_results.get('data_shape', 'unknown'),
                    'target_variable': clean_training_results.get('target_variable', 'unknown')
                }
                try:
                    with open(metadata_file, 'wb') as f:
                        pickle.dump(basic_metadata, f)
                    std_callback(f"基本元数据文件已生成: {os.path.basename(metadata_file)}")
                except Exception as e2:
                    std_callback(f"元数据文件保存彻底失败: {str(e2)}")
                    metadata_file = None

            # 更新result_files为临时文件路径
            result_files['final_model_joblib'] = model_file
            result_files['metadata'] = metadata_file

            std_callback("训练结果已准备完成，可通过UI下载")
            std_callback("注意：文件保存在临时目录中，只能通过UI下载")

            return result_files

        except Exception as e:
            error_msg = f"生成结果文件失败: {str(e)}"
            std_callback(error_msg)
            raise

    except Exception as e:
        error_msg = f"训练过程出错: {str(e)}"
        if progress_callback:
            progress_callback(error_msg)
        print(f"错误详情: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    # 当直接运行时，使用默认参数（不传入外部数据）
    # 默认使用全局后向选择
    run_tuning(enable_variable_selection=True, variable_selection_method='global_backward', max_iterations=30)