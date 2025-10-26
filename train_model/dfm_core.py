# -*- coding: utf-8 -*-
"""
核心 DFM 模型评估功能 - 轻量级版本

重要说明：
- 数据预处理（类型转换、NaN处理、方差检查等）已在data_prep模块完成
- 本模块仅进行DFM模型训练前的必要验证和模型相关操作
- 避免重复的数据质量检查，提高计算效率
- 信任data_prep模块的预处理结果
"""
import pandas as pd
import numpy as np
import time
import sys
import os
from typing import Tuple, List, Dict, Union, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error

# === 优化：添加静默控制机制 ===
_SILENT_WARNINGS = os.getenv('DFM_SILENT_WARNINGS', 'true').lower() == 'true'

def _conditional_print(*args, **kwargs):
    """静默模式下的条件打印函数"""
    if not _SILENT_WARNINGS:
        print(*args, **kwargs)

# 假设 apply_stationarity_transforms 在 data_utils.py 中定义
from dashboard.DFM.train_model.DynamicFactorModel import DFM_EMalgo
from dashboard.DFM.train_model.analysis_utils import calculate_metrics_with_lagged_target
from dashboard.DFM.train_model.evaluation_cache import DFMEvaluationCache

# === 辅助函数定义 ===

def _validate_inputs(
    variables: list[str],
    full_data: pd.DataFrame,
    target_variable: str,
    params: dict,
    validation_end: str,
    target_mean_original: float,
    target_std_original: float
) -> Tuple[bool, str, int]:
    """验证输入参数的有效性
    
    Returns:
        Tuple[bool, str, int]: (is_valid, error_message, k_factors)
    """
    # 验证 k_factors 参数
    k_factors = params.get('k_factors', None)
    if k_factors is None:
        return False, "错误: 参数字典中未提供 'k_factors'。", 0
    
    # 验证目标变量
    if target_variable not in variables:
        return False, f"错误: 目标变量 {target_variable} 不在当前变量列表中(len={len(variables)}): {variables[:5]}...", k_factors
    
    # 验证预测变量
    predictor_vars = [v for v in variables if v != target_variable]
    if not predictor_vars:
        return False, f"错误: 只有目标变量，没有预测变量: {variables}", k_factors
    
    # 验证数据有效性
    if not isinstance(full_data, pd.DataFrame) or full_data.empty:
        return False, "错误: 传入的 full_data 无效或为空。", k_factors
    
    if not isinstance(full_data.index, pd.DatetimeIndex):
        return False, "错误: full_data 的索引必须是 DatetimeIndex。", k_factors
    
    # 验证日期范围
    try:
        _ = full_data.loc[:validation_end]
    except (KeyError, TypeError, ValueError) as e:
        return False, f"错误: 无法在 full_data 中定位结束日期 '{validation_end}'。可用范围: {full_data.index.min()} 到 {full_data.index.max()}", k_factors
    
    # 验证原始目标变量统计数据
    if pd.isna(target_mean_original) or pd.isna(target_std_original) or target_std_original == 0:
        return False, f"错误: 传入的原始目标变量统计量无效 (Mean: {target_mean_original}, Std: {target_std_original})，无法反标准化。", k_factors
    
    return True, "", k_factors


def _prepare_data(
    variables: list[str],
    full_data: pd.DataFrame,
    target_variable: str,
    validation_end: str
) -> Tuple[bool, str, pd.DataFrame, pd.Series, list[str]]:
    """分离预测变量和目标变量
    
    按照标准DFM方法，将预测变量用于因子提取，目标变量用于载荷估计
    
    Returns:
        Tuple[bool, str, pd.DataFrame, pd.Series, list[str]]: (is_success, error_message, predictor_data, target_data, predictor_variables)
    """
    # 截取拟合所需数据段
    try:
        data_for_fitting = full_data.loc[:validation_end].copy()
    except KeyError:
        return False, f"错误: 无法在 full_data 中定位结束日期 '{validation_end}'。可用范围: {full_data.index.min()} 到 {full_data.index.max()}", pd.DataFrame(), pd.Series(), []
    
    if data_for_fitting.empty:
        return False, f"错误: 截取到 '{validation_end}' 的数据为空。", pd.DataFrame(), pd.Series(), []
    
    # 检查列存在性
    current_variables = list(variables)
    if not all(col in data_for_fitting.columns for col in current_variables):
        missing = [col for col in current_variables if col not in data_for_fitting.columns]
        return False, f"错误: 评估函数缺少列: {missing}。可用列: {data_for_fitting.columns.tolist()[:10]}...", pd.DataFrame(), pd.Series(), []
    
    # 按照标准DFM方法分离变量
    predictor_variables = [v for v in current_variables if v != target_variable]
    if not predictor_variables:
        return False, f"错误: 没有预测变量，无法训练模型。变量列表: {current_variables}", pd.DataFrame(), pd.Series(), []
    
    # 提取预测变量数据（用于DFM训练）
    predictor_data = data_for_fitting[predictor_variables].copy()
    
    # 提取目标变量数据（用于估计载荷）
    if target_variable not in data_for_fitting.columns:
        return False, f"错误: 目标变量 '{target_variable}' 不在数据中", pd.DataFrame(), pd.Series(), []
    target_data = data_for_fitting[target_variable].copy()
    
    if predictor_data.empty:
        return False, "错误: 预测变量数据为空", pd.DataFrame(), pd.Series(), []
    
    print(f"    数据分离完成 - 预测变量: {len(predictor_variables)} 个，观测: {predictor_data.shape[0]} 个")
    print(f"    目标变量 '{target_variable}' 已分离，将用于载荷估计")
    
    return True, "", predictor_data, target_data, predictor_variables


def _clean_and_validate_data(
    predictor_data: pd.DataFrame,
    predictor_variables: list[str],
    k_factors: int
) -> Tuple[bool, str, pd.DataFrame, list[str]]:
    """验证预测变量数据质量
    
    验证预测变量是否满足DFM训练要求
    
    Returns:
        Tuple[bool, str, pd.DataFrame, list[str]]: (is_success, error_message, validated_data, final_variables)
    """
    # 基本形状验证
    if predictor_data.empty:
        return False, f"错误: 预测变量={len(predictor_variables)}, n_factors={k_factors} -> 预测变量数据为空", pd.DataFrame(), []
    
    # 验证预测变量存在
    if not predictor_variables:
        return False, f"错误: 预测变量={len(predictor_variables)}, n_factors={k_factors} -> 没有预测变量", pd.DataFrame(), []
    
    # 基本维度检查 - 现在只需要预测变量数量足够
    if predictor_data.shape[1] < k_factors:
        return False, f"错误: 预测变量={len(predictor_variables)}, n_factors={k_factors} -> 预测变量数 ({predictor_data.shape[1]}) 不足因子数 ({k_factors})", pd.DataFrame(), []
    
    if predictor_data.shape[0] < k_factors:
        return False, f"错误: 预测变量={len(predictor_variables)}, n_factors={k_factors} -> 数据行数 ({predictor_data.shape[0]}) 不足因子数 ({k_factors})", pd.DataFrame(), []
    
    print(f"    预测变量验证完成: {len(predictor_variables)} 个变量，{predictor_data.shape[0]} 个观测")
    return True, "", predictor_data, predictor_variables


def _mask_seasonal_data(
    predictor_data: pd.DataFrame,
    target_data: pd.Series,
    target_variable: str
) -> Tuple[bool, str, pd.DataFrame, pd.Series]:
    """应用季节性掩码
    
    对目标变量的1-2月数据进行掩码，预测变量保持完整
    
    Returns:
        Tuple[bool, str, pd.DataFrame, pd.Series]: (is_success, error_message, predictor_data, masked_target_data)
    """
    predictor_data_copy = predictor_data.copy()  # 预测变量不需要掩码
    target_data_masked = target_data.copy()
    
    month_indices = target_data_masked.index.month
    mask_jan_feb = (month_indices == 1) | (month_indices == 2)
    
    target_nan_before_mask = target_data_masked.isna().sum()
    target_data_masked.loc[mask_jan_feb] = np.nan
    target_nan_after_mask = target_data_masked.isna().sum()
    
    print(f"    季节性掩码已应用：{target_variable} 在1-2月的值被掩码，新增NaN: {target_nan_after_mask - target_nan_before_mask}")
    print(f"    预测变量保持完整用于因子提取")
    
    return True, "", predictor_data_copy, target_data_masked


def _fit_dfm_model(
    predictor_data: pd.DataFrame,
    k_factors: int,
    max_iter: int,
    max_lags: int,
    train_end_date: str = None
) -> Tuple[bool, str, object, bool]:
    """从预测变量提取共同因子
    
    使用标准DFM方法，只从预测变量中提取共同因子和载荷
    
    Returns:
        Tuple[bool, str, object, bool]: (is_success, error_message, dfm_results, is_svd_error)
    """
    is_svd_error = False
    
    try:
        n_shocks = k_factors  # 假设 n_shocks 等于 k_factors
        print(f"    启动DFM训练：{predictor_data.shape[1]} 个预测变量，{predictor_data.shape[0]} 个观测，{k_factors} 个因子")
        print(f"    标准DFM方法：目标变量不参与因子提取")
        dfm_results = DFM_EMalgo(
            observation=predictor_data,  # 标准DFM：只使用预测变量进行因子提取
            n_factors=k_factors,
            n_shocks=n_shocks,
            n_iter=max_iter,
            train_end_date=train_end_date,  # 传递train_end_date以避免信息泄漏
            max_lags=max_lags
        )
    except Exception as dfm_e:
        error_msg = f"错误: DFM 运行失败: {dfm_e}"
        return False, error_msg, None, is_svd_error
    
    # 检查 DFM 结果
    if (not hasattr(dfm_results, 'x_sm') or dfm_results.x_sm is None or
        not isinstance(dfm_results.x_sm, (pd.DataFrame, pd.Series)) or dfm_results.x_sm.empty or
        not hasattr(dfm_results, 'Lambda') or dfm_results.Lambda is None or
        not isinstance(dfm_results.Lambda, np.ndarray)):
        return False, "错误: DFM 结果不完整或类型错误", None, is_svd_error
    
    return True, "", dfm_results, is_svd_error


def _calculate_nowcast(
    dfm_results: object,
    target_data: pd.Series,
    predictor_variables: list[str],
    target_variable: str,
    k_factors: int,
    target_mean_original: float,
    target_std_original: float,
    train_end_date: str
) -> Tuple[bool, str, pd.Series, pd.DataFrame]:
    """计算nowcast预测值 - 标准动态因子模型方法
    
    标准DFM流程：
    1. DFM模型从预测变量中提取共同因子和载荷
    2. 目标变量单独对这些共同因子进行OLS回归估计载荷
    3. 计算nowcast: factors @ lambda_target
    
    Returns:
        Tuple[bool, str, pd.Series, pd.DataFrame]: (is_success, error_message, nowcast_series, extended_lambda_df)
    """
    factors_sm = dfm_results.x_sm
    predictor_lambda_matrix = dfm_results.Lambda
    
    # 验证预测变量载荷矩阵维度
    if predictor_lambda_matrix.shape[1] != k_factors:
        return False, f"错误: 载荷矩阵列数 ({predictor_lambda_matrix.shape[1]}) 与因子数 ({k_factors}) 不符", pd.Series(), None
    if predictor_lambda_matrix.shape[0] != len(predictor_variables):
        return False, f"错误: 载荷矩阵行数 ({predictor_lambda_matrix.shape[0]}) 与预测变量数 ({len(predictor_variables)}) 不符", pd.Series(), None
    
    # 创建预测变量载荷DataFrame
    try:
        predictor_lambda_df = pd.DataFrame(
            predictor_lambda_matrix, 
            index=predictor_variables, 
            columns=[f'Factor{i+1}' for i in range(k_factors)]
        )
    except Exception as e_lambda_df:
        return False, f"错误: 创建载荷DataFrame失败: {e_lambda_df}", pd.Series(), None
    
    # 验证因子数据
    if not isinstance(factors_sm, pd.DataFrame):
        return False, f"错误: 因子数据类型错误 (期望DataFrame, 实际{type(factors_sm)})", pd.Series(), predictor_lambda_df
    
    try:
        print(f"    开始估计目标变量 '{target_variable}' 对共同因子的载荷（标准DFM方法）")
        
        # 准备训练数据，避免信息泄露
        if train_end_date:
            try:
                train_factors = factors_sm.loc[:train_end_date]
                train_target = target_data.loc[:train_end_date]
                print(f"    使用训练期数据估计载荷：{len(train_factors)} 个观测")
            except (KeyError, IndexError):
                print(f"    警告: 无法提取训练期数据，使用全样本")
                train_factors = factors_sm
                train_target = target_data
        else:
            print(f"    警告: 未指定训练期，使用全样本")
            train_factors = factors_sm
            train_target = target_data
        
        # 清理NaN数据
        valid_idx = ~(train_factors.isna().any(axis=1) | train_target.isna())
        if valid_idx.sum() < k_factors:
            return False, f"错误: 有效样本数 ({valid_idx.sum()}) 少于因子数 ({k_factors})", pd.Series(), predictor_lambda_df
        
        train_factors_clean = train_factors[valid_idx]
        train_target_clean = train_target[valid_idx]
        
        # 标准DFM方法：目标变量对共同因子的OLS回归
        # target_t = λ₁F₁ₜ + λ₂F₂ₜ + ... + λₖFₖₜ + εₜ
        from sklearn.linear_model import LinearRegression
        reg_model = LinearRegression(fit_intercept=False)  # 标准化数据无需截距
        reg_model.fit(train_factors_clean, train_target_clean)
        
        lambda_target = reg_model.coef_
        r2_score = reg_model.score(train_factors_clean, train_target_clean)

        print(f"    目标变量载荷估计完成，拟合优度 R²: {r2_score:.4f}")

        # 计算nowcast: y_t = F_t @ λ_target
        factors_array = factors_sm.to_numpy()

        # 维度一致性检查
        if factors_array.shape[1] != len(lambda_target):
            min_dim = min(factors_array.shape[1], len(lambda_target))
            if min_dim <= 0:
                return False, "错误: 无法匹配因子和载荷维度", pd.Series(), predictor_lambda_df

            print(f"    维度调整: 使用前{min_dim}个因子计算nowcast")
            nowcast_standardized = factors_array[:, :min_dim] @ lambda_target[:min_dim]
        else:
            nowcast_standardized = factors_array @ lambda_target
        
        # 反标准化到原始尺度
        if pd.isna(target_std_original) or pd.isna(target_mean_original) or target_std_original == 0:
            return False, f"错误: 反标准化参数无效 (μ={target_mean_original}, σ={target_std_original})", pd.Series(), predictor_lambda_df
        
        nowcast_series_orig = pd.Series(
            nowcast_standardized * target_std_original + target_mean_original, 
            index=factors_sm.index, 
            name='Nowcast_Orig'
        )
        
        print(f"    Nowcast计算完成，已反标准化到原始尺度")
        
        # 创建扩展载荷矩阵（包含目标变量载荷）
        target_row = pd.DataFrame(
            [lambda_target], 
            index=[target_variable], 
            columns=predictor_lambda_df.columns
        )
        extended_lambda_df = pd.concat([predictor_lambda_df, target_row])
        
    except Exception as e_nowcast:
        return False, f"错误: Nowcast计算失败: {e_nowcast}", pd.Series(), predictor_lambda_df
    
    return True, "", nowcast_series_orig, extended_lambda_df


def _calculate_metrics(
    nowcast_series_orig: pd.Series,
    full_data: pd.DataFrame,
    target_variable: str,
    validation_start: str,
    validation_end: str,
    train_end_date: str
) -> Tuple[bool, str, dict, pd.DataFrame]:
    """计算评估指标
    
    Returns:
        Tuple[bool, str, dict, pd.DataFrame]: (is_success, error_message, metrics_dict, aligned_df_monthly)
    """
    try:
        # 计算指标 (调用新函数，传递原始序列)
        original_target_series_full = full_data[target_variable].copy()
        metrics_dict, aligned_df_for_metrics = calculate_metrics_with_lagged_target(
            nowcast_series=nowcast_series_orig,
            target_series=original_target_series_full,
            validation_start=validation_start,
            validation_end=validation_end,
            train_end=train_end_date,
            target_variable_name=target_variable
        )
        
        return True, "", metrics_dict, aligned_df_for_metrics
        
    except Exception as e:
        return False, f"错误: 计算评估指标时出错: {e}", {}, pd.DataFrame()


def _handle_step_failure(step_name: str, error_msg: str, current_variables: list, k_factors: int, 
                        is_svd_error: bool = False, lambda_df: pd.DataFrame = None) -> tuple:
    """处理步骤失败的统一逻辑"""
    if "Vars=" not in error_msg:
        error_msg = f"Vars={len(current_variables)}, n_factors={k_factors} -> {error_msg}"
    print(f"    {error_msg}")
    return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, lambda_df, None)


def evaluate_dfm_params(
    variables: list[str],
    full_data: pd.DataFrame,
    target_variable: str,
    params: dict, # Now ONLY contains 'k_factors'
    # 移除 var_type_map 参数
    validation_start: str,
    validation_end: str,
    target_freq: str, # 假设 target_freq 仍被需要或将被使用
    train_end_date: str,
    target_mean_original: float,
    target_std_original: float,
    max_iter: int = 50,
    max_lags: int = 1,  # [HOT] 新增：因子自回归阶数参数
    use_cache: bool = True,  # 新增：是否使用缓存
    cache_instance: Optional[DFMEvaluationCache] = None,  # 新增：缓存实例
) -> Tuple[float, float, float, float, float, float, bool, pd.DataFrame | None, pd.DataFrame | None]:
    """评估DFM模型参数组合 - 标准动态因子模型方法
    
    标准DFM评估流程：
    1. 输入验证和数据准备
    2. 从预测变量中提取共同因子和载荷
    3. 目标变量OLS回归估计载荷，计算nowcast
    4. 评估指标计算和缓存管理
    
    返回:
        Tuple: (is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, 
               is_svd_error, lambda_df, aligned_df_monthly)
        - 指标值：成功时返回实际数值，失败时RMSE/MAE=np.inf, Hit Rate=-np.inf
        - lambda_df: 包含因子载荷的DataFrame，失败时为None
        - is_svd_error: 指示是否为SVD收敛问题
    """
    start_time = time.time()
    is_svd_error = False
    FAIL_RETURN = (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, None, None)
    
    # === 缓存机制 ===
    cache_key = None
    if use_cache:
        # 如果没有提供缓存实例，创建一个局部缓存
        if cache_instance is None:
            # 使用全局缓存或创建临时缓存
            from dashboard.DFM.train_model.evaluation_cache import get_global_cache
            cache_instance = get_global_cache()
        
        # 生成缓存键
        cache_key = cache_instance.generate_cache_key(
            variables=variables,
            params=params,
            target_variable=target_variable,
            validation_start=validation_start,
            validation_end=validation_end,
            train_end_date=train_end_date,
            target_freq=target_freq,
            max_iter=max_iter,
            target_mean_original=target_mean_original,
            target_std_original=target_std_original,
            max_lags=max_lags
        )
        
        # 尝试从缓存获取结果
        cached_result = cache_instance.get(cache_key, compute_time_estimate=2.149)
        if cached_result is not None:
            # 缓存命中，直接返回
            _conditional_print(f"    缓存命中: 变量数={len(variables)}, k={params.get('k_factors')}, 节省时间≈{time.time() - start_time:.3f}s")
            return cached_result

    try:
        # 步骤1: 验证输入
        is_valid, error_msg, k_factors = _validate_inputs(
            variables, full_data, target_variable, params, 
            validation_end, target_mean_original, target_std_original
        )
        if not is_valid:
            _conditional_print(f"    {error_msg}")
            return FAIL_RETURN

        # 步骤2: 分离预测变量和目标变量（标准DFM方法）
        success, error_msg, predictor_data, target_data, predictor_variables = _prepare_data(
            variables, full_data, target_variable, validation_end)
        if not success:
            print(f"    {error_msg}")
            return FAIL_RETURN
        
        # 步骤3: 验证预测变量数据质量
        success, error_msg, predictor_data_cleaned, predictor_variables = _clean_and_validate_data(
            predictor_data, predictor_variables, k_factors)
        if not success:
            print(f"    {error_msg}")
            return FAIL_RETURN
        
        # 步骤4: 应用季节性掩码
        success, error_msg, predictor_data_final, target_data_masked = _mask_seasonal_data(
            predictor_data_cleaned, target_data, target_variable)
        if not success:
            print(f"    {error_msg}")
            return FAIL_RETURN

        # 步骤5: 从预测变量提取共同因子和载荷
        success, error_msg, dfm_results, is_svd_error = _fit_dfm_model(
            predictor_data_final, k_factors, max_iter, max_lags, train_end_date)
        if not success:
            return _handle_step_failure("因子提取", error_msg, predictor_variables, k_factors, is_svd_error)

        # 步骤6: 估计目标变量载荷并计算nowcast
        success, error_msg, nowcast_series_orig, lambda_df = _calculate_nowcast(
            dfm_results, target_data_masked, predictor_variables, target_variable, k_factors,
            target_mean_original, target_std_original, train_end_date)
        if not success:
            return _handle_step_failure("目标载荷估计", error_msg, predictor_variables, k_factors, is_svd_error, lambda_df)

        # 步骤7: 计算评估指标
        success, error_msg, metrics_dict, aligned_df_monthly = _calculate_metrics(
            nowcast_series_orig, full_data, target_variable, validation_start, validation_end, train_end_date)
        if not success:
            return _handle_step_failure("指标计算", error_msg, predictor_variables, k_factors, is_svd_error, lambda_df)

        # 提取指标并返回成功结果
        result = (
            metrics_dict.get('is_rmse', np.nan), metrics_dict.get('oos_rmse', np.nan),
            metrics_dict.get('is_mae', np.nan), metrics_dict.get('oos_mae', np.nan),
            metrics_dict.get('is_hit_rate', np.nan), metrics_dict.get('oos_hit_rate', np.nan),
            is_svd_error, lambda_df, aligned_df_monthly
        )
        
        # 将结果存入缓存
        if use_cache and cache_key is not None and cache_instance is not None:
            cache_instance.put(cache_key, result)
            elapsed_time = time.time() - start_time
            _conditional_print(f"    计算完成: 变量数={len(variables)}, k={params.get('k_factors')}, 耗时={elapsed_time:.3f}s, 结果已缓存")
        
        return result

    except Exception as e:
        print(f"评估函数 evaluate_dfm_params (k={params.get('k_factors', 'N/A')}) 发生意外错误: {type(e).__name__}: {e}")
        return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False, None, None) 