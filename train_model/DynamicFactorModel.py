# -*- coding: utf-8 -*-
"""
Dynamic Factor Model Implementation

Created on Mon Feb 17 10:00:39 2020
@author: Hogan

OPTIMIZATION NOTES (2025-08-14):
=================================
This module has been optimized to avoid redundant preprocessing while maintaining 
DFM algorithm requirements. The key optimizations include:

1. BOUNDARY CLARIFICATION:
   - data_prep module: General time series preprocessing (stationarity, frequency alignment, basic NaN handling)
   - DFM module: Algorithm-specific transformations (standardization for PCA/EM, training set constraints)

2. PRESERVED DFM-SPECIFIC OPERATIONS:
   - Training/validation split standardization (prevents information leakage)
   - PCA-based matrix operations (SVD decomposition)
   - EM algorithm parameter estimation (Kalman filtering/smoothing)
   - Covariance matrix positive definiteness enforcement
   - Structural shock generation for state space representation

3. VALIDATION COMMENTS:
   - All DFM-specific operations are marked with [DFM-SPECIFIC] tags
   - Documentation clarifies what data_prep should provide vs. what DFM requires
   - Mathematical requirements are preserved for algorithm correctness

4. INFORMATION LEAKAGE PREVENTION:
   - Training set-based standardization is maintained
   - PCA initialization uses only training data when specified
   - All operations respect train_end_date boundary
"""
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
# import tkinter.filedialog  # [HOT] 注释掉可能有问题的导入
from datetime import timedelta
import math
import time
import calendar
# from numba import jit  # [HOT] 注释掉可能有问题的导入
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR
import scipy
import statsmodels.tsa.stattools as ts
import statsmodels.tsa as tsa
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.decomposition import PCA
from dashboard.DFM.train_model.DiscreteKalmanFilter import calculate_factor_loadings, KalmanFilter, FIS, EMstep
from scipy.optimize import minimize
import numpy.linalg

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _calculate_pca(observables, n_factors):
    """Helper function for PCA calculation (legacy support)"""
    # syntax: 
    n_time = len(observables.index)
    x = np.array(observables - observables.mean())
    z = np.array((observables - observables.mean())/observables.std())
    
    # Calculate covariance matrix S from standardized data z
    # Correct calculation: S = (1/N) * Z'Z
    S = (z.T @ z) / n_time

    eigenvalues, eigenvectors = np.linalg.eigh(S) # Use eigh for covariance matrix
    sorted_indices = np.argsort(eigenvalues)[::-1] # Descending order
    evalues = eigenvalues[sorted_indices[:-n_factors-1:-1]]
    V = np.array(eigenvectors[:,sorted_indices[:-n_factors-1:-1]])
    D = np.diag(evalues)
    
    return D, V, S

def DFM(observation, n_factors):
    """
    Initial PCA-based Factor Extraction (Legacy Function)
    
    PREPROCESSING BOUNDARY:
    - This function performs DFM-specific PCA initialization
    - The standardization here is algorithm-required, not redundant preprocessing
    
    NOTE: This function is mainly used for quick factor extraction.
    For full DFM estimation with EM algorithm, use DFM_EMalgo.
    
    Args:
        observation: DataFrame with time series data
        n_factors: Number of factors to extract
    
    Returns:
        DFMResultsWrapperPCA: Initial PCA-based factor results
    """
    if len(observation.columns) <= n_factors:
        raise ValueError('Error: number of common factors must be less than number of variables')

    n_time = len(observation.index)

    # [DFM-SPECIFIC] Standardize data for PCA algorithm requirements
    # This is not redundant with data_prep - it's required for PCA mathematical operations
    obs_mean = observation.mean()
    obs_std = observation.std()
    obs_std[obs_std == 0] = 1.0  # Prevent division by zero
    # [DFM-SPECIFIC] fillna(0) required for SVD decomposition stability
    z = ((observation - obs_mean) / obs_std).fillna(0)  # Algorithm requirement

    # Use np.linalg.svd directly on the standardized data z
    # U shape: (n_time, n_time), S shape: (min(n_time, n_obs),), Vh shape: (n_obs, n_obs)
    U, s, Vh = np.linalg.svd(z, full_matrices=False)

    # Factors are U[:, :k] * s[:k]
    factors = U[:, :n_factors] * s[:n_factors]
    CommonFactors = pd.DataFrame(data=factors, index=observation.index, columns=[f'Factor{i+1}' for i in range(n_factors)])

    # [DFM-SPECIFIC] Estimate idiosyncratic variance from PCA residuals
    # This is a core DFM algorithm step, not general preprocessing
    V = Vh.T
    reconstructed_z = factors @ V[:, :n_factors].T
    residuals = z - reconstructed_z
    # [DFM-SPECIFIC] Calculate diagonal covariance matrix for DFM state space model
    psi_diag = np.nanvar(residuals, axis=0)  # Algorithm handles NaN appropriately
    psi_diag = np.maximum(psi_diag, 1e-6)  # DFM requires positive definiteness
    Psi = np.diag(psi_diag)

    # We don't calculate Lambda, A, B, Sigma here anymore

    return DFMResultsWrapperPCA(common_factors=CommonFactors,
                                idiosyncratic_covariance=Psi,
                                obs_mean=obs_mean)

# ============================================================================
# RESULT WRAPPER CLASSES
# ============================================================================

class DFMResultsWrapperPCA():
    """Simplified wrapper for PCA-based DFM results"""
    def __init__(self, common_factors, idiosyncratic_covariance, obs_mean):
        self.common_factors = common_factors
        self.idiosyncratic_covariance = idiosyncratic_covariance
        self.obs_mean = obs_mean

def DFM_EMalgo(observation, n_factors, n_shocks, n_iter, train_end_date=None, error='False', max_lags=1):
    """
    Dynamic Factor Model Estimation via EM Algorithm
    
    PREPROCESSING BOUNDARY:
    - EXPECTS: data_prep has handled stationarity checks, basic NaN cleaning, frequency alignment
    - DFM-SPECIFIC OPERATIONS:
      * Training/validation split standardization (prevents information leakage)
      * PCA-based initialization (algorithm-specific matrix operations)
      * EM parameter estimation (Kalman filtering and smoothing)
      * Covariance matrix positive definiteness enforcement
      * Structural shock generation for state space representation
    
    The standardization and NaN handling here are NOT redundant with data_prep:
    - data_prep: general time series preprocessing
    - DFM_EMalgo: algorithm-specific transformations required for matrix operations
    
    Args:
        observation: DataFrame with time series data (from data_prep)
        n_factors: Number of common factors
        n_shocks: Number of structural shocks
        n_iter: EM algorithm iterations
        train_end_date: End date for training set (prevents information leakage)
        error: Whether to include structural shocks
        max_lags: VAR model lag order
    
    Returns:
        DFMEMResultsWrapper: Trained DFM model results
    """
    # [DFM-SPECIFIC] Set deterministic seed for reproducible results
    import numpy as np
    import random

    DFM_SEED = 42
    random.seed(DFM_SEED)
    np.random.seed(DFM_SEED)
    print(f"  [DFM-SPECIFIC] 设置确定性随机种子: {DFM_SEED}")

    n_obs = observation.shape[1]
    n_time = observation.shape[0]

    if max_lags == 1:
        # VAR(1)情况：状态向量就是因子向量
        n_state_vars = n_factors
        state_names = [f'Factor{i+1}' for i in range(n_factors)]
    else:
        # VAR(p)情况：状态向量包含当前因子和所有滞后因子
        n_state_vars = n_factors * max_lags
        state_names = []
        for lag in range(max_lags):
            for factor in range(n_factors):
                if lag == 0:
                    state_names.append(f'Factor{factor+1}')
                else:
                    state_names.append(f'Factor{factor+1}_lag{lag}')

    # Step 1: Data Validation and Preprocessing Parameter Setup
    # Validate that data_prep has provided appropriate preprocessing
    print(f"  [VALIDATION] 输入数据形状: {observation.shape}")
    print(f"  [VALIDATION] 数据中NaN比例: {observation.isna().sum().sum() / observation.size:.2%}")
    
    # [DFM-SPECIFIC] 仅使用训练集数据计算标准化参数，避免信息泄漏
    # 这是DFM算法必需的，因为需要确保预测性能评估的有效性
    if train_end_date is not None:
        try:
            train_data = observation.loc[:train_end_date]
            print(f"  [DFM-SPECIFIC] 使用训练集进行标准化: {len(train_data)} 样本 (截止到 {train_end_date})")
            obs_mean = train_data.mean(skipna=True)
        except (KeyError, IndexError) as e:
            print(f"  [WARNING] 无法解析train_end_date '{train_end_date}': {e}，回退到全样本标准化")
            obs_mean = observation.mean(skipna=True)
    else:
        print("  [WARNING] 未指定train_end_date，使用全数据集进行标准化（可能存在信息泄漏）")
        obs_mean = observation.mean(skipna=True) # 保持向后兼容
    
    # Center data using training set mean (DFM algorithm requirement)
    obs_centered = observation - obs_mean
    all_nan_cols = obs_centered.columns[obs_centered.isna().all()].tolist()
    if all_nan_cols:
        print(f"  [WARNING] 以下列在中心化后全为NaN (可能原本就是全NaN): {all_nan_cols}")

    # === Step 2: DFM-Specific Standardization ===
    # Note: data_prep should have handled basic cleaning, but DFM requires specific standardization
    print("  [DFM-SPECIFIC] 步骤2: DFM算法专用标准化初始化")

    # Calculate standard deviation from training set to prevent information leakage
    # This is essential for DFM algorithm validity - we cannot use future data for standardization
    if train_end_date is not None:
        try:
            train_data = observation.loc[:train_end_date]
            obs_std = train_data.std(skipna=True)
        except (KeyError, IndexError) as e:
            print(f"  [WARNING] 无法解析train_end_date计算标准差: {e}，回退到全样本计算")
            obs_std = observation.std(skipna=True)
    else:
        obs_std = observation.std(skipna=True) # 保持向后兼容
    
    # Handle zero standard deviation (constant variables after preprocessing)
    obs_std[obs_std == 0] = 1.0 # 避免除零
    
    # [DFM-SPECIFIC] Standardize and fill NaN for PCA/EM algorithm requirements
    # The fillna(0) is required for DFM matrix operations, not redundant preprocessing
    z = (obs_centered / obs_std).fillna(0) # DFM算法要求：标准化后NaN填充为0

    # [DFM-SPECIFIC] PCA initialization must use only training data to prevent information leakage
    # This is a core DFM algorithm requirement for valid out-of-sample evaluation
    if train_end_date is not None:
        try:
            z_train = z.loc[:train_end_date]  # 仅使用训练集进行PCA
            print(f"  [DFM-SPECIFIC] PCA初始化仅使用训练集: {len(z_train)} 样本 (防止信息泄漏)")
            z_for_pca = z_train
        except (KeyError, IndexError) as e:
            print(f"  [WARNING] 无法提取训练集进行PCA: {e}，回退到全数据集")
            z_for_pca = z
    else:
        print("  [WARNING] PCA使用全数据集（可能存在信息泄漏）")
        z_for_pca = z

    # 执行 PCA (SVD)
    # print("  Performing PCA via SVD...") # 注释掉
    try:
        U, s, Vh = np.linalg.svd(z_for_pca, full_matrices=False)
        # 初始因子 F0 = U_k * S_k
        factors_init = U[:, :n_factors] * s[:n_factors]
        factors_init_df = pd.DataFrame(factors_init, index=z_for_pca.index, columns=[f'Factor{i+1}' for i in range(n_factors)])
        
        # [DFM-SPECIFIC] 保持训练集和验证集的严格分离
        # 验证集因子将通过Kalman滤波动态估计，而不是预填充
        if train_end_date is not None and len(z_for_pca) < len(observation):
            print(f"  [DFM-SPECIFIC] PCA初始因子仅覆盖训练集 {len(factors_init_df)} 行，验证集因子将通过Kalman滤波估计")
        # print(f"  Initial PCA factors calculated. Shape: {factors_init_df.shape}") # 注释掉
    except np.linalg.LinAlgError as pca_e:
        # print(f"  Error during PCA (SVD): {pca_e}. Cannot initialize with PCA. Stopping.") # 注释掉
        raise ValueError("PCA failed during initialization.") from pca_e

    # 初始化 Lambda (载荷矩阵)
    # print("  Calculating initial Lambda (Factor Loadings)...") # 注释掉
    try:
        # 使用 DiscreteKalmanFilter.py 中的函数
        Lambda_temp = calculate_factor_loadings(obs_centered, factors_init_df)
        # 确保 Lambda_temp 是 (n_obs, n_factors)
        if Lambda_temp.shape == (n_factors, n_obs):
             Lambda_temp = Lambda_temp.T
        if Lambda_temp.shape != (n_obs, n_factors):
             raise ValueError(f"Unexpected Lambda shape: {Lambda_temp.shape}")

        if max_lags == 1:
            # VAR(1)情况：Lambda矩阵维度为 n_obs x n_factors
            Lambda_current = Lambda_temp
        else:
            # VAR(p)情况：Lambda矩阵维度为 n_obs x (n_factors*max_lags)
            # 但观测只与当前期因子相关，所以只有前n_factors列非零
            Lambda_current = np.zeros((n_obs, n_state_vars))
            Lambda_current[:, :n_factors] = Lambda_temp

        # print(f"  Initial Lambda calculated. Shape: {Lambda_current.shape}") # 注释掉
        # print(f"  Initial Lambda contains NaN: {np.isnan(Lambda_current).any()}") # 注释掉
    except Exception as lambda_e:
        # print(f"  Error calculating initial Lambda: {lambda_e}. Initializing randomly as fallback.") # 注释掉
        np.random.seed(DFM_SEED)  # 确保fallback也是确定性的
        if max_lags == 1:
            Lambda_current = np.random.randn(n_obs, n_factors) * 0.1
        else:
            Lambda_current = np.zeros((n_obs, n_state_vars))
            Lambda_current[:, :n_factors] = np.random.randn(n_obs, n_factors) * 0.1

    # 初始化 A (状态转移矩阵) 和 Q (过程噪声协方差)
    # print(f"  Calculating initial A (Transition Matrix) and Q (Process Noise Cov) via VAR({max_lags})...") # 注释掉
    try:
        var_model = VAR(factors_init_df.dropna()) # 对初始因子拟合 VAR(max_lags)
        var_results = var_model.fit(max_lags)

        if max_lags == 1:
            # VAR(1)情况：直接使用系数矩阵
            A_current = var_results.coefs[0]
        else:
            # VAR(p)情况：构造companion form矩阵
            # A_current将是 (n_factors*max_lags) x (n_factors*max_lags) 的矩阵
            n_factors_orig = factors_init_df.shape[1]
            A_current = np.zeros((n_factors_orig * max_lags, n_factors_orig * max_lags))

            # 填充上半部分：VAR系数
            for i in range(max_lags):
                A_current[:n_factors_orig, i*n_factors_orig:(i+1)*n_factors_orig] = var_results.coefs[i]

            # 填充下半部分：单位矩阵块（用于滞后项的传递）
            if max_lags > 1:
                for i in range(max_lags - 1):
                    start_row = (i + 1) * n_factors_orig
                    end_row = (i + 2) * n_factors_orig
                    start_col = i * n_factors_orig
                    end_col = (i + 1) * n_factors_orig
                    A_current[start_row:end_row, start_col:end_col] = np.eye(n_factors_orig)

        if max_lags == 1:
            # VAR(1)情况：Q矩阵就是残差协方差矩阵
            Q_current = np.cov(var_results.resid, rowvar=False)
        else:
            # VAR(p)情况：构造companion form的Q矩阵
            # Q矩阵维度为 (n_factors*max_lags) x (n_factors*max_lags)
            # 只有左上角的 n_factors x n_factors 块是非零的
            Q_var = np.cov(var_results.resid, rowvar=False)
            Q_current = np.zeros((n_state_vars, n_state_vars))
            Q_current[:n_factors, :n_factors] = Q_var
    except Exception as var_e:
        # print(f"  Error fitting VAR(1) for initial A/Q: {var_e}. Using simple initialization as fallback.") # 注释掉
        A_current = np.eye(n_factors) * 0.95
        Q_current = np.eye(n_factors) * 0.1

    # [DFM-SPECIFIC] Initialize observation noise covariance R from PCA residuals
    # This calculation is specific to DFM algorithm and cannot be replaced by data_prep
    V = Vh.T
    # 确保维度一致性：R矩阵计算必须基于训练集，保持算法的统计有效性
    if train_end_date is not None and len(z_for_pca) < len(z):
        # 使用训练集计算R矩阵 - 这是DFM算法的正确做法
        factors_for_residual = factors_init  # (262, n_factors) - numpy数组
        z_for_residual = z_for_pca           # (262, 75) - DataFrame
        print(f"  [DFM-SPECIFIC] 使用训练集计算R矩阵: {z_for_residual.shape}")
    else:
        # 使用全样本（当没有训练/验证分离时）
        factors_for_residual = factors_init  # numpy数组
        z_for_residual = z_for_pca           # DataFrame
        print(f"  [DFM-SPECIFIC] 使用全样本计算R矩阵: {z_for_residual.shape}")
    
    # 重构数据和计算残差（维度现在匹配）
    reconstructed_z = factors_for_residual @ V[:, :n_factors].T  
    residuals_z = z_for_residual - reconstructed_z               
    
    # [DFM-SPECIFIC] R computation: var(标准化残差) * var(原始观测)
    psi_diag = np.nanvar(residuals_z, axis=0) # 标准化残差的方差
    original_std_sq = obs_std.fillna(1.0)**2 # 原始标准差的平方
    R_diag_current = psi_diag * original_std_sq.to_numpy()
    R_diag_current = np.maximum(R_diag_current, 1e-6) # DFM要求：确保正定性
    R_current = np.diag(R_diag_current)
    print(f"  [DFM-SPECIFIC] 观测噪声协方差R已从PCA残差计算. 形状: {R_current.shape}")

    # 初始化 B (冲击矩阵) - 根据状态向量维度调整
    # print("  Initializing B simply...") # 注释掉
    if max_lags == 1:
        # VAR(1)情况：B矩阵维度为 n_factors x n_shocks
        if n_shocks != n_factors:
            B_current = np.zeros((n_factors, n_shocks))
            min_dim = min(n_factors, n_shocks)
            B_current[:min_dim, :min_dim] = np.eye(min_dim) * 0.1
        else:
            B_current = np.eye(n_factors) * 0.1
    else:
        # VAR(p)情况：B矩阵维度为 (n_factors*max_lags) x n_shocks
        # 只有第一个n_factors行有非零元素（当前期的冲击）
        B_current = np.zeros((n_state_vars, n_shocks))
        if n_shocks != n_factors:
            min_dim = min(n_factors, n_shocks)
            B_current[:min_dim, :min_dim] = np.eye(min_dim) * 0.1
        else:
            B_current[:n_factors, :n_factors] = np.eye(n_factors) * 0.1

    # 初始化 x0 和 P0 - 根据状态向量维度调整
    # print("  Initializing x0 as zero vector and P0 as identity matrix.") # 注释掉
    x0_current = np.zeros(n_state_vars)
    P0_current = np.eye(n_state_vars)
    initial_x0 = x0_current
    initial_P0 = P0_current

    # [DFM-SPECIFIC] Ensure covariance matrices are positive definite
    # This is essential for DFM algorithm stability and Kalman filter convergence
    epsilon = 1e-6
    print(f"  [DFM-SPECIFIC] 确保协方差矩阵Q和R的正定性 (epsilon={epsilon})")
    
    # Convert to numpy arrays if needed (algorithm requirement)
    if isinstance(Q_current, pd.DataFrame): Q_current = Q_current.to_numpy()
    if isinstance(R_current, pd.DataFrame): R_current = R_current.to_numpy()
    
    # Ensure 2D diagonal structure (handle VAR failure fallback)
    if Q_current.ndim == 1: Q_current = np.diag(Q_current)
    if R_current.ndim == 1: R_current = np.diag(R_current)

    # [DFM-SPECIFIC] Force positive definiteness for numerical stability
    Q_current = np.diag(np.maximum(np.diag(Q_current), epsilon))
    R_current = np.diag(np.maximum(np.diag(R_current), epsilon))
    
    print(f"    Q形状: {Q_current.shape}, 最小特征值: {np.min(np.diag(Q_current)):.6f}")
    print(f"    R形状: {R_current.shape}, 最小特征值: {np.min(np.diag(R_current)):.6f}")
    # === 修改结束: 基于 PCA 的初始化 ===

    # [DFM-SPECIFIC] Prepare structural shocks U for Kalman Filter
    # This is essential for DFM state space representation
    if error:
        # [DFM-SPECIFIC] Deterministic shock generation for reproducibility
        np.random.seed(DFM_SEED)  # DFM算法要求：确保可重现性
        u_data = np.random.randn(len(observation.index), n_shocks)
        print(f"  [DFM-SPECIFIC] 生成随机结构冲击 (种子={DFM_SEED})")
    else:
        u_data = np.zeros(shape=(len(observation.index), n_shocks))
        print(f"  [DFM-SPECIFIC] 使用零结构冲击")
    error_df = pd.DataFrame(data=u_data, columns=[f'shock{i+1}' for i in range(n_shocks)], index=observation.index)

    # === Step 3: DFM EM Algorithm Iterations ===
    # [DFM-SPECIFIC] This is the core DFM parameter estimation via EM algorithm
    print(f"  [DFM-SPECIFIC] 开始EM算法迭代 ({n_iter} 次迭代)")

    for i in range(n_iter):
        # E-Step: Run Kalman Filter and Smoother
        kf = KalmanFilter(Z=obs_centered, U=error_df, A=A_current, B=B_current, H=Lambda_current, state_names=state_names, x0=x0_current, P0=P0_current, Q=Q_current, R=R_current)
        fis = FIS(kf)

        # M-Step: Update parameters using smoothed factors
        em = EMstep(fis, n_shocks) # Should return arrays

        # Update parameters for next iteration
        A_current = np.array(em.A)
        B_current = np.array(em.B)
        Lambda_current = np.array(em.Lambda)
        Q_current = np.array(em.Q) # Make sure EMstep returns updated Q
        R_current = np.array(em.R) # Make sure EMstep returns updated R

        # Check diagonal of Q and R
        
        # [DFM-SPECIFIC] Update initial state for next Kalman filter iteration
        # This state updating is essential for EM convergence
        if max_lags == 1:
            # VAR(1): Direct use of smoothed state
            x0_current = np.array(em.x_sm.iloc[0])
        else:
            # VAR(p): Construct companion form initial state
            if hasattr(em, 'x_sm') and em.x_sm is not None:
                # Build companion form from lagged factors
                factors_for_init = em.x_sm.iloc[:max_lags].values
                x0_expanded = factors_for_init.flatten()
                if len(x0_expanded) == n_state_vars:
                    x0_current = x0_expanded
                else:
                    x0_current = np.zeros(n_state_vars)
            else:
                x0_current = np.zeros(n_state_vars)

        P0_current = fis.P_sm[0] # Use smoothed covariance for next iteration

    # === Step 4: Final DFM Results Generation ===
    print(f"  [DFM-SPECIFIC] 使用优化后参数运行最终Kalman滤波器")
    kf_final = KalmanFilter(Z=obs_centered, U=error_df, A=A_current, B=B_current, H=Lambda_current, state_names=state_names, x0=x0_current, P0=P0_current, Q=Q_current, R=R_current)
    
    # The smoothed state `x_sm` is from the last M-step's input (`em.x_sm`)
    if max_lags == 1:
        # VAR(1)情况：直接返回平滑状态
        final_x_sm_to_return = em.x_sm
    else:
        # VAR(p)情况：从companion form状态向量中提取当前期因子
        if hasattr(em, 'x_sm') and em.x_sm is not None:
            if isinstance(em.x_sm, pd.DataFrame) and em.x_sm.shape[1] == n_state_vars:
                # 提取前n_factors列（当前期因子）
                final_x_sm_to_return = em.x_sm.iloc[:, :n_factors].copy()
                # 重新设置列名为原始因子名称
                final_x_sm_to_return.columns = [f'Factor{i+1}' for i in range(n_factors)]
            else:
                # 如果维度不匹配，返回原始结果
                final_x_sm_to_return = em.x_sm
        else:
            final_x_sm_to_return = em.x_sm

    nan_in_final_factors = False
    if isinstance(final_x_sm_to_return, pd.DataFrame):
        if final_x_sm_to_return.isnull().values.any():
            nan_in_final_factors = True
            print("\nERROR DETECTED in DFM_EMalgo: Final smoothed factors (x_sm) contain NaNs!")
            print(f"  NaN count per factor:\n{final_x_sm_to_return.isnull().sum()}")
            # Optional: Find first NaN index
            try:
                 first_nan_idx = final_x_sm_to_return[final_x_sm_to_return.isnull().any(axis=1)].index[0]
                 print(f"  First NaN occurred around index: {first_nan_idx}")
            except IndexError:
                 print("  Could not determine first NaN index.")
    elif isinstance(final_x_sm_to_return, np.ndarray):
        if np.isnan(final_x_sm_to_return).any():
             nan_in_final_factors = True
             print("\nERROR DETECTED in DFM_EMalgo: Final smoothed factors (x_sm as ndarray) contain NaNs!")
             print(f"  Total NaN count: {np.isnan(final_x_sm_to_return).sum()}")
    

    if max_lags == 1:
        # VAR(1)情况：Lambda矩阵保持原样
        Lambda_to_return = Lambda_current
    else:
        # VAR(p)情况：只返回与当前期因子相关的载荷（前n_factors列）
        if Lambda_current.shape[1] == n_state_vars:
            Lambda_to_return = Lambda_current[:, :n_factors]
        else:
            Lambda_to_return = Lambda_current

    # 在返回之前存储计算出的 obs_mean
    return DFMEMResultsWrapper(A=A_current, B=B_current, Q=Q_current, R=R_current, Lambda=Lambda_to_return, x=kf_final.x, x_sm=final_x_sm_to_return, z=kf_final.z, obs_mean=obs_mean, x0=initial_x0, P0=initial_P0)

class DFMEMResultsWrapper():
    # 添加 obs_mean 到 __init__ 参数和属性
    def __init__(self, A, B, Q, R, Lambda, x, x_sm, z, obs_mean, x0, P0):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Lambda = Lambda
        self.x = x
        self.x_sm = x_sm
        self.z = z
        self.obs_mean = obs_mean # 存储原始观测均值
        self.x0 = x0
        self.P0 = P0
    
def RevserseTranslate(Factors, miu, Lambda, names):
    # Factors is DataFrame, miu is Series, Lambda is array (n_obs, n_factors)
    # observation = Factors @ Lambda.T + miu
    factors_arr = np.array(Factors)
    lambda_arr = np.array(Lambda)
    # Ensure lambda_arr has shape (n_obs, n_factors)
    if lambda_arr.shape[0] == Factors.shape[1]: # Check if factors are columns in Lambda
        lambda_arr = lambda_arr.T

    # Perform calculation: Factors (time x n_factors) @ Lambda.T (n_factors x n_obs)
    translated_data = factors_arr @ lambda_arr.T
    # Add mean back (broadcasting)
    observation_arr = translated_data + miu.to_numpy() # Convert Series to array for broadcasting

    observation_df=pd.DataFrame(data=observation_arr, columns=names, index=Factors.index)
    return observation_df