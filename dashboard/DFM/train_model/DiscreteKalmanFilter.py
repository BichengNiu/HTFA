# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:00:22 2020

@author: Hogan
"""
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from datetime import timedelta
import math
import time
import calendar
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR
import scipy
import statsmodels.tsa.stattools as ts
import statsmodels.tsa as tsa
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from numpy.random import randn
from scipy import linalg
import numpy.linalg # Needed for moved functions

DEBUG_KALMAN = False # 设置为 True 以打印详细的卡尔曼滤波器内部状态


def calculate_factor_loadings(observables, factors):
    """Calculates factor loadings (Lambda) using OLS, handling NaNs.

    Regresses each observable series onto the factors using only non-missing data points
    for that specific series.

    Args:
        observables (pd.DataFrame): Observation data (n_time x n_obs), potentially with NaNs.
                                      Assumed to be already centered.
        factors (pd.DataFrame): Factor data (n_time x n_factors), assumed no NaNs.

    Returns:
        np.ndarray: Lambda matrix (n_obs x n_factors).
    """
    n_obs = observables.shape[1]
    n_factors = factors.shape[1]
    Lambda = np.full((n_obs, n_factors), np.nan) # Initialize with NaNs

    # Ensure factors is numpy array for efficiency if needed later, though OLS takes DataFrame
    F_np = factors.to_numpy()

    for i in range(n_obs):
        y_i = observables.iloc[:, i]
        valid_idx = y_i.notna() & factors.notna().all(axis=1) # Ensure both y_i and all factors are valid

        y_i_valid = y_i[valid_idx]
        F_valid = factors[valid_idx]

        # Check if enough data points remain for OLS (at least n_factors + 1 if adding constant, or n_factors if not)
        # We don't add constant as data is assumed centered
        if len(y_i_valid) > n_factors:
            try:
                # Perform OLS: y_i ~ F (no constant needed for centered data)
                ols_model = sm.OLS(y_i_valid, F_valid)
                ols_results = ols_model.fit()
                Lambda[i, :] = ols_results.params.values
            except Exception as e:
                print(f"Warning: OLS failed for observable {i} ('{observables.columns[i]}'). Loadings set to NaN. Error: {e}")
                # Lambda[i, :] remains NaN due to initialization
        else:
             pass

    return Lambda # Shape (n_obs, n_factors)

def _calculate_prediction_matrix(factors):
    n_time = len(factors.index)
    F = np.array(factors)

    # Correct matrix calculation for A = (F_t' F_{t-1})(F_{t-1}' F_{t-1})^-1
    F_t = F[1:, :]      # Shape (n_time-1, n_factors)
    F_tm1 = F[:-1, :]   # Shape (n_time-1, n_factors)

    Ft_Ftm1 = F_t.T @ F_tm1     # Shape: (n_factors, n_time-1) @ (n_time-1, n_factors) -> (n_factors, n_factors)
    Ftm1_Ftm1 = F_tm1.T @ F_tm1 # Shape: (n_factors, n_time-1) @ (n_time-1, n_factors) -> (n_factors, n_factors)

    # Use scipy.linalg.solve for better numerical stability
    try:
        A = scipy.linalg.solve((Ftm1_Ftm1 + np.eye(Ftm1_Ftm1.shape[0]) * 1e-7).T, Ft_Ftm1.T, assume_a='pos').T
    except scipy.linalg.LinAlgError:
        # Fallback to pseudo-inverse if solve fails
        A = Ft_Ftm1 @ np.linalg.pinv(Ftm1_Ftm1 + np.eye(Ftm1_Ftm1.shape[0]) * 1e-7)

    return A

def _calculate_shock_matrix(factors, prediction_matrix, n_shocks):
    n_time = len(factors.index)
    F = np.array(factors)
    A = np.array(prediction_matrix)
    
    # Calculate F_{t-1}' F_{t-1} efficiently
    F_tm1 = F[:-1, :] # Shape (n_time-1, n_factors)
    temp = F_tm1.T @ F_tm1 # Shape (n_factors, n_factors)
    
    # Calculate F_t' F_t efficiently
    F_t = F[1:, :] # Shape (n_time-1, n_factors)
    term1 = F_t.T @ F_t # Shape (n_factors, n_factors)
    term1 = term1 / (n_time - 1)
    
    # Calculate Sigma = E[F_t F_t'] - A E[F_{t-1} F_{t-1}'] A'
    term2 = A @ (temp / (n_time - 1)) @ A.T
    Sigma = term1 - term2 # This is the estimated Q matrix
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma) # Use eigh for symmetric matrices
        # Replace negative eigenvalues with a small positive number
        min_eig_val = 1e-7 # Floor for eigenvalues
        eigenvalues_corrected = np.maximum(eigenvalues, min_eig_val)
        
        # Reconstruct Sigma using corrected eigenvalues (optional, but good practice)
        Sigma_corrected = eigenvectors @ np.diag(eigenvalues_corrected) @ eigenvectors.T
        
        # Calculate B using the corrected eigenvalues
        # Sort eigenvalues descending to pick largest shocks
        sorted_indices = np.argsort(eigenvalues_corrected)[::-1]
        # Select top n_shocks eigenvalues and corresponding eigenvectors
        evalues_selected = eigenvalues_corrected[sorted_indices[:n_shocks]] 
        M = eigenvectors[:, sorted_indices[:n_shocks]]
        
        # Calculate B = M * sqrt(diag(selected eigenvalues))
        # Ensure sqrt is applied only to non-negative values (should be guaranteed by correction)
        B = M @ np.diag(np.sqrt(evalues_selected))
        
        # Use the corrected Sigma as the returned Q
        Q = Sigma_corrected

    except np.linalg.LinAlgError as e:
        print(f"Warning: Eigenvalue decomposition failed for Sigma (Q) in calculate_shock_matrix: {e}. Falling back to identity matrix for Q and B.")
        Q = np.eye(A.shape[0]) * 1e-6 # Small identity matrix as fallback Q
        # Fallback B: Adjust shape based on n_shocks
        B = np.zeros((A.shape[0], n_shocks))
        min_dim_fallback = min(A.shape[0], n_shocks)
        B[:min_dim_fallback, :min_dim_fallback] = np.eye(min_dim_fallback) * np.sqrt(1e-6)
    
    return B, Q # Return corrected Q (Sigma_corrected)


class KalmanFilterResultsWrapper():
    def __init__(self, x_minus, x, z, Kalman_gain, P, P_minus, state_names, A):
        self.x_minus = x_minus
        self.x = x
        self.z = z
        self.Kalman_gain = Kalman_gain
        self.P = P
        self.state_names = state_names
        self.A = A
        self.P_minus = P_minus
        
def KalmanFilter(Z, U, A, B, H, state_names, x0, P0, Q, R):
    #x_t = A*x_{t-1} + B*u_t + Q
    #z_t = H*x_t + R
    # Q is process noise covariance
    # R is measurement noice covariance
    
    measurement_names = Z.columns
    timestamp = Z.index
    n_time = len(Z.index)
    n_state = len(state_names)
    # Convert inputs to NumPy arrays
    z = np.array(Z.to_numpy())
    u = np.array(U.to_numpy())
    A = np.array(A)
    B = np.array(B)
    H = np.array(H)
    x0 = np.array(x0)
    P0 = np.array(P0)
    Q = np.array(Q)
    R = np.array(R)

    "out initialization"
    # Use np.zeros directly which returns arrays
    x = np.zeros(shape=(n_time, n_state))
    x[0, :] = x0 # Assign initial state
    x_minus = np.zeros(shape=(n_time, n_state))
    x_minus[0, :] = x0 # Assign initial prediction

    # Factor errors - Store as list of arrays
    P = [np.zeros_like(P0) for _ in range(n_time)]
    P[0] = P0
    P_minus = [np.zeros_like(P0) for _ in range(n_time)]
    P_minus[0] = P0

    # Kalman gains - Store as list of arrays (or determine shape if fixed)
    # Assuming K shape is (n_state, n_measurements_available)
    # Size might vary, so list of arrays is safer
    K = [None] * n_time # Initialize with None or appropriate zeros

    for i in range(1, n_time):
        ix = np.where(~np.isnan(z[i, :]))[0]

        if len(ix) == 0:
            x_prev_col = x[i-1, :].reshape(-1, 1)
            u_col = u[i, :].reshape(-1, 1)
            x_minus_pred = A @ x_prev_col + B @ u_col
            x_minus[i, :] = x_minus_pred.flatten()
            P_minus_raw = A @ P[i-1] @ A.T + Q
            p_jitter = np.eye(P_minus_raw.shape[0]) * 1e-6 # Small jitter value
            P_minus[i] = P_minus_raw + p_jitter
            x[i, :] = x_minus[i, :]
            P[i] = P_minus[i]
            K[i] = np.zeros((n_state, H.shape[0]))
            continue

        z_t = z[i, ix]
        H_t = H[ix, :]
        R_t = R[np.ix_(ix, ix)]

        x_prev_col = x[i-1, :].reshape(-1, 1)
        u_col = u[i, :].reshape(-1, 1)

        "prediction step"
        x_minus_pred = A @ x_prev_col + B @ u_col
        x_minus[i, :] = x_minus_pred.flatten()

        P_minus_raw = A @ P[i-1] @ A.T + Q
        p_jitter = np.eye(P_minus_raw.shape[0]) * 1e-6 # Small jitter value
        P_minus[i] = P_minus_raw + p_jitter

        # DEBUG: 输出第1个时间步的预测步结果
        if i == 1:
            print(f"\n[KALMAN-DEBUG] 老代码 i={i} 预测步:")
            print(f"  x[{i-1},:] = {x[i-1, :]}")
            print(f"  u[{i},:] = {u[i, :]}")
            print(f"  x_minus[{i},:] = {x_minus[i, :]}")
            print(f"  P[{i-1}] diag = {np.diag(P[i-1])}")
            print(f"  P_minus_raw diag = {np.diag(P_minus_raw)}")
            print(f"  P_minus[{i}] diag = {np.diag(P_minus[i])}")

        "update step"
        innovation_cov = H_t @ P_minus[i] @ H_t.T + R_t
        jitter = np.eye(innovation_cov.shape[0]) * 1e-4

        # DEBUG: 输出第1个时间步的更新步中间结果
        if i == 1:
            print(f"\n[KALMAN-DEBUG] 老代码 i={i} 更新步:")
            print(f"  有效观测数: {len(ix)}")
            print(f"  z_t[:5] = {z_t[:5]}")
            print(f"  H_t @ x_minus (前5) = {(H_t @ x_minus[i, :].reshape(-1, 1)).flatten()[:5]}")
            x_minus_col = x_minus[i, :].reshape(-1, 1)
            z_t_col = z_t.reshape(-1, 1)
            innovation_temp = z_t_col - H_t @ x_minus_col
            print(f"  innovation[:5] = {innovation_temp.flatten()[:5]}")
            print(f"  innovation_cov diag[:5] = {np.diag(innovation_cov)[:5]}")
        
        try:
            # Use scipy.linalg.solve for better numerical stability
            K_t_effective = scipy.linalg.solve((innovation_cov + jitter).T, (P_minus[i] @ H_t.T).T, assume_a='pos').T

            n_obs = H.shape[0] # Total number of observables
            K_t_full = np.zeros((n_state, n_obs)) # Initialize full gain with zeros
            K_t_full[:, ix] = K_t_effective # Fill in columns for observed variables
            K[i] = K_t_full # Store the full gain matrix

        except np.linalg.LinAlgError as svd_error:
            print(f"\n--- KF SVD Error Details (Iter {i}) ---")
            print(f"Timestamp: {timestamp[i]}")
            print(f"Available observation indices (ix): {ix}")
            print(f"Shape of H_t: {H_t.shape}")
            print(f"Shape of P_minus[{i}]: {P_minus[i].shape}")
            print(f"Shape of R_t: {R_t.shape}")
            print(f"Innovation Cov Matrix (before jitter):\n{innovation_cov}")
            cond_innov_cov = np.linalg.cond(innovation_cov)
            print(f"Innovation Cov Condition Number (before jitter): {cond_innov_cov}")
            raise # Re-raise the exception

        x_minus_col = x_minus[i, :].reshape(-1, 1)
        z_t_col = z_t.reshape(-1, 1)
        innovation = z_t_col - H_t @ x_minus_col

        x_updated = x_minus_col + K_t_effective @ innovation
        x[i, :] = x_updated.flatten()

        I_mat = np.eye(n_state)
        P[i] = (I_mat - K_t_effective @ H_t) @ P_minus[i]
        # Symmetrize P
        P[i] = (P[i] + P[i].T) / 2.0

        # DEBUG: 输出第1个时间步的最终滤波结果
        if i == 1:
            print(f"\n[KALMAN-DEBUG] 老代码 i={i} 滤波结果:")
            print(f"  K_t_effective[0,:] = {K_t_effective[0, :]}")  # 只显示第一行
            print(f"  x[{i},:] = {x[i, :]}")
            print(f"  P[{i}] diag = {np.diag(P[i])}")

    x = pd.DataFrame(data=x, index=Z.index, columns=state_names)
    x_minus = pd.DataFrame(data=x_minus, index=Z.index, columns=state_names)

    
    return KalmanFilterResultsWrapper(x_minus=x_minus, x=x, z=Z, Kalman_gain=K, P=P, P_minus=P_minus, state_names = state_names, A=A)

def FIS(res_KF):
    N = len(res_KF.x.index)
    n_state = len(res_KF.x.columns)
    # Convert inputs to arrays
    x = np.array(res_KF.x)
    x_minus = np.array(res_KF.x_minus)
    # P and P_minus are lists of arrays from KF
    P = res_KF.P
    P_minus = res_KF.P_minus
    A = np.array(res_KF.A)

    # Initialize smoothed results as arrays
    x_sm = np.zeros((N, n_state))
    x_sm[N-1, :] = x[N-1, :]

    P_sm = [np.zeros_like(P[0]) for _ in range(N)]
    P_sm[N-1] = P[N-1]

    J = [None] * (N - 1) # Smoother gains

    for i in reversed(range(N-1)):
        try:
            cond_p_minus = np.linalg.cond(P_minus[i+1])
            if cond_p_minus > 1e14:
                 print(f"  [FIS Smoother Iter {i}] WARNING: P_minus[{i+1}] is ill-conditioned (Cond: {cond_p_minus:.2e})!")

            # Use scipy.linalg.solve for better numerical stability
            try:
                J_k = scipy.linalg.solve(P_minus[i+1].T, (P[i] @ A.T).T, assume_a='pos').T
            except scipy.linalg.LinAlgError:
                # Fallback to pseudo-inverse if solve fails
                P_minus_inv = np.linalg.pinv(P_minus[i+1])
                J_k = P[i] @ A.T @ P_minus_inv
            J[i] = J_k
        except np.linalg.LinAlgError as inv_error:
            print(f"  [FIS Smoother Iter {i}] Error inverting P_minus[{i+1}] for smoother: {inv_error}")
            print(f"  [FIS Smoother Iter {i}] P_minus[{i+1}] on error:\n{P_minus[i+1]}")
            raise

        delta_P = P_sm[i+1] - P_minus[i+1]
        P_update_term = J_k @ delta_P @ J_k.T

        P_sm[i] = P[i] + P_update_term

        x_col = x[i, :].reshape(-1, 1)
        x_sm_next_col = x_sm[i+1, :].reshape(-1, 1)
        x_minus_next_col = x_minus[i+1, :].reshape(-1, 1)
        delta_x = x_sm_next_col - x_minus_next_col
        x_update_term = J_k @ delta_x

        x_sm_updated = x_col + x_update_term
        x_sm[i, :] = x_sm_updated.flatten()


    x_sm = pd.DataFrame(data=x_sm, index=res_KF.x.index, columns=res_KF.x.columns)
    
    return SKFResultsWrapper(x_sm=x_sm, P_sm=P_sm,z=res_KF.z)
        
class SKFResultsWrapper():
    def __init__(self, x_sm, P_sm, z):
        self.x_sm = x_sm
        self.P_sm = P_sm
        self.z = z
    

def EMstep(res_SKF, n_shocks):
    """Performs the M-step of the EM algorithm, updating parameters.

    Handles NaNs in observables when calculating R.
    Assumes _calculate_factor_loadings handles NaNs for Lambda.
    Might still face issues if _calculate_shock_matrix produces non-PSD Q.
    """
    f = res_SKF.x_sm # Smoothed factors (DataFrame, n_time x n_factors)
    y = res_SKF.z    # Original centered observables (DataFrame, n_time x n_obs, with NaNs)
    n_obs = y.shape[1]
    n_time = y.shape[0]
    n_factors = f.shape[1]

 
    # Calculate Lambda (Factor Loadings)
    # Lambda = funcs.calculate_factor_loadings(y, f) # Old call
    Lambda = calculate_factor_loadings(y, f) # New call to internal helper

    # Calculate A (Prediction Matrix)
    # A = funcs.calculate_prediction_matrix(f) # Old call
    A = _calculate_prediction_matrix(f) # New call to internal helper

    # Calculate B (Shock Matrix) and Q (Process Noise Covariance)
    # B, Q = funcs.calculate_shock_matrix(f, A, n_shocks) # Old call
    B, Q = _calculate_shock_matrix(f, A, n_shocks) # New call to internal helper

    # Use Lambda just calculated
    # R_diag = E[(y_t - Lambda f_t)^2] for each variable
    
    if isinstance(Lambda, pd.DataFrame):
         Lambda_np = Lambda.to_numpy()
    else:
         Lambda_np = np.array(Lambda) # Ensure it's an array

    if np.isnan(Lambda_np).any():
        print("  [EMstep] 警告: 计算出的 Lambda 包含 NaN。R 的计算可能不准确或失败。")
        # 可以选择填充 Lambda 中的 NaN (例如，用 0 或列均值)，但这可能引入偏差
        # Lambda_np = np.nan_to_num(Lambda_np) # 简单的填充方法
        # 或者，在计算残差时跳过包含 NaN 的行/列

    if isinstance(f, pd.DataFrame):
         f_np = f.to_numpy()
    else:
         f_np = np.array(f)
         
    if isinstance(y, pd.DataFrame):
         y_np = y.to_numpy()
    else:
         y_np = np.array(y)

    # Calculate predicted values: Lambda @ f.T (shape: n_obs x n_time)
    try:
        if Lambda_np.shape[1] != f_np.shape[1]:
            print(f"  [EMstep] 警告: 维度不匹配 - Lambda列数={Lambda_np.shape[1]}, f列数={f_np.shape[1]}")
            print(f"    Lambda shape: {Lambda_np.shape}, f shape: {f_np.shape}")

            # 尝试修复：使用最小公共维度
            min_factors = min(Lambda_np.shape[1], f_np.shape[1])
            if min_factors > 0:
                print(f"    尝试使用前{min_factors}个因子进行计算...")
                Lambda_truncated = Lambda_np[:, :min_factors]
                f_truncated = f_np[:, :min_factors]  # [HOT] 修复：使用numpy切片而不是iloc
                predicted_y = Lambda_truncated @ f_truncated.T
                print(f"    维度修复成功")
            else:
                raise ValueError("无法修复维度不匹配，无有效因子维度")
        else:
            # 维度匹配，正常计算
            predicted_y = Lambda_np @ f_np.T # Shape: (n_obs, n_factors) @ (n_factors, n_time) -> (n_obs, n_time)
    except ValueError as ve_matmul:
         print(f"  [EMstep] 错误: 计算预测值时矩阵乘法失败 (Lambda @ f.T): {ve_matmul}")
         print(f"    Lambda shape: {Lambda_np.shape}, f shape: {f_np.shape}")
         # 如果失败，无法计算 R，可以返回当前的 R 或引发错误
         # For now, return an identity matrix as a fallback, but this is not ideal
         R_fallback = np.eye(n_obs) * 1e-6
         # 不再使用fallback值，直接抛出错误
         print(f"  [EMstep] 错误: R矩阵计算失败，无法继续EM步骤")
         raise RuntimeError("R矩阵计算失败，EM步骤无法完成")
    except Exception as e_predy:
         print(f"  [EMstep] 错误: 计算预测值时发生意外错误: {e_predy}")
         # 不再使用fallback值，直接抛出错误
         raise RuntimeError(f"EM步骤计算失败: {e_predy}")


    # Calculate residuals: y - predicted_y.T (shape: n_time x n_obs)
    residuals = y_np - predicted_y.T

    # Calculate variance of residuals for each observable, ignoring NaNs
    # np.nanvar computes variance along specified axis, ignoring NaNs
    R_diag = np.nanvar(residuals, axis=0) # axis=0 calculates variance for each column (observable)

    # Ensure R is positive definite (set floor for diagonal)
    R_diag_corrected = np.maximum(R_diag, 1e-7) # Use a small positive floor
    R = np.diag(R_diag_corrected)

    # Return updated parameters as NumPy arrays
    return EMstepResultsWrapper(Lambda=np.array(Lambda), A=np.array(A), B=np.array(B), Q=np.array(Q), R=np.array(R), x_sm=f, z=y)

class EMstepResultsWrapper():
    def __init__(self, Lambda, A, B, Q, R, x_sm, z):
        self.Lambda = Lambda
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.x_sm = x_sm
        self.z = z

    