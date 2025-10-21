# -*- coding: utf-8 -*-
"""
卡尔曼滤波一致性测试

对比 train_ref/core/kalman.py 与 train_model/DiscreteKalmanFilter.py
验证计算结果的一致性
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加项目路径 - 需要向上6层到达HTFA根目录
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入新代码
from dashboard.DFM.train_ref.core.kalman import KalmanFilter, kalman_filter, kalman_smoother

# 导入老代码
from dashboard.DFM.train_model.DiscreteKalmanFilter import KalmanFilter as OldKalmanFilter, FIS as OldFIS


def create_test_data():
    """创建测试数据"""
    np.random.seed(42)

    n_time = 50
    n_states = 3
    n_obs = 5

    # 状态空间参数
    A = np.array([[0.8, 0.1, 0.0],
                  [0.1, 0.7, 0.1],
                  [0.0, 0.1, 0.6]])

    B = np.eye(n_states)

    H = np.random.randn(n_obs, n_states) * 0.5

    Q = np.eye(n_states) * 0.1
    R = np.eye(n_obs) * 0.2

    x0 = np.zeros(n_states)
    P0 = np.eye(n_states)

    # 生成观测数据
    true_states = np.zeros((n_states, n_time))
    observations = np.zeros((n_obs, n_time))

    x = x0.copy()
    for t in range(n_time):
        x = A @ x + np.random.multivariate_normal(np.zeros(n_states), Q)
        true_states[:, t] = x
        observations[:, t] = H @ x + np.random.multivariate_normal(np.zeros(n_obs), R)

    U = np.zeros((n_states, n_time))

    return A, B, H, Q, R, x0, P0, observations, U, true_states


def test_kalman_filter_consistency():
    """测试卡尔曼滤波的一致性"""
    print("="*80)
    print("测试1: 卡尔曼滤波一致性")
    print("="*80)

    A, B, H, Q, R, x0, P0, Z, U, true_states = create_test_data()

    print(f"\n参数设置:")
    print(f"  状态维度: {A.shape[0]}")
    print(f"  观测维度: {H.shape[0]}")
    print(f"  时间步数: {Z.shape[1]}")

    # 新代码
    print("\n运行新代码 (train_ref)...")
    kf_new = KalmanFilter(A, B, H, Q, R, x0, P0)
    result_new = kf_new.filter(Z, U)

    # 老代码 - 需要DataFrame格式
    print("运行老代码 (train_model)...")
    state_names = [f'x{i}' for i in range(A.shape[0])]

    # 转换为DataFrame，老代码期望(n_time, n_features)格式
    Z_df = pd.DataFrame(Z.T, columns=[f'obs{i}' for i in range(Z.shape[0])])
    U_df = pd.DataFrame(U.T, columns=state_names)

    result_old = OldKalmanFilter(Z_df, U_df, A, B, H, state_names, x0, P0, Q, R)

    # 对比滤波状态
    print("\n对比滤波状态估计:")
    x_filt_new = result_new.x_filtered  # (n_states, n_time)
    x_filt_old = result_old.x.values.T  # DataFrame (n_time, n_states) -> (n_states, n_time)

    diff_filt = np.abs(x_filt_new - x_filt_old)
    max_diff_filt = np.max(diff_filt)
    mean_diff_filt = np.mean(diff_filt)

    print(f"  最大差异: {max_diff_filt:.10f}")
    print(f"  平均差异: {mean_diff_filt:.10f}")
    print(f"  相对误差: {mean_diff_filt / (np.abs(x_filt_old).mean() + 1e-10):.10f}")

    # 对比预测状态
    print("\n对比预测状态估计:")
    x_pred_new = result_new.x_predicted  # (n_states, n_time+1)
    x_pred_old = result_old.x_minus.values.T  # DataFrame (n_time, n_states) -> (n_states, n_time)

    # 新代码有n_time+1个预测，老代码只有n_time个，只比较前n_time个
    diff_pred = np.abs(x_pred_new[:, :x_pred_old.shape[1]] - x_pred_old)
    max_diff_pred = np.max(diff_pred)
    mean_diff_pred = np.mean(diff_pred)

    print(f"  最大差异: {max_diff_pred:.10f}")
    print(f"  平均差异: {mean_diff_pred:.10f}")

    # 对比似然 - 老代码没有返回loglikelihood，跳过这个比较
    print("\n注意: 老代码没有返回对数似然，跳过似然比较")
    print(f"  新代码对数似然: {result_new.loglikelihood:.6f}")

    # 判断是否通过
    threshold = 1e-6
    passed = (max_diff_filt < threshold and max_diff_pred < threshold)

    status = "通过" if passed else "失败"
    print(f"\n[{status}] 测试结果 (阈值: {threshold})")

    return passed


def test_kalman_smoother_consistency():
    """测试卡尔曼平滑的一致性"""
    print("\n" + "="*80)
    print("测试2: 卡尔曼平滑一致性")
    print("="*80)

    A, B, H, Q, R, x0, P0, Z, U, true_states = create_test_data()

    # 先运行滤波
    kf_new = KalmanFilter(A, B, H, Q, R, x0, P0)
    filter_result_new = kf_new.filter(Z, U)

    state_names = [f'x{i}' for i in range(A.shape[0])]

    # 转换为DataFrame
    Z_df = pd.DataFrame(Z.T, columns=[f'obs{i}' for i in range(Z.shape[0])])
    U_df = pd.DataFrame(U.T, columns=state_names)

    filter_result_old = OldKalmanFilter(Z_df, U_df, A, B, H, state_names, x0, P0, Q, R)

    # 新代码平滑
    print("\n运行新代码平滑...")
    smoother_result_new = kf_new.smooth(filter_result_new)

    # 老代码平滑
    print("运行老代码平滑...")
    smoother_result_old = OldFIS(filter_result_old)

    # 对比平滑状态
    print("\n对比平滑状态估计:")
    x_smooth_new = smoother_result_new.x_smoothed  # (n_states, n_time)
    x_smooth_old = smoother_result_old.x_sm.values.T  # DataFrame (n_time, n_states) -> (n_states, n_time)

    diff_smooth = np.abs(x_smooth_new - x_smooth_old)
    max_diff_smooth = np.max(diff_smooth)
    mean_diff_smooth = np.mean(diff_smooth)

    print(f"  最大差异: {max_diff_smooth:.10f}")
    print(f"  平均差异: {mean_diff_smooth:.10f}")
    print(f"  相对误差: {mean_diff_smooth / (np.abs(x_smooth_old).mean() + 1e-10):.10f}")

    # 对比平滑协方差
    print("\n对比平滑协方差:")
    P_smooth_new = smoother_result_new.P_smoothed  # (n_states, n_states, n_time)
    P_smooth_old = np.array(smoother_result_old.P_sm)  # list of arrays -> (n_time, n_states, n_states)

    # 转换老代码协方差格式 (n_time, n_states, n_states) -> (n_states, n_states, n_time)
    P_smooth_old_transposed = np.transpose(P_smooth_old, (1, 2, 0))

    diff_P = np.abs(P_smooth_new - P_smooth_old_transposed)
    max_diff_P = np.max(diff_P)
    mean_diff_P = np.mean(diff_P)

    print(f"  最大差异: {max_diff_P:.10f}")
    print(f"  平均差异: {mean_diff_P:.10f}")

    # 判断是否通过
    threshold = 1e-6
    passed = (max_diff_smooth < threshold and max_diff_P < threshold)

    status = "通过" if passed else "失败"
    print(f"\n[{status}] 测试结果 (阈值: {threshold})")

    return passed


def test_with_missing_data():
    """测试带缺失数据的情况"""
    print("\n" + "="*80)
    print("测试3: 带缺失数据的卡尔曼滤波")
    print("="*80)

    A, B, H, Q, R, x0, P0, Z, U, true_states = create_test_data()

    # 添加缺失数据
    Z_missing = Z.copy()
    missing_indices = np.random.choice(Z.shape[1], size=10, replace=False)
    for idx in missing_indices:
        Z_missing[:, idx] = np.nan

    print(f"\n添加了 {len(missing_indices)} 个时间步的缺失观测")

    # 新代码
    print("\n运行新代码...")
    kf_new = KalmanFilter(A, B, H, Q, R, x0, P0)
    result_new = kf_new.filter(Z_missing, U)

    # 老代码 - 需要DataFrame格式
    print("运行老代码...")
    state_names = [f'x{i}' for i in range(A.shape[0])]

    # 转换为DataFrame
    Z_missing_df = pd.DataFrame(Z_missing.T, columns=[f'obs{i}' for i in range(Z_missing.shape[0])])
    U_df = pd.DataFrame(U.T, columns=state_names)

    result_old = OldKalmanFilter(Z_missing_df, U_df, A, B, H, state_names, x0, P0, Q, R)

    # 对比
    print("\n对比滤波结果:")
    diff = np.abs(result_new.x_filtered - result_old.x.values.T)

    # 只对比非缺失时间步
    valid_indices = [i for i in range(Z.shape[1]) if i not in missing_indices]
    diff_valid = diff[:, valid_indices]

    max_diff = np.max(diff_valid)
    mean_diff = np.mean(diff_valid)

    print(f"  有效时间步最大差异: {max_diff:.10f}")
    print(f"  有效时间步平均差异: {mean_diff:.10f}")

    threshold = 1e-6
    passed = max_diff < threshold

    status = "通过" if passed else "失败"
    print(f"\n[{status}] 测试结果 (阈值: {threshold})")

    return passed


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("卡尔曼滤波一致性测试套件")
    print("对比 train_ref vs train_model")
    print("="*80)

    results = []

    try:
        results.append(("卡尔曼滤波", test_kalman_filter_consistency()))
    except Exception as e:
        print(f"\n[失败] 卡尔曼滤波测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("卡尔曼滤波", False))

    try:
        results.append(("卡尔曼平滑", test_kalman_smoother_consistency()))
    except Exception as e:
        print(f"\n[失败] 卡尔曼平滑测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("卡尔曼平滑", False))

    try:
        results.append(("缺失数据处理", test_with_missing_data()))
    except Exception as e:
        print(f"\n[失败] 缺失数据测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("缺失数据处理", False))

    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)

    for name, passed in results:
        status = "[通过]" if passed else "[失败]"
        print(f"  {name}: {status}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\n总计: {total_passed}/{total_tests} 测试通过")

    if total_passed == total_tests:
        print("\n>>> 所有测试通过！train_ref卡尔曼滤波与老代码一致。")
        return 0
    else:
        print("\n>>> 部分测试失败，需要检查差异。")
        return 1


if __name__ == "__main__":
    exit(main())
