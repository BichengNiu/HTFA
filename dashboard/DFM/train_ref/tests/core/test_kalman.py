# -*- coding: utf-8 -*-
"""
卡尔曼滤波器单元测试

测试KalmanFilter类的正确性和数值稳定性
"""

import pytest
import numpy as np
import pandas as pd
from dashboard.DFM.train_ref.core.kalman import (
    KalmanFilter,
    KalmanFilterResult,
    KalmanSmootherResult,
    kalman_filter,
    kalman_smoother
)


class TestKalmanFilter:
    """卡尔曼滤波器测试"""

    @pytest.fixture
    def simple_system(self):
        """简单的1维系统"""
        n_time = 50
        n_states = 1
        n_obs = 1

        # 系统参数
        A = np.array([[0.9]])
        B = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[0.5]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])

        # 生成模拟数据
        np.random.seed(42)
        true_states = np.zeros((n_states, n_time))
        observations = np.zeros((n_obs, n_time))

        x = x0.copy()
        for t in range(n_time):
            # 状态转移
            x = A @ x + np.random.multivariate_normal(np.zeros(n_states), Q)
            true_states[:, t] = x

            # 观测
            z = H @ x + np.random.multivariate_normal(np.zeros(n_obs), R)
            observations[:, t] = z

        return {
            'A': A, 'B': B, 'H': H, 'Q': Q, 'R': R,
            'x0': x0, 'P0': P0,
            'true_states': true_states,
            'observations': observations,
            'n_time': n_time
        }

    def test_filter_basic(self, simple_system):
        """测试基础滤波功能"""
        kf = KalmanFilter(
            A=simple_system['A'],
            B=simple_system['B'],
            H=simple_system['H'],
            Q=simple_system['Q'],
            R=simple_system['R'],
            x0=simple_system['x0'],
            P0=simple_system['P0']
        )

        result = kf.filter(simple_system['observations'])

        # 验证结果结构
        assert isinstance(result, KalmanFilterResult)
        assert result.x_filtered.shape == (1, simple_system['n_time'])
        assert result.P_filtered.shape == (1, 1, simple_system['n_time'])
        assert isinstance(result.loglikelihood, float)

        # 验证数值合理性
        assert not np.any(np.isnan(result.x_filtered))
        assert result.loglikelihood < 0  # 对数似然应为负值

    def test_smoother_basic(self, simple_system):
        """测试基础平滑功能"""
        kf = KalmanFilter(
            A=simple_system['A'],
            B=simple_system['B'],
            H=simple_system['H'],
            Q=simple_system['Q'],
            R=simple_system['R'],
            x0=simple_system['x0'],
            P0=simple_system['P0']
        )

        filter_result = kf.filter(simple_system['observations'])
        smooth_result = kf.smooth(filter_result)

        # 验证结果结构
        assert isinstance(smooth_result, KalmanSmootherResult)
        assert smooth_result.x_smoothed.shape == (1, simple_system['n_time'])

        # 验证平滑误差小于等于滤波误差
        filter_error = np.mean((filter_result.x_filtered - simple_system['true_states']) ** 2)
        smooth_error = np.mean((smooth_result.x_smoothed - simple_system['true_states']) ** 2)
        assert smooth_error <= filter_error * 1.1

    def test_missing_observations(self, simple_system):
        """测试缺失观测处理"""
        Z = simple_system['observations'].copy()

        # 随机设置30%的观测为NaN
        np.random.seed(42)
        mask = np.random.rand(*Z.shape) < 0.3
        Z[mask] = np.nan

        kf = KalmanFilter(
            A=simple_system['A'],
            B=simple_system['B'],
            H=simple_system['H'],
            Q=simple_system['Q'],
            R=simple_system['R'],
            x0=simple_system['x0'],
            P0=simple_system['P0']
        )

        result = kf.filter(Z)

        # 验证滤波器不崩溃且结果有效
        assert not np.any(np.isnan(result.x_filtered))
        assert result.loglikelihood < 0
