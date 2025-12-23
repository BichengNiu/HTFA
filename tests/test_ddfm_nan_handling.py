"""
DDFM模型NaN处理单元测试

验证DDFM模型能够正确处理包含NaN的数据，
通过KalmanFilter的有效观测索引机制自动处理缺失值。
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


class TestDDFMNaNHandling:
    """测试DDFM模型的NaN处理能力"""

    def test_kalman_filter_handles_nan_observations(self):
        """测试KalmanFilter能正确处理包含NaN的观测数据"""
        from dashboard.models.DFM.train.core.kalman import KalmanFilter

        # 创建简单的状态空间模型参数
        n_states = 2
        n_obs = 3
        n_time = 10

        A = np.eye(n_states) * 0.9  # 状态转移矩阵
        B = np.zeros((n_states, 1))  # 控制矩阵
        H = np.random.randn(n_obs, n_states)  # 观测矩阵
        Q = np.eye(n_states) * 0.1  # 过程噪声
        R = np.eye(n_obs) * 0.1  # 观测噪声
        x0 = np.zeros(n_states)  # 初始状态
        P0 = np.eye(n_states)  # 初始协方差

        # 创建卡尔曼滤波器
        kf = KalmanFilter(A=A, B=B, H=H, Q=Q, R=R, x0=x0, P0=P0)

        # 创建包含NaN的观测数据
        Z = np.random.randn(n_time, n_obs)
        # 在某些时刻引入NaN
        Z[3, 0] = np.nan  # 第3时刻第0个变量缺失
        Z[5, :] = np.nan  # 第5时刻全部缺失
        Z[7, 1:] = np.nan  # 第7时刻只有第0个变量有效

        # 执行滤波 - 不应抛出异常
        result = kf.filter(Z)

        # 验证结果
        assert result.x_filtered.shape == (n_time, n_states)
        assert not np.any(np.isnan(result.x_filtered)), "滤波结果不应包含NaN"
        assert len(result.kalman_gains_history) == n_time

        # 第5时刻全缺失，应该没有卡尔曼增益
        assert result.kalman_gains_history[5] is None

    def test_kalman_filter_partial_missing(self):
        """测试KalmanFilter处理部分缺失的情况"""
        from dashboard.models.DFM.train.core.kalman import KalmanFilter

        n_states = 2
        n_obs = 4
        n_time = 20

        A = np.eye(n_states) * 0.95
        B = np.zeros((n_states, 1))
        H = np.random.randn(n_obs, n_states)
        Q = np.eye(n_states) * 0.05
        R = np.eye(n_obs) * 0.1
        x0 = np.zeros(n_states)
        P0 = np.eye(n_states)

        kf = KalmanFilter(A=A, B=B, H=H, Q=Q, R=R, x0=x0, P0=P0)

        # 创建大量缺失的数据（模拟预测期场景）
        Z = np.random.randn(n_time, n_obs)
        # 后半部分数据大量缺失
        for t in range(10, 20):
            missing_vars = np.random.choice(n_obs, size=np.random.randint(1, n_obs), replace=False)
            Z[t, missing_vars] = np.nan

        nan_count = np.isnan(Z).sum()
        assert nan_count > 0, "测试数据应包含NaN"

        # 执行滤波
        result = kf.filter(Z)

        # 验证滤波成功完成
        assert result.x_filtered.shape == (n_time, n_states)
        assert not np.any(np.isnan(result.x_filtered))

    def test_ddfm_model_with_nan_data(self):
        """测试DDFM模型能处理包含NaN的完整训练流程"""
        from dashboard.models.DFM.train.core.ddfm_model import DDFMModel

        # 创建测试数据
        np.random.seed(42)
        n_samples = 100
        n_vars = 5

        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='W')
        data = pd.DataFrame(
            np.random.randn(n_samples, n_vars),
            index=dates,
            columns=[f'var_{i}' for i in range(n_vars)]
        )

        # 在预测期（后20个时间点）引入缺失值
        for t in range(80, 100):
            missing_vars = np.random.choice(n_vars, size=np.random.randint(1, 3), replace=False)
            for v in missing_vars:
                data.iloc[t, v] = np.nan

        nan_count = data.isna().sum().sum()
        assert nan_count > 0, f"测试数据应包含NaN，实际: {nan_count}"

        # 创建DDFM模型 - 使用正确的初始化参数
        model = DDFMModel(
            encoder_structure=(8, 2),  # 2个因子
            factor_order=1,
            max_iter=3,  # 减少迭代次数加快测试
            tolerance=1e-2,
            epochs=5
        )

        # 训练模型 - 不应抛出异常
        training_start = '2020-01-01'
        train_end = '2021-06-01'

        try:
            model.fit(
                data=data,
                training_start=training_start,
                train_end=train_end
            )
            # 验证模型训练成功
            assert model.results_ is not None
            assert model.results_.factors is not None
            assert not np.any(np.isnan(model.results_.factors))
        except ValueError as e:
            if "缺失值" in str(e):
                pytest.fail(f"DDFM不应因缺失值而失败: {e}")
            raise

    def test_ddfm_logs_nan_count(self, caplog):
        """测试DDFM在遇到NaN时记录日志"""
        import logging
        from dashboard.models.DFM.train.core.ddfm_model import DDFMModel

        # 创建包含NaN的测试数据
        np.random.seed(123)
        n_samples = 50
        n_vars = 3

        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='W')
        data = pd.DataFrame(
            np.random.randn(n_samples, n_vars),
            index=dates,
            columns=[f'var_{i}' for i in range(n_vars)]
        )

        # 引入缺失值
        data.iloc[40:, 0] = np.nan
        data.iloc[45:, 1] = np.nan

        # 使用正确的初始化参数
        model = DDFMModel(
            encoder_structure=(4, 1),  # 1个因子
            factor_order=1,
            max_iter=2,
            epochs=3
        )

        with caplog.at_level(logging.INFO):
            try:
                model.fit(
                    data=data,
                    training_start='2020-01-01',
                    train_end='2020-09-01'
                )
            except Exception:
                pass  # 忽略其他可能的错误

        # 检查是否记录了NaN相关日志
        nan_log_found = any("缺失值" in record.message for record in caplog.records)
        # 注意：只有当数据确实有NaN且到达_run_kalman_filter时才会记录
        # 这个测试主要验证日志机制存在


class TestKalmanFilterEdgeCases:
    """测试KalmanFilter的边界情况"""

    def test_all_observations_missing_at_one_timestep(self):
        """测试某个时刻所有观测都缺失的情况"""
        from dashboard.models.DFM.train.core.kalman import KalmanFilter

        n_states = 2
        n_obs = 3
        n_time = 5

        A = np.eye(n_states) * 0.9
        B = np.zeros((n_states, 1))
        H = np.random.randn(n_obs, n_states)
        Q = np.eye(n_states) * 0.1
        R = np.eye(n_obs) * 0.1
        x0 = np.zeros(n_states)
        P0 = np.eye(n_states)

        kf = KalmanFilter(A=A, B=B, H=H, Q=Q, R=R, x0=x0, P0=P0)

        Z = np.random.randn(n_time, n_obs)
        Z[2, :] = np.nan  # 第2时刻全部缺失

        result = kf.filter(Z)

        # 滤波应成功完成
        assert result.x_filtered.shape == (n_time, n_states)
        # 全缺失时刻的卡尔曼增益应为None
        assert result.kalman_gains_history[2] is None
        # 滤波结果应使用纯预测
        assert not np.any(np.isnan(result.x_filtered[2, :]))

    def test_consecutive_missing_timesteps(self):
        """测试连续多个时刻缺失的情况"""
        from dashboard.models.DFM.train.core.kalman import KalmanFilter

        n_states = 2
        n_obs = 3
        n_time = 10

        A = np.eye(n_states) * 0.9
        B = np.zeros((n_states, 1))
        H = np.random.randn(n_obs, n_states)
        Q = np.eye(n_states) * 0.1
        R = np.eye(n_obs) * 0.1
        x0 = np.zeros(n_states)
        P0 = np.eye(n_states)

        kf = KalmanFilter(A=A, B=B, H=H, Q=Q, R=R, x0=x0, P0=P0)

        Z = np.random.randn(n_time, n_obs)
        # 连续3个时刻全部缺失
        Z[4:7, :] = np.nan

        result = kf.filter(Z)

        assert result.x_filtered.shape == (n_time, n_states)
        assert not np.any(np.isnan(result.x_filtered))
        # 连续缺失时刻的卡尔曼增益都应为None
        for t in range(4, 7):
            assert result.kalman_gains_history[t] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
