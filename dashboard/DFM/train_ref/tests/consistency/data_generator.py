# -*- coding: utf-8 -*-
"""
模拟数据生成器

生成符合DFM理论的模拟时间序列数据,用于算法验证测试。

状态空间模型:
    因子过程: F_t = A * F_{t-1} + eta_t,  eta_t ~ N(0, Q)
    观测方程: Z_t = Lambda * F_t + eps_t,  eps_t ~ N(0, R)

关键特性:
- 固定随机种子(SEED=42)确保可重现性
- 已知真实参数,便于验证估计精度
- 支持多种数据规模和因子数配置
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional


# 全局固定随机种子
DFM_SEED = 42


@dataclass
class DFMSimulationConfig:
    """DFM模拟配置"""
    n_time: int                # 时间点数
    n_obs: int                 # 观测变量数
    n_factors: int             # 因子数
    ar_coef: float = 0.8       # AR(1)系数(单因子)或对角元(多因子)
    noise_std: float = 0.3     # 过程噪声标准差
    obs_noise_std: float = 0.5 # 观测噪声标准差
    missing_rate: float = 0.0  # 缺失值比例
    seed: int = DFM_SEED       # 随机种子


@dataclass
class DFMSimulationResult:
    """DFM模拟结果"""
    # 观测数据
    Z: pd.DataFrame           # 观测数据 (n_time × n_obs)

    # 真实参数(用于验证估计精度)
    true_factors: pd.DataFrame  # 真实因子 (n_time × n_factors)
    true_Lambda: np.ndarray     # 真实载荷矩阵 (n_obs × n_factors)
    true_A: np.ndarray          # 真实转移矩阵 (n_factors × n_factors)
    true_Q: np.ndarray          # 真实过程噪声协方差 (n_factors × n_factors)
    true_R: np.ndarray          # 真实观测噪声协方差 (n_obs × n_obs)

    # 配置信息
    config: DFMSimulationConfig


class DFMDataGenerator:
    """DFM模拟数据生成器"""

    def __init__(self, config: DFMSimulationConfig):
        """初始化生成器

        Args:
            config: 模拟配置
        """
        self.config = config

        # 设置随机种子确保可重现性
        np.random.seed(config.seed)
        import random
        random.seed(config.seed)

    def generate(self) -> DFMSimulationResult:
        """生成DFM模拟数据

        Returns:
            DFMSimulationResult: 模拟结果(包含观测数据和真实参数)
        """
        # 生成真实参数
        true_Lambda = self._generate_loading_matrix()
        true_A = self._generate_transition_matrix()
        true_Q = self._generate_process_noise_cov()
        true_R = self._generate_obs_noise_cov()

        # 生成因子序列
        true_factors = self._generate_factor_sequence(true_A, true_Q)

        # 生成观测数据
        Z = self._generate_observations(true_factors, true_Lambda, true_R)

        # 注入缺失值(如果需要)
        if self.config.missing_rate > 0:
            Z = self._inject_missing_values(Z)

        return DFMSimulationResult(
            Z=Z,
            true_factors=true_factors,
            true_Lambda=true_Lambda,
            true_A=true_A,
            true_Q=true_Q,
            true_R=true_R,
            config=self.config
        )

    def _generate_loading_matrix(self) -> np.ndarray:
        """生成因子载荷矩阵Lambda

        策略: 稀疏结构,每个变量主要受1-2个因子影响

        Returns:
            np.ndarray: 载荷矩阵 (n_obs × n_factors)
        """
        n_obs = self.config.n_obs
        n_factors = self.config.n_factors

        Lambda = np.zeros((n_obs, n_factors))

        # 为每个变量分配主导因子
        vars_per_factor = n_obs // n_factors

        for i in range(n_obs):
            # 主导因子
            主导因子 = i // vars_per_factor
            if 主导因子 >= n_factors:
                主导因子 = n_factors - 1

            # 主导因子载荷: 较大值
            Lambda[i, 主导因子] = np.random.uniform(0.7, 1.5)

            # 其他因子载荷: 较小值或0
            for j in range(n_factors):
                if j != 主导因子:
                    if np.random.rand() < 0.3:  # 30%概率有次要因子
                        Lambda[i, j] = np.random.uniform(0.1, 0.4)

        return Lambda

    def _generate_transition_matrix(self) -> np.ndarray:
        """生成状态转移矩阵A

        策略: 对角占优的AR结构,确保平稳性

        Returns:
            np.ndarray: 转移矩阵 (n_factors × n_factors)
        """
        n_factors = self.config.n_factors
        ar_coef = self.config.ar_coef

        if n_factors == 1:
            # 单因子: AR(1)
            A = np.array([[ar_coef]])
        else:
            # 多因子: 对角占优,特征值 < 1确保平稳性
            A = np.eye(n_factors) * ar_coef

            # 添加少量非对角元素
            for i in range(n_factors):
                for j in range(n_factors):
                    if i != j and np.random.rand() < 0.2:  # 20%概率
                        A[i, j] = np.random.uniform(-0.1, 0.1)

            # 确保最大特征值 < 1
            eigenvalues = np.linalg.eigvals(A)
            max_eigenvalue = np.max(np.abs(eigenvalues))
            if max_eigenvalue >= 1:
                A = A * 0.9 / max_eigenvalue

        return A

    def _generate_process_noise_cov(self) -> np.ndarray:
        """生成过程噪声协方差矩阵Q

        策略: 对角矩阵,方差较小

        Returns:
            np.ndarray: 过程噪声协方差 (n_factors × n_factors)
        """
        n_factors = self.config.n_factors
        noise_std = self.config.noise_std

        # 对角矩阵
        Q = np.eye(n_factors) * (noise_std ** 2)

        return Q

    def _generate_obs_noise_cov(self) -> np.ndarray:
        """生成观测噪声协方差矩阵R

        策略: 对角矩阵,方差适中

        Returns:
            np.ndarray: 观测噪声协方差 (n_obs × n_obs)
        """
        n_obs = self.config.n_obs
        obs_noise_std = self.config.obs_noise_std

        # 对角矩阵,每个变量可以有不同的噪声水平
        noise_stddevs = np.random.uniform(
            obs_noise_std * 0.8,
            obs_noise_std * 1.2,
            n_obs
        )
        R = np.diag(noise_stddevs ** 2)

        return R

    def _generate_factor_sequence(
        self,
        A: np.ndarray,
        Q: np.ndarray
    ) -> pd.DataFrame:
        """生成因子时间序列

        使用状态空间方程: F_t = A * F_{t-1} + eta_t

        Args:
            A: 转移矩阵
            Q: 过程噪声协方差

        Returns:
            pd.DataFrame: 因子序列 (n_time × n_factors)
        """
        n_time = self.config.n_time
        n_factors = self.config.n_factors

        # 初始化因子矩阵
        F = np.zeros((n_time, n_factors))

        # 初始状态: 平稳分布(近似为0)
        F[0, :] = np.random.multivariate_normal(
            mean=np.zeros(n_factors),
            cov=Q
        )

        # 生成后续时间点
        for t in range(1, n_time):
            # F_t = A * F_{t-1} + eta_t
            eta_t = np.random.multivariate_normal(
                mean=np.zeros(n_factors),
                cov=Q
            )
            F[t, :] = A @ F[t-1, :] + eta_t

        # 转换为DataFrame
        factor_names = [f'Factor_{i+1}' for i in range(n_factors)]
        F_df = pd.DataFrame(
            F,
            columns=factor_names,
            index=pd.RangeIndex(n_time, name='time')
        )

        return F_df

    def _generate_observations(
        self,
        factors: pd.DataFrame,
        Lambda: np.ndarray,
        R: np.ndarray
    ) -> pd.DataFrame:
        """生成观测数据

        使用观测方程: Z_t = Lambda * F_t + eps_t

        Args:
            factors: 因子序列
            Lambda: 载荷矩阵
            R: 观测噪声协方差

        Returns:
            pd.DataFrame: 观测数据 (n_time × n_obs)
        """
        n_time = self.config.n_time
        n_obs = self.config.n_obs

        F = factors.values
        Z = np.zeros((n_time, n_obs))

        for t in range(n_time):
            # Z_t = Lambda * F_t + eps_t
            eps_t = np.random.multivariate_normal(
                mean=np.zeros(n_obs),
                cov=R
            )
            Z[t, :] = Lambda @ F[t, :] + eps_t

        # 转换为DataFrame
        var_names = [f'Var_{i+1}' for i in range(n_obs)]
        Z_df = pd.DataFrame(
            Z,
            columns=var_names,
            index=factors.index
        )

        return Z_df

    def _inject_missing_values(self, Z: pd.DataFrame) -> pd.DataFrame:
        """注入缺失值

        Args:
            Z: 原始观测数据

        Returns:
            pd.DataFrame: 包含缺失值的数据
        """
        Z_missing = Z.copy()
        missing_rate = self.config.missing_rate

        # 随机选择缺失位置
        n_total = Z.size
        n_missing = int(n_total * missing_rate)

        missing_indices = np.random.choice(
            n_total,
            size=n_missing,
            replace=False
        )

        # 将选中位置设为NaN
        Z_flat = Z_missing.values.ravel()
        Z_flat[missing_indices] = np.nan
        Z_missing = pd.DataFrame(
            Z_flat.reshape(Z.shape),
            columns=Z.columns,
            index=Z.index
        )

        return Z_missing


def generate_standard_datasets(output_dir: str = "fixtures"):
    """生成标准测试数据集

    生成5种标准配置的测试数据集:
    - small: 快速单元测试
    - medium: 标准集成测试
    - large: 性能和稳定性测试
    - single_factor: 边界情况测试
    - high_dim: 高维模型测试

    Args:
        output_dir: 输出目录
    """
    import os

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 配置定义
    configs = {
        'small': DFMSimulationConfig(
            n_time=50, n_obs=10, n_factors=2,
            ar_coef=0.8, noise_std=0.2, obs_noise_std=0.4
        ),
        'medium': DFMSimulationConfig(
            n_time=200, n_obs=30, n_factors=3,
            ar_coef=0.8, noise_std=0.3, obs_noise_std=0.5
        ),
        'large': DFMSimulationConfig(
            n_time=500, n_obs=50, n_factors=5,
            ar_coef=0.8, noise_std=0.3, obs_noise_std=0.5
        ),
        'single_factor': DFMSimulationConfig(
            n_time=100, n_obs=20, n_factors=1,
            ar_coef=0.8, noise_std=0.3, obs_noise_std=0.5
        ),
        'high_dim': DFMSimulationConfig(
            n_time=300, n_obs=100, n_factors=10,
            ar_coef=0.7, noise_std=0.3, obs_noise_std=0.5
        )
    }

    # 生成并保存数据集
    for name, config in configs.items():
        print(f"生成{name}数据集...")
        generator = DFMDataGenerator(config)
        result = generator.generate()

        # 保存为npz格式(包含所有数组和元数据)
        output_path = os.path.join(output_dir, f'{name}_dataset.npz')
        np.savez(
            output_path,
            Z=result.Z.values,
            true_factors=result.true_factors.values,
            true_Lambda=result.true_Lambda,
            true_A=result.true_A,
            true_Q=result.true_Q,
            true_R=result.true_R,
            # 元数据
            Z_columns=result.Z.columns.tolist(),
            factor_columns=result.true_factors.columns.tolist(),
            n_time=config.n_time,
            n_obs=config.n_obs,
            n_factors=config.n_factors
        )

        print(f"  保存到: {output_path}")
        print(f"  数据维度: Z{result.Z.shape}, F{result.true_factors.shape}")
        print(f"  参数维度: Lambda{result.true_Lambda.shape}, A{result.true_A.shape}")

    print(f"\n所有标准数据集已生成到{output_dir}/目录")


if __name__ == '__main__':
    # 生成标准测试数据集
    import sys
    import os

    # 设置输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fixtures_dir = os.path.join(script_dir, 'fixtures')

    print("=" * 60)
    print("DFM模拟数据生成器")
    print("=" * 60)
    print(f"随机种子: {DFM_SEED}")
    print(f"输出目录: {fixtures_dir}")
    print()

    generate_standard_datasets(fixtures_dir)

    print("\n生成完成!")
