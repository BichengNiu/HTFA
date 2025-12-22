# -*- coding: utf-8 -*-
"""
深度动态因子模型 (DDFM)

使用神经网络自编码器提取因子，复用项目的KalmanFilter
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Callable

from dashboard.models.DFM.train.core.models import DFMModelResult
from dashboard.models.DFM.train.utils.ddfm_utils import (
    mse_missing,
    convergence_checker,
    convert_decoder_to_numpy,
    get_transition_params,
    get_idio
)
from dashboard.models.DFM.train.utils.logger import get_logger

logger = get_logger(__name__)


class DDFMModel:
    """
    深度动态因子模型

    使用神经网络自编码器提取非线性因子，然后构建线性状态空间模型进行滤波和预测。
    训练采用MCMC迭代方法。
    """

    def __init__(
        self,
        encoder_structure: Tuple[int, ...] = (16, 4),
        decoder_structure: Optional[Tuple[int, ...]] = None,
        use_bias: bool = True,
        factor_order: int = 2,
        lags_input: int = 0,
        batch_norm: bool = True,
        activation: str = 'relu',
        learning_rate: float = 0.005,
        optimizer: str = 'Adam',
        decay_learning_rate: bool = True,
        epochs: int = 100,
        batch_size: int = 100,
        max_iter: int = 200,
        tolerance: float = 0.0005,
        display_interval: int = 10,
        seed: int = 3,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        optimize_cpu: bool = True,
        num_threads: Optional[int] = None
    ):
        """
        初始化DDFM模型

        Args:
            encoder_structure: 编码器层结构，最后一个数为因子数
            decoder_structure: 解码器层结构(None=对称单层线性)
            use_bias: 解码器最后一层是否使用偏置
            factor_order: 因子AR阶数(1或2)
            lags_input: 输入滞后期数
            batch_norm: 是否使用批量归一化
            activation: 激活函数
            learning_rate: 学习率
            optimizer: 优化器
            decay_learning_rate: 是否使用学习率衰减
            epochs: 每次MCMC迭代的epoch数
            batch_size: 批量大小
            max_iter: MCMC最大迭代次数
            tolerance: MCMC收敛阈值
            display_interval: 显示间隔
            seed: 随机种子
            progress_callback: 进度回调函数，签名(message: str, progress: float)
            optimize_cpu: 是否启用CPU多核优化
            num_threads: CPU线程数（None=自动检测）
        """
        # 保存progress_callback
        self.progress_callback = progress_callback

        # TensorFlow延迟导入
        try:
            from tensorflow import keras
            import tensorflow as tf
            self.keras = keras
            self.tf = tf

            # CPU多核优化
            if optimize_cpu:
                n_cpus = num_threads or os.cpu_count() or 4
                tf.config.threading.set_intra_op_parallelism_threads(n_cpus)
                tf.config.threading.set_inter_op_parallelism_threads(n_cpus)
                logger.info(f"TensorFlow CPU优化: 使用 {n_cpus} 个线程")

        except ImportError:
            raise ImportError(
                "DDFM需要TensorFlow。请安装: pip install tensorflow"
            )

        # 验证因子阶数
        if factor_order not in [1, 2]:
            raise ValueError('factor_order必须为1或2')

        self.n_factors = encoder_structure[-1]  # 因子数由编码器最后一层决定
        self.encoder_structure = encoder_structure
        self.decoder_structure = decoder_structure
        self.use_bias = use_bias
        self.factor_order = factor_order
        self.lags_input = lags_input
        self.batch_norm = batch_norm
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.decay_learning_rate = decay_learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.display_interval = display_interval
        self.seed = seed

        # 内部状态
        self.rng = np.random.RandomState(seed)
        self.initializer = self.keras.initializers.GlorotNormal(seed=seed)

        # 模型组件（训练后填充）
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.optimizer = None

        # 训练结果
        self.mean_z = None
        self.sigma_z = None
        self.factors = None
        self.factors_filtered = None
        self.eps = None
        self.loss_now = None
        self.state_space_dict = {}

        # 结果对象
        self.results_: Optional[DFMModelResult] = None

    def _report_progress(self, message: str, progress: float):
        """
        报告进度到callback和logger

        Args:
            message: 进度消息
            progress: 进度值 (0.0 - 1.0)
        """
        if self.progress_callback:
            self.progress_callback(message, progress)
        logger.info(message)

    def fit(
        self,
        data: pd.DataFrame,
        training_start: str,
        train_end: str
    ) -> DFMModelResult:
        """
        训练DDFM模型

        Args:
            data: 观测数据 (时间 × 变量)
            training_start: 训练期开始日期
            train_end: 训练期结束日期

        Returns:
            DFMModelResult: 与经典DFM兼容的结果对象
        """
        self._report_progress("开始DDFM训练...", 0.0)
        self._report_progress(f"编码器结构: {self.encoder_structure}, 因子数: {self.n_factors}", 0.01)

        # 数据排序
        data = data.sort_index()

        # 切分训练期数据
        train_data = data.loc[training_start:train_end].copy()

        if train_data.empty:
            raise ValueError(
                f"训练期({training_start}至{train_end})内没有数据。"
                f"数据时间范围：{data.index.min()}至{data.index.max()}"
            )

        self._report_progress("数据标准化处理中...", 0.02)

        # 数据标准化
        self.mean_z = train_data.mean().values
        self.sigma_z = train_data.std().values
        # 避免除零
        self.sigma_z[self.sigma_z == 0] = 1.0

        normalized_data = (train_data - self.mean_z) / self.sigma_z

        # 记录缺失值位置
        self.bool_miss = normalized_data.isnull()[self.lags_input:].values
        self.bool_no_miss = ~self.bool_miss

        # 创建数据副本
        self.data_mod_only_miss = normalized_data.copy()
        self.data_mod = normalized_data.copy()
        self.z_actual = normalized_data[self.lags_input:].values

        # 构建模型（_build_inputs会设置self.data_tmp）
        self._build_inputs(normalized_data)
        self._build_model(normalized_data.shape[1])

        # 创建优化器
        self._create_optimizer()
        self._report_progress("数据准备完成", 0.05)

        # 预训练
        self._report_progress("预训练自编码器...", 0.05)
        self._pre_train()
        self._report_progress("预训练完成", 0.15)

        # MCMC训练
        self._report_progress("开始MCMC迭代训练...", 0.15)
        self._train()

        # 构建状态空间模型
        self._report_progress("构建状态空间模型...", 0.90)
        self._build_state_space()

        # 使用项目的KalmanFilter进行滤波
        self._report_progress("执行卡尔曼滤波...", 0.95)
        self._run_kalman_filter(data, training_start, train_end)

        self._report_progress("DDFM训练完成", 1.0)
        return self.results_

    def _build_inputs(self, data: pd.DataFrame, interpolate: bool = True) -> None:
        """构建输入数据（包含滞后变量）"""
        new_dict = {}
        for col_name in data.columns:
            new_dict[col_name] = data[col_name]
            for lag in range(self.lags_input):
                new_dict[f'{col_name}_lag{lag + 1}'] = data[col_name].shift(lag + 1)

        self.data_tmp = pd.DataFrame(new_dict, index=data.index)
        self.data_tmp = self.data_tmp[self.lags_input:]

        if interpolate and self.data_tmp.isna().sum().sum() > 0:
            self.data_tmp.interpolate(method='spline', limit_direction='both', inplace=True, order=3)

    def _build_model(self, n_variables: int) -> None:
        """构建Keras自编码器模型"""
        layers = self.keras.layers

        # 编码器
        input_dim = int((self.lags_input + 1) * n_variables)
        inputs_ = self.keras.Input(shape=(input_dim,))

        if len(self.encoder_structure) > 1:
            encoded = layers.Dense(
                self.encoder_structure[0],
                activation=self.activation,
                bias_initializer='zeros',
                kernel_initializer=self.initializer
            )(inputs_)
            for j in self.encoder_structure[1:]:
                if self.batch_norm:
                    encoded = layers.BatchNormalization()(encoded)
                encoded = layers.Dense(
                    j,
                    activation=self.activation,
                    kernel_initializer=self.initializer,
                    bias_initializer='zeros'
                )(encoded)
        else:
            encoded = layers.Dense(
                self.encoder_structure[0],
                bias_initializer='zeros',
                kernel_initializer=self.initializer
            )(inputs_)

        self.encoder = self.keras.Model(inputs_, encoded)

        # 解码器
        latent_inputs = self.keras.Input(shape=(self.encoder_structure[-1],))
        if self.decoder_structure:
            decoded = layers.Dense(
                self.decoder_structure[0],
                activation=self.activation,
                kernel_initializer=self.initializer,
                bias_initializer='zeros'
            )(latent_inputs)
            for j in self.decoder_structure[1:]:
                decoded = layers.Dense(
                    j,
                    activation=self.activation,
                    kernel_initializer=self.initializer,
                    bias_initializer='zeros'
                )(decoded)
            output_ = layers.Dense(
                n_variables,
                bias_initializer='zeros',
                kernel_initializer=self.initializer,
                use_bias=self.use_bias
            )(decoded)
        else:
            output_ = layers.Dense(
                n_variables,
                bias_initializer='zeros',
                kernel_initializer=self.initializer,
                use_bias=self.use_bias
            )(latent_inputs)

        self.decoder = self.keras.Model(latent_inputs, output_)

        # 自编码器
        outputs_ = self.decoder(self.encoder(inputs_))
        self.autoencoder = self.keras.Model(inputs_, outputs_)

    def _create_optimizer(self) -> None:
        """创建优化器"""
        lr = self.learning_rate
        if self.decay_learning_rate:
            lr = self.keras.optimizers.schedules.ExponentialDecay(
                lr, decay_steps=self.epochs, decay_rate=0.96, staircase=True
            )

        if self.optimizer_name == 'SGD':
            self.optimizer = self.keras.optimizers.SGD(learning_rate=lr)
        elif self.optimizer_name == 'Adam':
            self.optimizer = self.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise KeyError(f"优化器必须是SGD或Adam，当前值: {self.optimizer_name}")

    def _pre_train(self, min_obs: int = 50, mult_epoch_pre: int = 1) -> None:
        """预训练自编码器"""
        self._build_inputs(self.data_mod, interpolate=False)

        if len(self.data_tmp.dropna()) >= min_obs:
            inpt_pre_train = self.data_tmp.dropna().values
            self.autoencoder.compile(optimizer=self.optimizer, loss='mse')
        else:
            self._build_inputs(self.data_mod)
            inpt_pre_train = self.data_tmp.dropna().values
            self.autoencoder.compile(optimizer=self.optimizer, loss=mse_missing)

        oupt_pre_train = self.data_tmp.dropna()[self.data_mod.columns].values

        self.autoencoder.fit(
            inpt_pre_train, oupt_pre_train,
            epochs=self.epochs * mult_epoch_pre,
            batch_size=self.batch_size,
            verbose=0
        )

    def _train(self) -> None:
        """MCMC迭代训练"""
        self.autoencoder.compile(optimizer=self.optimizer, loss=mse_missing)
        self._build_inputs(self.data_mod)

        prediction_iter = self.autoencoder.predict(self.data_tmp.values, verbose=0)
        self.data_mod_only_miss.values[self.lags_input:][self.bool_miss] = prediction_iter[self.bool_miss]

        self.eps = self.data_tmp[self.data_mod.columns].values - prediction_iter

        iter_count = 0
        not_converged = True
        prediction_prev_iter = None

        while not_converged and iter_count < self.max_iter:
            # 获取特质项分布
            phi, mu_eps, std_eps = get_idio(self.eps, self.bool_no_miss)

            # 减去条件AR特质项均值
            self.data_mod.values[self.lags_input + 1:] = (
                self.data_mod_only_miss.values[self.lags_input + 1:] -
                self.eps[:-1, :] @ phi
            )
            self.data_mod.values[:self.lags_input + 1] = self.data_mod_only_miss.values[:self.lags_input + 1]

            self._build_inputs(self.data_mod)

            # 验证协方差矩阵正定性
            if not np.all(std_eps > 0):
                invalid_indices = np.where(std_eps <= 0)[0]
                raise ValueError(
                    f"特质项标准差包含非正值，无法构建协方差矩阵。"
                    f"问题变量索引: {invalid_indices.tolist()}"
                )

            # 生成MC样本
            eps_draws = self.rng.multivariate_normal(
                mu_eps, np.diag(std_eps),
                (self.epochs, self.data_tmp.shape[0])
            )

            x_sim_den = np.zeros((
                eps_draws.shape[0], eps_draws.shape[1],
                eps_draws.shape[2] * (self.lags_input + 1)
            ))

            for i in range(self.epochs):
                x_sim_den[i, :, :] = self.data_tmp.values.copy()
                x_sim_den[i, :, :eps_draws[i, :, :].shape[1]] = (
                    x_sim_den[i, :, :eps_draws[i, :, :].shape[1]] - eps_draws[i, :, :]
                )
                self.autoencoder.fit(
                    x_sim_den[i, :, :], self.z_actual,
                    epochs=1, batch_size=self.batch_size, verbose=0
                )

            # 更新因子（确保转换为numpy数组）
            self.factors = np.array([
                self.encoder(x_sim_den[i, :, :]).numpy()
                for i in range(x_sim_den.shape[0])
            ])

            # 检查收敛（确保转换为numpy数组）
            prediction_iter = np.mean(np.array([
                self.decoder(self.factors[i, :, :]).numpy()
                for i in range(self.factors.shape[0])
            ]), axis=0)

            if iter_count > 1 and prediction_prev_iter is not None:
                delta, self.loss_now = convergence_checker(
                    prediction_prev_iter, prediction_iter, self.z_actual
                )

                # 计算当前进度 (15% - 90%)
                progress = 0.15 + (iter_count / self.max_iter) * 0.75

                if iter_count % self.display_interval == 0:
                    msg = f'MCMC迭代 {iter_count}/{self.max_iter}: loss={self.loss_now:.6f}, delta={delta:.6f}'
                    self._report_progress(msg, progress)

                if delta < self.tolerance:
                    not_converged = False
                    msg = f'收敛于迭代 {iter_count}/{self.max_iter}: loss={self.loss_now:.6f}'
                    self._report_progress(msg, 0.90)

            prediction_prev_iter = prediction_iter.copy()

            # 更新缺失值
            self.data_mod_only_miss.values[self.lags_input:][self.bool_miss] = prediction_iter[self.bool_miss]
            self.eps = self.data_mod_only_miss.values[self.lags_input:] - prediction_iter
            iter_count += 1

        if not_converged:
            self._report_progress(f"未在{self.max_iter}次迭代内收敛，继续处理...", 0.90)

        # 获取最后一层神经元（确保转换为numpy数组）
        if self.decoder_structure is None:
            self.last_neurons = self.factors
        else:
            decoder_for_last = self.keras.Model(
                self.decoder.input,
                self.decoder.get_layer(self.decoder.layers[-2].name).output
            )
            self.last_neurons = np.array([
                decoder_for_last(self.encoder(x_sim_den[i, :, :]).numpy()).numpy()
                for i in range(x_sim_den.shape[0])
            ])

    def _build_state_space(self) -> None:
        """从自编码器构建状态空间模型"""
        # 提取公共因子
        f_t = np.mean(self.factors, axis=0)
        eps_t = self.eps

        # 从解码器获取观测方程参数
        bs, H = convert_decoder_to_numpy(
            self.decoder, self.use_bias, self.factor_order,
            structure_decoder=self.decoder_structure
        )

        # 修正均值（加入偏置项）
        self.mean_z = self.mean_z + bs * self.sigma_z

        # 获取状态转移方程参数
        A, Q, mu_0, Sigma_0, x_t = get_transition_params(
            f_t, eps_t, factor_order=self.factor_order, bool_no_miss=self.bool_no_miss
        )

        # 观测噪声协方差（设为很小的值）
        R = np.eye(eps_t.shape[1]) * 1e-15

        # 保存状态空间参数
        self.state_space_dict["transition"] = {
            "A": A,
            "Q": Q,
            "mu_0": mu_0,
            "Sigma_0": Sigma_0
        }
        self.state_space_dict["measurement"] = {
            "H": H,
            "R": R
        }

    def _run_kalman_filter(
        self,
        full_data: pd.DataFrame,
        training_start: str,
        train_end: str
    ) -> None:
        """使用项目的KalmanFilter进行滤波"""
        from dashboard.models.DFM.train.core.kalman import KalmanFilter

        # 获取状态空间参数
        A = self.state_space_dict["transition"]["A"]
        Q = self.state_space_dict["transition"]["Q"]
        H = self.state_space_dict["measurement"]["H"]
        R = self.state_space_dict["measurement"]["R"]
        mu_0 = self.state_space_dict["transition"]["mu_0"]
        Sigma_0 = self.state_space_dict["transition"]["Sigma_0"]

        # 标准化全量数据
        full_normalized = (full_data - self.mean_z) / self.sigma_z

        # 处理缺失值
        Z = full_normalized.values.copy()
        nan_count = np.isnan(Z).sum()
        if nan_count > 0:
            raise ValueError(
                f"预测期数据包含{nan_count}个缺失值，DDFM无法处理预测期缺失数据。"
                "请确保预测变量在预测期内有完整观测值。"
            )

        # 创建控制矩阵B（零矩阵）
        n_states = A.shape[0]
        B = np.zeros((n_states, 1))

        # 创建卡尔曼滤波器
        kf = KalmanFilter(
            A=A,
            B=B,
            H=H,
            Q=Q,
            R=R,
            x0=mu_0,
            P0=Sigma_0
        )

        # 执行滤波
        kf_result = kf.filter(Z)

        # 提取因子（前n_factors列）
        factors = kf_result.x_filtered[:, :self.n_factors].T
        kalman_gains_history = kf_result.kalman_gains_history

        # 构建结果对象
        self.results_ = DFMModelResult(
            A=A,
            Q=Q,
            H=H,
            R=R,
            factors=factors,
            factors_smooth=factors,
            kalman_gains_history=kalman_gains_history,
            converged=self.loss_now is not None,
            iterations=self.max_iter,
            log_likelihood=-self.loss_now if self.loss_now else -np.inf
        )
