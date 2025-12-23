# -*- coding: utf-8 -*-
"""
DDFM模型后端测试脚本

测试目标：
1. 验证DDFM模型训练流程
2. 验证因子提取和卡尔曼滤波
3. 验证预测生成逻辑
4. 验证评估指标计算

使用数据：data/DFM预处理数据.xlsx的前150行
默认参数：DDFM配置文件中的默认值
"""

import os
import sys
import time
import warnings

# 设置项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 忽略TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
import numpy as np


def load_test_data(data_path: str, n_rows: int = 150):
    """
    加载测试数据

    Args:
        data_path: 数据文件路径
        n_rows: 使用的行数

    Returns:
        tuple: (data, target_variable, predictor_vars)
    """
    print("=" * 50)
    print("加载测试数据")
    print("=" * 50)

    # 读取Excel文件
    df = pd.read_excel(data_path, index_col=0)

    # 转换索引为日期时间
    df.index = pd.to_datetime(df.index)

    # 按日期升序排列（原始数据是逆序的）
    df = df.sort_index()

    # 取前n_rows行
    df = df.iloc[:n_rows].copy()

    # 识别目标变量
    target_variable = '中国:工业增加值:规模以上工业企业:当月同比(1-2月拆分)'
    if target_variable not in df.columns:
        # 尝试查找类似的列
        for col in df.columns:
            if '工业增加值' in col and '规模以上' in col and '当月同比' in col:
                target_variable = col
                break

    if target_variable not in df.columns:
        raise ValueError(f"目标变量 {target_variable} 不在数据中")

    # 筛选有足够数据的列（缺失率<50%）
    valid_cols = []
    for col in df.columns:
        if col == target_variable:
            valid_cols.append(col)
            continue
        missing_rate = df[col].isna().sum() / len(df)
        if missing_rate < 0.5:  # 缺失率小于50%
            valid_cols.append(col)

    df = df[valid_cols].copy()

    # 用前向填充+后向填充处理缺失值（避免spline插值问题）
    df = df.ffill().bfill()

    # 获取预测变量列表（排除目标变量）
    predictor_vars = [col for col in df.columns if col != target_variable]

    print(f"数据形状: {df.shape}")
    print(f"日期范围: {df.index.min()} 到 {df.index.max()}")
    print(f"目标变量: {target_variable}")
    print(f"预测变量数: {len(predictor_vars)}")
    print()

    return df, target_variable, predictor_vars


def run_ddfm_test():
    """
    运行DDFM模型测试
    """
    print("=" * 60)
    print("DDFM模型后端测试")
    print("=" * 60)
    print()

    # 数据路径
    data_path = os.path.join(PROJECT_ROOT, 'data', 'DFM预处理数据.xlsx')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    # 加载数据
    df, target_variable, predictor_vars = load_test_data(data_path, n_rows=150)

    # 数据切分：前100行训练期，后50行观察期
    train_size = 100
    train_data = df.iloc[:train_size]
    obs_data = df.iloc[train_size:]

    training_start = train_data.index.min().strftime('%Y-%m-%d')
    train_end = train_data.index.max().strftime('%Y-%m-%d')
    validation_start = obs_data.index.min().strftime('%Y-%m-%d')
    validation_end = obs_data.index.max().strftime('%Y-%m-%d')

    print("=" * 50)
    print("时间段划分")
    print("=" * 50)
    print(f"训练期: {training_start} 到 {train_end} ({len(train_data)} 个样本)")
    print(f"观察期: {validation_start} 到 {validation_end} ({len(obs_data)} 个样本)")
    print()

    # 导入DDFM训练函数
    from dashboard.models.DFM.train.training.model_ops import (
        train_ddfm_with_forecast,
        evaluate_model_performance
    )

    # 准备数据
    predictor_data = df[predictor_vars]
    target_data = df[target_variable]

    # 定义进度回调函数
    def progress_callback(message: str):
        print(message)

    # DDFM默认参数
    encoder_structure = (16, 4)  # 4个因子

    print("=" * 50)
    print("开始DDFM训练")
    print("=" * 50)
    print(f"编码器结构: {encoder_structure}")
    print(f"因子数: {encoder_structure[-1]}")
    print()

    start_time = time.time()

    try:
        # 训练DDFM模型（使用默认参数）
        model_result = train_ddfm_with_forecast(
            predictor_data=predictor_data,
            target_data=target_data,
            encoder_structure=encoder_structure,
            training_start=training_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            observation_end=None,  # 无额外观察期
            # 以下为默认参数
            decoder_structure=None,
            use_bias=True,
            factor_order=2,
            lags_input=0,
            batch_norm=True,
            activation='relu',
            learning_rate=0.005,
            optimizer='Adam',
            decay_learning_rate=True,
            epochs=100,
            batch_size=100,
            max_iter=200,
            tolerance=0.0005,
            display_interval=10,
            seed=3,
            progress_callback=progress_callback
        )

        training_time = time.time() - start_time

        print()
        print("=" * 50)
        print("训练完成")
        print("=" * 50)
        print(f"训练时间: {training_time:.2f} 秒")
        print(f"收敛状态: {model_result.converged}")
        print(f"迭代次数: {model_result.iterations}")
        print(f"因子形状: {model_result.factors.shape if model_result.factors is not None else 'None'}")
        print()

        # 断言验证模型结果
        assert model_result.factors is not None, "因子矩阵不应为None"
        assert model_result.factors.shape[0] == encoder_structure[-1], f"因子数应为{encoder_structure[-1]}"
        assert model_result.H is not None, "观测矩阵H不应为None"
        assert model_result.A is not None, "状态转移矩阵A不应为None"

        # 评估模型
        print("=" * 50)
        print("模型评估")
        print("=" * 50)

        metrics = evaluate_model_performance(
            model_result=model_result,
            target_data=target_data,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            observation_end=None,
            alignment_mode='next_month'
        )

        print(f"训练期 RMSE: {metrics.is_rmse:.4f}")
        print(f"训练期 Win Rate: {metrics.is_win_rate:.2f}%")
        print(f"观察期 RMSE: {metrics.oos_rmse:.4f}")
        print(f"观察期 Win Rate: {metrics.oos_win_rate:.2f}%")
        print()

        # 断言验证评估指标
        assert metrics.is_rmse < np.inf, "训练期RMSE不应为无穷大"
        assert metrics.oos_rmse < np.inf, "观察期RMSE不应为无穷大"
        assert not np.isnan(metrics.is_win_rate), "训练期Win Rate不应为NaN"
        assert not np.isnan(metrics.oos_win_rate), "观察期Win Rate不应为NaN"
        assert 0 <= metrics.is_win_rate <= 100, "训练期Win Rate应在0-100之间"
        assert 0 <= metrics.oos_win_rate <= 100, "观察期Win Rate应在0-100之间"

        # 检查预测结果
        print("=" * 50)
        print("预测结果检查")
        print("=" * 50)

        if model_result.forecast_is is not None:
            print(f"训练期预测: {len(model_result.forecast_is)} 个数据点")
            print(f"  范围: [{model_result.forecast_is.min():.4f}, {model_result.forecast_is.max():.4f}]")
            # 断言验证训练期预测
            assert len(model_result.forecast_is) == train_size, f"训练期预测数应为{train_size}"
        else:
            print("训练期预测: None")
            raise AssertionError("训练期预测不应为None")

        if model_result.forecast_oos is not None:
            print(f"观察期预测: {len(model_result.forecast_oos)} 个数据点")
            print(f"  范围: [{model_result.forecast_oos.min():.4f}, {model_result.forecast_oos.max():.4f}]")
            # 断言验证观察期预测
            assert len(model_result.forecast_oos) == len(obs_data), f"观察期预测数应为{len(obs_data)}"
        else:
            print("观察期预测: None")
            raise AssertionError("观察期预测不应为None")

        print()
        print("=" * 50)
        print("测试完成")
        print("=" * 50)

        return True

    except Exception as e:
        print()
        print("=" * 50)
        print("测试失败")
        print("=" * 50)
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_ddfm_test()
    sys.exit(0 if success else 1)
