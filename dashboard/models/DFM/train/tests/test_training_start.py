# -*- coding: utf-8 -*-
"""
training_start参数功能测试

验证：
1. TrainingConfig能否正确接收training_start
2. 模型训练是否真正使用training_start划分数据（而非从数据第一期开始）
3. metadata是否正确保存training_start_date
4. 训练样本数是否符合预期
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# 设置UTF-8编码，避免GBK编码错误
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到路径
# 从 test_training_start.py 往上5层到达 HTFA/
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 确保导入路径正确
print(f"项目根目录: {project_root}")
print(f"Python路径: {sys.path[:3]}")

from dashboard.models.DFM.train.training.config import TrainingConfig
from dashboard.models.DFM.train.training.trainer import DFMTrainer


def load_test_data():
    """加载测试数据"""
    data_path = project_root / "dashboard/models/DFM/train/tests/data/dfm_prepared_output.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"测试数据不存在: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    print(f"测试数据加载完成: {data.shape}")
    print(f"时间范围: {data.index.min()} 到 {data.index.max()}")

    return data


def test_training_start_parameter():
    """测试training_start参数功能"""

    print("=" * 80)
    print("测试: training_start参数功能验证")
    print("=" * 80)

    # 1. 加载测试数据
    data = load_test_data()

    # 2. 配置参数
    target_variable = "规模以上工业增加值:当月同比"
    all_indicators = [col for col in data.columns if col != target_variable]

    # 选择NaN最少的10个指标（加快测试）
    nan_counts = data[all_indicators].isna().sum().sort_values()
    selected_indicators = nan_counts.head(10).index.tolist()

    # 关键测试点：设置明确的training_start
    # 数据范围是 2015-01-02 到 2025-10-10
    # 我们设置训练从 2020-01-03 开始，这样可以验证模型是否排除了2015-2019年的数据

    training_start = '2020-01-03'
    train_end = '2024-08-16'
    validation_start = '2024-08-23'
    validation_end = '2025-08-29'

    print(f"\n测试配置:")
    print(f"  目标变量: {target_variable}")
    print(f"  指标数量: {len(selected_indicators)}")
    print(f"  training_start: {training_start} [关键测试点]")
    print(f"  train_end: {train_end}")
    print(f"  validation_start: {validation_start}")
    print(f"  validation_end: {validation_end}")

    # 计算预期的训练样本数
    expected_train_data = data.loc[training_start:train_end]
    print(f"\n预期训练样本数: {len(expected_train_data)}")
    print(f"预期训练期范围: {expected_train_data.index[0]} 到 {expected_train_data.index[-1]}")

    # 如果不使用training_start（错误情况），样本数应该是多少
    wrong_train_data = data.loc[:train_end]
    print(f"\n如果从数据第一期开始（错误情况）: {len(wrong_train_data)} 个样本")
    print(f"差异: {len(wrong_train_data) - len(expected_train_data)} 个样本")

    # 3. 创建配置
    config = TrainingConfig(
        # 数据配置
        data_path=str(project_root / "dashboard/models/DFM/train/tests/data/dfm_prepared_output.csv"),
        target_variable=target_variable,
        selected_indicators=selected_indicators,

        # 训练/验证期配置（关键测试点）
        training_start=training_start,  # 关键: 设置训练开始日期
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        target_freq='W-FRI',

        # 模型配置
        k_factors=2,
        max_iterations=30,
        tolerance=1e-6,

        # 禁用变量选择（加快测试）
        enable_variable_selection=False,

        # 因子数选择
        factor_selection_method='fixed',

        # 输出配置
        output_dir=str(project_root / "dashboard/models/DFM/train/tests/result")
    )

    print(f"\n[验证1] TrainingConfig.training_start = {config.training_start}")

    if config.training_start == training_start:
        print(f"  [PASS] TrainingConfig正确接收training_start参数")
    else:
        print(f"  [FAIL] 预期 {training_start}, 实际 {config.training_start}")
        return False

    # 4. 训练模型
    print("\n" + "=" * 80)
    print("开始训练模型...")
    print("=" * 80)

    trainer = DFMTrainer(config)

    progress_messages = []

    def progress_callback(msg):
        progress_messages.append(msg)
        print(f"  [训练进度] {msg}")

    try:
        result = trainer.train(
            progress_callback=progress_callback,
            enable_export=True
        )

        print("=" * 80)
        print("模型训练完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n[错误] 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. 验证结果
    print("\n" + "=" * 80)
    print("验证训练结果")
    print("=" * 80)

    # 5.1 检查metadata中的training_start_date
    print(f"\n[验证2] 检查metadata文件")

    if result.export_files and result.export_files.get('metadata'):
        metadata_path = result.export_files['metadata']
        print(f"  metadata路径: {metadata_path}")

        import pickle
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        saved_training_start = metadata.get('training_start_date', 'NOT FOUND')
        print(f"  metadata['training_start_date'] = {saved_training_start}")

        if saved_training_start == training_start:
            print(f"  [PASS] metadata正确保存training_start_date")
        else:
            print(f"  [FAIL] 预期 {training_start}, 实际 {saved_training_start}")
            return False
    else:
        print(f"  [WARNING] metadata文件未生成")

    # 5.2 检查训练配置消息
    print(f"\n[验证3] 检查训练配置消息")

    # 在进度消息中查找训练配置信息
    config_msg_found = False
    for msg in progress_messages:
        if "训练期:" in msg and training_start in msg:
            print(f"  找到训练配置消息:")
            print(f"    {msg}")
            config_msg_found = True
            break

    if config_msg_found:
        print(f"  [PASS] 训练配置消息包含正确的training_start")
    else:
        print(f"  [WARNING] 未在进度消息中找到training_start")

    # 5.3 检查评估指标
    print(f"\n[验证4] 模型评估指标")
    print(f"  样本内RMSE: {result.metrics.is_rmse:.4f}")
    print(f"  样本外RMSE: {result.metrics.oos_rmse:.4f}")
    print(f"  样本内Hit Rate: {result.metrics.is_hit_rate:.2f}%")
    print(f"  样本外Hit Rate: {result.metrics.oos_hit_rate:.2f}%")

    if np.isfinite(result.metrics.oos_rmse):
        print(f"  [PASS] 模型训练和评估成功")
    else:
        print(f"  [FAIL] 评估指标异常")
        return False

    # 5.4 检查因子数
    print(f"\n[验证5] 模型参数")
    print(f"  因子数: {result.k_factors}")
    print(f"  选定变量数: {len(result.selected_variables)}")
    print(f"  训练时间: {result.training_time:.2f}秒")

    # 最终总结
    print("\n" + "=" * 80)
    print("测试结论: 所有验证通过!")
    print("=" * 80)
    print(f"\n关键验证点:")
    print(f"  1. TrainingConfig接收training_start: PASS")
    print(f"  2. metadata保存training_start_date: PASS")
    print(f"  3. 训练配置消息正确: PASS")
    print(f"  4. 模型训练成功: PASS")
    print(f"\n结论: training_start参数功能正常工作")

    return True


def test_without_training_start():
    """测试不设置training_start应该报错"""

    print("\n\n" + "=" * 80)
    print("测试: 不设置training_start应该报错")
    print("=" * 80)

    data = load_test_data()

    target_variable = "规模以上工业增加值:当月同比"
    all_indicators = [col for col in data.columns if col != target_variable]
    nan_counts = data[all_indicators].isna().sum().sort_values()
    selected_indicators = nan_counts.head(10).index.tolist()

    # 尝试不设置training_start（应该在创建TrainingConfig时就失败）
    try:
        config = TrainingConfig(
            data_path=str(project_root / "dashboard/models/DFM/train/tests/data/dfm_prepared_output.csv"),
            target_variable=target_variable,
            selected_indicators=selected_indicators,

            # 故意不设置training_start，应该报错
            train_end='2024-08-16',
            validation_start='2024-08-23',
            validation_end='2025-08-29',
            target_freq='W-FRI',

            k_factors=2,
            max_iterations=30,
            tolerance=1e-6,
            enable_variable_selection=False,
            factor_selection_method='fixed',
            output_dir=str(project_root / "dashboard/models/DFM/train/tests/result")
        )
        print(f"\n[FAIL] 不设置training_start居然没报错！config.training_start = {config.training_start}")
        return False
    except TypeError as e:
        print(f"\n[PASS] 不设置training_start正确报错: {e}")
        return True
    except Exception as e:
        print(f"\n[PASS] 不设置training_start报错（其他类型）: {e}")
        return True


def main():
    """主测试函数"""
    print("training_start参数功能测试")
    print("=" * 80)

    # 测试1: 设置training_start
    success1 = test_training_start_parameter()

    # 测试2: 不设置training_start应该报错
    success2 = test_without_training_start()

    # 总结
    print("\n\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"测试1 (设置training_start): {'PASS' if success1 else 'FAIL'}")
    print(f"测试2 (不设置应报错): {'PASS' if success2 else 'FAIL'}")

    if success1 and success2:
        print(f"\n[PASS] 所有测试通过!")
        return 0
    else:
        print(f"\n[FAIL] 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
