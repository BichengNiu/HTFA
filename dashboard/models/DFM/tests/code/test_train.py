# -*- coding: utf-8 -*-
"""
DFM训练模块后端测试接口

支持命令行参数和配置文件两种方式进行模型训练测试

使用示例:
    # 使用配置文件
    python dashboard/models/DFM/tests/code/test_train.py --config example_config.json

    # 使用命令行参数
    python dashboard/models/DFM/tests/code/test_train.py --data-path "../data/dfm_prepared_output.csv" --target "目标变量"

    # 快速测试模式（使用tests/data目录下的测试数据）
    python dashboard/models/DFM/tests/code/test_train.py --quick-test
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入训练模块
from dashboard.models.DFM.train import DFMTrainer, TrainingConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='DFM模型训练后端测试接口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 1. 使用配置文件
  python dashboard/models/DFM/tests/code/test_train.py --config example_config.json

  # 2. 快速测试模式（使用tests/data目录下的测试数据）
  python dashboard/models/DFM/tests/code/test_train.py --quick-test

  # 3. 使用命令行参数
  python dashboard/models/DFM/tests/code/test_train.py \\
    --data-path "../data/dfm_prepared_output.csv" \\
    --target "规模以上工业增加值:当月同比" \\
    --k-factors 3 \\
    --train-end "2023-12-31" \\
    --validation-end "2024-06-30"
        """
    )

    # 配置文件参数
    parser.add_argument('--config', type=str, help='配置文件路径（JSON格式）')

    # 快速测试模式
    parser.add_argument('--quick-test', action='store_true',
                       help='快速测试模式（使用tests目录下的测试数据）')

    # 数据配置
    parser.add_argument('--data-path', type=str, help='数据文件路径')
    parser.add_argument('--target', '--target-variable', type=str,
                       dest='target_variable', help='目标变量名称')
    parser.add_argument('--indicators', type=str, nargs='*',
                       help='选中的指标列表（空格分隔，不指定则使用全部）')

    # 日期配置
    parser.add_argument('--train-end', type=str, help='训练期结束日期（YYYY-MM-DD）')
    parser.add_argument('--validation-start', type=str, help='验证期开始日期（YYYY-MM-DD）')
    parser.add_argument('--validation-end', type=str, help='验证期结束日期（YYYY-MM-DD）')
    parser.add_argument('--target-freq', type=str, default='W-FRI',
                       help='目标频率（默认: W-FRI）')

    # 模型参数
    parser.add_argument('--k-factors', type=int, default=3, help='因子数量（默认: 3）')
    parser.add_argument('--max-iter', type=int, default=30,
                       help='EM算法最大迭代次数（默认: 30）')
    parser.add_argument('--max-lags', type=int, default=1,
                       help='因子自回归阶数（默认: 1）')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='收敛容差（默认: 1e-6）')

    # 变量选择
    parser.add_argument('--enable-selection', action='store_true',
                       help='启用变量选择')
    parser.add_argument('--min-variables', type=int,
                       help='变量选择后的最少变量数')

    # 因子选择
    parser.add_argument('--factor-method', type=str, default='fixed',
                       choices=['fixed', 'cumulative'],
                       help='因子数选择方法（默认: fixed）')
    parser.add_argument('--pca-threshold', type=float, default=0.8,
                       help='PCA累积方差阈值（默认: 0.8）')

    # 输出配置
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--no-export', action='store_true',
                       help='不导出结果文件（仅显示结果）')

    # 其他选项
    parser.add_argument('--verbose', action='store_true', help='显示详细输出')
    parser.add_argument('--show-config', action='store_true',
                       help='显示配置并退出（不执行训练）')

    return parser.parse_args()


def load_config_from_file(config_path: str) -> Dict:
    """从JSON文件加载配置"""
    config_file = Path(config_path)

    # 如果是相对路径，相对于当前目录或train目录查找
    if not config_file.is_absolute():
        # 尝试相对于当前工作目录
        if not config_file.exists():
            # 尝试相对于train目录
            train_dir = Path(__file__).parent
            config_file = train_dir / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    print(f"加载配置文件: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def create_quick_test_config() -> Dict:
    """创建快速测试配置（使用tests/data目录下的测试数据）"""
    # 当前文件在 dashboard/models/DFM/tests/code/test_train.py
    # 数据在 dashboard/models/DFM/tests/data/dfm_prepared_output.csv
    current_file = Path(__file__)
    tests_dir = current_file.parent.parent  # tests目录
    data_dir = tests_dir / 'data'

    data_path = data_dir / 'dfm_prepared_output.csv'

    if not data_path.exists():
        raise FileNotFoundError(
            f"测试数据文件不存在: {data_path}\n"
            f"请确保 dashboard/models/DFM/tests/data/dfm_prepared_output.csv 文件存在"
        )

    # 读取数据以确定日期范围
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # 筛选从2020-01-01开始的数据
    df = df[df.index >= '2020-01-01']

    # 只选择前10个预测变量（除了目标变量）
    target_var = "规模以上工业增加值:当月同比"
    all_vars = [col for col in df.columns if col != target_var]
    selected_vars = all_vars[:10]  # 只取前10个变量

    # 自动确定训练和验证期
    # 使用80%作为训练集，20%作为验证集
    total_rows = len(df)
    train_ratio = 0.8
    train_end_idx = int(total_rows * train_ratio)

    train_end = df.index[train_end_idx].strftime('%Y-%m-%d')
    validation_start = df.index[train_end_idx + 1].strftime('%Y-%m-%d')
    validation_end = df.index[-1].strftime('%Y-%m-%d')

    print(f"\n快速测试模式配置:")
    print(f"  数据文件: {data_path}")
    print(f"  数据范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  总样本数: {total_rows}")
    print(f"  训练期: {df.index[0].strftime('%Y-%m-%d')} ~ {train_end} ({train_end_idx} 样本)")
    print(f"  验证期: {validation_start} ~ {validation_end} ({total_rows - train_end_idx - 1} 样本)")
    print(f"  选择变量数: {len(selected_vars)}")

    return {
        "data_path": str(data_path),
        "target_variable": target_var,
        "selected_indicators": selected_vars,  # 只使用前10个变量
        "train_end": train_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "target_freq": "W-FRI",
        "k_factors": 3,
        "max_iterations": 30,
        "max_lags": 1,  # 因子自回归阶数（与UI默认值一致）
        "tolerance": 1e-6,
        "enable_variable_selection": True,  # 启用变量选择以便测试完整流程
        "min_variables_after_selection": 5,  # 最少保留5个变量
        "factor_selection_method": "fixed",
        "pca_threshold": 0.8,  # 累积方差阈值（与UI默认值一致）
        "output_dir": None
    }


def build_config_from_args(args) -> Dict:
    """从命令行参数构建配置字典"""
    # 快速测试模式
    if args.quick_test:
        return create_quick_test_config()

    # 从配置文件加载
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = {}

    # 命令行参数覆盖配置文件
    if args.data_path:
        config['data_path'] = args.data_path
    if args.target_variable:
        config['target_variable'] = args.target_variable
    if args.indicators:
        config['selected_indicators'] = args.indicators

    # 日期配置
    if args.train_end:
        config['train_end'] = args.train_end
    if args.validation_start:
        config['validation_start'] = args.validation_start
    if args.validation_end:
        config['validation_end'] = args.validation_end
    if args.target_freq:
        config['target_freq'] = args.target_freq

    # 模型参数
    if args.k_factors is not None:
        config['k_factors'] = args.k_factors
    if args.max_iter is not None:
        config['max_iterations'] = args.max_iter
    if args.max_lags is not None:
        config['max_lags'] = args.max_lags
    if args.tolerance is not None:
        config['tolerance'] = args.tolerance

    # 变量选择
    if args.enable_selection:
        config['enable_variable_selection'] = True
    if args.min_variables:
        config['min_variables_after_selection'] = args.min_variables

    # 因子选择
    if args.factor_method:
        config['factor_selection_method'] = args.factor_method
    if args.pca_threshold:
        config['pca_threshold'] = args.pca_threshold

    # 输出目录
    if args.output_dir:
        config['output_dir'] = args.output_dir

    return config


def print_config(config: Dict):
    """打印配置信息"""
    print("\n" + "=" * 60)
    print("训练配置")
    print("=" * 60)

    print(f"\n数据配置:")
    print(f"  数据路径: {config.get('data_path')}")
    print(f"  目标变量: {config.get('target_variable')}")
    print(f"  选中指标数: {len(config.get('selected_indicators', []))} "
          f"({'全部' if not config.get('selected_indicators') else '部分'})")

    print(f"\n日期配置:")
    print(f"  训练期结束: {config.get('train_end')}")
    print(f"  验证期开始: {config.get('validation_start')}")
    print(f"  验证期结束: {config.get('validation_end')}")
    print(f"  目标频率: {config.get('target_freq', 'W-FRI')}")

    print(f"\n模型参数:")
    print(f"  因子数: {config.get('k_factors', 3)}")
    print(f"  最大迭代: {config.get('max_iterations', 30)}")
    print(f"  收敛容差: {config.get('tolerance', 1e-6)}")

    print(f"\n变量选择:")
    print(f"  启用: {config.get('enable_variable_selection', False)}")
    if config.get('min_variables_after_selection'):
        print(f"  最少变量数: {config.get('min_variables_after_selection')}")

    print(f"\n因子选择:")
    print(f"  方法: {config.get('factor_selection_method', 'fixed')}")
    if config.get('factor_selection_method') == 'cumulative':
        print(f"  PCA阈值: {config.get('pca_threshold', 0.9)}")

    print(f"\n输出配置:")
    print(f"  输出目录: {config.get('output_dir', '临时目录')}")

    print("=" * 60 + "\n")


def print_results(result):
    """打印训练结果"""
    print("\n" + "=" * 60)
    print("训练结果")
    print("=" * 60)

    print(f"\n模型信息:")
    print(f"  最终变量数: {len(result.selected_variables) - 1}")  # 减去目标变量
    print(f"  因子数: {result.k_factors}")
    print(f"  迭代次数: {result.model_result.iterations}")
    print(f"  是否收敛: {result.model_result.converged}")

    print(f"\n评估指标:")
    print(f"  训练期RMSE: {result.metrics.is_rmse:.6f}")
    print(f"  验证期RMSE: {result.metrics.oos_rmse:.6f}")
    print(f"  训练期MAE: {result.metrics.is_mae:.6f}")
    print(f"  验证期MAE: {result.metrics.oos_mae:.6f}")
    print(f"  训练期Hit Rate: {result.metrics.is_hit_rate:.2f}%")
    print(f"  验证期Hit Rate: {result.metrics.oos_hit_rate:.2f}%")

    print(f"\n训练统计:")
    print(f"  训练时间: {result.training_time:.2f} 秒")
    print(f"  总评估次数: {result.total_evaluations}")
    if result.svd_error_count > 0:
        print(f"  SVD错误数: {result.svd_error_count}")

    if result.export_files:
        print(f"\n导出文件:")
        for file_type, file_path in result.export_files.items():
            if file_path:
                print(f"  {file_type}: {file_path}")

    print("=" * 60 + "\n")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    try:
        # 构建配置
        config_dict = build_config_from_args(args)

        # 打印配置
        if args.verbose or args.show_config:
            print_config(config_dict)

        # 如果只是显示配置，则退出
        if args.show_config:
            print("配置检查完成，退出。")
            return 0

        # 创建训练配置对象
        print("\n创建训练配置...")
        config = TrainingConfig.from_dict(config_dict)

        # 创建训练器
        print("初始化训练器...")
        trainer = DFMTrainer(config)

        # 定义进度回调函数
        def progress_callback(message: str):
            """进度回调，打印到控制台"""
            print(message)

        # 开始训练
        print("\n开始训练...\n")
        start_time = datetime.now()

        result = trainer.train(
            progress_callback=progress_callback,
            enable_export=not args.no_export,
            export_dir=config.output_dir
        )

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        # 打印结果
        print_results(result)

        print(f"训练完成! 总耗时: {elapsed:.2f} 秒\n")

        return 0

    except FileNotFoundError as e:
        print(f"\n错误: {e}\n", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"\n配置错误: {e}\n", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n训练失败: {e}\n", file=sys.stderr)
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
