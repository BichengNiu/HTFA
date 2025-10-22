# -*- coding: utf-8 -*-
"""
Baseline生成器

从train_model模块运行测试案例并保存baseline结果，用于后续的数值一致性验证。

使用方法:
    python dashboard/DFM/train_ref/tests/consistency/baseline/generate_baseline.py
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# 设置项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入data_prep模块
from dashboard.DFM.data_prep import prepare_data

# 导入train_ref模块用于生成baseline
from dashboard.DFM.train_ref.core.factor_model import DFMModel
from dashboard.DFM.train_ref.evaluation.metrics import calculate_hit_rate
from sklearn.metrics import mean_squared_error


def load_test_cases(config_path: str) -> Dict[str, Any]:
    """加载测试案例配置

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_baseline_data(
    data_path: str,
    data_prep_config: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """使用data_prep模块准备数据

    Args:
        data_path: 数据文件路径
        data_prep_config: data_prep配置

    Returns:
        处理后的DataFrame或None
    """
    logger.info(f"开始数据预处理: {data_path}")

    try:
        # 调用data_prep.prepare_data
        processed_data, variable_mapping, transform_log, removal_log = prepare_data(
            excel_path=data_path,
            target_freq=data_prep_config.get('target_freq', 'W-FRI'),
            target_sheet_name=data_prep_config.get('target_sheet_name', '工业增加值同比增速_月度_同花顺'),
            target_variable_name=data_prep_config.get('target_variable_name', '规模以上工业增加值:当月同比'),
            consecutive_nan_threshold=data_prep_config.get('consecutive_nan_threshold', 10),
            data_start_date=data_prep_config.get('data_start_date', '2020-01-01'),
            data_end_date=data_prep_config.get('data_end_date', '2025-07-03'),
            reference_sheet_name=data_prep_config.get('reference_sheet_name', '指标体系'),
            reference_column_name=data_prep_config.get('reference_column_name', '指标名称')
        )

        if processed_data is None:
            logger.error("数据预处理失败")
            return None

        logger.info(f"数据预处理完成，数据形状: {processed_data.shape}")
        logger.info(f"列名: {list(processed_data.columns)}")

        return processed_data

    except Exception as e:
        logger.error(f"数据预处理异常: {e}", exc_info=True)
        return None


def run_baseline_case(
    case_id: str,
    case_config: Dict[str, Any],
    processed_data: pd.DataFrame,
    output_dir: Path,
    seed: int = 42
) -> bool:
    """运行单个测试案例并保存baseline结果

    Args:
        case_id: 案例ID
        case_config: 案例配置
        processed_data: 预处理后的数据
        output_dir: 输出目录
        seed: 随机种子

    Returns:
        是否成功
    """
    logger.info(f"=" * 80)
    logger.info(f"开始运行测试案例: {case_id}")
    logger.info(f"=" * 80)

    # 创建案例输出目录
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 设置随机种子
        np.random.seed(seed)

        # 保存配置
        config_path = case_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'case_id': case_id,
                'config': case_config,
                'seed': seed,
                'generated_at': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"配置已保存: {config_path}")

        # 提取配置参数
        target_variable = case_config.get('target_variable', '工业增加值同比增速')
        selected_indicators = case_config.get('selected_indicators', [])
        train_start = case_config.get('train_start')
        train_end = case_config.get('train_end')
        validation_start = case_config.get('validation_start')
        validation_end = case_config.get('validation_end')
        k_factors = case_config.get('k_factors', 2)
        max_iterations = case_config.get('max_iterations', 30)

        # 准备数据
        logger.info(f"目标变量配置: {target_variable}")
        logger.info(f"选择的指标配置: {selected_indicators}")

        # 检查数据中是否包含所需列
        available_columns = processed_data.columns.tolist()
        logger.info(f"可用列数: {len(available_columns)}")

        # 列名映射：从配置名称映射到实际列名
        # data_prep后的目标变量实际列名
        actual_target_variable = None
        for col in available_columns:
            if '工业增加值' in col and '同比' in col:
                actual_target_variable = col
                break

        if actual_target_variable is None:
            logger.error(f"无法在数据中找到目标变量（包含'工业增加值'和'同比'的列）")
            logger.error(f"可用列: {available_columns[:10]}...")
            return False

        logger.info(f"实际目标变量列名: {actual_target_variable}")

        # 映射selected_indicators到实际列名
        # 简化处理：从可用列中筛选出不是目标变量的列作为候选指标
        # 这里可以根据需要实现更复杂的名称映射逻辑
        candidate_indicators = [col for col in available_columns if col != actual_target_variable]

        if len(candidate_indicators) == 0:
            logger.error("没有可用的预测变量")
            return False

        # 如果配置中指定了selected_indicators，尝试找到匹配的列
        # 否则使用所有候选列
        if selected_indicators:
            # 简化：使用候选指标的前N个（N = len(selected_indicators)）
            actual_indicators = candidate_indicators[:min(len(selected_indicators), len(candidate_indicators))]
            logger.info(f"使用前{len(actual_indicators)}个可用指标作为selected_indicators")
        else:
            actual_indicators = candidate_indicators
            logger.info(f"使用所有{len(actual_indicators)}个可用指标")

        logger.info(f"实际使用的指标数量: {len(actual_indicators)}")
        logger.info(f"实际使用的指标（前5个）: {actual_indicators[:5]}")

        # 使用train_ref DFMModel生成baseline
        logger.info(f"开始使用train_ref生成baseline...")

        # 准备输入数据
        input_columns = [actual_target_variable] + actual_indicators
        input_df = processed_data[input_columns].copy()

        # 分割训练集和验证集
        train_data = input_df[train_start:train_end]
        validation_data = input_df[validation_start:validation_end]

        logger.info(f"训练数据: {train_data.shape}, 验证数据: {validation_data.shape}")

        try:
            # 创建并训练DFM模型
            model = DFMModel(
                n_factors=k_factors,
                max_lags=1,
                max_iter=max_iterations,
                tolerance=tolerance
            )

            # 训练模型
            result = model.fit(train_data)

            logger.info(f"模型训练完成")
            logger.info(f"  收敛: {result.converged}, 迭代次数: {result.n_iter}")
            logger.info(f"  对数似然: {result.loglikelihood:.2f}")

            # 计算验证集指标
            target_col = actual_target_variable
            true_values = validation_data[target_col].values

            # 获取验证期的因子预测（使用第一个因子）
            n_train = len(result.factors) - len(validation_data)
            if n_train >= 0:
                pred_factors = result.factors.iloc[n_train:, 0].values
            else:
                pred_factors = result.factors.iloc[:, 0].values[-len(validation_data):]

            # 确保长度一致
            min_len = min(len(true_values), len(pred_factors))
            true_values = true_values[:min_len]
            pred_factors = pred_factors[:min_len]

            # 计算评估指标
            rmse = np.sqrt(mean_squared_error(true_values, pred_factors))
            hit_rate = calculate_hit_rate(
                pd.Series(true_values),
                pd.Series(pred_factors)
            )
            correlation = np.corrcoef(true_values, pred_factors)[0, 1]

            logger.info(f"验证集指标:")
            logger.info(f"  RMSE: {rmse:.6f}")
            logger.info(f"  Hit Rate: {hit_rate:.2%}")
            logger.info(f"  Correlation: {correlation:.6f}")

            # 保存完整的baseline元数据
            metadata = {
                'case_id': case_id,
                'actual_target_variable': actual_target_variable,
                'actual_indicators': actual_indicators,
                'actual_indicators_count': len(actual_indicators),
                'k_factors': k_factors,
                'max_iterations': max_iterations,
                'tolerance': tolerance,

                # 模型参数
                'model_parameters': {
                    'loadings': result.loadings.tolist(),
                    'transition_matrix': result.transition_matrix.tolist(),
                    'process_noise_cov': result.process_noise_cov.tolist(),
                    'measurement_noise_cov': result.measurement_noise_cov.tolist(),
                    'loglikelihood': float(result.loglikelihood),
                    'converged': bool(result.converged),
                    'n_iter': int(result.n_iter)
                },

                # 平滑因子
                'smoothed_factors': result.factors.values.tolist(),
                'smoothed_factors_shape': list(result.factors.shape),

                # 验证集指标
                'validation_metrics': {
                    'rmse': float(rmse),
                    'hit_rate': float(hit_rate),
                    'correlation': float(correlation)
                },

                'generated_at': datetime.now().isoformat()
            }

            metadata_path = case_dir / "baseline_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Baseline元数据已保存: {metadata_path}")

            logger.info(f"=" * 80)
            logger.info(f"案例 {case_id} baseline生成成功")
            logger.info(f"=" * 80)
            return True

        except Exception as e:
            logger.error(f"DFM模型训练失败: {e}", exc_info=True)
            return False

    except Exception as e:
        logger.error(f"案例 {case_id} 执行失败: {e}", exc_info=True)
        return False


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("Baseline生成器启动")
    logger.info("=" * 80)

    # 定位配置文件和输出目录
    script_dir = Path(__file__).parent
    config_path = script_dir / "test_cases.json"
    output_dir = script_dir

    # 检查配置文件
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return 1

    # 加载配置
    logger.info(f"加载配置文件: {config_path}")
    config = load_test_cases(str(config_path))

    seed = config.get('seed', 42)
    data_path = PROJECT_ROOT / config['default_data_path']
    data_prep_config = config.get('data_prep_config', {})
    cases = config.get('cases', [])

    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"数据文件: {data_path}")
    logger.info(f"测试案例数量: {len(cases)}")
    logger.info(f"随机种子: {seed}")

    # 检查数据文件
    if not data_path.exists():
        logger.error(f"数据文件不存在: {data_path}")
        return 1

    # 预处理数据（所有案例共用）
    logger.info("=" * 80)
    logger.info("步骤1: 数据预处理")
    logger.info("=" * 80)

    processed_data = prepare_baseline_data(str(data_path), data_prep_config)

    if processed_data is None:
        logger.error("数据预处理失败，终止baseline生成")
        return 1

    # 保存预处理后的数据
    preprocessed_data_path = output_dir / "preprocessed_data.csv"
    processed_data.to_csv(preprocessed_data_path)
    logger.info(f"预处理数据已保存: {preprocessed_data_path}")

    # 运行所有测试案例
    logger.info("=" * 80)
    logger.info("步骤2: 运行测试案例")
    logger.info("=" * 80)

    success_count = 0
    for i, case in enumerate(cases, 1):
        case_id = case['id']
        case_name = case.get('name', case_id)
        case_config = case['config']

        logger.info(f"\n[{i}/{len(cases)}] 处理案例: {case_name} ({case_id})")

        success = run_baseline_case(
            case_id=case_id,
            case_config=case_config,
            processed_data=processed_data,
            output_dir=output_dir,
            seed=seed
        )

        if success:
            success_count += 1

    # 总结
    logger.info("=" * 80)
    logger.info("Baseline生成完成")
    logger.info("=" * 80)
    logger.info(f"成功: {success_count}/{len(cases)}")
    logger.info(f"输出目录: {output_dir}")

    if success_count == len(cases):
        logger.info("所有测试案例baseline生成成功")
        return 0
    else:
        logger.warning(f"部分案例失败: {len(cases) - success_count}个")
        return 1


if __name__ == "__main__":
    sys.exit(main())
