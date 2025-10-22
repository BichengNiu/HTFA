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

# 导入train_model模块（作为参考baseline）
# 注意：这里只是导入用于生成baseline，不修改原代码
from dashboard.DFM.train_model import tune_dfm


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
        logger.info(f"目标变量: {target_variable}")
        logger.info(f"选择的指标: {selected_indicators}")

        # 检查数据中是否包含所需列
        available_columns = processed_data.columns.tolist()
        logger.info(f"可用列数: {len(available_columns)}")

        # 注意：这里需要根据实际的train_model接口调整
        # 由于train_model是一个复杂的模块，这里只是示例框架
        # 实际实现需要调用具体的训练函数

        logger.warning(f"案例 {case_id}: train_model调用接口需要根据实际情况实现")
        logger.warning("当前仅保存配置，未执行实际训练")

        # TODO: 实际调用train_model进行训练
        # 这需要了解train_model的具体接口
        # 示例:
        # result = train_dfm_model(
        #     data=processed_data,
        #     target_variable=target_variable,
        #     selected_indicators=selected_indicators,
        #     train_end=train_end,
        #     validation_start=validation_start,
        #     validation_end=validation_end,
        #     k_factors=k_factors,
        #     max_iterations=max_iterations
        # )

        # 保存占位符结果
        placeholder_path = case_dir / "placeholder.txt"
        with open(placeholder_path, 'w') as f:
            f.write(f"Baseline generation for {case_id} - implementation pending\n")
            f.write(f"Configuration saved successfully\n")
            f.write(f"Data shape: {processed_data.shape}\n")

        logger.info(f"案例 {case_id} 配置已保存（实际训练待实现）")
        return True

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
        logger.info("所有测试案例配置已保存")
        logger.warning("注意：实际的train_model训练逻辑需要补充实现")
        return 0
    else:
        logger.warning(f"部分案例失败: {len(cases) - success_count}个")
        return 1


if __name__ == "__main__":
    sys.exit(main())
