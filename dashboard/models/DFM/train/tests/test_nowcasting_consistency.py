# -*- coding: utf-8 -*-
"""
Nowcasting值一致性测试

验证新代码（dashboard/models/DFM/train/）与老代码（train_model/）
在nowcasting计算上的机器精度级一致性
"""

import sys
import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pathlib import Path

# 设置UTF-8编码，避免GBK编码错误
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 老代码导入 - 创建虚拟包结构以满足老代码的导入需求
import importlib.util
import types

# 步骤1: 创建虚拟的包层级结构（用于老代码导入）
# 策略：先导入真实的dashboard包，然后在其下创建虚拟的DFM.train_model子包

# 导入真实的dashboard包（包含models子包）
import dashboard

# 在dashboard下创建虚拟的DFM子包（如果不存在）
if not hasattr(dashboard, 'DFM'):
    dfm_pkg = types.ModuleType('dashboard.DFM')
    dfm_pkg.__path__ = []
    dashboard.DFM = dfm_pkg
    sys.modules['dashboard.DFM'] = dfm_pkg

# 在dashboard.DFM下创建虚拟的train_model子包
train_model_pkg = types.ModuleType('dashboard.DFM.train_model')
train_model_pkg.__path__ = [str(project_root / "train_model")]
dashboard.DFM.train_model = train_model_pkg
sys.modules['dashboard.DFM.train_model'] = train_model_pkg

def load_module_directly(name, path):
    """直接从路径加载模块，避免__init__.py"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # 先注册避免循环导入
    spec.loader.exec_module(module)
    return module

# 步骤2: 按依赖顺序加载老代码模块
# 首先加载DiscreteKalmanFilter（最底层依赖）
DiscreteKalmanFilter_path = project_root / "train_model/DiscreteKalmanFilter.py"
DiscreteKalmanFilter = load_module_directly("dashboard.DFM.train_model.DiscreteKalmanFilter", str(DiscreteKalmanFilter_path))

# 加载analysis_utils
analysis_utils_path = project_root / "train_model/analysis_utils.py"
analysis_utils = load_module_directly("dashboard.DFM.train_model.analysis_utils", str(analysis_utils_path))

# 加载evaluation_cache
evaluation_cache_path = project_root / "train_model/evaluation_cache.py"
evaluation_cache = load_module_directly("dashboard.DFM.train_model.evaluation_cache", str(evaluation_cache_path))

# 加载DynamicFactorModel
DynamicFactorModel_path = project_root / "train_model/DynamicFactorModel.py"
DynamicFactorModel = load_module_directly("dashboard.DFM.train_model.DynamicFactorModel", str(DynamicFactorModel_path))

# 最后加载dfm_core
dfm_core_path = project_root / "train_model/dfm_core.py"
dfm_core = load_module_directly("dashboard.DFM.train_model.dfm_core", str(dfm_core_path))

# 导入所需函数
_prepare_data = dfm_core._prepare_data
_clean_and_validate_data = dfm_core._clean_and_validate_data
_mask_seasonal_data = dfm_core._mask_seasonal_data
_fit_dfm_model = dfm_core._fit_dfm_model
_calculate_nowcast = dfm_core._calculate_nowcast

# 现在可以安全导入新代码了（老代码已经加载完成）
from dashboard.models.DFM.train import DFMTrainer, TrainingConfig


def load_test_data():
    """加载测试数据"""
    data_path = project_root / "dashboard/models/DFM/tests/data/dfm_prepared_output.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"测试数据不存在: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    print(f"测试数据加载完成: {data.shape}")
    print(f"时间范围: {data.index.min()} 到 {data.index.max()}")

    return data


def run_new_code(data, config_dict):
    """运行新代码获取nowcast"""
    print("\n" + "="*60)
    print("运行新代码（dashboard/models/DFM/train/）")
    print("="*60)

    # 启用debug日志
    import logging
    logging.basicConfig(level=logging.DEBUG, force=True)
    from dashboard.models.DFM.train.utils.logger import get_logger
    logger = get_logger('dashboard.models.DFM.train.core.factor_model')
    logger.setLevel(logging.DEBUG)

    # 创建配置
    config = TrainingConfig(
        data_path=config_dict['data_path'],
        target_variable=config_dict['target_variable'],
        selected_indicators=config_dict['selected_indicators'],
        train_end=config_dict['train_end'],
        validation_start=config_dict['validation_start'],
        validation_end=config_dict['validation_end'],
        k_factors=config_dict['k_factors'],
        max_iterations=config_dict['max_iterations'],
        tolerance=config_dict['tolerance'],
        enable_variable_selection=False,  # 关闭变量选择，直接比对
        factor_selection_method='fixed'
    )

    # 训练模型
    trainer = DFMTrainer(config)
    result = trainer.train(progress_callback=None, enable_export=False)

    # 提取nowcast和关键中间结果
    nowcast_is = result.model_result.forecast_is
    nowcast_oos = result.model_result.forecast_oos
    factors_new = result.model_result.factors  # (n_factors, n_time)

    print(f"新代码结果:")
    print(f"  IS nowcast: {len(nowcast_is)} 个点")
    print(f"  OOS nowcast: {len(nowcast_oos) if nowcast_oos is not None else 0} 个点")
    print(f"  IS nowcast 范围: [{nowcast_is.min():.6f}, {nowcast_is.max():.6f}]")
    if nowcast_oos is not None:
        print(f"  OOS nowcast 范围: [{nowcast_oos.min():.6f}, {nowcast_oos.max():.6f}]")

    print(f"\n[调试] 新代码关键变量尺度:")
    print(f"  factors.shape = {factors_new.shape} (n_factors x n_time)")
    factors_T = factors_new.T  # 转置为 (n_time, n_factors)
    print(f"  factors前3行:\n{factors_T[:3]}")
    print(f"  factors均值: {factors_T.mean(axis=0)}")
    print(f"  factors标准差: {factors_T.std(axis=0)}")

    # 打印nowcast前5个值
    print(f"  nowcast_is前5个值: {nowcast_is[:5]}")

    return nowcast_is, nowcast_oos


def run_old_code(data, config_dict):
    """运行老代码获取nowcast

    注意：老代码无法直接返回nowcast序列，需要重新训练获取
    """
    print("\n" + "="*60)
    print("运行老代码（train_model/）")
    print("="*60)

    # 准备老代码所需的参数
    target_variable = config_dict['target_variable']
    selected_indicators = config_dict['selected_indicators']

    # 构造变量列表（包含目标变量）
    variables = [target_variable] + selected_indicators

    # 计算目标变量原始统计量
    target_train = data.loc[:config_dict['train_end'], target_variable]
    target_mean_original = target_train.mean()
    target_std_original = target_train.std()

    print(f"目标变量原始统计量: mean={target_mean_original:.6f}, std={target_std_original:.6f}")

    # 手动执行老代码的完整流程以获取nowcast
    # 步骤1: 准备数据
    success, error_msg, predictor_data, target_data, predictor_variables = _prepare_data(
        variables, data, target_variable, config_dict['validation_end']
    )
    if not success:
        raise ValueError(f"数据准备失败: {error_msg}")

    # 步骤2: 清理和验证数据
    success, error_msg, predictor_data_cleaned, predictor_variables = _clean_and_validate_data(
        predictor_data, predictor_variables, config_dict['k_factors']
    )
    if not success:
        raise ValueError(f"数据验证失败: {error_msg}")

    # 步骤3: 应用季节性掩码
    success, error_msg, predictor_data_final, target_data_masked = _mask_seasonal_data(
        predictor_data_cleaned, target_data, target_variable
    )
    if not success:
        raise ValueError(f"掩码应用失败: {error_msg}")

    # 步骤4: 训练DFM
    success, error_msg, dfm_results, is_svd_error = _fit_dfm_model(
        predictor_data_final,
        config_dict['k_factors'],
        config_dict['max_iterations'],
        max_lags=1,
        train_end_date=config_dict['train_end']
    )
    if not success:
        raise ValueError(f"DFM训练失败: {error_msg}")

    # 调试：打印因子和目标变量的尺度
    factors_sm = dfm_results.x_sm
    print(f"\n[调试] 老代码关键变量尺度:")
    print(f"  factors_sm.shape = {factors_sm.shape}")
    print(f"  factors_sm前3行:\n{factors_sm.iloc[:3].values}")
    print(f"  factors_sm均值: {factors_sm.mean().values}")
    print(f"  factors_sm标准差: {factors_sm.std().values}")
    print(f"  target_data_masked前5个值: {target_data_masked.iloc[:5].values}")
    print(f"  target_data_masked均值: {target_data_masked.mean():.4f}")
    print(f"  target_data_masked标准差: {target_data_masked.std():.4f}")
    print(f"  target_mean_original: {target_mean_original:.4f}")
    print(f"  target_std_original: {target_std_original:.4f}")

    # 步骤5: 计算nowcast
    success, error_msg, nowcast_series_orig, lambda_df = _calculate_nowcast(
        dfm_results,
        target_data_masked,
        predictor_variables,
        target_variable,
        config_dict['k_factors'],
        target_mean_original,
        target_std_original,
        config_dict['train_end']
    )
    if not success:
        raise ValueError(f"Nowcast计算失败: {error_msg}")

    # 分割IS和OOS
    train_end_date = pd.to_datetime(config_dict['train_end'])

    # 找到训练期在nowcast_series中的位置
    train_mask = nowcast_series_orig.index <= train_end_date
    nowcast_is = nowcast_series_orig[train_mask].values

    if config_dict['validation_start'] and config_dict['validation_end']:
        val_start = pd.to_datetime(config_dict['validation_start'])
        val_end = pd.to_datetime(config_dict['validation_end'])
        val_mask = (nowcast_series_orig.index >= val_start) & (nowcast_series_orig.index <= val_end)
        nowcast_oos = nowcast_series_orig[val_mask].values
    else:
        val_mask = nowcast_series_orig.index > train_end_date
        nowcast_oos = nowcast_series_orig[val_mask].values

    print(f"老代码结果:")
    print(f"  IS nowcast: {len(nowcast_is)} 个点")
    print(f"  OOS nowcast: {len(nowcast_oos) if nowcast_oos is not None else 0} 个点")
    print(f"  IS nowcast 范围: [{nowcast_is.min():.6f}, {nowcast_is.max():.6f}]")
    if nowcast_oos is not None and len(nowcast_oos) > 0:
        print(f"  OOS nowcast 范围: [{nowcast_oos.min():.6f}, {nowcast_oos.max():.6f}]")

    return nowcast_is, nowcast_oos


def compare_nowcast(new_is, new_oos, old_is, old_oos):
    """比对nowcast值"""
    print("\n" + "="*60)
    print("比对结果")
    print("="*60)

    # 长度检查
    print(f"\n长度比对:")
    print(f"  IS: 新={len(new_is)}, 老={len(old_is)}, 差异={abs(len(new_is)-len(old_is))}")
    if new_oos is not None and old_oos is not None:
        print(f"  OOS: 新={len(new_oos)}, 老={len(old_oos)}, 差异={abs(len(new_oos)-len(old_oos))}")

    # 对齐长度（以较短的为准）
    min_is_len = min(len(new_is), len(old_is))
    new_is_aligned = new_is[:min_is_len]
    old_is_aligned = old_is[:min_is_len]

    # 样本内比对
    print(f"\n样本内(IS)比对 ({min_is_len} 个点):")

    # 计算差异
    abs_diff_is = np.abs(new_is_aligned - old_is_aligned)
    rel_diff_is = np.abs((new_is_aligned - old_is_aligned) / (old_is_aligned + 1e-15))

    print(f"  绝对差异: 最大={abs_diff_is.max():.2e}, 均值={abs_diff_is.mean():.2e}, 中位={np.median(abs_diff_is):.2e}")
    print(f"  相对差异: 最大={rel_diff_is.max():.2e}, 均值={rel_diff_is.mean():.2e}, 中位={np.median(rel_diff_is):.2e}")

    # 相关系数
    corr_is = np.corrcoef(new_is_aligned, old_is_aligned)[0, 1]
    print(f"  相关系数: {corr_is:.15f}")

    # 样本外比对
    if new_oos is not None and old_oos is not None and len(new_oos) > 0 and len(old_oos) > 0:
        min_oos_len = min(len(new_oos), len(old_oos))
        new_oos_aligned = new_oos[:min_oos_len]
        old_oos_aligned = old_oos[:min_oos_len]

        print(f"\n样本外(OOS)比对 ({min_oos_len} 个点):")

        abs_diff_oos = np.abs(new_oos_aligned - old_oos_aligned)
        rel_diff_oos = np.abs((new_oos_aligned - old_oos_aligned) / (old_oos_aligned + 1e-15))

        print(f"  绝对差异: 最大={abs_diff_oos.max():.2e}, 均值={abs_diff_oos.mean():.2e}, 中位={np.median(abs_diff_oos):.2e}")
        print(f"  相对差异: 最大={rel_diff_oos.max():.2e}, 均值={rel_diff_oos.mean():.2e}, 中位={np.median(rel_diff_oos):.2e}")

        corr_oos = np.corrcoef(new_oos_aligned, old_oos_aligned)[0, 1]
        print(f"  相关系数: {corr_oos:.15f}")

    # 机器精度验证
    print(f"\n机器精度验证:")
    tolerance = 1e-10  # 机器精度容差

    try:
        assert_allclose(
            new_is_aligned, old_is_aligned,
            rtol=tolerance, atol=tolerance,
            err_msg="样本内nowcast不一致"
        )
        print(f"  ✓ 样本内(IS)通过: 差异 < {tolerance}")
    except AssertionError as e:
        print(f"  ✗ 样本内(IS)失败: {e}")
        return False

    if new_oos is not None and old_oos is not None and len(new_oos) > 0 and len(old_oos) > 0:
        try:
            assert_allclose(
                new_oos_aligned, old_oos_aligned,
                rtol=tolerance, atol=tolerance,
                err_msg="样本外nowcast不一致"
            )
            print(f"  ✓ 样本外(OOS)通过: 差异 < {tolerance}")
        except AssertionError as e:
            print(f"  ✗ 样本外(OOS)失败: {e}")
            return False

    return True


def main():
    """主测试函数"""
    print("="*60)
    print("Nowcasting值一致性测试")
    print("="*60)

    # 加载测试数据
    data = load_test_data()

    # 配置参数
    target_variable = "规模以上工业增加值:当月同比"

    # 选择部分指标进行测试（选择NaN较少的指标）
    all_indicators = [col for col in data.columns if col != target_variable]

    # 按NaN数量排序，选择NaN最少的30个指标
    nan_counts = data[all_indicators].isna().sum().sort_values()
    selected_indicators = nan_counts.head(30).index.tolist()

    # 关键修复：使用数据的实际最大日期作为validation_end，确保新老代码使用相同的数据范围
    max_date = data.index.max().strftime('%Y-%m-%d')

    config = {
        'data_path': str(project_root / "dashboard/models/DFM/tests/data/dfm_prepared_output.csv"),
        'target_variable': target_variable,
        'selected_indicators': selected_indicators,
        'train_end': '2024-08-16',
        'validation_start': '2024-08-23',
        'validation_end': max_date,  # 使用数据最大日期，确保新老代码数据一致
        'k_factors': 3,
        'max_iterations': 30,
        'tolerance': 1e-6
    }

    print(f"\n测试配置:")
    print(f"  目标变量: {config['target_variable']}")
    print(f"  指标数量: {len(selected_indicators)}")
    print(f"  训练期结束: {config['train_end']}")
    print(f"  验证期: {config['validation_start']} ~ {config['validation_end']}")
    print(f"  k_factors: {config['k_factors']}")

    # 运行新代码
    new_is, new_oos = run_new_code(data, config)

    # 运行老代码
    old_is, old_oos = run_old_code(data, config)

    # 比对结果
    success = compare_nowcast(new_is, new_oos, old_is, old_oos)

    # 总结
    print("\n" + "="*60)
    if success:
        print("✓ 测试通过！新老代码nowcasting值在机器精度内完全一致")
    else:
        print("✗ 测试失败！新老代码nowcasting值存在差异")
    print("="*60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
