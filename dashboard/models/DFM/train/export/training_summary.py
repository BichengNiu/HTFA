# -*- coding: utf-8 -*-
"""
训练摘要信息生成模块

生成用户可读的训练信息文本文件，包含：
- 目标变量
- 预测变量（进入训练、最终保留）
- 行业信息
- 模型参数
- 评估指标
- 训练统计
"""

from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime


def generate_training_summary(
    result,  # TrainingResult or TwoStageTrainingResult
    config,  # TrainingConfig
    timestamp: Optional[str] = None
) -> str:
    """
    生成训练摘要文本

    Args:
        result: 训练结果对象（TrainingResult 或 TwoStageTrainingResult）
        config: 训练配置对象
        timestamp: 时间戳字符串（可选）

    Returns:
        str: 格式化的训练摘要文本
    """
    from dashboard.models.DFM.train.core.models import TwoStageTrainingResult

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 判断是否为二次估计法
    is_two_stage = isinstance(result, TwoStageTrainingResult)

    lines = []
    lines.append("=" * 80)
    lines.append("DFM 模型训练摘要")
    lines.append("=" * 80)
    lines.append(f"生成时间: {timestamp}")
    lines.append("")

    # 估计方法
    lines.append("[估计方法]")
    if is_two_stage:
        lines.append("  估计方法: 二次估计法 (Two-Stage Estimation)")
        lines.append("")
    else:
        lines.append("  估计方法: 一次估计法 (Single-Stage Estimation)")
        lines.append("")

    # 目标变量
    lines.append("[目标变量]")
    if is_two_stage:
        # 二次估计法：显示第二阶段目标变量
        second_stage_target = result.second_stage_result.selected_variables[0] if result.second_stage_result and result.second_stage_result.selected_variables else "未知"
        lines.append(f"  第二阶段目标变量（总量）: {second_stage_target}")

        # 显示第一阶段目标变量（各行业）
        first_stage_targets = []
        for industry, ind_result in result.first_stage_results.items():
            if ind_result.selected_variables:
                first_stage_targets.append(f"{industry}: {ind_result.selected_variables[0]}")

        if first_stage_targets:
            lines.append(f"  第一阶段目标变量数量（各行业）: {len(first_stage_targets)}")
            lines.append("  第一阶段目标变量明细:")
            for target in first_stage_targets:
                lines.append(f"    - {target}")
    else:
        # 一次估计法：直接显示目标变量
        lines.append(f"  目标变量: {config.target_variable}")
    lines.append("")

    # 预测变量信息
    lines.append("[预测变量]")

    if is_two_stage:
        # 二次估计法：第一阶段和第二阶段分别统计
        lines.append("  第一阶段（分行业）:")

        # 进入训练的预测变量总数（各行业初始选择的指标数之和）
        initial_total = 0
        for industry, ind_result in result.first_stage_results.items():
            # 从config获取初始选择的指标数
            initial_count = len(config.selected_indicators) if hasattr(config, 'selected_indicators') else 0
            initial_total += initial_count

        lines.append(f"    进入训练的预测变量总数: {initial_total}")

        # 最终保留的预测变量总数（各行业最终变量数之和）
        final_total = 0
        industry_details = []
        for industry, ind_result in result.first_stage_results.items():
            # 排除目标变量（第一个变量）
            predictors = ind_result.selected_variables[1:] if len(ind_result.selected_variables) > 1 else []
            final_total += len(predictors)
            industry_details.append({
                'industry': industry,
                'count': len(predictors),
                'variables': predictors
            })

        lines.append(f"    最终保留的预测变量总数: {final_total}")
        lines.append("")
        lines.append("    分行业明细:")
        for detail in industry_details:
            lines.append(f"      {detail['industry']}: {detail['count']}个变量")
            for var in detail['variables']:
                lines.append(f"        - {var}")

        lines.append("")
        lines.append("  第二阶段（总量）:")
        if result.second_stage_result:
            # 第二阶段进入训练的预测变量 = 各行业nowcasting + 额外预测变量
            second_predictors = result.second_stage_result.selected_variables[1:] if len(result.second_stage_result.selected_variables) > 1 else []
            extra_predictors = config.second_stage_extra_predictors if hasattr(config, 'second_stage_extra_predictors') else []

            industry_nowcast_count = len(result.first_stage_results)  # 行业数 = nowcasting数

            lines.append(f"    进入训练的预测变量数: {industry_nowcast_count + len(extra_predictors)}")
            lines.append(f"      - 各行业Nowcasting: {industry_nowcast_count}个")
            if extra_predictors:
                lines.append(f"      - 额外宏观指标: {len(extra_predictors)}个")
                for var in extra_predictors:
                    lines.append(f"          {var}")

            lines.append(f"    最终保留的预测变量数: {len(second_predictors)}")
            if second_predictors:
                lines.append("    最终保留的预测变量明细:")
                for var in second_predictors:
                    lines.append(f"      - {var}")
    else:
        # 一次估计法：简单统计
        initial_indicators = config.selected_indicators if hasattr(config, 'selected_indicators') else []
        lines.append(f"  进入训练的预测变量数: {len(initial_indicators)}")
        if initial_indicators:
            lines.append("  进入训练的预测变量明细:")
            for var in initial_indicators:
                lines.append(f"    - {var}")

        lines.append("")
        final_predictors = [v for v in result.selected_variables if v != config.target_variable]
        lines.append(f"  最终保留的预测变量数: {len(final_predictors)}")
        if final_predictors:
            lines.append("  最终保留的预测变量明细:")
            for var in final_predictors:
                lines.append(f"    - {var}")

    lines.append("")

    # 行业信息
    lines.append("[行业信息]")
    if is_two_stage:
        lines.append(f"  行业总数: {len(result.first_stage_results)}")
        lines.append("  各行业因子数:")
        for industry, k_factors in result.industry_k_factors_used.items():
            lines.append(f"    {industry}: {k_factors}个因子")
    else:
        # 一次估计法：从最终保留变量统计涉及的行业数
        if hasattr(config, 'industry_map') and config.industry_map:
            # 获取最终保留的预测变量
            final_predictors = [v for v in result.selected_variables if v != config.target_variable]
            # 只统计最终保留变量涉及的行业
            involved_industries = set()
            for var in final_predictors:
                if var in config.industry_map:
                    involved_industries.add(config.industry_map[var])

            if involved_industries:
                lines.append(f"  涉及行业数: {len(involved_industries)}")
                lines.append("  行业列表:")
                for industry in sorted(involved_industries):
                    lines.append(f"    - {industry}")
            else:
                lines.append("  涉及行业数: 0 (变量未映射到行业)")
        else:
            lines.append("  行业信息: 未提供行业映射")
    lines.append("")

    # 模型参数
    lines.append("[模型参数]")

    if is_two_stage:
        # 第一阶段参数
        lines.append("  第一阶段（分行业）:")
        lines.append(f"    因子数: 各行业独立设置（见上方行业信息）")

        # 第二阶段参数
        lines.append("")
        lines.append("  第二阶段（总量）:")
        if result.second_stage_result:
            lines.append(f"    因子数: {result.second_stage_result.k_factors}")
            lines.append(f"    因子选择策略: {result.second_stage_result.factor_selection_method}")
    else:
        lines.append(f"  因子数: {result.k_factors}")
        lines.append(f"  因子选择策略: {result.factor_selection_method}")

    # 通用参数
    lines.append("")
    lines.append("  通用参数:")
    lines.append(f"    最大迭代次数: {config.max_iterations}")
    lines.append(f"    因子AR阶数: {config.max_lags}")
    lines.append(f"    收敛容差: {config.tolerance}")
    lines.append(f"    目标频率: {config.target_freq}")

    # 变量选择方法
    if config.enable_variable_selection:
        lines.append(f"    变量选择方法: {config.variable_selection_method}")
    else:
        lines.append("    变量选择方法: 无筛选（使用全部已选变量）")

    # 并行配置
    if config.enable_parallel:
        lines.append(f"    并行计算: 启用 (n_jobs={config.n_jobs}, backend={config.parallel_backend})")
    else:
        lines.append("    并行计算: 未启用")

    # 目标配对模式
    if hasattr(config, 'target_alignment_mode'):
        alignment_desc = "下月值" if config.target_alignment_mode == 'next_month' else "本月值"
        lines.append(f"    目标配对模式: {alignment_desc} ({config.target_alignment_mode})")

    lines.append("")

    # 训练/验证期设置
    lines.append("[训练与验证期设置]")
    lines.append(f"  训练期: {config.training_start} 至 {config.train_end}")
    lines.append(f"  验证期: {config.validation_start} 至 {config.validation_end}")
    lines.append("")

    # 评估指标
    lines.append("[评估指标]")

    if is_two_stage:
        # 第一阶段指标汇总
        lines.append("  第一阶段（分行业）平均指标:")
        if result.first_stage_results:
            is_rmse_list = []
            oos_rmse_list = []
            is_hr_list = []
            oos_hr_list = []

            for ind_result in result.first_stage_results.values():
                if ind_result.metrics:
                    is_rmse_list.append(ind_result.metrics.is_rmse)
                    oos_rmse_list.append(ind_result.metrics.oos_rmse)
                    is_hr_list.append(ind_result.metrics.is_hit_rate)
                    oos_hr_list.append(ind_result.metrics.oos_hit_rate)

            if is_rmse_list:
                lines.append(f"    样本内RMSE (平均): {np.mean(is_rmse_list):.4f}")
                lines.append(f"    样本外RMSE (平均): {np.mean(oos_rmse_list):.4f}")
                lines.append(f"    样本内命中率 (平均): {np.mean(is_hr_list):.2f}%")
                lines.append(f"    样本外命中率 (平均): {np.mean(oos_hr_list):.2f}%")

        lines.append("")
        lines.append("  第二阶段（总量）指标:")
        if result.second_stage_result and result.second_stage_result.metrics:
            metrics = result.second_stage_result.metrics
            lines.append(f"    样本内RMSE: {metrics.is_rmse:.4f}")
            lines.append(f"    样本外RMSE: {metrics.oos_rmse:.4f}")
            lines.append(f"    样本内MAE: {metrics.is_mae:.4f}")
            lines.append(f"    样本外MAE: {metrics.oos_mae:.4f}")

            # Hit Rate处理（可能为nan）
            is_hr = metrics.is_hit_rate
            oos_hr = metrics.oos_hit_rate
            is_hr_str = f"{is_hr:.2f}%" if not np.isnan(is_hr) and not np.isinf(is_hr) else "N/A"
            oos_hr_str = f"{oos_hr:.2f}%" if not np.isnan(oos_hr) and not np.isinf(oos_hr) else "N/A"

            lines.append(f"    样本内命中率: {is_hr_str}")
            lines.append(f"    样本外命中率: {oos_hr_str}")

            # 观察期指标（如果存在）
            if metrics.obs_rmse != np.inf:
                obs_hr = metrics.obs_hit_rate
                obs_hr_str = f"{obs_hr:.2f}%" if not np.isnan(obs_hr) and not np.isinf(obs_hr) else "N/A"
                lines.append(f"    观察期RMSE: {metrics.obs_rmse:.4f}")
                lines.append(f"    观察期MAE: {metrics.obs_mae:.4f}")
                lines.append(f"    观察期命中率: {obs_hr_str}")
    else:
        # 一次估计法指标
        if result.metrics:
            metrics = result.metrics
            lines.append(f"  样本内RMSE: {metrics.is_rmse:.4f}")
            lines.append(f"  样本外RMSE: {metrics.oos_rmse:.4f}")
            lines.append(f"  样本内MAE: {metrics.is_mae:.4f}")
            lines.append(f"  样本外MAE: {metrics.oos_mae:.4f}")

            # Hit Rate处理
            is_hr = metrics.is_hit_rate
            oos_hr = metrics.oos_hit_rate
            is_hr_str = f"{is_hr:.2f}%" if not np.isnan(is_hr) and not np.isinf(is_hr) else "N/A"
            oos_hr_str = f"{oos_hr:.2f}%" if not np.isnan(oos_hr) and not np.isinf(oos_hr) else "N/A"

            lines.append(f"  样本内命中率: {is_hr_str}")
            lines.append(f"  样本外命中率: {oos_hr_str}")

            # 观察期指标
            if metrics.obs_rmse != np.inf:
                obs_hr = metrics.obs_hit_rate
                obs_hr_str = f"{obs_hr:.2f}%" if not np.isnan(obs_hr) and not np.isinf(obs_hr) else "N/A"
                lines.append(f"  观察期RMSE: {metrics.obs_rmse:.4f}")
                lines.append(f"  观察期MAE: {metrics.obs_mae:.4f}")
                lines.append(f"  观察期命中率: {obs_hr_str}")

    lines.append("")

    # 训练统计
    lines.append("[训练统计]")

    if is_two_stage:
        lines.append(f"  第一阶段训练耗时: {result.first_stage_time:.2f} 秒")
        lines.append(f"  第二阶段训练耗时: {result.second_stage_time:.2f} 秒")
        lines.append(f"  总训练耗时: {result.total_training_time:.2f} 秒")

        if result.second_stage_result:
            lines.append(f"  第二阶段模型迭代次数: {result.second_stage_result.model_result.iterations if result.second_stage_result.model_result else 'N/A'}")
            lines.append(f"  第二阶段模型收敛状态: {'已收敛' if result.second_stage_result.model_result and result.second_stage_result.model_result.converged else '未收敛'}")
    else:
        lines.append(f"  训练耗时: {result.training_time:.2f} 秒")
        if result.model_result:
            lines.append(f"  模型迭代次数: {result.model_result.iterations}")
            lines.append(f"  模型收敛状态: {'已收敛' if result.model_result.converged else '未收敛'}")

        if result.total_evaluations > 0:
            lines.append(f"  变量选择评估次数: {result.total_evaluations}")
        if result.svd_error_count > 0:
            lines.append(f"  SVD错误次数: {result.svd_error_count}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("摘要结束")
    lines.append("=" * 80)

    return "\n".join(lines)


__all__ = ['generate_training_summary']
