# -*- coding: utf-8 -*-
"""
后向逐步变量选择器

实现后向逐步变量剔除算法,支持RMSE和Hit Rate作为优化目标。
参考train_model/variable_selection.py的核心逻辑,但简化为更清晰的面向对象设计。
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """变量选择结果"""
    selected_variables: List[str]  # 最终选中的变量列表
    selection_history: List[Dict]  # 选择历史记录
    final_score: Tuple[float, float]  # 最终得分 (HR, -RMSE)
    total_evaluations: int  # 总评估次数
    svd_error_count: int  # SVD错误次数


class BackwardSelector:
    """
    后向逐步变量选择器

    算法流程:
    1. 从全部变量开始
    2. 逐个尝试剔除每个变量
    3. 评估剔除后的模型性能
    4. 选择性能提升最大的变量剔除
    5. 重复直到无法提升

    优化目标: HR -> -RMSE (先优化命中率,再优化RMSE)
    """

    def __init__(
        self,
        evaluator_func: Callable,
        criterion: str = 'rmse',
        min_variables: int = 1
    ):
        """
        Args:
            evaluator_func: 评估函数,签名为 (variables, **kwargs) -> (
                is_rmse, oos_rmse, _, _, is_hit_rate, oos_hit_rate,
                is_svd_error, _, _
            )
            criterion: 优化准则,'rmse'或'hit_rate'(当前固定为HR -> -RMSE)
            min_variables: 最少保留的变量数
        """
        self.evaluator_func = evaluator_func
        self.criterion = criterion
        self.min_variables = max(1, min_variables)

    def select(
        self,
        initial_variables: List[str],
        target_variable: str,
        full_data: pd.DataFrame,
        params: Dict,
        validation_start: str,
        validation_end: str,
        target_freq: str,
        train_end_date: str,
        target_mean_original: float,
        target_std_original: float,
        max_iter: int,
        max_lags: int = 1,
        progress_callback: Optional[Callable[[str], None]] = None,
        use_optimization: bool = False  # 暂不支持优化版本
    ) -> SelectionResult:
        """
        执行后向变量选择

        Args:
            initial_variables: 初始变量列表(包含目标变量)
            target_variable: 目标变量名称
            full_data: 完整数据DataFrame
            params: DFM参数字典(包含k_factors等)
            validation_start: 验证集开始日期
            validation_end: 验证集结束日期
            target_freq: 目标频率
            train_end_date: 训练集结束日期
            target_mean_original: 目标变量原始均值
            target_std_original: 目标变量原始标准差
            max_iter: 最大EM迭代次数
            max_lags: 最大滞后阶数
            progress_callback: 进度回调函数
            use_optimization: 是否使用预计算优化

        Returns:
            SelectionResult对象
        """
        total_evaluations = 0
        svd_error_count = 0
        selection_history = []

        # 1. 初始化当前最优变量集(仅预测变量)
        current_predictors = sorted([v for v in initial_variables if v != target_variable])
        if not current_predictors:
            logger.error("初始预测变量列表为空,无法进行筛选")
            return SelectionResult(
                selected_variables=initial_variables,
                selection_history=[],
                final_score=(-np.inf, np.inf),
                total_evaluations=0,
                svd_error_count=0
            )

        # 2. 计算初始基准性能
        if progress_callback:
            progress_callback(f"[SELECTION] 计算初始基准性能,变量数: {len(current_predictors)}")

        logger.info(f"计算初始基准性能,变量数: {len(current_predictors)}")

        initial_vars = [target_variable] + current_predictors
        try:
            result_tuple = self.evaluator_func(
                variables=initial_vars,
                full_data=full_data,
                target_variable=target_variable,
                params=params,
                validation_start=validation_start,
                validation_end=validation_end,
                target_freq=target_freq,
                train_end_date=train_end_date,
                target_mean_original=target_mean_original,
                target_std_original=target_std_original,
                max_iter=max_iter,
                max_lags=max_lags
            )
            total_evaluations += 1

            # 解析评估结果
            if len(result_tuple) != 9:
                logger.error(f"评估函数返回了{len(result_tuple)}个值(预期9)")
                logger.error(f"但继续尝试变量选择，使用默认分数")
                # 使用默认分数继续，而不是直接返回
                current_best_score = (0.0, -999999.0)  # HR=0%, RMSE=999999

            is_rmse, oos_rmse, _, _, is_hit_rate, oos_hit_rate, is_svd_error, _, _ = result_tuple
            if is_svd_error:
                svd_error_count += 1

            # 计算组合得分
            current_best_score = self._calculate_combined_score(
                is_rmse, oos_rmse, is_hit_rate, oos_hit_rate
            )

            if not np.isfinite(current_best_score[0]) or not np.isfinite(current_best_score[1]):
                logger.warning(
                    f"初始基准评估返回无效分数: HR={current_best_score[0]}, "
                    f"RMSE={-current_best_score[1]}, 但仍继续尝试变量选择"
                )
                # 使用最差的初始分数，确保任何有效候选都能被选中
                current_best_score = (-np.inf, -np.inf)  # 最差分数

            logger.info(
                f"初始基准得分 (HR={current_best_score[0]:.2f}%, "
                f"RMSE={-current_best_score[1]:.6f}), 变量数: {len(current_predictors)}"
            )

        except Exception as e:
            logger.error(f"计算初始基准性能时出错: {e}")
            import traceback
            traceback.print_exc()
            # 即使初始评估失败，也尝试继续（设置最差的初始分数）
            logger.warning("初始基准评估失败，使用默认分数继续尝试变量选择")
            current_best_score = (-np.inf, -np.inf)  # 最差分数

        # 3. 迭代移除变量
        iteration = 0
        logger.info(
            f"准备进入变量选择循环: current_predictors数={len(current_predictors)}, "
            f"min_variables={self.min_variables}, 循环条件={len(current_predictors) > self.min_variables}"
        )
        logger.info(f"k_factors={params.get('k_factors', 1)}")

        # 强制执行至少一次变量选择，用于调试
        force_debug_run = len(current_predictors) >= 3  # 如果有3个或更多变量，强制运行

        while len(current_predictors) > self.min_variables or (force_debug_run and iteration == 0):
            iteration += 1
            logger.info(f"第{iteration}轮变量选择 (当前变量数: {len(current_predictors)})")

            if progress_callback:
                progress_callback(
                    f"[SELECTION] 第{iteration}轮: 评估{len(current_predictors)}个变量的剔除效果"
                )

            # 找到本轮最佳移除候选
            # 初始值设为最差分数，确保任何有效候选都能被选中
            best_score_this_iter = (-np.inf, -np.inf)  # (HR, -RMSE)，-RMSE越大越好
            best_removal_var = None
            valid_removals = 0

            for var_to_remove in current_predictors:
                temp_predictors = [v for v in current_predictors if v != var_to_remove]
                if not temp_predictors:
                    logger.debug(f"跳过{var_to_remove}: 移除后无剩余变量")
                    continue

                temp_variables = [target_variable] + temp_predictors

                # 检查因子数是否仍然小于变量数(N > K)
                k_factors = params.get('k_factors', 1)
                if k_factors >= len(temp_variables):
                    logger.debug(
                        f"跳过{var_to_remove}: k_factors({k_factors}) >= "
                        f"剩余变量数({len(temp_variables)})"
                    )
                    continue

                valid_removals += 1

                # 评估剔除该变量后的性能
                try:
                    result_tuple = self.evaluator_func(
                        variables=temp_variables,
                        full_data=full_data,
                        target_variable=target_variable,
                        params=params,
                        validation_start=validation_start,
                        validation_end=validation_end,
                        target_freq=target_freq,
                        train_end_date=train_end_date,
                        target_mean_original=target_mean_original,
                        target_std_original=target_std_original,
                        max_iter=max_iter,
                        max_lags=max_lags
                    )
                    total_evaluations += 1

                    if len(result_tuple) != 9:
                        logger.warning(f"评估返回了{len(result_tuple)}个值,跳过{var_to_remove}")
                        continue

                    is_rmse, oos_rmse, _, _, is_hit_rate, oos_hit_rate, is_svd_error, _, _ = result_tuple
                    if is_svd_error:
                        svd_error_count += 1

                    # 计算得分
                    score = self._calculate_combined_score(
                        is_rmse, oos_rmse, is_hit_rate, oos_hit_rate
                    )

                    logger.debug(f"候选{var_to_remove}: score={score}, isfinite(HR)={np.isfinite(score[0])}, isfinite(RMSE)={np.isfinite(score[1])}")

                    # 只要RMSE是有限的就参与比较（即使HR是-inf也可以）
                    # 这允许在命中率无效时，仍然可以基于RMSE进行变量选择
                    if np.isfinite(score[1]):  # 只检查RMSE（第二个元素）是否有限
                        # 比较得分 (HR, -RMSE) - 先比较HR,再比较-RMSE
                        if score > best_score_this_iter:
                            best_score_this_iter = score
                            best_removal_var = var_to_remove
                            logger.info(f"找到更好的候选: {var_to_remove}, score={score}")
                    else:
                        logger.warning(f"跳过{var_to_remove}: RMSE不是有限值 (score={score})")

                except Exception as e:
                    logger.error(f"评估移除{var_to_remove}时出错: {e}")
                    continue

            # 检查是否有改进
            logger.info(f"第{iteration}轮评估完成: valid_removals={valid_removals}, best_removal_var={best_removal_var}")
            logger.info(f"  best_score_this_iter={best_score_this_iter}, current_best_score={current_best_score}")

            if best_removal_var is None:
                logger.warning(
                    f"本轮无可行的评估任务,筛选结束 "
                    f"(valid_removals={valid_removals}, "
                    f"current_predictors={len(current_predictors)}, "
                    f"k_factors={params.get('k_factors', 1)})"
                )
                break

            # 检查是否有性能提升
            if best_score_this_iter <= current_best_score:
                logger.warning(
                    f"移除任何变量都无法提升性能 "
                    f"(当前: HR={current_best_score[0]:.2f}%, RMSE={-current_best_score[1]:.6f}; "
                    f"最佳候选: HR={best_score_this_iter[0]:.2f}%, RMSE={-best_score_this_iter[1]:.6f}), "
                    f"筛选结束"
                )
                break

            # 执行移除
            current_predictors.remove(best_removal_var)
            current_best_score = best_score_this_iter

            # 记录选择历史
            selection_history.append({
                'iteration': iteration,
                'removed_variable': best_removal_var,
                'score': best_score_this_iter,
                'remaining_vars': current_predictors.copy(),
                'remaining_count': len(current_predictors)
            })

            logger.info(
                f"第{iteration}轮: 剔除'{best_removal_var}', "
                f"剩余{len(current_predictors)}个变量, "
                f"得分 (HR={best_score_this_iter[0]:.2f}%, RMSE={-best_score_this_iter[1]:.6f})"
            )

            if progress_callback:
                progress_callback(
                    f"[SELECTION] 剔除: {best_removal_var}, "
                    f"剩余: {len(current_predictors)}, "
                    f"HR={best_score_this_iter[0]:.2f}%"
                )

        # 4. 返回结果
        final_variables = [target_variable] + current_predictors
        logger.info(
            f"变量选择完成: 从{len(initial_variables)-1}个变量剔除到{len(current_predictors)}个, "
            f"最终得分 (HR={current_best_score[0]:.2f}%, RMSE={-current_best_score[1]:.6f})"
        )

        return SelectionResult(
            selected_variables=final_variables,
            selection_history=selection_history,
            final_score=current_best_score,
            total_evaluations=total_evaluations,
            svd_error_count=svd_error_count
        )

    def _calculate_combined_score(
        self,
        is_rmse: float,
        oos_rmse: float,
        is_hit_rate: float,
        oos_hit_rate: float
    ) -> Tuple[float, float]:
        """
        计算组合得分 (HR, -RMSE)

        Args:
            is_rmse: 样本内RMSE
            oos_rmse: 样本外RMSE
            is_hit_rate: 样本内命中率
            oos_hit_rate: 样本外命中率

        Returns:
            (combined_hit_rate, -combined_rmse)
        """
        # 计算平均RMSE
        finite_rmses = [r for r in [is_rmse, oos_rmse] if np.isfinite(r)]
        combined_rmse = np.mean(finite_rmses) if finite_rmses else np.inf

        # 计算平均命中率
        finite_hrs = [hr for hr in [is_hit_rate, oos_hit_rate] if np.isfinite(hr)]
        combined_hr = np.mean(finite_hrs) if finite_hrs else -np.inf

        return (combined_hr, -combined_rmse)
