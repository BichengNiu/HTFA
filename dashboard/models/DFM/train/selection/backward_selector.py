# -*- coding: utf-8 -*-
"""
后向逐步变量选择器

实现后向逐步变量剔除算法,以RMSE作为优化目标。
参考train_model/variable_selection.py的核心逻辑,但简化为更清晰的面向对象设计。
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Optional
from dashboard.models.DFM.train.utils.logger import get_logger
from dashboard.models.DFM.train.core.models import SelectionResult
from dashboard.models.DFM.train.evaluation.metrics import calculate_combined_score

logger = get_logger(__name__)


class BackwardSelector:
    """
    后向逐步变量选择器

    算法流程:
    1. 从全部变量开始
    2. 逐个尝试剔除每个变量
    3. 评估剔除后的模型性能
    4. 选择性能提升最大的变量剔除
    5. 重复直到无法提升

    优化目标: 最小化RMSE
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
            criterion: 优化准则,'rmse'(Hit Rate已弃用)
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
        use_optimization: bool = False
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
        # 保存评估参数供辅助方法使用
        self._eval_params = {
            'full_data': full_data,
            'target_variable': target_variable,
            'params': params,
            'validation_start': validation_start,
            'validation_end': validation_end,
            'target_freq': target_freq,
            'train_end_date': train_end_date,
            'target_mean_original': target_mean_original,
            'target_std_original': target_std_original,
            'max_iter': max_iter,
            'max_lags': max_lags
        }

        total_evaluations = 0
        svd_error_count = 0
        selection_history = []

        # 1. 初始化预测变量
        current_predictors = self._initialize_predictors(initial_variables, target_variable)
        if not current_predictors:
            return SelectionResult(
                selected_variables=initial_variables,
                selection_history=[],
                final_score=(-np.inf, np.inf),
                total_evaluations=0,
                svd_error_count=0
            )

        # 2. 计算初始基准性能
        current_best_score, eval_count, svd_count, current_is_rmse, current_oos_rmse = self._evaluate_baseline(
            current_predictors, progress_callback
        )
        total_evaluations += eval_count
        svd_error_count += svd_count

        # 3. 迭代移除变量
        iteration = 0
        while self._should_continue_selection(current_predictors, iteration):
            iteration += 1
            self._log_iteration_start(iteration, len(current_predictors), progress_callback)

            # 找到本轮最佳移除候选
            best_removal, best_score_this_iter, eval_count, svd_count, best_is_rmse, best_oos_rmse = self._find_best_removal_candidate(
                current_predictors
            )
            total_evaluations += eval_count
            svd_error_count += svd_count

            # 检查是否找到有效的移除候选
            if best_removal is None:
                logger.warning("本轮无可行的移除候选，筛选结束")
                break

            # 检查是否有性能提升
            if best_score_this_iter <= current_best_score:
                logger.warning(
                    f"移除任何变量都无法提升性能 "
                    f"(当前RMSE={-current_best_score[1]:.6f}; 最佳候选RMSE={-best_score_this_iter[1]:.6f})"
                )
                break

            # 应用移除并记录历史
            self._apply_removal_and_record(
                current_predictors, best_removal, best_score_this_iter,
                current_best_score, iteration, selection_history, progress_callback,
                current_is_rmse, current_oos_rmse, best_is_rmse, best_oos_rmse
            )

            # 更新当前最佳得分和RMSE
            current_best_score = best_score_this_iter
            current_is_rmse = best_is_rmse
            current_oos_rmse = best_oos_rmse

        # 4. 返回结果
        return self._build_selection_result(
            current_predictors, selection_history,
            current_best_score, total_evaluations, svd_error_count,
            len(initial_variables)
        )

    def _generate_progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """生成简单的进度条"""
        if total <= 0:
            return "[====================]"
        percent = min(1.0, current / total)
        filled = int(width * percent)
        bar = '=' * filled + '-' * (width - filled)
        return f"[{bar}]"

    def _initialize_predictors(
        self,
        initial_variables: List[str],
        target_variable: str
    ) -> List[str]:
        """初始化预测变量列表"""
        predictors = sorted([v for v in initial_variables if v != target_variable])
        if not predictors:
            logger.error("初始预测变量列表为空，无法进行筛选")
        return predictors

    def _evaluate_baseline(
        self,
        current_predictors: List[str],
        progress_callback: Optional[Callable]
    ) -> Tuple[Tuple[float, float], int, int, float, float]:
        """计算初始基准性能（扩展返回is_rmse和oos_rmse）"""
        # logger.info(f"计算初始基准性能，变量数: {len(current_predictors)}")

        target_variable = self._eval_params['target_variable']
        initial_vars = [target_variable] + current_predictors

        try:
            result_tuple = self.evaluator_func(variables=initial_vars, **self._eval_params)

            if len(result_tuple) != 9:
                logger.error(f"评估函数返回了{len(result_tuple)}个值(预期9)，使用默认分数")
                return ((-np.inf, -np.inf), 1, 0, np.inf, np.inf)

            is_rmse, oos_rmse, _, _, is_hit_rate, oos_hit_rate, is_svd_error, _, _ = result_tuple
            svd_count = 1 if is_svd_error else 0

            score = calculate_combined_score(is_rmse, oos_rmse, is_hit_rate, oos_hit_rate)

            if not np.isfinite(score[0]) or not np.isfinite(score[1]):
                logger.warning(f"初始基准评估返回无效分数，使用最差分数")
                score = (-np.inf, -np.inf)

            # logger.info(
            #     f"初始基准得分 (RMSE={-score[1]:.6f}), "
            #     f"变量数: {len(current_predictors)}"
            # )

            # 输出基线模型RMSE（精简格式）
            if progress_callback:
                progress_callback(
                    f"========== 变量选择 ==========\n"
                    f"基线模型 - 训练期RMSE: {is_rmse:.4f}, 验证期RMSE: {oos_rmse:.4f}"
                )

            return (score, 1, svd_count, is_rmse, oos_rmse)

        except Exception as e:
            logger.error(f"计算初始基准性能时出错: {e}")
            logger.warning("使用默认分数继续")
            return ((-np.inf, -np.inf), 1, 0, np.inf, np.inf)

    def _should_continue_selection(
        self,
        current_predictors: List[str],
        iteration: int
    ) -> bool:
        """判断是否应该继续变量选择"""
        # 基本条件：变量数大于最小要求
        if len(current_predictors) > self.min_variables:
            return True

        # 调试模式：至少执行一次
        force_debug_run = len(current_predictors) >= 3
        return force_debug_run and iteration == 0

    def _log_iteration_start(
        self,
        iteration: int,
        n_vars: int,
        progress_callback: Optional[Callable]
    ):
        """记录迭代开始信息"""
        # logger.info(f"\n{'='*60}")
        # logger.info(f"变量选择 - 第{iteration}轮 (当前{n_vars}个变量)")
        # logger.info(f"{'='*60}")

        # 进度显示（简化格式）
        if progress_callback:
            max_rounds = n_vars - self.min_variables
            progress_bar = self._generate_progress_bar(iteration, max_rounds)
            progress_callback(f"{progress_bar} 第{iteration}轮 (当前{n_vars}个变量)")

    def _find_best_removal_candidate(
        self,
        current_predictors: List[str]
    ) -> Tuple[Optional[str], Tuple[float, float], int, int, Optional[float], Optional[float]]:
        """找到本轮最佳移除候选（扩展返回is_rmse和oos_rmse）"""
        best_score = (-np.inf, -np.inf)
        best_var = None
        best_is_rmse = None
        best_oos_rmse = None
        total_evals = 0
        total_svd_errors = 0

        target_variable = self._eval_params['target_variable']
        k_factors = self._eval_params['params'].get('k_factors', 1)

        for var in current_predictors:
            temp_predictors = [v for v in current_predictors if v != var]
            if not temp_predictors:
                continue

            temp_variables = [target_variable] + temp_predictors

            # 检查因子数约束
            if k_factors >= len(temp_variables):
                logger.debug(f"跳过{var}: k_factors({k_factors}) >= 剩余变量数({len(temp_variables)})")
                continue

            # 评估移除后的性能
            try:
                result_tuple = self.evaluator_func(variables=temp_variables, **self._eval_params)
                total_evals += 1

                if len(result_tuple) != 9:
                    logger.warning(f"评估返回了{len(result_tuple)}个值，跳过{var}")
                    continue

                is_rmse, oos_rmse, _, _, is_hit_rate, oos_hit_rate, is_svd_error, _, _ = result_tuple
                if is_svd_error:
                    total_svd_errors += 1

                score = calculate_combined_score(is_rmse, oos_rmse, is_hit_rate, oos_hit_rate)

                # 只要RMSE有限就参与比较
                if np.isfinite(score[1]) and score > best_score:
                    best_score = score
                    best_var = var
                    best_is_rmse = is_rmse
                    best_oos_rmse = oos_rmse

            except Exception as e:
                logger.error(f"评估移除{var}时出错: {e}")
                continue

        return (best_var, best_score, total_evals, total_svd_errors, best_is_rmse, best_oos_rmse)

    def _apply_removal_and_record(
        self,
        current_predictors: List[str],
        removed_var: str,
        new_score: Tuple[float, float],
        old_score: Tuple[float, float],
        iteration: int,
        history: List[Dict],
        progress_callback: Optional[Callable],
        old_is_rmse: float,
        old_oos_rmse: float,
        new_is_rmse: float,
        new_oos_rmse: float
    ):
        """应用变量移除并记录历史（精简输出）"""
        current_predictors.remove(removed_var)

        # 记录选择历史
        history.append({
            'iteration': iteration,
            'removed_variable': removed_var,
            'score': new_score,
            'remaining_vars': current_predictors.copy(),
            'remaining_count': len(current_predictors)
        })

        # 记录得分改善
        delta_rmse = -(new_score[1] - old_score[1])

        # logger.info(
        #     f"第{iteration}轮完成: 移除'{removed_var}', 剩余{len(current_predictors)}个变量\n"
        #     f"  移除前RMSE -> {-old_score[1]:.6f}\n"
        #     f"  移除后RMSE -> {-new_score[1]:.6f}\n"
        #     f"  改善量ΔRMSE -> {delta_rmse:.6f}"
        # )

        # 精简的回调输出格式
        if progress_callback:
            progress_callback(
                f"第{iteration}轮: 移除'{removed_var}', 剩余{len(current_predictors)}个变量\n"
                f"  训练期RMSE: {old_is_rmse:.4f} -> {new_is_rmse:.4f}\n"
                f"  验证期RMSE: {old_oos_rmse:.4f} -> {new_oos_rmse:.4f}"
            )

    def _build_selection_result(
        self,
        final_predictors: List[str],
        history: List[Dict],
        final_score: Tuple[float, float],
        total_evals: int,
        svd_errors: int,
        n_initial_vars: int
    ) -> SelectionResult:
        """构建选择结果对象"""
        target_variable = self._eval_params['target_variable']
        final_variables = [target_variable] + final_predictors

        logger.info(
            f"变量选择完成: 从{n_initial_vars-1}个变量剔除到{len(final_predictors)}个, "
            f"最终RMSE={-final_score[1]:.6f}"
        )

        return SelectionResult(
            selected_variables=final_variables,
            selection_history=history,
            final_score=final_score,
            total_evaluations=total_evals,
            svd_error_count=svd_errors
        )

