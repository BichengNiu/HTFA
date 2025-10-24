# -*- coding: utf-8 -*-
"""
分析报告生成器

生成各种Excel报告，包括PCA分析、R²分析、贡献度分解等
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any
from dashboard.DFM.train_ref.analysis.analysis_utils import (
    calculate_pca_variance,
    calculate_individual_variable_r2,
    calculate_industry_r2,
    calculate_factor_contributions
)
from dashboard.DFM.train_ref.utils.logger import get_logger


logger = get_logger(__name__)


class AnalysisReporter:
    """分析报告生成器

    生成DFM模型训练结果的各种分析报告
    """

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_pca_report(
        self,
        data: pd.DataFrame,
        n_components: int,
        filename: str = 'pca_analysis.xlsx'
    ) -> str:
        """生成PCA方差贡献分析报告

        Args:
            data: 输入数据
            n_components: 主成分数量
            filename: 输出文件名

        Returns:
            报告文件路径
        """
        logger.info(f"生成PCA分析报告: {n_components}个主成分")

        # 计算PCA方差
        explained_var, explained_ratio, cumulative_ratio = calculate_pca_variance(
            data, n_components
        )

        # 构建报告DataFrame
        pca_df = pd.DataFrame({
            '主成分': [f'PC{i+1}' for i in range(n_components)],
            '方差': explained_var,
            '方差贡献率(%)': explained_ratio * 100,
            '累计方差贡献率(%)': cumulative_ratio * 100
        })

        # 保存到Excel
        output_path = self.output_dir / filename
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            pca_df.to_excel(writer, sheet_name='PCA分析', index=False)
            self._format_excel_sheet(writer, 'PCA分析', pca_df)

        logger.info(f"PCA报告已保存: {output_path}")
        return str(output_path)

    def generate_r2_report(
        self,
        observables: pd.DataFrame,
        factors: pd.DataFrame,
        loadings: np.ndarray,
        variable_industry_map: Optional[Dict[str, str]] = None,
        filename: str = 'r2_analysis.xlsx'
    ) -> str:
        """生成R²分析报告

        Args:
            observables: 观测变量
            factors: 因子
            loadings: 载荷矩阵
            variable_industry_map: 变量到行业的映射
            filename: 输出文件名

        Returns:
            报告文件路径
        """
        logger.info("生成R²分析报告")

        # 计算个体变量R²
        individual_r2 = calculate_individual_variable_r2(observables, factors, loadings)

        individual_df = pd.DataFrame({
            '变量': individual_r2.index,
            'R²': individual_r2.values,
            'R²(%)': individual_r2.values * 100
        })

        # 保存到Excel
        output_path = self.output_dir / filename
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 个体R²
            individual_df.to_excel(writer, sheet_name='个体变量R²', index=False)
            self._format_excel_sheet(writer, '个体变量R²', individual_df)

            # 行业R²（如果提供了映射）
            if variable_industry_map:
                industry_r2 = calculate_industry_r2(
                    observables, factors, loadings, variable_industry_map
                )
                industry_df = pd.DataFrame({
                    '行业': industry_r2.index,
                    'R²': industry_r2.values,
                    'R²(%)': industry_r2.values * 100
                })
                industry_df.to_excel(writer, sheet_name='行业R²', index=False)
                self._format_excel_sheet(writer, '行业R²', industry_df)

        logger.info(f"R²报告已保存: {output_path}")
        return str(output_path)

    def generate_contribution_report(
        self,
        factors: pd.DataFrame,
        loadings: np.ndarray,
        target_loading: np.ndarray,
        target_name: str = '目标变量',
        filename: str = 'factor_contribution.xlsx'
    ) -> str:
        """生成因子贡献度报告

        Args:
            factors: 因子时间序列
            loadings: 载荷矩阵
            target_loading: 目标变量载荷
            target_name: 目标变量名称
            filename: 输出文件名

        Returns:
            报告文件路径
        """
        logger.info("生成因子贡献度报告")

        # 计算贡献度
        contributions = calculate_factor_contributions(factors, loadings, target_loading)

        # 保存到Excel
        output_path = self.output_dir / filename
        contributions.to_excel(output_path, sheet_name='因子贡献度')

        logger.info(f"贡献度报告已保存: {output_path}")
        return str(output_path)

    def generate_report_with_params(
        self,
        training_result: Any,
        model_result: Any,
        data: pd.DataFrame,
        target: pd.Series,
        filename: str = 'dfm_综合报告.xlsx'
    ) -> str:
        """生成参数化综合报告

        Args:
            training_result: 训练结果对象
            model_result: 模型结果对象
            data: 输入数据
            target: 目标变量
            filename: 输出文件名

        Returns:
            报告文件路径
        """
        logger.info("生成综合报告")

        output_path = self.output_dir / filename

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: 模型参数
            params_df = pd.DataFrame({
                '参数': ['因子数', '迭代次数', '是否收敛', '对数似然'],
                '值': [
                    training_result.k_factors,
                    model_result.iterations,
                    model_result.converged,
                    model_result.log_likelihood
                ]
            })
            params_df.to_excel(writer, sheet_name='模型参数', index=False)

            # Sheet 2: 评估指标
            if training_result.metrics:
                metrics_df = pd.DataFrame({
                    '指标': ['样本内RMSE', '样本外RMSE', '样本内命中率(%)', '样本外命中率(%)'],
                    '值': [
                        training_result.metrics.is_rmse,
                        training_result.metrics.oos_rmse,
                        training_result.metrics.is_hit_rate,
                        training_result.metrics.oos_hit_rate
                    ]
                })
                metrics_df.to_excel(writer, sheet_name='评估指标', index=False)

            # Sheet 3: 因子载荷
            if model_result.H is not None:
                loadings_df = pd.DataFrame(
                    model_result.H,
                    columns=[f'Factor{i+1}' for i in range(training_result.k_factors)],
                    index=training_result.selected_variables[1:]  # 排除目标变量
                )
                loadings_df.to_excel(writer, sheet_name='因子载荷')

            # Sheet 4: 转移矩阵
            if model_result.A is not None:
                A_df = pd.DataFrame(
                    model_result.A,
                    columns=[f'Factor{i+1}' for i in range(training_result.k_factors)],
                    index=[f'Factor{i+1}' for i in range(training_result.k_factors)]
                )
                A_df.to_excel(writer, sheet_name='状态转移矩阵')

        logger.info(f"综合报告已保存: {output_path}")
        return str(output_path)

    def _format_excel_sheet(
        self,
        writer: pd.ExcelWriter,
        sheet_name: str,
        df: pd.DataFrame
    ):
        """格式化Excel工作表

        Args:
            writer: ExcelWriter对象
            sheet_name: 工作表名称
            df: DataFrame
        """
        try:
            from openpyxl.styles import Font, Alignment, PatternFill

            worksheet = writer.sheets[sheet_name]

            # 设置表头样式
            header_font = Font(bold=True, size=11)
            header_fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
            header_alignment = Alignment(horizontal="center", vertical="center")

            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_alignment

            # 自动调整列宽
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if cell.value:
                            max_length = max(max_length, len(str(cell.value)))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

        except ImportError:
            logger.warning("openpyxl不可用，跳过Excel格式化")
        except Exception as e:
            logger.warning(f"Excel格式化失败: {e}")


def generate_report(*args, **kwargs):
    """报告生成函数（向后兼容）"""
    if 'output_dir' not in kwargs:
        kwargs['output_dir'] = './output'

    reporter = AnalysisReporter(kwargs['output_dir'])
    return reporter.generate_report_with_params(*args, **kwargs)
