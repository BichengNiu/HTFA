# -*- coding: utf-8 -*-
"""
结果可视化器

支持Plotly和Matplotlib两种后端，生成各种分析图表
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Literal, Tuple, Dict, Any
from dashboard.DFM.train_ref.utils.logger import get_logger


logger = get_logger(__name__)


class ResultVisualizer:
    """DFM结果可视化器

    支持Plotly（交互式）和Matplotlib（静态）两种后端
    """

    def __init__(
        self,
        backend: Literal['plotly', 'matplotlib'] = 'plotly',
        output_dir: Optional[str] = None
    ):
        """
        Args:
            backend: 绘图后端，'plotly'或'matplotlib'
            output_dir: 输出目录（用于保存静态图片）
        """
        self.backend = backend
        self.output_dir = Path(output_dir) if output_dir else None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # 延迟导入绘图库
        self._setup_backend()

    def _setup_backend(self):
        """设置绘图后端"""
        if self.backend == 'plotly':
            try:
                import plotly.graph_objects as go
                import plotly.express as px
                from plotly.subplots import make_subplots
                self.go = go
                self.px = px
                self.make_subplots = make_subplots
            except ImportError:
                logger.warning("plotly未安装，回退到matplotlib")
                self.backend = 'matplotlib'

        if self.backend == 'matplotlib':
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                self.plt = plt
                self.sns = sns
                sns.set_style("whitegrid")
            except ImportError:
                raise ImportError("matplotlib和seaborn必须安装至少一个绘图库")

    def plot_forecast_vs_actual(
        self,
        actual: pd.Series,
        forecast: pd.Series,
        title: str = "预测 vs 实际",
        save_path: Optional[str] = None
    ) -> Any:
        """绘制预测vs实际对比图

        Args:
            actual: 实际值序列
            forecast: 预测值序列
            title: 图表标题
            save_path: 保存路径（可选）

        Returns:
            图表对象（Plotly Figure或Matplotlib Figure）
        """
        logger.info(f"绘制预测对比图: {title}")

        if self.backend == 'plotly':
            fig = self.go.Figure()

            fig.add_trace(self.go.Scatter(
                x=actual.index,
                y=actual.values,
                mode='lines',
                name='实际值',
                line=dict(color='blue', width=2)
            ))

            fig.add_trace(self.go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode='lines',
                name='预测值',
                line=dict(color='red', width=2, dash='dash')
            ))

            fig.update_layout(
                title=title,
                xaxis_title='时间',
                yaxis_title='值',
                hovermode='x unified',
                template='plotly_white'
            )

            if save_path:
                fig.write_html(save_path)

            return fig

        else:  # matplotlib
            fig, ax = self.plt.subplots(figsize=(12, 6))

            ax.plot(actual.index, actual.values, label='实际值',
                   color='blue', linewidth=2)
            ax.plot(forecast.index, forecast.values, label='预测值',
                   color='red', linewidth=2, linestyle='--')

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('时间', fontsize=12)
            ax.set_ylabel('值', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            self.plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            return fig

    def plot_residuals(
        self,
        residuals: pd.Series,
        title: str = "残差分析",
        save_path: Optional[str] = None
    ) -> Any:
        """绘制残差分析图（时序图 + 直方图 + QQ图）

        Args:
            residuals: 残差序列
            title: 图表标题
            save_path: 保存路径

        Returns:
            图表对象
        """
        logger.info(f"绘制残差分析图: {title}")

        if self.backend == 'plotly':
            from scipy import stats

            # 创建子图
            fig = self.make_subplots(
                rows=2, cols=2,
                subplot_titles=('残差时序图', '残差直方图',
                              '残差ACF', '残差QQ图'),
                specs=[[{"colspan": 2}, None],
                       [{}, {}]]
            )

            # 1. 残差时序图
            fig.add_trace(
                self.go.Scatter(x=residuals.index, y=residuals.values,
                              mode='lines', name='残差',
                              line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

            # 2. 残差直方图
            fig.add_trace(
                self.go.Histogram(x=residuals.values, name='残差分布',
                                marker_color='steelblue', nbinsx=30),
                row=2, col=1
            )

            # 3. QQ图
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
            sample_quantiles = np.sort(residuals.values)

            fig.add_trace(
                self.go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                              mode='markers', name='QQ图',
                              marker=dict(color='blue', size=5)),
                row=2, col=2
            )

            # QQ图参考线
            fig.add_trace(
                self.go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles,
                              mode='lines', name='参考线',
                              line=dict(color='red', dash='dash')),
                row=2, col=2
            )

            fig.update_layout(
                title_text=title,
                showlegend=False,
                height=800,
                template='plotly_white'
            )

            if save_path:
                fig.write_html(save_path)

            return fig

        else:  # matplotlib
            from scipy import stats

            fig, axes = self.plt.subplots(2, 2, figsize=(14, 10))

            # 1. 残差时序图
            axes[0, 0].plot(residuals.index, residuals.values, color='blue')
            axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
            axes[0, 0].set_title('残差时序图', fontweight='bold')
            axes[0, 0].set_xlabel('时间')
            axes[0, 0].set_ylabel('残差')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 残差直方图
            axes[0, 1].hist(residuals.values, bins=30, color='steelblue',
                          edgecolor='black', alpha=0.7)
            axes[0, 1].set_title('残差分布', fontweight='bold')
            axes[0, 1].set_xlabel('残差')
            axes[0, 1].set_ylabel('频数')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 残差ACF
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(residuals.dropna(), ax=axes[1, 0], lags=20)
            axes[1, 0].set_title('残差自相关图', fontweight='bold')

            # 4. QQ图
            stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('残差QQ图', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)

            fig.suptitle(title, fontsize=16, fontweight='bold')
            self.plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            return fig

    def plot_pca_variance(
        self,
        explained_variance_ratio: np.ndarray,
        cumulative_variance_ratio: np.ndarray,
        title: str = "PCA方差贡献分析",
        save_path: Optional[str] = None
    ) -> Any:
        """绘制PCA方差贡献图

        Args:
            explained_variance_ratio: 各主成分方差贡献率
            cumulative_variance_ratio: 累计方差贡献率
            title: 图表标题
            save_path: 保存路径

        Returns:
            图表对象
        """
        logger.info(f"绘制PCA方差贡献图: {title}")

        n_components = len(explained_variance_ratio)
        components = [f'PC{i+1}' for i in range(n_components)]

        if self.backend == 'plotly':
            fig = self.go.Figure()

            # 方差贡献率柱状图
            fig.add_trace(self.go.Bar(
                x=components,
                y=explained_variance_ratio * 100,
                name='方差贡献率',
                marker_color='steelblue',
                yaxis='y1'
            ))

            # 累计方差贡献率折线图
            fig.add_trace(self.go.Scatter(
                x=components,
                y=cumulative_variance_ratio * 100,
                name='累计方差贡献率',
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=8),
                yaxis='y2'
            ))

            fig.update_layout(
                title=title,
                xaxis_title='主成分',
                yaxis=dict(title='方差贡献率 (%)', side='left'),
                yaxis2=dict(title='累计方差贡献率 (%)',
                           side='right', overlaying='y', range=[0, 105]),
                template='plotly_white',
                hovermode='x unified'
            )

            if save_path:
                fig.write_html(save_path)

            return fig

        else:  # matplotlib
            fig, ax1 = self.plt.subplots(figsize=(12, 6))

            # 方差贡献率柱状图
            x_pos = np.arange(n_components)
            ax1.bar(x_pos, explained_variance_ratio * 100,
                   color='steelblue', alpha=0.7, label='方差贡献率')
            ax1.set_xlabel('主成分', fontsize=12)
            ax1.set_ylabel('方差贡献率 (%)', fontsize=12, color='steelblue')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(components)
            ax1.tick_params(axis='y', labelcolor='steelblue')

            # 累计方差贡献率折线图
            ax2 = ax1.twinx()
            ax2.plot(x_pos, cumulative_variance_ratio * 100,
                    color='red', marker='o', linewidth=2,
                    markersize=8, label='累计方差贡献率')
            ax2.set_ylabel('累计方差贡献率 (%)', fontsize=12, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim([0, 105])

            # 图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            self.plt.title(title, fontsize=14, fontweight='bold')
            self.plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            return fig

    def plot_factor_loadings(
        self,
        loadings: pd.DataFrame,
        title: str = "因子载荷热力图",
        save_path: Optional[str] = None
    ) -> Any:
        """绘制因子载荷热力图

        Args:
            loadings: 载荷矩阵DataFrame (变量 x 因子)
            title: 图表标题
            save_path: 保存路径

        Returns:
            图表对象
        """
        logger.info(f"绘制因子载荷热力图: {title}")

        if self.backend == 'plotly':
            fig = self.go.Figure(data=self.go.Heatmap(
                z=loadings.values,
                x=loadings.columns,
                y=loadings.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(loadings.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="载荷值")
            ))

            fig.update_layout(
                title=title,
                xaxis_title='因子',
                yaxis_title='变量',
                template='plotly_white',
                height=max(400, len(loadings) * 20)
            )

            if save_path:
                fig.write_html(save_path)

            return fig

        else:  # matplotlib
            fig, ax = self.plt.subplots(figsize=(10, max(8, len(loadings) * 0.3)))

            self.sns.heatmap(
                loadings,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': '载荷值'},
                ax=ax
            )

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('因子', fontsize=12)
            ax.set_ylabel('变量', fontsize=12)

            self.plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            return fig

    def plot_factor_loading_clustermap(
        self,
        loadings: pd.DataFrame,
        title: str = "因子载荷聚类图",
        save_path: Optional[str] = None
    ) -> Any:
        """绘制因子载荷聚类图（层次聚类热力图）

        Args:
            loadings: 载荷矩阵DataFrame
            title: 图表标题
            save_path: 保存路径

        Returns:
            图表对象
        """
        logger.info(f"绘制因子载荷聚类图: {title}")

        if self.backend == 'matplotlib':
            g = self.sns.clustermap(
                loadings,
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt='.2f',
                figsize=(12, max(10, len(loadings) * 0.4)),
                dendrogram_ratio=0.15,
                cbar_kws={'label': '载荷值'}
            )

            g.fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

            if save_path:
                g.savefig(save_path, dpi=300, bbox_inches='tight')

            return g

        else:
            logger.warning("聚类图仅支持matplotlib后端，请切换backend='matplotlib'")
            return self.plot_factor_loadings(loadings, title, save_path)

    def plot_industry_vs_driving_factor(
        self,
        industry_data: pd.Series,
        factor_data: pd.Series,
        industry_name: str,
        factor_name: str = "驱动因子",
        save_path: Optional[str] = None
    ) -> Any:
        """绘制行业指标vs驱动因子对比图

        Args:
            industry_data: 行业数据
            factor_data: 因子数据
            industry_name: 行业名称
            factor_name: 因子名称
            save_path: 保存路径

        Returns:
            图表对象
        """
        logger.info(f"绘制行业对比图: {industry_name} vs {factor_name}")

        # 数据标准化（用于可视化）
        industry_std = (industry_data - industry_data.mean()) / industry_data.std()
        factor_std = (factor_data - factor_data.mean()) / factor_data.std()

        if self.backend == 'plotly':
            fig = self.make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                self.go.Scatter(x=industry_std.index, y=industry_std.values,
                              name=industry_name, mode='lines',
                              line=dict(color='blue', width=2)),
                secondary_y=False
            )

            fig.add_trace(
                self.go.Scatter(x=factor_std.index, y=factor_std.values,
                              name=factor_name, mode='lines',
                              line=dict(color='red', width=2, dash='dash')),
                secondary_y=True
            )

            fig.update_layout(
                title=f"{industry_name} vs {factor_name} (标准化)",
                xaxis_title="时间",
                hovermode='x unified',
                template='plotly_white'
            )

            fig.update_yaxes(title_text=industry_name, secondary_y=False)
            fig.update_yaxes(title_text=factor_name, secondary_y=True)

            if save_path:
                fig.write_html(save_path)

            return fig

        else:  # matplotlib
            fig, ax = self.plt.subplots(figsize=(12, 6))

            ax.plot(industry_std.index, industry_std.values,
                   label=industry_name, color='blue', linewidth=2)
            ax.plot(factor_std.index, factor_std.values,
                   label=factor_name, color='red', linewidth=2, linestyle='--')

            ax.set_title(f"{industry_name} vs {factor_name} (标准化)",
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('时间', fontsize=12)
            ax.set_ylabel('标准化值', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            self.plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            return fig

    def plot_aligned_loading_comparison(
        self,
        loading1: np.ndarray,
        loading2: np.ndarray,
        labels: list,
        title: str = "因子载荷对比",
        save_path: Optional[str] = None
    ) -> Any:
        """绘制因子载荷对比图（用于比较不同模型）

        Args:
            loading1: 第一组载荷
            loading2: 第二组载荷
            labels: 变量标签
            title: 图表标题
            save_path: 保存路径

        Returns:
            图表对象
        """
        logger.info(f"绘制载荷对比图: {title}")

        if self.backend == 'plotly':
            fig = self.go.Figure()

            fig.add_trace(self.go.Bar(
                x=labels,
                y=loading1,
                name='模型1',
                marker_color='steelblue'
            ))

            fig.add_trace(self.go.Bar(
                x=labels,
                y=loading2,
                name='模型2',
                marker_color='coral'
            ))

            fig.update_layout(
                title=title,
                xaxis_title='变量',
                yaxis_title='载荷值',
                barmode='group',
                template='plotly_white'
            )

            if save_path:
                fig.write_html(save_path)

            return fig

        else:  # matplotlib
            fig, ax = self.plt.subplots(figsize=(12, 6))

            x = np.arange(len(labels))
            width = 0.35

            ax.bar(x - width/2, loading1, width, label='模型1', color='steelblue')
            ax.bar(x + width/2, loading2, width, label='模型2', color='coral')

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('变量', fontsize=12)
            ax.set_ylabel('载荷值', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

            self.plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            return fig


def plot_results(*args, **kwargs):
    """绘图函数（向后兼容）"""
    visualizer = ResultVisualizer(backend=kwargs.get('backend', 'plotly'))

    plot_type = kwargs.get('plot_type', 'forecast_vs_actual')

    if plot_type == 'forecast_vs_actual':
        return visualizer.plot_forecast_vs_actual(*args, **kwargs)
    elif plot_type == 'residuals':
        return visualizer.plot_residuals(*args, **kwargs)
    elif plot_type == 'pca_variance':
        return visualizer.plot_pca_variance(*args, **kwargs)
    elif plot_type == 'factor_loadings':
        return visualizer.plot_factor_loadings(*args, **kwargs)
    else:
        raise ValueError(f"未知绘图类型: {plot_type}")
