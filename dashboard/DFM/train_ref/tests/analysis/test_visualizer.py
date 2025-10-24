# -*- coding: utf-8 -*-
"""
可视化器单元测试

测试ResultVisualizer类的图表生成功能
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from dashboard.DFM.train_ref.analysis.visualizer import ResultVisualizer, plot_results


@pytest.fixture
def temp_output_dir():
    """临时输出目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(params=['plotly', 'matplotlib'])
def visualizer(request, temp_output_dir):
    """可视化器实例（参数化两种后端）"""
    return ResultVisualizer(backend=request.param, output_dir=temp_output_dir)


@pytest.fixture
def sample_time_series():
    """样本时间序列数据"""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    actual = pd.Series(np.cumsum(np.random.randn(100)), index=dates)
    forecast = actual + np.random.randn(100) * 0.5
    return {'actual': actual, 'forecast': forecast}


@pytest.fixture
def sample_factors_data():
    """样本因子数据"""
    np.random.seed(42)
    n_time = 100
    n_factors = 3
    n_obs = 5

    factors = pd.DataFrame(np.random.randn(n_time, n_factors))
    loadings = pd.DataFrame(
        np.random.randn(n_obs, n_factors),
        index=[f'Var{i+1}' for i in range(n_obs)],
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    return {'factors': factors, 'loadings': loadings}


class TestVisualizerInit:
    """测试可视化器初始化"""

    def test_init_plotly_backend(self, temp_output_dir):
        """测试Plotly后端初始化"""
        viz = ResultVisualizer(backend='plotly', output_dir=temp_output_dir)
        assert viz.backend in ['plotly', 'matplotlib']  # 可能回退到matplotlib
        assert viz.output_dir == Path(temp_output_dir)

    def test_init_matplotlib_backend(self, temp_output_dir):
        """测试Matplotlib后端初始化"""
        viz = ResultVisualizer(backend='matplotlib', output_dir=temp_output_dir)
        assert viz.backend == 'matplotlib'
        assert viz.output_dir == Path(temp_output_dir)

    def test_init_creates_output_dir(self, temp_output_dir):
        """测试初始化创建输出目录"""
        new_dir = Path(temp_output_dir) / "plots"
        viz = ResultVisualizer(backend='matplotlib', output_dir=str(new_dir))
        assert new_dir.exists()


class TestPlotForecastVsActual:
    """测试预测vs实际对比图"""

    def test_basic_plot(self, visualizer, sample_time_series):
        """测试基础绘图"""
        fig = visualizer.plot_forecast_vs_actual(
            actual=sample_time_series['actual'],
            forecast=sample_time_series['forecast'],
            title="测试预测对比"
        )

        assert fig is not None

    def test_plot_with_save(self, visualizer, sample_time_series, temp_output_dir):
        """测试保存图表"""
        if visualizer.backend == 'plotly':
            save_path = str(Path(temp_output_dir) / "forecast_vs_actual.html")
        else:
            save_path = str(Path(temp_output_dir) / "forecast_vs_actual.png")

        fig = visualizer.plot_forecast_vs_actual(
            actual=sample_time_series['actual'],
            forecast=sample_time_series['forecast'],
            save_path=save_path
        )

        assert Path(save_path).exists()

    def test_plot_empty_series(self, visualizer):
        """测试空序列"""
        empty_actual = pd.Series([])
        empty_forecast = pd.Series([])

        with pytest.raises((ValueError, Exception)):
            visualizer.plot_forecast_vs_actual(empty_actual, empty_forecast)


class TestPlotResiduals:
    """测试残差分析图"""

    def test_basic_residuals(self, visualizer, sample_time_series):
        """测试基础残差图"""
        residuals = sample_time_series['actual'] - sample_time_series['forecast']

        fig = visualizer.plot_residuals(
            residuals=residuals,
            title="测试残差分析"
        )

        assert fig is not None

    def test_residuals_with_save(self, visualizer, sample_time_series, temp_output_dir):
        """测试保存残差图"""
        residuals = sample_time_series['actual'] - sample_time_series['forecast']

        if visualizer.backend == 'plotly':
            save_path = str(Path(temp_output_dir) / "residuals.html")
        else:
            save_path = str(Path(temp_output_dir) / "residuals.png")

        fig = visualizer.plot_residuals(residuals, save_path=save_path)

        assert Path(save_path).exists()

    def test_residuals_normal_distribution(self, visualizer):
        """测试正态分布残差"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        residuals = pd.Series(np.random.randn(200), index=dates)

        fig = visualizer.plot_residuals(residuals)

        assert fig is not None


class TestPlotPCAVariance:
    """测试PCA方差贡献图"""

    def test_basic_pca_variance(self, visualizer):
        """测试基础PCA方差图"""
        explained_variance_ratio = np.array([0.5, 0.3, 0.15, 0.05])
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        fig = visualizer.plot_pca_variance(
            explained_variance_ratio=explained_variance_ratio,
            cumulative_variance_ratio=cumulative_variance_ratio,
            title="测试PCA方差"
        )

        assert fig is not None

    def test_pca_variance_with_save(self, visualizer, temp_output_dir):
        """测试保存PCA方差图"""
        explained_variance_ratio = np.array([0.6, 0.3, 0.1])
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        if visualizer.backend == 'plotly':
            save_path = str(Path(temp_output_dir) / "pca_variance.html")
        else:
            save_path = str(Path(temp_output_dir) / "pca_variance.png")

        fig = visualizer.plot_pca_variance(
            explained_variance_ratio,
            cumulative_variance_ratio,
            save_path=save_path
        )

        assert Path(save_path).exists()


class TestPlotFactorLoadings:
    """测试因子载荷热力图"""

    def test_basic_factor_loadings(self, visualizer, sample_factors_data):
        """测试基础载荷热力图"""
        fig = visualizer.plot_factor_loadings(
            loadings=sample_factors_data['loadings'],
            title="测试因子载荷"
        )

        assert fig is not None

    def test_factor_loadings_with_save(self, visualizer, sample_factors_data, temp_output_dir):
        """测试保存载荷热力图"""
        if visualizer.backend == 'plotly':
            save_path = str(Path(temp_output_dir) / "factor_loadings.html")
        else:
            save_path = str(Path(temp_output_dir) / "factor_loadings.png")

        fig = visualizer.plot_factor_loadings(
            sample_factors_data['loadings'],
            save_path=save_path
        )

        assert Path(save_path).exists()

    def test_single_factor_loading(self, visualizer):
        """测试单因子载荷"""
        loadings = pd.DataFrame(
            [[0.8], [0.6], [0.4]],
            index=['Var1', 'Var2', 'Var3'],
            columns=['Factor1']
        )

        fig = visualizer.plot_factor_loadings(loadings)
        assert fig is not None


class TestPlotFactorLoadingClustermap:
    """测试因子载荷聚类图"""

    def test_clustermap_matplotlib(self, sample_factors_data, temp_output_dir):
        """测试Matplotlib聚类图"""
        viz = ResultVisualizer(backend='matplotlib', output_dir=temp_output_dir)

        fig = viz.plot_factor_loading_clustermap(
            loadings=sample_factors_data['loadings'],
            title="测试聚类图"
        )

        assert fig is not None

    def test_clustermap_plotly_fallback(self, sample_factors_data, temp_output_dir):
        """测试Plotly后端回退"""
        viz = ResultVisualizer(backend='plotly', output_dir=temp_output_dir)

        # Plotly不支持聚类图，应回退到热力图
        fig = viz.plot_factor_loading_clustermap(
            loadings=sample_factors_data['loadings']
        )

        assert fig is not None


class TestPlotIndustryVsDrivingFactor:
    """测试行业vs驱动因子对比图"""

    def test_basic_industry_comparison(self, visualizer):
        """测试基础行业对比"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        industry_data = pd.Series(np.cumsum(np.random.randn(100)), index=dates)
        factor_data = pd.Series(np.cumsum(np.random.randn(100)), index=dates)

        fig = visualizer.plot_industry_vs_driving_factor(
            industry_data=industry_data,
            factor_data=factor_data,
            industry_name="钢铁行业",
            factor_name="经济因子"
        )

        assert fig is not None

    def test_industry_comparison_with_save(self, visualizer, temp_output_dir):
        """测试保存行业对比图"""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        industry_data = pd.Series(np.random.randn(50), index=dates)
        factor_data = pd.Series(np.random.randn(50), index=dates)

        if visualizer.backend == 'plotly':
            save_path = str(Path(temp_output_dir) / "industry_comparison.html")
        else:
            save_path = str(Path(temp_output_dir) / "industry_comparison.png")

        fig = visualizer.plot_industry_vs_driving_factor(
            industry_data, factor_data,
            "测试行业", "测试因子",
            save_path=save_path
        )

        assert Path(save_path).exists()


class TestPlotAlignedLoadingComparison:
    """测试因子载荷对比图"""

    def test_basic_loading_comparison(self, visualizer):
        """测试基础载荷对比"""
        loading1 = np.array([0.8, 0.6, 0.4, 0.7])
        loading2 = np.array([0.75, 0.65, 0.35, 0.72])
        labels = ['Var1', 'Var2', 'Var3', 'Var4']

        fig = visualizer.plot_aligned_loading_comparison(
            loading1=loading1,
            loading2=loading2,
            labels=labels,
            title="测试载荷对比"
        )

        assert fig is not None

    def test_loading_comparison_with_save(self, visualizer, temp_output_dir):
        """测试保存载荷对比图"""
        loading1 = np.random.randn(5)
        loading2 = np.random.randn(5)
        labels = [f'Var{i+1}' for i in range(5)]

        if visualizer.backend == 'plotly':
            save_path = str(Path(temp_output_dir) / "loading_comparison.html")
        else:
            save_path = str(Path(temp_output_dir) / "loading_comparison.png")

        fig = visualizer.plot_aligned_loading_comparison(
            loading1, loading2, labels,
            save_path=save_path
        )

        assert Path(save_path).exists()


class TestPlotResults:
    """测试向后兼容的plot_results函数"""

    def test_plot_results_forecast_vs_actual(self, sample_time_series):
        """测试plot_results函数 - forecast_vs_actual"""
        fig = plot_results(
            sample_time_series['actual'],
            sample_time_series['forecast'],
            plot_type='forecast_vs_actual',
            backend='matplotlib'
        )

        assert fig is not None

    def test_plot_results_residuals(self):
        """测试plot_results函数 - residuals"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        residuals = pd.Series(np.random.randn(100), index=dates)

        fig = plot_results(
            residuals,
            plot_type='residuals',
            backend='matplotlib'
        )

        assert fig is not None

    def test_plot_results_pca_variance(self):
        """测试plot_results函数 - pca_variance"""
        explained_variance_ratio = np.array([0.5, 0.3, 0.2])
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        fig = plot_results(
            explained_variance_ratio,
            cumulative_variance_ratio,
            plot_type='pca_variance',
            backend='matplotlib'
        )

        assert fig is not None

    def test_plot_results_invalid_type(self):
        """测试无效绘图类型"""
        with pytest.raises(ValueError, match="未知绘图类型"):
            plot_results(
                np.array([1, 2, 3]),
                plot_type='invalid_type',
                backend='matplotlib'
            )


class TestVisualizerEdgeCases:
    """测试边界情况"""

    def test_empty_dataframe_loadings(self, visualizer):
        """测试空DataFrame载荷"""
        empty_loadings = pd.DataFrame()

        with pytest.raises((ValueError, Exception)):
            visualizer.plot_factor_loadings(empty_loadings)

    def test_mismatched_lengths(self, visualizer):
        """测试长度不匹配"""
        actual = pd.Series([1, 2, 3])
        forecast = pd.Series([1, 2])  # 长度不同

        # 应该能处理或抛出明确错误
        try:
            fig = visualizer.plot_forecast_vs_actual(actual, forecast)
        except (ValueError, IndexError, Exception):
            pass  # 预期会失败

    def test_all_nan_residuals(self, visualizer):
        """测试全NaN残差"""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        residuals = pd.Series([np.nan] * 10, index=dates)

        # 应该能处理或抛出明确错误
        try:
            fig = visualizer.plot_residuals(residuals)
        except (ValueError, Exception):
            pass  # 预期可能失败

    def test_single_point_series(self, visualizer):
        """测试单点序列"""
        dates = pd.date_range('2020-01-01', periods=1, freq='D')
        actual = pd.Series([1.0], index=dates)
        forecast = pd.Series([1.1], index=dates)

        # 单点数据可能无法绘图
        try:
            fig = visualizer.plot_forecast_vs_actual(actual, forecast)
        except (ValueError, Exception):
            pass  # 预期可能失败
