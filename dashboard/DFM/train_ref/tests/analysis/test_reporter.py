# -*- coding: utf-8 -*-
"""
报告生成器单元测试

测试AnalysisReporter类的Excel报告生成功能
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from dashboard.DFM.train_ref.analysis.reporter import AnalysisReporter


@pytest.fixture
def temp_output_dir():
    """临时输出目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def reporter(temp_output_dir):
    """报告生成器实例"""
    return AnalysisReporter(output_dir=temp_output_dir)


@pytest.fixture
def sample_data():
    """样本数据"""
    np.random.seed(42)
    n_time = 100
    n_factors = 2
    n_obs = 5

    data = pd.DataFrame(np.random.randn(n_time, n_obs))
    factors = pd.DataFrame(np.random.randn(n_time, n_factors))
    loadings = np.random.randn(n_obs, n_factors)
    target_loading = np.array([1.5, -0.8])

    return {
        'data': data,
        'factors': factors,
        'loadings': loadings,
        'target_loading': target_loading,
        'n_time': n_time,
        'n_factors': n_factors,
        'n_obs': n_obs
    }


class TestAnalysisReporterInit:
    """测试AnalysisReporter初始化"""

    def test_init_creates_directory(self, temp_output_dir):
        """测试初始化创建输出目录"""
        new_dir = Path(temp_output_dir) / "reports"
        reporter = AnalysisReporter(output_dir=str(new_dir))

        assert new_dir.exists()
        assert reporter.output_dir == new_dir

    def test_init_with_existing_directory(self, temp_output_dir):
        """测试使用已存在目录初始化"""
        reporter = AnalysisReporter(output_dir=temp_output_dir)

        assert reporter.output_dir == Path(temp_output_dir)


class TestGeneratePCAReport:
    """测试PCA报告生成"""

    def test_basic_pca_report(self, reporter, sample_data):
        """测试基础PCA报告生成"""
        output_path = reporter.generate_pca_report(
            data=sample_data['data'],
            n_components=3,
            filename="test_pca.xlsx"
        )

        # 验证文件生成
        assert Path(output_path).exists()
        assert Path(output_path).suffix == '.xlsx'

        # 读取验证内容
        df = pd.read_excel(output_path, sheet_name='PCA分析')
        assert len(df) == 3
        assert '主成分' in df.columns
        assert '方差' in df.columns
        assert '方差贡献率(%)' in df.columns
        assert '累计方差贡献率(%)' in df.columns

    def test_pca_report_default_components(self, reporter, sample_data):
        """测试默认主成分数"""
        output_path = reporter.generate_pca_report(
            data=sample_data['data'],
            filename="test_pca_default.xlsx"
        )

        df = pd.read_excel(output_path, sheet_name='PCA分析')
        expected_components = min(sample_data['data'].shape)
        assert len(df) == expected_components


class TestGenerateR2Report:
    """测试R²报告生成"""

    def test_basic_r2_report(self, reporter, sample_data):
        """测试基础R²报告生成"""
        observables = pd.DataFrame(
            sample_data['factors'].values @ sample_data['loadings'].T,
            columns=['Var1', 'Var2', 'Var3', 'Var4', 'Var5']
        )

        output_path = reporter.generate_r2_report(
            observables=observables,
            factors=sample_data['factors'],
            loadings=sample_data['loadings'],
            filename="test_r2.xlsx"
        )

        # 验证文件生成
        assert Path(output_path).exists()

        # 读取验证内容
        df = pd.read_excel(output_path, sheet_name='个体变量R²')
        assert len(df) == 5
        assert '变量名称' in df.columns
        assert 'R²' in df.columns

    def test_r2_report_with_industry_map(self, reporter, sample_data):
        """测试带行业映射的R²报告"""
        observables = pd.DataFrame(
            sample_data['factors'].values @ sample_data['loadings'].T,
            columns=['钢铁1', '钢铁2', '有色1', '有色2', '煤炭1']
        )

        variable_industry_map = {
            '钢铁1': '钢铁',
            '钢铁2': '钢铁',
            '有色1': '有色金属',
            '有色2': '有色金属',
            '煤炭1': '煤炭'
        }

        output_path = reporter.generate_r2_report(
            observables=observables,
            factors=sample_data['factors'],
            loadings=sample_data['loadings'],
            variable_industry_map=variable_industry_map,
            filename="test_r2_industry.xlsx"
        )

        # 验证生成两个sheet
        excel_file = pd.ExcelFile(output_path)
        assert '个体变量R²' in excel_file.sheet_names
        assert '行业聚合R²' in excel_file.sheet_names

        # 验证行业sheet内容
        industry_df = pd.read_excel(output_path, sheet_name='行业聚合R²')
        assert '行业名称' in industry_df.columns
        assert 'R²' in industry_df.columns
        assert len(industry_df) == 3  # 3个行业


class TestGenerateContributionReport:
    """测试因子贡献度报告生成"""

    def test_basic_contribution_report(self, reporter, sample_data):
        """测试基础贡献度报告"""
        dates = pd.date_range('2020-01-01', periods=sample_data['n_time'], freq='D')
        factors = sample_data['factors'].copy()
        factors.index = dates

        output_path = reporter.generate_contribution_report(
            factors=factors,
            loadings=sample_data['loadings'],
            target_loading=sample_data['target_loading'],
            filename="test_contribution.xlsx"
        )

        # 验证文件生成
        assert Path(output_path).exists()

        # 读取验证内容
        df = pd.read_excel(output_path, sheet_name='因子贡献度')
        assert len(df) == sample_data['n_time']
        assert '0_contribution' in df.columns
        assert '1_contribution' in df.columns
        assert 'Total_forecast' in df.columns

    def test_contribution_report_with_target_actual(self, reporter, sample_data):
        """测试带实际值的贡献度报告"""
        dates = pd.date_range('2020-01-01', periods=sample_data['n_time'], freq='D')
        factors = sample_data['factors'].copy()
        factors.index = dates

        target_actual = pd.Series(
            np.random.randn(sample_data['n_time']),
            index=dates
        )

        output_path = reporter.generate_contribution_report(
            factors=factors,
            loadings=sample_data['loadings'],
            target_loading=sample_data['target_loading'],
            target_actual=target_actual,
            filename="test_contribution_with_actual.xlsx"
        )

        df = pd.read_excel(output_path, sheet_name='因子贡献度')
        assert 'Target_actual' in df.columns


class TestGenerateReportWithParams:
    """测试参数化综合报告生成"""

    def test_comprehensive_report(self, reporter, sample_data):
        """测试综合报告生成"""
        from dashboard.DFM.train_ref.training.config import DFMConfig

        # 准备训练结果
        training_result = {
            'A': np.random.randn(2, 2),
            'Q': np.random.randn(2, 2),
            'H': sample_data['loadings'],
            'R': np.random.randn(5, 5),
            'target_loading': sample_data['target_loading'],
            'converged': True,
            'n_iterations': 15
        }

        # 准备评估结果
        model_result = {
            'is_rmse': 0.123,
            'oos_rmse': 0.156,
            'is_hit_rate': 85.5,
            'oos_hit_rate': 78.3,
            'is_correlation': 0.92,
            'oos_correlation': 0.88
        }

        config = DFMConfig(
            k_factors=2,
            max_iterations=30,
            tolerance=1e-6,
            smoothing=True
        )

        output_path = reporter.generate_report_with_params(
            training_result=training_result,
            model_result=model_result,
            config=config,
            filename="test_comprehensive.xlsx"
        )

        # 验证文件生成
        assert Path(output_path).exists()

        # 验证所有sheet存在
        excel_file = pd.ExcelFile(output_path)
        expected_sheets = ['模型参数', '评估指标', '因子载荷', '状态转移矩阵']
        for sheet in expected_sheets:
            assert sheet in excel_file.sheet_names

        # 验证模型参数sheet
        params_df = pd.read_excel(output_path, sheet_name='模型参数')
        assert '参数名称' in params_df.columns
        assert '参数值' in params_df.columns

        # 验证评估指标sheet
        metrics_df = pd.read_excel(output_path, sheet_name='评估指标')
        assert '指标名称' in metrics_df.columns
        assert '样本内' in metrics_df.columns
        assert '样本外' in metrics_df.columns

        # 验证因子载荷sheet
        loadings_df = pd.read_excel(output_path, sheet_name='因子载荷')
        assert loadings_df.shape[0] == sample_data['n_obs']

        # 验证状态转移矩阵sheet
        transition_df = pd.read_excel(output_path, sheet_name='状态转移矩阵')
        assert transition_df.shape[0] == 2
        assert transition_df.shape[1] == 2

    def test_comprehensive_report_with_variable_names(self, reporter, sample_data):
        """测试带变量名的综合报告"""
        training_result = {
            'A': np.random.randn(2, 2),
            'Q': np.random.randn(2, 2),
            'H': sample_data['loadings'],
            'R': np.random.randn(5, 5),
            'target_loading': sample_data['target_loading'],
            'converged': True,
            'n_iterations': 10
        }

        model_result = {
            'is_rmse': 0.1,
            'oos_rmse': 0.15
        }

        from dashboard.DFM.train_ref.training.config import DFMConfig
        config = DFMConfig(k_factors=2)

        variable_names = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5']

        output_path = reporter.generate_report_with_params(
            training_result=training_result,
            model_result=model_result,
            config=config,
            variable_names=variable_names,
            filename="test_with_varnames.xlsx"
        )

        # 验证因子载荷使用了变量名
        loadings_df = pd.read_excel(output_path, sheet_name='因子载荷')
        assert loadings_df.iloc[:, 0].tolist() == variable_names


class TestFormatExcelSheet:
    """测试Excel格式化"""

    def test_format_excel_sheet(self, reporter, temp_output_dir):
        """测试Excel格式化功能"""
        # 创建测试数据
        df = pd.DataFrame({
            '列1': [1, 2, 3],
            '列2': [4, 5, 6],
            '列3': [7, 8, 9]
        })

        output_path = Path(temp_output_dir) / "test_format.xlsx"

        # 写入并格式化
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='测试', index=False)
            reporter._format_excel_sheet(writer, '测试', df)

        # 验证文件存在
        assert output_path.exists()

        # 读取验证（格式化不影响数据）
        df_read = pd.read_excel(output_path, sheet_name='测试')
        pd.testing.assert_frame_equal(df, df_read)


class TestReporterEdgeCases:
    """测试边界情况"""

    def test_empty_data(self, reporter):
        """测试空数据"""
        empty_data = pd.DataFrame()

        with pytest.raises((ValueError, Exception)):
            reporter.generate_pca_report(empty_data, n_components=2)

    def test_invalid_filename(self, reporter, sample_data):
        """测试无效文件名"""
        # 使用无后缀名，应自动添加.xlsx
        output_path = reporter.generate_pca_report(
            sample_data['data'],
            n_components=2,
            filename="test_no_extension"
        )

        assert Path(output_path).suffix == '.xlsx'

    def test_overwrite_existing_file(self, reporter, sample_data):
        """测试覆盖已存在文件"""
        filename = "test_overwrite.xlsx"

        # 第一次生成
        output_path1 = reporter.generate_pca_report(
            sample_data['data'],
            n_components=2,
            filename=filename
        )

        # 第二次生成（覆盖）
        output_path2 = reporter.generate_pca_report(
            sample_data['data'],
            n_components=3,
            filename=filename
        )

        assert output_path1 == output_path2
        assert Path(output_path2).exists()

        # 验证被覆盖（主成分数变为3）
        df = pd.read_excel(output_path2, sheet_name='PCA分析')
        assert len(df) == 3
