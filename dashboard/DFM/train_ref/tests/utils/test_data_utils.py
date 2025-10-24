# -*- coding: utf-8 -*-
"""
data_utils单元测试

测试内容:
1. load_data: 数据加载
2. prepare_data: 数据准备
3. split_train_validation: 数据分割
4. standardize_data: 数据标准化
5. destandardize_series: 反标准化
6. apply_seasonal_mask: 季节性掩码
7. verify_alignment: 数据对齐验证
8. check_data_quality: 数据质量检查
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from dashboard.DFM.train_ref.utils.data_utils import (
    load_data,
    prepare_data,
    split_train_validation,
    standardize_data,
    destandardize_series,
    apply_seasonal_mask,
    verify_alignment,
    check_data_quality
)


class TestLoadData:
    """load_data函数测试"""

    @pytest.fixture
    def temp_excel_file(self, tmp_path):
        """创建临时Excel文件"""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({'var1': range(10), 'var2': range(10, 20)}, index=dates)
        file_path = tmp_path / "test_data.xlsx"
        data.to_excel(file_path)
        return str(file_path)

    @pytest.fixture
    def temp_csv_file(self, tmp_path):
        """创建临时CSV文件"""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({'var1': range(10), 'var2': range(10, 20)}, index=dates)
        file_path = tmp_path / "test_data.csv"
        data.to_csv(file_path)
        return str(file_path)

    def test_load_data_excel(self, temp_excel_file):
        """测试加载Excel文件"""
        df = load_data(temp_excel_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert 'var1' in df.columns
        assert 'var2' in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_data_csv(self, temp_csv_file):
        """测试加载CSV文件"""
        df = load_data(temp_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_data_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.xlsx")

    def test_load_data_unsupported_format(self, tmp_path):
        """测试不支持的文件格式"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test data")

        with pytest.raises(ValueError, match="不支持的文件格式"):
            load_data(str(file_path))


class TestPrepareData:
    """prepare_data函数测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'target': np.random.randn(100),
            'var1': np.random.randn(100),
            'var2': np.random.randn(100),
            'var3': np.random.randn(100)
        }, index=dates)
        return data

    def test_prepare_data_basic(self, sample_data):
        """测试基础数据准备"""
        prepared_data, predictor_vars = prepare_data(
            sample_data,
            target_variable='target'
        )

        assert isinstance(prepared_data, pd.DataFrame)
        assert 'target' in prepared_data.columns
        assert len(predictor_vars) == 3
        assert 'target' not in predictor_vars

    def test_prepare_data_with_selected_variables(self, sample_data):
        """测试指定变量"""
        prepared_data, predictor_vars = prepare_data(
            sample_data,
            target_variable='target',
            selected_variables=['target', 'var1', 'var2']
        )

        assert len(prepared_data.columns) == 3
        assert 'var1' in prepared_data.columns
        assert 'var2' in prepared_data.columns
        assert 'var3' not in prepared_data.columns
        assert len(predictor_vars) == 2

    def test_prepare_data_with_end_date(self, sample_data):
        """测试截止日期"""
        prepared_data, predictor_vars = prepare_data(
            sample_data,
            target_variable='target',
            end_date='2020-02-01'
        )

        assert len(prepared_data) <= 32  # 最多32天

    def test_prepare_data_missing_target(self, sample_data):
        """测试目标变量不存在"""
        with pytest.raises(ValueError, match="目标变量.*不在数据中"):
            prepare_data(sample_data, target_variable='nonexistent')


class TestSplitTrainValidation:
    """split_train_validation函数测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({'var1': range(100), 'var2': range(100, 200)}, index=dates)
        return data

    def test_split_train_validation_basic(self, sample_data):
        """测试基础分割"""
        train_data, val_data = split_train_validation(
            sample_data,
            train_end='2020-02-15'
        )

        assert isinstance(train_data, pd.DataFrame)
        assert isinstance(val_data, pd.DataFrame)
        assert len(train_data) < len(sample_data)
        assert len(train_data) + len(val_data) >= len(sample_data)

    def test_split_train_validation_with_dates(self, sample_data):
        """测试指定验证集日期范围"""
        train_data, val_data = split_train_validation(
            sample_data,
            train_end='2020-02-15',
            validation_start='2020-02-16',
            validation_end='2020-03-01'
        )

        assert train_data.index.max() <= pd.Timestamp('2020-02-15')
        assert val_data.index.min() >= pd.Timestamp('2020-02-16')
        assert val_data.index.max() <= pd.Timestamp('2020-03-01')

    def test_split_train_validation_no_overlap(self, sample_data):
        """测试训练集和验证集不重叠"""
        train_data, val_data = split_train_validation(
            sample_data,
            train_end='2020-02-15',
            validation_start='2020-02-16'
        )

        # 确保没有重叠日期
        assert train_data.index.max() < val_data.index.min()


class TestStandardizeData:
    """standardize_data函数测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'var1': np.random.randn(100) + 5.0,
            'var2': np.random.randn(100) * 2.0 + 10.0
        }, index=dates)
        return data

    def test_standardize_data_basic(self, sample_data):
        """测试基础标准化"""
        standardized, stats = standardize_data(sample_data)

        assert isinstance(standardized, pd.DataFrame)
        assert standardized.shape == sample_data.shape
        assert len(stats) == 2

        # 标准化后均值接近0，标准差接近1
        for col in standardized.columns:
            assert standardized[col].mean() == pytest.approx(0.0, abs=1e-10)
            assert standardized[col].std() == pytest.approx(1.0, abs=1e-10)

    def test_standardize_data_with_fit_data(self, sample_data):
        """测试使用不同数据计算统计量"""
        train_data = sample_data.iloc[:50]
        test_data = sample_data.iloc[50:]

        standardized, stats = standardize_data(test_data, fit_data=train_data)

        # 统计量应该来自train_data
        for col in sample_data.columns:
            mean_val, std_val = stats[col]
            assert mean_val == pytest.approx(train_data[col].mean(), abs=1e-6)
            assert std_val == pytest.approx(train_data[col].std(), abs=1e-6)

    def test_standardize_data_with_nan(self):
        """测试包含NaN的数据"""
        data = pd.DataFrame({
            'var1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'var2': [5.0, 6.0, 7.0, 8.0, 9.0]
        })

        standardized, stats = standardize_data(data)

        # 应该能处理NaN
        assert standardized.shape == data.shape
        assert len(stats) == 2

    def test_standardize_data_constant_series(self):
        """测试常数序列"""
        data = pd.DataFrame({
            'var1': [5.0] * 10,
            'var2': [1.0, 2.0, 3.0, 4.0, 5.0] * 2
        })

        standardized, stats = standardize_data(data)

        # 常数序列应保持不变或被处理
        assert 'var1' in stats
        assert stats['var1'][1] == 1.0  # std设为1.0


class TestDestandardizeSeries:
    """destandardize_series函数测试"""

    def test_destandardize_series_basic(self):
        """测试基础反标准化"""
        standardized = pd.Series([0.0, 1.0, -1.0])
        mean = 5.0
        std = 2.0

        destandardized = destandardize_series(standardized, mean, std)

        expected = pd.Series([5.0, 7.0, 3.0])
        pd.testing.assert_series_equal(destandardized, expected)

    def test_destandardize_series_identity(self):
        """测试标准化和反标准化互为逆操作"""
        original = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = original.mean()
        std = original.std()

        standardized = (original - mean) / std
        destandardized = destandardize_series(standardized, mean, std)

        pd.testing.assert_series_equal(destandardized, original)


class TestApplySeasonalMask:
    """apply_seasonal_mask函数测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        data = pd.DataFrame({
            'target': np.random.randn(365),
            'var1': np.random.randn(365)
        }, index=dates)
        return data

    def test_apply_seasonal_mask_default(self, sample_data):
        """测试默认掩码（1、2月）"""
        masked = apply_seasonal_mask(sample_data, target_variable='target')

        # 1、2月的target应该被设为NaN
        jan_feb_mask = masked.index.month.isin([1, 2])
        assert masked.loc[jan_feb_mask, 'target'].isna().all()

        # 其他月份不受影响
        other_mask = ~jan_feb_mask
        assert masked.loc[other_mask, 'target'].notna().all()

    def test_apply_seasonal_mask_custom_months(self, sample_data):
        """测试自定义掩码月份"""
        masked = apply_seasonal_mask(
            sample_data,
            target_variable='target',
            months_to_mask=[6, 7, 8]
        )

        # 6、7、8月应该被掩码
        summer_mask = masked.index.month.isin([6, 7, 8])
        assert masked.loc[summer_mask, 'target'].isna().all()

    def test_apply_seasonal_mask_only_target(self, sample_data):
        """测试只掩码目标变量"""
        masked = apply_seasonal_mask(sample_data, target_variable='target')

        # var1不应该被掩码
        assert masked['var1'].notna().all()


class TestVerifyAlignment:
    """verify_alignment函数测试"""

    @pytest.fixture
    def aligned_data(self):
        """创建对齐的数据"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'var1': np.random.randn(100),
            'var2': np.random.randn(100),
            'var3': np.random.randn(100)
        }, index=dates)
        return data

    @pytest.fixture
    def misaligned_data(self):
        """创建未对齐的数据（含缺失值）"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'var1': np.random.randn(100),
            'var2': np.random.randn(100),
            'var3': np.random.randn(100)
        }, index=dates)
        # 添加一些缺失值
        data.loc[data.index[:10], 'var1'] = np.nan
        data.loc[data.index[20:30], 'var2'] = np.nan
        return data

    def test_verify_alignment_aligned(self, aligned_data):
        """测试对齐的数据"""
        is_aligned, diagnosis = verify_alignment(
            aligned_data,
            variables=['var1', 'var2', 'var3'],
            strict=True
        )

        assert is_aligned is True
        assert diagnosis['common_dates'] == 100
        assert len(diagnosis['missing_variables']) == 0

    def test_verify_alignment_with_missing(self, misaligned_data):
        """测试有缺失值的数据（严格模式）"""
        is_aligned, diagnosis = verify_alignment(
            misaligned_data,
            variables=['var1', 'var2', 'var3'],
            strict=True
        )

        assert is_aligned is False
        assert len(diagnosis['issues']) > 0
        assert diagnosis['missing_counts']['var1'] == 10
        assert diagnosis['missing_counts']['var2'] == 10

    def test_verify_alignment_non_strict(self, misaligned_data):
        """测试非严格模式"""
        is_aligned, diagnosis = verify_alignment(
            misaligned_data,
            variables=['var1', 'var2', 'var3'],
            strict=False
        )

        # 非严格模式可能通过（如果共同日期足够多）
        assert diagnosis['common_dates'] > 0

    def test_verify_alignment_missing_variable(self, aligned_data):
        """测试缺失变量"""
        is_aligned, diagnosis = verify_alignment(
            aligned_data,
            variables=['var1', 'var2', 'nonexistent'],
            strict=True
        )

        assert is_aligned is False
        assert 'nonexistent' in diagnosis['missing_variables']


class TestCheckDataQuality:
    """check_data_quality函数测试"""

    @pytest.fixture
    def good_quality_data(self):
        """创建高质量数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'var1': np.random.randn(100) + 5.0,
            'var2': np.random.randn(100) * 2.0
        }, index=dates)
        return data

    @pytest.fixture
    def poor_quality_data(self):
        """创建低质量数据"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'var1': [1.0] * 100,  # 常数序列
            'var2': [np.nan] * 50 + list(np.random.randn(50)),  # 50%缺失
            'var3': np.random.randn(100)
        }, index=dates)
        return data

    def test_check_data_quality_good(self, good_quality_data):
        """测试高质量数据"""
        passed, problematic, diagnosis = check_data_quality(good_quality_data)

        assert passed is True
        assert len(problematic) == 0
        assert diagnosis['total_variables'] == 2

    def test_check_data_quality_poor(self, poor_quality_data):
        """测试低质量数据"""
        passed, problematic, diagnosis = check_data_quality(
            poor_quality_data,
            max_missing_ratio=0.3
        )

        assert passed is False
        assert len(problematic) >= 2  # var1和var2应该有问题
        assert 'var1' in problematic  # 常数序列
        assert 'var2' in problematic  # 缺失值过多

    def test_check_data_quality_custom_thresholds(self, poor_quality_data):
        """测试自定义阈值"""
        # 放宽阈值
        passed, problematic, diagnosis = check_data_quality(
            poor_quality_data,
            max_missing_ratio=0.6,  # 允许60%缺失
            min_variance=1e-10
        )

        # var2可能通过，但var1仍然是常数
        assert 'var1' in problematic

    def test_check_data_quality_specific_variables(self, poor_quality_data):
        """测试指定变量检查"""
        passed, problematic, diagnosis = check_data_quality(
            poor_quality_data,
            variables=['var3']  # 只检查var3
        )

        assert passed is True
        assert len(problematic) == 0
        assert diagnosis['total_variables'] == 1

    def test_check_data_quality_metrics(self, good_quality_data):
        """测试诊断指标"""
        passed, problematic, diagnosis = check_data_quality(good_quality_data)

        for var in good_quality_data.columns:
            metrics = diagnosis['quality_metrics'][var]
            assert 'missing_count' in metrics
            assert 'missing_ratio' in metrics
            assert 'variance' in metrics
            assert 'is_constant' in metrics
            assert metrics['is_constant'] == False  # 使用==而非is比较
