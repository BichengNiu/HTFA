"""
测试DFM模型训练页面UI布局修改
"""
import pytest


class TestUILayoutChanges:
    """测试UI布局修改"""

    def test_removed_section_headers(self):
        """验证已删除的section标题"""
        with open(
            r"C:\Users\NIU\Desktop\HTFA\dashboard\models\DFM\train\ui\pages\model_training_page.py",
            "r",
            encoding="utf-8"
        ) as f:
            content = f.read()

        removed_headers = [
            'st_instance.markdown("**因子参数**")',
            'st_instance.markdown("**EM算法参数**")',
            'st_instance.markdown("**筛选参数**")',
            'st_instance.markdown("**评分权重**")',
            'st_instance.markdown("**判定阈值**")',
            'st_instance.markdown("**变量保留限制**")',
        ]

        for header in removed_headers:
            assert header not in content, f"标题未删除: {header}"

    def test_three_column_layout_exists(self):
        """验证三列布局存在"""
        with open(
            r"C:\Users\NIU\Desktop\HTFA\dashboard\models\DFM\train\ui\pages\model_training_page.py",
            "r",
            encoding="utf-8"
        ) as f:
            content = f.read()

        assert "weight_col1, weight_col2, weight_col3 = st_instance.columns(3)" in content

    def test_em_and_min_vars_two_column_layout(self):
        """验证EM算法最大迭代次数和最少保留变量数在两列布局"""
        with open(
            r"C:\Users\NIU\Desktop\HTFA\dashboard\models\DFM\train\ui\pages\model_training_page.py",
            "r",
            encoding="utf-8"
        ) as f:
            content = f.read()

        assert "em_var_col1, em_var_col2 = st_instance.columns(2)" in content

    def test_invalid_n_jobs_raises_error(self):
        """验证无效并行任务数抛出ValueError"""
        with open(
            r"C:\Users\NIU\Desktop\HTFA\dashboard\models\DFM\train\ui\pages\model_training_page.py",
            "r",
            encoding="utf-8"
        ) as f:
            content = f.read()

        assert 'raise ValueError(f"无效的并行任务数: {current_n_jobs}，有效值: {n_jobs_options}")' in content

    def test_no_fallback_pattern_for_n_jobs(self):
        """验证没有n_jobs的回退模式"""
        with open(
            r"C:\Users\NIU\Desktop\HTFA\dashboard\models\DFM\train\ui\pages\model_training_page.py",
            "r",
            encoding="utf-8"
        ) as f:
            content = f.read()

        assert "if current_n_jobs in n_jobs_options else 0" not in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
