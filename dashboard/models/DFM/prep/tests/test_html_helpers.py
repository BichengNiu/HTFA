# -*- coding: utf-8 -*-
"""
html_helpers模块单元测试

测试:
1. render_tag - 单个标签渲染
2. render_tag_group - 标签组渲染
3. render_grouped_tags - 分组标签渲染
"""

import sys
import os
from pathlib import Path

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dashboard.models.DFM.prep.utils.html_helpers import (
    DEFAULT_TAG_STYLE,
    render_tag,
    render_tag_group,
    render_grouped_tags
)


class TestRenderTag:
    """render_tag函数测试"""

    def test_basic_rendering(self):
        """测试基本渲染"""
        result = render_tag("test")
        assert "<span" in result
        assert "test</span>" in result
        assert DEFAULT_TAG_STYLE in result

    def test_chinese_content(self):
        """测试中文内容"""
        result = render_tag("变量A")
        assert "变量A" in result
        assert "<span" in result

    def test_custom_style(self):
        """测试自定义样式"""
        custom_style = "color:red;"
        result = render_tag("test", style=custom_style)
        assert custom_style in result
        assert DEFAULT_TAG_STYLE not in result

    def test_special_characters(self):
        """测试特殊字符"""
        result = render_tag("GDP:同比增速(%)")
        assert "GDP:同比增速(%)" in result

    def test_empty_content(self):
        """测试空内容"""
        result = render_tag("")
        assert "<span" in result
        assert "</span>" in result


class TestRenderTagGroup:
    """render_tag_group函数测试"""

    def test_multiple_items(self):
        """测试多个项目"""
        items = ["变量A", "变量B", "变量C"]
        result = render_tag_group(items)

        for item in items:
            assert item in result

        # 应包含3个span标签
        assert result.count("<span") == 3
        assert result.count("</span>") == 3

    def test_empty_list(self):
        """测试空列表"""
        result = render_tag_group([])
        assert result == ""

    def test_single_item(self):
        """测试单个项目"""
        result = render_tag_group(["唯一变量"])
        assert "唯一变量" in result
        assert result.count("<span") == 1

    def test_custom_style(self):
        """测试自定义样式"""
        custom_style = "background:blue;"
        result = render_tag_group(["A", "B"], style=custom_style)
        assert result.count(custom_style) == 2


class TestRenderGroupedTags:
    """render_grouped_tags函数测试"""

    def test_basic_grouping(self):
        """测试基本分组"""
        groups = {
            "原因A": ["变量1", "变量2"],
            "原因B": ["变量3"]
        }
        result = render_grouped_tags(groups)

        assert len(result) == 2
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)

    def test_group_format(self):
        """测试分组格式"""
        groups = {"测试组": ["项目1", "项目2", "项目3"]}
        result = render_grouped_tags(groups)

        title, html = result[0]
        assert "测试组" in title
        assert "3个" in title
        assert "项目1" in html

    def test_custom_format(self):
        """测试自定义格式"""
        groups = {"组A": ["a", "b"]}
        custom_format = "[{key}] - 共{count}项"
        result = render_grouped_tags(groups, group_format=custom_format)

        title, _ = result[0]
        assert "[组A]" in title
        assert "共2项" in title

    def test_empty_groups(self):
        """测试空分组"""
        result = render_grouped_tags({})
        assert result == []

    def test_preserves_order(self):
        """测试保持顺序(Python 3.7+字典有序)"""
        groups = {"第一组": ["a"], "第二组": ["b"], "第三组": ["c"]}
        result = render_grouped_tags(groups)

        titles = [item[0] for item in result]
        assert "第一组" in titles[0]
        assert "第二组" in titles[1]
        assert "第三组" in titles[2]


def run_tests():
    """运行所有测试"""
    import traceback

    test_classes = [TestRenderTag, TestRenderTagGroup, TestRenderGroupedTags]
    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        instance = test_class()
        print(f"\n{'='*50}")
        print(f"运行测试类: {test_class.__name__}")
        print('='*50)

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"  [PASS] {method_name}")
                    passed_tests += 1
                except AssertionError as e:
                    print(f"  [FAIL] {method_name}: {e}")
                    failed_tests.append((test_class.__name__, method_name, str(e)))
                except Exception as e:
                    print(f"  [ERROR] {method_name}: {e}")
                    traceback.print_exc()
                    failed_tests.append((test_class.__name__, method_name, str(e)))

    print(f"\n{'='*50}")
    print(f"测试结果: {passed_tests}/{total_tests} 通过")
    print('='*50)

    if failed_tests:
        print("\n失败的测试:")
        for class_name, method, error in failed_tests:
            print(f"  - {class_name}.{method}: {error}")
        return 1
    return 0


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
