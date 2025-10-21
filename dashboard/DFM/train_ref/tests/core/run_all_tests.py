# -*- coding: utf-8 -*-
"""
运行所有核心层一致性测试

一次性执行所有测试并生成报告
"""

import sys
from pathlib import Path
import time
import io

# 添加项目路径 - 需要向上6层到达HTFA根目录
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置输出编码为UTF-8，避免Windows下的编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def run_test_module(module_name, test_func_name="main"):
    """运行单个测试模块

    Args:
        module_name: 模块名
        test_func_name: 测试函数名

    Returns:
        int: 返回码（0=成功，非0=失败）
    """
    try:
        module = __import__(module_name, fromlist=[test_func_name])
        test_func = getattr(module, test_func_name)
        return test_func()
    except Exception as e:
        print(f"\n[失败] 模块 {module_name} 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """运行所有测试"""
    print("="*80)
    print("DFM train_ref 核心层一致性测试套件")
    print("="*80)
    print(f"\n开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n本测试套件将验证train_ref与train_model的计算一致性")
    print("包括: 卡尔曼滤波、DFM模型、参数估计\n")

    start_time = time.time()

    test_modules = [
        ("test_kalman_consistency", "卡尔曼滤波一致性"),
        ("test_estimator_consistency", "参数估计一致性"),
        ("test_dfm_consistency", "DFM模型一致性"),
    ]

    results = []

    for module_name, description in test_modules:
        print("\n" + "="*80)
        print(f"运行: {description}")
        print("="*80)

        module_start = time.time()
        return_code = run_test_module(module_name)
        module_duration = time.time() - module_start

        passed = (return_code == 0)
        results.append((description, passed, module_duration))

        status = "[通过]" if passed else "[失败]"
        print(f"\n{description}: {status} (耗时: {module_duration:.2f}秒)")

    # 生成总结报告
    total_duration = time.time() - start_time

    print("\n" + "="*80)
    print("总体测试报告")
    print("="*80)

    print(f"\n结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_duration:.2f}秒\n")

    print("测试结果:")
    print("-" * 60)
    print(f"{'测试模块':<30} {'状态':<10} {'耗时(秒)':<10}")
    print("-" * 60)

    for description, passed, duration in results:
        status = "[通过]" if passed else "[失败]"
        print(f"{description:<30} {status:<10} {duration:>8.2f}")

    print("-" * 60)

    total_passed = sum(1 for _, passed, _ in results if passed)
    total_tests = len(results)

    print(f"\n总计: {total_passed}/{total_tests} 测试模块通过")

    if total_passed == total_tests:
        print("\n" + "="*80)
        print("恭喜！所有测试通过！")
        print("="*80)
        print("\ntrain_ref核心层实现与train_model完全一致")
        print("可以安全地使用新代码进行后续开发")
        print("\n" + "="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("部分测试失败")
        print("="*80)
        print(f"\n失败的模块:")
        for description, passed, _ in results:
            if not passed:
                print(f"  - {description}")
        print("\n请检查失败的测试，修复问题后重新运行")
        print("\n" + "="*80)
        return 1


if __name__ == "__main__":
    exit(main())
