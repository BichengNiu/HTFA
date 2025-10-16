# -*- coding: utf-8 -*-
"""
DFM优化系统性能基准测试执行脚本

统一执行所有性能测试并生成完整报告：
1. 综合基准测试
2. 详细组件分析
3. 边界条件测试
4. 完整报告生成
"""

import sys
import os
import time
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_performance_analysis():
    """运行完整的性能分析"""
    print("=" * 80)
    print("DFM优化系统完整性能基准测试")
    print("=" * 80)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = {}

    # 1. 运行综合基准测试
    print("1. 运行综合基准测试...")
    print("-" * 40)

    from dashboard.DFM.train_model.performance_benchmark import DFMBenchmarkSuite

    benchmark_suite = DFMBenchmarkSuite()
    benchmark_results = benchmark_suite.run_full_benchmark_suite()

    results['benchmark_suite'] = benchmark_suite
    results['benchmark_results'] = benchmark_results

    print(f"[成功] 综合基准测试完成，共 {len(benchmark_results)} 项测试")

    # 显示综合基准测试报告
    benchmark_report = benchmark_suite.generate_performance_report()
    print("\n" + "=" * 50)
    print("综合基准测试报告")
    print("=" * 50)
    print(benchmark_report)

    # 保存综合基准报告
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    benchmark_filename = f"dfm_benchmark_report_{timestamp}.txt"
    benchmark_suite.save_report_to_file(benchmark_filename)

    print("\n" + "=" * 80)

    # 2. 运行详细组件分析
    print("2. 运行详细组件分析...")
    print("-" * 40)

    from dashboard.DFM.train_model.detailed_performance_analyzer import DetailedDFMAnalyzer, run_detailed_analysis

    analyzer, baseline_results, optimized_results = run_detailed_analysis()

    results['detailed_analyzer'] = analyzer
    results['baseline_results'] = baseline_results
    results['optimized_results'] = optimized_results

    print("[成功] 详细组件分析完成")

    # 显示详细分析报告
    detailed_report = analyzer.generate_detailed_report()
    print("\n" + "=" * 50)
    print("详细组件分析报告")
    print("=" * 50)
    print(detailed_report)

    # 保存详细分析报告
    detailed_filename = f"dfm_detailed_analysis_{timestamp}.txt"
    with open(detailed_filename, 'w', encoding='utf-8') as f:
        f.write(detailed_report)
    print(f"详细分析报告已保存到: {detailed_filename}")

    print("\n" + "=" * 80)

    # 3. 生成综合总结报告
    print("3. 生成综合总结报告...")
    print("-" * 40)

    comprehensive_report = generate_comprehensive_report(results)

    print("\n" + "=" * 50)
    print("综合性能评估总结")
    print("=" * 50)
    print(comprehensive_report)

    # 保存综合报告
    comprehensive_filename = f"dfm_comprehensive_performance_report_{timestamp}.txt"
    with open(comprehensive_filename, 'w', encoding='utf-8') as f:
        f.write(comprehensive_report)
    print(f"综合报告已保存到: {comprehensive_filename}")

    results['comprehensive_report'] = comprehensive_report

    print("\n" + "=" * 80)
    print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("所有性能测试执行完成！")
    print("=" * 80)

    return results


def generate_comprehensive_report(results: dict) -> str:
    """生成综合性能评估报告"""
    report_lines = [
        "=" * 80,
        "DFM优化系统综合性能评估报告",
        "=" * 80,
        f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]

    # 执行摘要
    report_lines.extend([
        "执行摘要:",
        "-" * 20
    ])

    # 综合基准测试结果摘要
    if 'benchmark_results' in results and results['benchmark_results']:
        benchmark_results = results['benchmark_results']
        successful_tests = [r for r in benchmark_results if r.success]

        if successful_tests:
            avg_speedup = sum(r.improvement_ratios.get('time_speedup', 1) for r in successful_tests) / len(successful_tests)
            avg_memory_reduction = sum(r.improvement_ratios.get('memory_reduction_pct', 0) for r in successful_tests) / len(successful_tests)

            report_lines.extend([
                f"[成功] 综合基准测试: {len(successful_tests)}/{len(benchmark_results)} 项测试成功",
                f"  平均时间加速比: {avg_speedup:.2f}x",
                f"  平均内存节省: {avg_memory_reduction:.1f}%"
            ])
        else:
            report_lines.append("[失败] 综合基准测试: 所有测试都失败了")
    else:
        report_lines.append("[失败] 综合基准测试: 未执行或执行失败")

    # 详细分析结果摘要
    if 'detailed_analyzer' in results:
        analyzer = results['detailed_analyzer']
        comparison = analyzer.compare_performance()

        if comparison and 'overall_improvement' in comparison:
            overall = comparison['overall_improvement']
            report_lines.extend([
                f"[成功] 详细组件分析: 执行成功",
                f"  总体加速比: {overall.get('total_speedup', 0):.2f}x",
                f"  总体时间节省: {overall.get('total_time_saved', 0):.3f}秒",
                f"  总体时间减少: {overall.get('total_reduction_percent', 0):.1f}%"
            ])
        else:
            report_lines.append("[成功] 详细组件分析: 执行成功，但缺少比较数据")
    else:
        report_lines.append("[失败] 详细组件分析: 未执行或执行失败")

    report_lines.append("")

    # 关键发现
    report_lines.extend([
        "关键发现:",
        "-" * 20
    ])

    key_findings = []

    # 从综合测试中提取关键发现
    if 'benchmark_results' in results:
        benchmark_results = results['benchmark_results']
        successful_tests = [r for r in benchmark_results if r.success]

        if successful_tests:
            # 找到最佳和最差性能测试
            best_test = max(successful_tests, key=lambda r: r.improvement_ratios.get('time_speedup', 1))
            worst_test = min(successful_tests, key=lambda r: r.improvement_ratios.get('time_speedup', 1))

            key_findings.extend([
                f"1. 最佳优化效果: {best_test.test_name}",
                f"   加速比: {best_test.improvement_ratios.get('time_speedup', 1):.2f}x",
                f"   变量数: {best_test.test_parameters.get('n_variables', 'N/A')}",
                "",
                f"2. 最小优化效果: {worst_test.test_name}",
                f"   加速比: {worst_test.improvement_ratios.get('time_speedup', 1):.2f}x",
                f"   变量数: {worst_test.test_parameters.get('n_variables', 'N/A')}",
                ""
            ])

            # 可扩展性分析
            scalability_tests = [r for r in successful_tests if "可扩展性" in r.test_name]
            if scalability_tests:
                key_findings.extend([
                    "3. 可扩展性表现:",
                ])
                for test in sorted(scalability_tests, key=lambda r: r.test_parameters.get('n_variables', 0)):
                    speedup = test.improvement_ratios.get('time_speedup', 1)
                    n_vars = test.test_parameters.get('n_variables', 0)
                    key_findings.append(f"   {n_vars}变量: {speedup:.2f}x 加速")
                key_findings.append("")

    # 从详细分析中提取关键发现
    if 'detailed_analyzer' in results:
        analyzer = results['detailed_analyzer']
        comparison = analyzer.compare_performance()

        if comparison and 'component_improvements' in comparison:
            component_improvements = comparison['component_improvements']

            if component_improvements:
                best_component = max(
                    component_improvements.items(),
                    key=lambda x: x[1].get('speedup_ratio', 0)
                )

                key_findings.extend([
                    f"4. 最优化组件: {best_component[0]}",
                    f"   加速比: {best_component[1].get('speedup_ratio', 0):.2f}x",
                    f"   时间节省: {best_component[1].get('time_saved_seconds', 0):.3f}秒",
                    ""
                ])

    if not key_findings:
        key_findings = ["未能提取关键发现，请查看详细报告。"]

    report_lines.extend(key_findings)

    # 性能改进建议
    report_lines.extend([
        "性能改进建议:",
        "-" * 30
    ])

    recommendations = []

    # 基于测试结果的建议
    if 'benchmark_results' in results:
        benchmark_results = results['benchmark_results']
        successful_tests = [r for r in benchmark_results if r.success]

        if successful_tests:
            avg_speedup = sum(r.improvement_ratios.get('time_speedup', 1) for r in successful_tests) / len(successful_tests)

            if avg_speedup > 2.0:
                recommendations.append("1. [推荐] 优化效果显著，强烈建议在生产环境启用优化")
            elif avg_speedup > 1.5:
                recommendations.append("1. [推荐] 优化效果良好，建议在生产环境启用优化")
            else:
                recommendations.append("1. [注意] 优化效果一般，建议根据具体场景决定是否启用")

            # 基于变量数的建议
            large_scale_tests = [r for r in successful_tests if r.test_parameters.get('n_variables', 0) > 15]
            if large_scale_tests:
                avg_large_speedup = sum(r.improvement_ratios.get('time_speedup', 1) for r in large_scale_tests) / len(large_scale_tests)
                recommendations.append(f"2. 对于大规模数据集（>15变量），优化效果更明显（{avg_large_speedup:.2f}x加速）")

            recommendations.append("3. 建议设置 use_optimization=True 参数以获得最佳性能")
            recommendations.append("4. 对于频繁的变量选择操作，优化带来的收益更显著")

    if not recommendations:
        recommendations = [
            "1. 基于测试结果，建议谨慎使用优化功能",
            "2. 建议在特定场景下进行更多测试"
        ]

    report_lines.extend(recommendations)

    # 技术细节总结
    report_lines.extend([
        "",
        "技术细节总结:",
        "-" * 30,
        "1. 优化机制:",
        "   - 预计算上下文: 避免重复数据预处理",
        "   - 智能缓存: 减少重复计算",
        "   - 优化评估器: 加速DFM评估过程",
        "",
        "2. 主要改进组件:",
        "   - 数据准备步骤的预计算",
        "   - DFM评估结果的缓存",
        "   - 内存使用优化",
        "",
        "3. 适用场景:",
        "   - 中大规模变量选择（变量数 > 8）",
        "   - 多次迭代的变量选择过程",
        "   - 对性能要求较高的应用"
    ])

    # 报告结尾
    report_lines.extend([
        "",
        "=" * 80,
        "报告结束",
        "=" * 80,
        f"注: 详细数据和分析请参考单独的基准测试报告和组件分析报告。",
        f"测试环境: Python {sys.version.split()[0]}, 系统时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80
    ])

    return "\n".join(report_lines)


if __name__ == "__main__":
    results = run_full_performance_analysis()
