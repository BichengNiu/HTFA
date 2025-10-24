"""
端到端配置测试 - 使用Playwright MCP验证train_ref在不同参数下的完整训练流程

测试范围:
- 数据上传和data_prep预处理
- 12个参数组合的模型训练
- 结果合理性验证
- 控制台输出监控

使用方法:
这是一个测试脚本模板,需要通过Claude Code的Playwright MCP工具手动执行。
不需要安装pytest或其他依赖。

执行流程:
1. 启动Streamlit应用: http://localhost:8501
2. 通过Playwright MCP按测试用例配置执行
3. 记录结果到test_results_{timestamp}.md

测试数据:
- 文件: data/经济数据库1017.xlsx
- 开始日期: 2020-01-01
- 预测变量: 约10-12个
"""

# 测试用例配置
TEST_CONFIGURATIONS = [
    {
        "id": "T1",
        "name": "基线_小因子数",
        "num_variables": 10,
        "variable_selection": "禁用",
        "factor_selection_method": "fixed",
        "k_factors": 2,
        "em_iterations": 30,
        "expected_k": 2,
        "description": "基准测试: k=2固定因子数"
    },
    {
        "id": "T2",
        "name": "基线_中因子数",
        "num_variables": 10,
        "variable_selection": "禁用",
        "factor_selection_method": "fixed",
        "k_factors": 3,
        "em_iterations": 30,
        "expected_k": 3,
        "description": "基准测试: k=3固定因子数"
    },
    {
        "id": "T3",
        "name": "基线_大因子数",
        "num_variables": 10,
        "variable_selection": "禁用",
        "factor_selection_method": "fixed",
        "k_factors": 5,
        "em_iterations": 30,
        "expected_k": 5,
        "description": "基准测试: k=5固定因子数"
    },
    {
        "id": "T4",
        "name": "PCA自动选择_标准阈值",
        "num_variables": 10,
        "variable_selection": "禁用",
        "factor_selection_method": "cumulative",
        "pca_threshold": 0.85,
        "k_factors": None,  # auto
        "em_iterations": 30,
        "expected_k": "auto (2-5)",
        "description": "PCA累积方差85%自动选择因子数"
    },
    {
        "id": "T5",
        "name": "PCA自动选择_高阈值",
        "num_variables": 10,
        "variable_selection": "禁用",
        "factor_selection_method": "cumulative",
        "pca_threshold": 0.90,
        "k_factors": None,  # auto
        "em_iterations": 30,
        "expected_k": "auto (3-7)",
        "description": "PCA累积方差90%自动选择因子数"
    },
    {
        "id": "T6",
        "name": "Elbow方法自动选择",
        "num_variables": 10,
        "variable_selection": "禁用",
        "factor_selection_method": "elbow",
        "k_factors": None,  # auto
        "em_iterations": 30,
        "expected_k": "auto (2-4)",
        "description": "使用Elbow方法自动确定因子数"
    },
    {
        "id": "T7",
        "name": "变量选择_固定因子",
        "num_variables": 12,
        "variable_selection": "backward",
        "factor_selection_method": "fixed",
        "k_factors": 3,
        "em_iterations": 30,
        "expected_k": 3,
        "expected_variables": "8-12",
        "description": "后向逐步变量选择 + k=3固定因子"
    },
    {
        "id": "T8",
        "name": "变量选择_PCA自动",
        "num_variables": 12,
        "variable_selection": "backward",
        "factor_selection_method": "cumulative",
        "pca_threshold": 0.85,
        "k_factors": None,  # auto
        "em_iterations": 30,
        "expected_k": "auto (2-5)",
        "expected_variables": "8-12",
        "description": "后向逐步变量选择 + PCA自动选择因子"
    },
    {
        "id": "T9",
        "name": "边界_单因子",
        "num_variables": 10,
        "variable_selection": "禁用",
        "factor_selection_method": "fixed",
        "k_factors": 1,
        "em_iterations": 30,
        "expected_k": 1,
        "description": "边界测试: 单因子模型"
    },
    {
        "id": "T10",
        "name": "边界_高维因子",
        "num_variables": 10,
        "variable_selection": "禁用",
        "factor_selection_method": "fixed",
        "k_factors": 10,
        "em_iterations": 30,
        "expected_k": 10,
        "description": "边界测试: k=10高维因子(等于变量数)"
    },
    {
        "id": "T11",
        "name": "EM迭代_少",
        "num_variables": 10,
        "variable_selection": "禁用",
        "factor_selection_method": "fixed",
        "k_factors": 3,
        "em_iterations": 10,
        "expected_k": 3,
        "description": "EM迭代次数测试: 10次(可能未收敛)"
    },
    {
        "id": "T12",
        "name": "EM迭代_多",
        "num_variables": 10,
        "variable_selection": "禁用",
        "factor_selection_method": "fixed",
        "k_factors": 3,
        "em_iterations": 50,
        "expected_k": 3,
        "description": "EM迭代次数测试: 50次(确保收敛)"
    },
]

# 验证规则
VALIDATION_RULES = {
    "training_success": {
        "status": "completed",
        "no_exceptions": True,
        "results_displayed": True
    },
    "metrics_validity": {
        "rmse": "> 0",
        "hit_rate": "[0, 1]",
        "correlation": "[-1, 1]",
        "selected_variables": "<= initial_variables",
        "k_factors": "符合配置"
    },
    "console_monitoring": {
        "no_errors": True,
        "check_keywords": ["ERROR", "WARNING", "Exception"],
        "check_convergence": True
    }
}

# 预期测试变量列表(示例 - 从经济数据库1017.xlsx中选择)
COMMON_TEST_VARIABLES = [
    "钢铁产量",
    "发电量",
    "货运量",
    "水泥产量",
    "汽车产量",
    "房地产投资",
    "制造业PMI",
    "进出口总额",
    "社会消费品零售",
    "固定资产投资",
    "CPI",
    "PPI"
]


def get_test_config(test_id: str) -> dict:
    """获取指定测试用例的配置"""
    for config in TEST_CONFIGURATIONS:
        if config["id"] == test_id:
            return config
    raise ValueError(f"未找到测试用例: {test_id}")


def validate_results(test_id: str, results: dict) -> dict:
    """
    验证测试结果是否符合预期

    Args:
        test_id: 测试用例ID
        results: 训练结果字典

    Returns:
        验证结果字典 {passed: bool, issues: list}
    """
    issues = []
    config = get_test_config(test_id)

    # 1. 训练成功性检查
    if results.get("status") != "completed":
        issues.append(f"训练状态异常: {results.get('status')}")

    # 2. 指标合理性检查
    rmse = results.get("rmse_oos", 0)
    if rmse <= 0:
        issues.append(f"RMSE无效: {rmse}")

    hit_rate = results.get("hit_rate_oos", -1)
    if not (0 <= hit_rate <= 1):
        issues.append(f"Hit Rate超出范围: {hit_rate}")

    correlation = results.get("correlation_oos", -2)
    if not (-1 <= correlation <= 1):
        issues.append(f"相关系数超出范围: {correlation}")

    # 3. 因子数检查
    actual_k = results.get("k_factors", 0)
    if config.get("k_factors") is not None:
        if actual_k != config["k_factors"]:
            issues.append(f"因子数不符: 期望{config['k_factors']}, 实际{actual_k}")

    # 4. 变量数检查
    actual_vars = results.get("num_selected_variables", 0)
    if actual_vars > config["num_variables"]:
        issues.append(f"选定变量数超出初始数量: {actual_vars} > {config['num_variables']}")

    # 5. 控制台错误检查
    if results.get("console_errors", 0) > 0:
        issues.append(f"发现{results['console_errors']}个控制台ERROR")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "warnings": results.get("console_warnings", [])
    }


def generate_test_report_template(test_id: str) -> str:
    """生成测试报告模板"""
    config = get_test_config(test_id)

    template = f"""
## 测试用例 {test_id}: {config['name']}

### 配置
- 变量数: {config['num_variables']}
- 变量选择: {config['variable_selection']}
- 因子选择方法: {config['factor_selection_method']}
- 因子数(k): {config.get('k_factors', 'auto')}
- EM最大迭代: {config['em_iterations']}
- 说明: {config['description']}

### 执行步骤
1. 导航到DFM模块
2. 上传数据文件(data/经济数据库1017.xlsx)
3. 设置开始日期为2020-01-01
4. 运行data_prep
5. 切换到模型训练tab
6. 选择{config['num_variables']}个预测变量
7. 配置训练参数
8. 开始训练
9. 记录结果

### 训练结果
- 状态: [待填写]
- 样本外RMSE: [待填写]
- 样本外Hit Rate: [待填写]
- 样本外相关系数: [待填写]
- 选定变量数: [待填写]
- 选定因子数: [待填写]
- 训练用时: [待填写]

### 控制台输出
- ERROR数量: [待填写]
- WARNING数量: [待填写]
- EM收敛信息: [待填写]

### 验证结果
- [ ] 训练成功完成
- [ ] 结果指标合理
- [ ] 无控制台ERROR
- [ ] 因子数符合配置

### 问题记录
[如有问题请在此记录]

---
"""
    return template


if __name__ == "__main__":
    # 打印所有测试用例配置
    print("train_ref端到端配置测试套件")
    print("=" * 60)
    print(f"\n共{len(TEST_CONFIGURATIONS)}个测试用例:\n")

    for i, config in enumerate(TEST_CONFIGURATIONS, 1):
        print(f"{i}. [{config['id']}] {config['name']}")
        print(f"   {config['description']}")
        print()

    print("=" * 60)
    print("\n使用Playwright MCP执行测试用例,记录结果到markdown报告")
