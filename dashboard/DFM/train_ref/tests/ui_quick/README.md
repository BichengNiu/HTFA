# train_ref快速UI验证测试

## 目的

使用Playwright MCP快速验证train_ref模块在不同参数配置下的稳定性和结果合理性。

## 测试范围

测试3个核心参数组合:
1. T1: k=2, fixed, 10个变量
2. T2: k=3, fixed, 10个变量
3. T7: k=3, fixed, 12个变量 + backward变量选择

## 使用说明

1. 确保Streamlit应用正在运行: `http://localhost:8501`
2. 运行测试: 直接通过Claude Code的Playwright MCP执行test_quick_validation.md中的测试步骤

## 测试数据

- 文件: `data/经济数据库1017.xlsx`
- 开始日期: 2020-01-01

## 预期结果

所有测试应:
- 训练成功完成
- RMSE > 0
- Hit Rate ∈ [0, 1]
- 无ERROR消息(控制台)
