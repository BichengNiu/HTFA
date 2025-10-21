# 核心层一致性测试

本目录包含train_ref核心层与train_model的一致性测试。

## 测试文件

### 1. test_kalman_consistency.py

测试卡尔曼滤波和平滑算法的一致性

**测试内容：**
- 卡尔曼滤波基本功能
- 卡尔曼平滑算法
- 带缺失数据的处理
- 对数似然计算

**对比模块：**
- 新代码: `train_ref/core/kalman.py`
- 老代码: `train_model/DiscreteKalmanFilter.py`

### 2. test_estimator_consistency.py

测试参数估计功能的一致性

**测试内容：**
- 载荷矩阵估计
- 目标变量载荷估计
- 带训练期切分的估计
- 状态转移矩阵估计
- 缺失数据处理

**对比模块：**
- 新代码: `train_ref/core/estimator.py`
- 老代码: `train_model/DiscreteKalmanFilter.py` (calculate_factor_loadings)

### 3. test_dfm_consistency.py

测试DFM模型完整流程的一致性

**测试内容：**
- DFM基本拟合
- 带训练期切分的拟合
- 收敛行为验证
- 多种参数配置测试

**对比模块：**
- 新代码: `train_ref/core/factor_model.py`
- 老代码: `train_model/DynamicFactorModel.py` (DFM_EMalgo)

### 4. run_all_tests.py

运行所有测试的主脚本

## 运行测试

### 运行所有测试

```bash
cd "E:\同步文件夹\HTFA"
python -m dashboard.DFM.train_ref.tests.core.run_all_tests
```

### 运行单个测试

```bash
# 卡尔曼滤波测试
python dashboard/DFM/train_ref/tests/core/test_kalman_consistency.py

# 参数估计测试
python dashboard/DFM/train_ref/tests/core/test_estimator_consistency.py

# DFM模型测试
python dashboard/DFM/train_ref/tests/core/test_dfm_consistency.py
```

## 测试标准

### 数值精度

- **严格一致**: 差异 < 1e-8 (用于线性代数运算)
- **算法一致**: 差异 < 1e-6 (用于迭代算法)
- **统计一致**: 相关系数 > 0.95 (用于因子提取)

### 通过标准

所有测试模块都必须通过，确保：

1. **计算正确性**: 与老代码数值结果一致
2. **功能完整性**: 所有关键功能都已实现
3. **边界条件**: 缺失数据、特殊配置等情况正确处理

## 测试结果解读

### 成功输出示例

```
==========================================
🎉 恭喜！所有测试通过！
==========================================

train_ref核心层实现与train_model完全一致
可以安全地使用新代码进行后续开发
```

### 失败处理

如果测试失败，会显示：

1. 失败的具体测试
2. 数值差异详情
3. 错误堆栈跟踪

需要检查：

- 算法实现是否正确
- 数据处理流程是否一致
- 边界条件处理是否完善

## 注意事项

1. **随机种子**: 所有测试使用固定随机种子(42)确保可重现
2. **浮点精度**: 由于浮点运算，允许微小差异（< 1e-8）
3. **因子符号**: 因子可能有符号差异，使用相关性比较
4. **迭代算法**: EM算法可能在不同平台有微小差异，使用相对误差

## 开发建议

在修改核心层代码后，务必：

1. 运行完整测试套件
2. 确保所有测试通过
3. 如有失败，分析原因并修复
4. 添加新测试覆盖新功能

## 性能基准

参考测试耗时（测试机器配置相关）：

- 卡尔曼滤波测试: ~5秒
- 参数估计测试: ~3秒
- DFM模型测试: ~15秒
- 总计: ~25秒

如果测试时间显著增加，可能需要优化实现。
