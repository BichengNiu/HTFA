# DFM核心算法一致性问题记录

本文档记录在验证train_model和train_ref一致性过程中发现的所有问题、根因分析和解决方案。

## 重要原则

**问题处理流程**:
1. **发现问题**: 任何数值差异(即使很小)都必须记录
2. **根因分析**: 追踪到具体代码行,分析差异产生的原因
3. **修复验证**: 修复train_ref代码,回归测试确认有效
4. **文档记录**: 详细记录问题和解决方案,便于后续参考

**禁止行为**:
- ❌ 放宽容差以"通过"测试
- ❌ 跳过失败的测试
- ❌ 使用近似方法"制造"一致性
- ❌ 忽略"小"差异

---

## Phase 2: PCA算法一致性问题

### 问题 2.1: 浮点数计算的固有数值误差

**发现时间**: 2025-10-23

**测试场景**:
- `test_pca_consistency.py::test_002_covariance_matrix_consistency`
- `test_pca_consistency.py::test_004_eigenvalue_decomposition_consistency`

**问题描述**:
在零容差验证PCA算法时,以下测试失败:

**test_002 - 协方差矩阵计算**:
```
协方差矩阵 S[0,0]:
  expected (eigh方法): 0.9799999999999999
  actual   (SVD重构):  0.9799999999999985
  差异: -1.3322676295501878e-15
```

**test_004 - 特征值计算**:
```
特征值(因子1):
  expected (eigh方法): 2.4301111078415376
  actual   (SVD方法):  2.4301111078415363
  差异: -1.3322676295501878e-15
```

**关键观察**:
- ✓ test_001 (标准化): **完全一致** (0差异)
- ✗ test_002 (协方差矩阵): 失败 (~1e-15差异)
- ✗ test_004 (特征值): 失败 (~1e-15差异)
- ✓ test_005 (因子提取): **完全一致** (0差异)
- ✓ test_006 (载荷矩阵): **完全一致** (0差异)

**根因分析**:

**1. 数学理论等价性**:
两种计算方法在数学上完全等价:
- 方法1(eigh): S = (1/N) Z'Z, 然后对S进行特征分解
- 方法2(SVD): Z的SVD分解, 然后 S = V diag(s²/N) V'

**2. 代码路径对比**:

train_model使用eigh方法 (`DynamicFactorModel.py:72`):
```python
S = (z.T @ z) / n_time  # 协方差矩阵
eigenvalues, eigenvectors = np.linalg.eigh(S)
```

测试中的SVD等价计算 (`test_pca_consistency.py:142`):
```python
U, s, Vh = np.linalg.svd(z, full_matrices=False)
S_svd = Vh.T @ np.diag(s**2 / n_time) @ Vh
```

**3. 数值差异根源**:

**本质原因**: **浮点数运算的不结合性**(IEEE 754标准固有特性)

```
方法1: z.T @ z → 一次矩阵乘法, 然后除以n_time
方法2: Vh.T @ diag(...) @ Vh → 三次矩阵乘法

不同的运算顺序 → 不同的舍入误差累积 → 约10^-15量级差异
```

**数值分析证据**:
- 差异量级: 1.33e-15 ≈ 6 × εmachine (float64精度 ε ≈ 2.22e-16)
- 确定性: 多次运行产生相同差异
- 一致性: 所有失败测试的差异都在~1e-15量级

**4. 为什么test_005和test_006通过?**

关键区别: **这两个测试中train_model和train_ref使用完全相同的算法**

- test_005: 两边都用 `U[:, :k] * s[:k]` 提取因子 → 计算路径完全相同 → 0差异
- test_006: 两边都用OLS回归估计载荷 → 算法完全相同 → 0差异

这证明:
- 差异**不是代码bug**
- 差异来自**数学上等价但数值上不同**的计算路径
- 当算法完全相同时,结果**逐位完全相等**

**解决方案评估**:

**选项1: 调整验证策略为数值容差(推荐)**

```python
# 当前: 零容差
np.array_equal(actual, expected)

# 建议: 相对/绝对容差
np.allclose(actual, expected, rtol=1e-10, atol=1e-14)
```

**容差标准建议**:
- rtol=1e-10: 相对误差 (比通用标准严格10000倍)
- atol=1e-14: 绝对误差 (比machine epsilon大100倍)

**理由**:
- 1e-15误差在物理意义上完全可忽略
- 数值分析标准通常用sqrt(ε) ≈ 1e-8
- 我们的1e-14标准比通用标准严格1000倍
- 符合"数学等价即可接受"的工程原则

**选项2: 强制代码完全一致(不推荐)**

修改train_ref,确保与train_model使用完全相同的算法(如都用eigh,不用SVD)

**缺点**:
- 限制代码重构自由度
- 无法利用不同算法优势(如SVD数值稳定性更好)
- 过度严格,不符合工程实际

**选项3: 接受零容差失败,标记为已知限制**

继续Phase 3-7,在最终报告中说明此限制

**决策依据**:
- test_005/006已证明核心算法一致
- 浮点数误差不影响实际模型效果

**当前状态**: **已解决** - 2025-10-23

**用户决策**:
1. ✓ 认同这是"浮点数运算的固有限制"
2. ✓ 验证目标是"数学逻辑一致"
3. ✓ 选择选项1: 调整验证策略为极严格数值容差

**解决方案**: **选项1 - 极严格数值容差验证**

实施步骤:
1. ✓ 更新proposal.md: 将"零容差"改为"极严格数值容差(rtol=1e-10, atol=1e-14)"
2. ✓ 更新tasks.md: 同上,并详细说明容差设计理由
3. ✓ 在base.py中添加`assert_allclose_strict()`方法
4. ✓ 在base.py中添加`assert_eigenvectors_equal_up_to_sign()`方法
5. ✓ 修改test_pca_consistency.py使用新验证方法:
   - test_002: 协方差矩阵 → assert_allclose_strict
   - test_003: 奇异向量 → assert_eigenvectors_equal_up_to_sign
   - test_004: 特征值 → assert_allclose_strict

**修复验证**: **✓ 100%通过**

重新运行测试结果:
```
6 passed in 2.67s

test_001_standardization_consistency      PASSED
test_002_covariance_matrix_consistency    PASSED  ← 之前失败,现已通过
test_003_svd_decomposition_consistency    PASSED  ← 之前失败,现已通过
test_004_eigenvalue_decomposition_consistency PASSED  ← 之前失败,现已通过
test_005_factor_extraction_consistency    PASSED
test_006_loading_matrix_estimation_consistency PASSED
```

**关键指标**:
- 测试通过率: 100% (6/6)
- 最大浮点数误差: ~1e-15 (符合预期)
- 验证标准: rtol=1e-10, atol=1e-14 (比NumPy默认值严格10万倍)

**相关提交**: [待git commit]

---

## Phase 3: 卡尔曼滤波一致性问题

### 总结: **无问题发现** - 完美一致性 ✓

**测试时间**: 2025-10-23

**测试范围**:
- Phase 3.1: 卡尔曼滤波器一致性(4个测试)
- Phase 3.2: 卡尔曼平滑器一致性(4个测试)

**测试结果**: **8/8 测试通过 (100%)**

**关键发现**:

**1. 卡尔曼滤波完美一致性**:
```
test_001_single_step_prediction_consistency      PASSED  (0差异)
test_002_single_step_update_consistency          PASSED  (0差异)
test_003_full_filtering_consistency              PASSED  (0差异)
test_004_missing_data_handling_consistency       PASSED  (0差异)

最大差异: 0.000000e+00
验证标准: rtol=1e-10, atol=1e-14
```

**2. 卡尔曼平滑完美一致性**:
```
test_001_rts_smoother_backward_consistency       PASSED  (0差异)
test_002_lag_covariance_consistency              PASSED  (0差异)
test_003_boundary_conditions_consistency         PASSED  (0差异)
test_004_full_filter_smoother_consistency        PASSED  (0差异)

最大差异: 0.000000e+00
验证标准: rtol=1e-10, atol=1e-14
```

**3. 为什么Phase 3没有浮点数误差?**

与Phase 2的PCA不同,Phase 3的卡尔曼滤波/平滑算法在train_model和train_ref中:

**算法路径完全相同**:
- 预测步: `x_pred = A @ x_filt + B @ u`
- 更新步: `K = solve((S + jitter).T, (P @ H.T).T).T`
- 滤波步: `x_filt = x_pred + K @ innovation`
- 平滑步: `J = solve(P_pred.T, (P_filt @ A.T).T).T`

**数值稳定性处理一致**:
- 协方差对称化: `P = (P + P.T) / 2`
- jitter添加: `P_pred += eye * 1e-6`
- scipy.linalg.solve使用方式完全相同

**结论**: 由于两个版本的计算路径**逐行完全一致**,因此产生**逐位相同**的结果(0差异)

**4. 验证的关键属性**:

- **边界条件**: `x_sm[T-1] == x_filt[T-1]` (完全相等)
- **协方差性质**: `trace(P_sm[t]) <= trace(P_filt[t])` for all t (理论性质满足)
- **缺失数据**: 10%缺失数据下依然0差异

**当前状态**: **Phase 3完成** - 2025-10-23

**结论**:
- ✓ train_model和train_ref的卡尔曼滤波/平滑实现**完全一致**
- ✓ 没有算法差异,没有数值误差
- ✓ 可以安全进入Phase 4 (EM参数估计测试)

**最后更新**: 2025-10-23

---

## Phase 4: EM参数估计一致性问题

### 总结: **无问题发现** - 完美一致性 ✓

**测试时间**: 2025-10-23

**测试范围**: Phase 4.1 - EM参数估计函数一致性(5个测试)

**测试结果**: **5/5 测试通过 (100%)**

**关键发现**:

**1. EM参数估计完美一致性**:
```
test_001_loadings_estimation_consistency                 PASSED  (差异~8.88e-16)
test_002_transition_matrix_estimation_consistency        PASSED  (差异0)
test_003_process_noise_covariance_estimation_consistency PASSED  (差异0)
test_004_observation_noise_covariance_estimation_consistency PASSED  (差异~2.78e-17)
test_005_positive_definite_guarantee_consistency         PASSED  (差异0)

验证标准: rtol=1e-10, atol=1e-14
```

**2. 各参数估计函数对比**:

**载荷矩阵Lambda估计**:
- train_model: `sm.OLS` (statsmodels)
- train_ref: `LinearRegression` (sklearn)
- **结果**: 最大差异 8.88e-16 (机器精度级别)
- **结论**: 两种OLS实现数值完全等价

**状态转移矩阵A估计**:
- train_model: `_calculate_prediction_matrix()` + `scipy.linalg.solve`
- train_ref: `estimate_transition_matrix()` + `scipy.linalg.solve`
- **结果**: 0差异 (完美一致)
- **结论**: 公式和数值稳定性策略完全相同

**过程噪声协方差Q估计**:
- train_model: `_calculate_shock_matrix()` + 特征值调整
- train_ref: `estimate_covariance_matrices()` + `_ensure_positive_definite()`
- **结果**: 0差异 (完美一致)
- **结论**: Q矩阵计算和正定性保证完全相同

**观测噪声协方差R估计**:
- train_model: EMstep中的残差方差计算
- train_ref: `estimate_covariance_matrices()`中的残差方差
- **结果**: 差异 ~2.78e-17 (机器精度级别)
- **结论**: 残差计算路径完全相同

**正定性保证函数**:
- train_model: 特征值调整 `max(λ, 1e-7)`
- train_ref: `_ensure_positive_definite(epsilon=1e-7)`
- **结果**: 0差异 (完美一致)
- **结论**: 特征值调整策略完全相同

**3. 为什么Phase 4.1达到近乎完美一致性?**

与Phase 3类似,Phase 4.1的参数估计函数在两个版本中:

**算法路径完全相同**:
- OLS回归: 虽然使用不同库(sm vs sklearn),但底层都是最小二乘法
- scipy.linalg.solve: 完全相同的调用方式和参数
- 特征值调整: 相同的公式和epsilon值(1e-7)

**数值稳定性策略一致**:
- 正则化: `(F'F + 1e-7*I)`
- 残差方差: `np.maximum(R_diag, 1e-7)`
- 特征值下界: `np.maximum(eigenvalues, 1e-7)`

**当前状态**: **Phase 4.1完成** - 2025-10-23

**结论**:
- ✓ train_model和train_ref的EM参数估计函数**完全一致**
- ✓ 没有算法差异,仅有机器精度级别的数值误差
- ✓ 可以安全进入Phase 4.2 (完整EM迭代测试)

**最后更新**: 2025-10-23

---

## Phase 5: 全流程集成测试一致性问题

### 问题 5.1: [待填写]

**发现时间**:

**测试场景**:

**问题描述**:

**根因分析**:

**解决方案**:

**修复验证**:

**相关提交**:

---

## Phase 6: 真实数据验证问题

### 问题 6.1: [待填写]

**发现时间**:

**测试场景**:

**问题描述**:

**根因分析**:

**解决方案**:

**修复验证**:

**相关提交**:

---

## 统计摘要

**总问题数**: 0 (待更新)

**按Phase统计**:
- Phase 2 (PCA): 0
- Phase 3 (卡尔曼): 0
- Phase 4 (EM): 0
- Phase 5 (集成): 0
- Phase 6 (真实数据): 0

**按类型统计**:
- 算法实现差异: 0
- 数值计算顺序差异: 0
- 数据类型差异: 0
- 随机种子问题: 0
- 边界条件处理差异: 0
- 其他: 0

**当前状态**: Phase 1 (基础设施搭建中)

**最后更新**: YYYY-MM-DD
