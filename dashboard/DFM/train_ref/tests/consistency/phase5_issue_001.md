# Phase 5 Issue #001: Kalman滤波器数据格式不一致

**发现时间**: 2025-10-23
**问题级别**: 🔴 **严重** - 阻塞Phase 5所有测试
**状态**: ⬜ 待修复

---

## 问题描述

在Phase 5端到端测试中,发现train_model和train_ref的Kalman滤波器对输入数据Z的形状期望**完全相反**,导致:
- 第1次EM迭代后平滑因子就出现差异
- 最终模型参数差异巨大(Lambda: 0.128, A: 0.261, 远超1e-10目标容差)

## 根本原因

**数据形状不匹配**:

| 模块 | Z的期望形状 | 示例 | 代码位置 |
|------|------------|------|----------|
| train_model | `(n_time, n_obs)` | `(50, 10)` | DiscreteKalmanFilter.py:167<br/>`z = np.array(Z.to_numpy())` |
| train_ref | `(n_obs, n_time)` | `(10, 50)` | kalman.py:84<br/>`Z: 观测序列 (n_obs, n_time)` |

**调用方式对比**:

```python
# train_model (DynamicFactorModel.py:472)
kf = KalmanFilter(
    Z=obs_centered,  # DataFrame (50, 10) → ndarray (50, 10)
    ...
)

# train_ref (factor_model.py:319)
kf = KalmanFilter(...)
result = kf.filter(
    Z=obs_centered.values.T  # DataFrame (50, 10) → ndarray (10, 50)
    ...
)
```

## 数值影响

**第1次EM迭代E步平滑因子**:
```
train_model第1次: [[-0.03744514  0.36153377]
                     [-0.20065461  2.1039572 ]
                     [ 0.05335588  0.53388721]]

train_ref第1次:   [[-0.02372962  0.3868709 ]
                     [-0.21267013  2.0742688 ]
                     [ 0.06846635  0.5436317 ]]
```

**最终参数差异**:
- Lambda最大差异: 0.128 (超目标容差1.28e7倍)
- A最大差异: 0.261
- 平滑因子最大差异: 1.694

## 与Phase 3的矛盾

**困惑点**: Phase 3的Kalman滤波器一致性测试**全部通过(0差异)**,为什么Phase 5会出现这个问题?

**可能原因**:
1. Phase 3的测试可能使用了相同的数据格式(都是train_ref格式)
2. Phase 3可能没有测试完整的train_model调用路径
3. Phase 3的数据准备方式与Phase 5不同

**需要验证**: 检查test_kalman_filter_consistency.py中的数据准备方式

## 解决方案选项

### 选项A: 修改train_ref匹配train_model (推荐)

**优点**:
- 符合"train_ref与train_model完全一致"的任务目标
- 保持DataFrame (n_time, n_obs)的自然顺序

**缺点**:
- 需要修改kalman.py中的所有索引逻辑
- 可能影响Phase 3的测试(需要回归验证)
- 影响范围大

**实施步骤**:
1. 修改`kalman.py`的`filter()`和`smooth()`方法,期望Z为(n_time, n_obs)
2. 修改`factor_model.py`中的调用,移除`.T`
3. 回归运行Phase 3所有测试
4. 重新运行Phase 5测试

### 选项B: 修改train_model匹配train_ref

**优点**:
- train_ref的设计更符合矩阵标准表示(观测在前)

**缺点**:
- **违反任务目标**: 不应该修改train_model
- 需要修改train_model的多个文件
- 影响现有生产代码

**结论**: ❌ 不推荐

### 选项C: 在Phase 5测试中转置数据

**优点**:
- 不修改核心代码
- 快速验证

**缺点**:
- **治标不治本**: 隐藏了真实的设计不一致
- 无法验证生产环境中的真实行为
- 违反tasks.md的严格验证要求

**结论**: ❌ 不可接受

## 决策

**推荐**: **选项A - 修改train_ref匹配train_model**

**理由**:
1. 符合openspec任务目标: "train_ref必须与train_model完全一致"
2. 问题根源在train_ref的设计决策
3. 尽管影响范围大,但这是唯一正确的解决方案

## 下一步行动

1. **[ ] 确认选项A**
2. **[ ] 修改kalman.py的filter()方法**
   - 将`Z: (n_obs, n_time)`改为`Z: (n_time, n_obs)`
   - 修改`n_time = Z.shape[1]`为`n_time = Z.shape[0]`
   - 修改所有涉及Z索引的代码
3. **[ ] 修改kalman.py的smooth()方法**
4. **[ ] 修改factor_model.py中的调用**
   - 移除`Z=obs_centered.values.T`中的`.T`
5. **[ ] 回归测试Phase 3**
   - 运行test_kalman_filter_consistency.py
   - 运行test_kalman_smoother_consistency.py
6. **[ ] 重新运行Phase 5测试**

## 预期结果

修复后,Phase 5测试应该出现:
- 第1次EM迭代的平滑因子差异 < 1e-14
- 最终参数差异 < 1e-10

---

**报告生成时间**: 2025-10-23
**优先级**: P0 (最高)
**阻塞**: Phase 5.3, 5.4, 5.5, 5.6, 5.7
