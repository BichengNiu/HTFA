# algorithm-validation Specification

## Purpose
TBD - created by archiving change validate-algorithm-consistency. Update Purpose after archive.
## Requirements
### Requirement: 模拟数据生成能力
系统MUST支持生成符合DFM理论的模拟时间序列数据,用于算法验证测试。

#### Scenario: 生成标准DFM过程数据
**Given** 配置参数: 时间点数T=200, 变量数N=30, 因子数k=3
**When** 调用模拟数据生成器
**Then** 生成数据应满足:
- 因子过程遵循AR(1): F_t = A * F_{t-1} + eta_t
- 观测数据遵循: Z_t = Lambda * F_t + eps_t
- 所有矩阵维度正确: Z(200×30), F(200×3), Lambda(30×3), A(3×3)
- 噪声协方差矩阵Q和R为正定矩阵
- 使用固定随机种子(SEED=42)确保可重现性

#### Scenario: 生成包含缺失值的数据
**Given** 基础模拟数据和缺失率参数(如20%)
**When** 注入随机缺失值
**Then** 应满足:
- 缺失值比例在容差范围内(20% ± 2%)
- 缺失位置随机分布(非系统性缺失)
- 保留原始数据的副本用于对比

#### Scenario: 生成多种规模的测试数据集
**Given** 预定义的数据集配置(小、中、大、高维等)
**When** 批量生成测试数据集
**Then** 应生成:
- small: 50×10×2 (快速测试)
- medium: 200×30×3 (标准测试)
- large: 500×50×5 (性能测试)
- single_factor: 100×20×1 (边界测试)
- high_dim: 300×100×10 (高维测试)
- 所有数据集保存为.npz格式到fixtures/目录

---

### Requirement: PCA算法数值一致性验证
系统MUST验证train_ref的PCA初始化算法与train_model完全一致。

#### Scenario: 协方差矩阵计算一致性
**Given** 相同的标准化数据矩阵Z (200×30)和固定随机种子(SEED=42)
**When** 分别使用train_model和train_ref计算协方差矩阵S
**Then** 应满足:
- 两个矩阵**完全相等**(使用np.array_equal(S1, S2))
- 如有任何差异,测试失败并要求根因分析
- 两个矩阵均为对称正定矩阵

#### Scenario: 特征值分解一致性
**Given** 相同的协方差矩阵S
**When** 执行特征值分解
**Then** 应满足:
- 特征值**完全相等**(np.array_equal(eigenvalues1, eigenvalues2))
- 特征值排序一致(降序)
- 特征向量处理符号不确定性:
  - 对每列验证: `v1[:, i] == v2[:, i]` OR `v1[:, i] == -v2[:, i]`
  - 如两者均不满足,则为真正不一致,测试失败

#### Scenario: 因子载荷初始化一致性
**Given** 相同的特征值和特征向量
**When** 计算初始Lambda矩阵
**Then** 应满足:
- Lambda矩阵**完全相等**(考虑符号传播: `Lambda1 == Lambda2` OR `Lambda1 == -Lambda2`)
- 方差贡献度**完全相等**(np.array_equal)
- 对于k=1,2,3,5,10不同因子数配置均通过
- 任一配置失败必须修复后才能测试下一个

#### Scenario: 因子提取一致性
**Given** 相同的数据Z和特征向量V
**When** 计算初始因子F = Z * V
**Then** 应满足:
- 因子矩阵**完全相等**(考虑符号传播)
- 因子方差贡献度**完全相等**(np.array_equal)

---

### Requirement: 卡尔曼滤波数值一致性验证
系统MUST验证卡尔曼滤波算法的每个步骤与train_model完全一致。

#### Scenario: 单步预测一致性
**Given** 相同的状态空间参数(A, H, Q, R, x0, P0)和观测数据Z
**When** 执行单时间步的预测步骤
**Then** 应满足:
- 预测状态x_pred[t]**完全相等**(np.array_equal,对每个时间步t)
- 预测协方差P_pred[t]**完全相等**(np.array_equal,对每个时间步t)
- 对所有时间点t=1,2,...,T均成立
- 任一时间步差异立即暂停并根因分析

#### Scenario: 单步更新一致性
**Given** 相同的预测结果和观测值z[t]
**When** 执行卡尔曼更新步骤
**Then** 应满足:
- 卡尔曼增益K[t]**完全相等**(np.array_equal)
- 滤波状态x_filt[t]**完全相等**(np.array_equal)
- 滤波协方差P_filt[t]**完全相等**(np.array_equal)
- 新息序列innovation[t]**完全相等**(np.array_equal)
- 任一时间步差异立即暂停并根因分析

#### Scenario: 完整滤波序列一致性
**Given** 相同的观测序列Z (T×N)
**When** 运行完整的前向卡尔曼滤波
**Then** 应满足:
- 所有时间点的滤波状态**完全相等**
- 对数似然**完全相等**(np.allclose with rtol=0, atol=0)
- 数值稳定性处理完全一致(协方差对称化、正定性保证策略)

#### Scenario: 缺失数据处理一致性
**Given** 包含固定位置缺失值的观测数据(确保可重现)
**When** 运行卡尔曼滤波
**Then** 应满足:
- 缺失时间点的跳过逻辑完全一致
- 滤波状态在缺失前后**完全相等**
- 最终结果**完全相等**(np.array_equal)

---

### Requirement: 卡尔曼平滑数值一致性验证
系统MUST验证RTS平滑算法与train_model完全一致。

#### Scenario: RTS平滑增益计算一致性
**Given** 滤波结果(x_filt, P_filt, x_pred, P_pred)
**When** 反向计算平滑增益C[t]
**Then** 应满足:
- 平滑增益矩阵C[t]**完全相等**(np.array_equal,反向每个时间步t)
- 对所有时间点t=T-1,T-2,...,0均成立
- 任一时间步差异立即暂停并根因分析

#### Scenario: 平滑状态估计一致性
**Given** 滤波结果和平滑增益
**When** 执行反向平滑
**Then** 应满足:
- 平滑状态x_sm[t]**完全相等**(np.array_equal,反向每个时间步t)
- 平滑协方差P_sm[t]**完全相等**(np.array_equal,反向每个时间步t)
- 边界条件: assert x_sm[T] == x_filt[T] (完全相等)

#### Scenario: 滞后协方差计算一致性
**Given** 平滑结果
**When** 计算滞后协方差P_lag_sm[t]
**Then** 应满足:
- 滞后协方差矩阵P_lag_sm[t]**完全相等**(np.array_equal,每个时间步t)
- 用于EM算法M步的数据完全一致

---

### Requirement: EM参数估计数值一致性验证
系统MUST验证EM算法的M步参数更新与train_model完全一致。

#### Scenario: 载荷矩阵估计一致性
**Given** 观测数据Z和平滑因子F
**When** 使用OLS估计载荷矩阵Lambda
**Then** 应满足:
- Lambda矩阵**完全相等**(np.array_equal)
- 对不同变量数(10, 30, 50, 100)均通过
- 任一配置失败必须修复后才能测试下一个

#### Scenario: 转移矩阵估计一致性
**Given** 平滑因子和滞后协方差
**When** 估计状态转移矩阵A
**Then** 应满足:
- A矩阵**完全相等**(np.array_equal)
- 特征值**完全相等**
- 满足平稳性约束(特征值 < 1)

#### Scenario: 协方差矩阵估计一致性
**Given** 平滑结果和模型残差
**When** 估计Q和R矩阵
**Then** 应满足:
- Q矩阵**完全相等**(np.array_equal)
- R矩阵**完全相等**(np.array_equal)
- 两矩阵均为正定矩阵(特征值完全相等)

#### Scenario: 正定性保证机制一致性
**Given** 可能非正定的协方差矩阵估计
**When** 应用ensure_positive_definite()
**Then** 应满足:
- 特征值调整策略完全一致
- 调整后矩阵**完全相等**(np.array_equal)
- 最小特征值阈值设置相同
- 调整后的特征值**完全相等**

---

### Requirement: 完整EM迭代一致性验证
系统MUST验证多次EM迭代的参数演化与train_model完全一致。

#### Scenario: 单次EM迭代一致性
**Given** 固定的初始参数(Lambda, A, Q, R)
**When** 执行一次完整的EM迭代(E步+M步)
**Then** 应满足:
- E步输出(x_sm, P_sm, P_lag_sm)**完全相等**(np.array_equal)
- M步输出(Lambda_new, A_new, Q_new, R_new)**完全相等**(np.array_equal)
- 对数似然**完全相等**

#### Scenario: 多次EM迭代演化一致性
**Given** PCA初始化参数
**When** 运行5次EM迭代
**Then** 应满足:
- 每次迭代的参数更新**完全相等**
- 迭代1-5的参数演化轨迹**完全一致**
- 收敛速度相同(相同迭代次数收敛)
- 对数似然序列**完全相等**

#### Scenario: 收敛判定一致性
**Given** EM迭代序列
**When** 判断是否收敛(|loglik[i]-loglik[i-1]| < tolerance)
**Then** 应满足:
- 收敛判定时机相同(相同迭代次数)
- 最终对数似然**完全相等**
- 最终参数(Lambda, A, Q, R)**完全相等**(np.array_equal)

---

### Requirement: 全流程集成测试一致性验证
系统MUST验证完整训练流程在不同配置下的一致性。

#### Scenario: 标准配置全流程一致性
**Given** 模拟数据(200×30, k=3)和标准配置(max_iter=30, tolerance=1e-6)
**When** 运行完整训练流程(PCA初始化→EM迭代→收敛)
**Then** 应满足:
- 最终Lambda矩阵**完全相等**(np.array_equal)
- 最终A, Q, R矩阵**完全相等**(np.array_equal)
- 平滑因子F**完全相等**(np.array_equal)
- 样本内预测值**完全相等**
- RMSE, Hit Rate, 相关系数**完全相等**

#### Scenario: 不同因子数配置一致性
**Given** 相同数据但不同因子数(k=1, 2, 3, 5, 10)
**When** 分别训练模型
**Then** 应满足:
- 每个配置的最终参数**完全相等**(np.array_equal)
- 单因子模型(k=1)特殊处理完全一致
- 高维模型(k=10)数值稳定性处理完全一致
- 任一配置失败必须修复后才能测试下一个

#### Scenario: 不同数据规模一致性
**Given** 小样本(50×10)、中等(200×30)、大样本(500×50)
**When** 分别训练模型
**Then** 应满足:
- 所有规模的最终参数**完全相等**(np.array_equal)
- 执行时间增长趋势一致(记录但不强制要求)
- 内存占用趋势一致(记录但不强制要求)

#### Scenario: 缺失数据场景一致性
**Given** 包含固定位置5%和20%缺失值的数据
**When** 训练模型
**Then** 应满足:
- 缺失数据处理逻辑完全一致
- 最终参数**完全相等**(np.array_equal)
- 有效样本数计算完全相同

---

### Requirement: 真实数据验证测试
系统MUST使用真实经济数据验证算法在生产场景下的一致性。

#### Scenario: 经济数据库完整训练一致性
**Given** 经济数据库1017.xlsx, 数据维度(1342×88), 固定随机种子
**When** 执行完整训练流程(数据预处理→变量选择→模型训练)
**Then** 应满足:
- 最终模型参数(Lambda, A, Q, R)**完全相等**(np.array_equal)
- 平滑因子x_sm**完全相等**
- 样本内预测值**完全相等**
- 样本外预测值**完全相等**
- RMSE, Hit Rate, 相关系数**完全相等**
- **注意**: 不允许放宽容差,真实数据也必须完全一致

#### Scenario: 变量选择流程一致性
**Given** 12个行业指标和后向选择方法, 固定随机种子
**When** 执行变量选择
**Then** 应满足:
- 剔除顺序完全一致(每步选择的变量相同)
- 每步的RMSE/Hit Rate评分**完全相等**
- 最终选中变量集合完全相同
- 任一步骤差异立即暂停并根因分析

#### Scenario: 因子数自动选择一致性
**Given** PCA累积方差阈值(如85%)或Elbow方法
**When** 自动选择因子数
**Then** 应满足:
- PCA方差贡献曲线**完全相等**(np.array_equal)
- 选择的因子数完全相同
- BIC准则计算**完全相等**
- 累积方差计算**完全相等**

---

### Requirement: 验证报告生成能力
系统MUST生成详细的一致性验证报告,包含统计摘要和可视化对比。

#### Scenario: 生成综合验证报告
**Given** 所有测试套件执行完毕
**When** 调用报告生成器
**Then** 应生成:
- 执行摘要(测试日期、版本、总体通过率)
- 测试覆盖率统计(单元测试、集成测试)
- 数值一致性结果(PCA、卡尔曼、EM、全流程)
- 可视化对比图表(差异分布、时间序列对比、矩阵热力图)
- 问题列表和修复建议
- 结论和后续行动建议

#### Scenario: 生成HTML交互式报告
**Given** 测试结果JSON数据
**When** 生成HTML报告
**Then** 应包含:
- 可折叠的详细对比表格
- 交互式图表(Plotly)
- 超链接导航(章节跳转)
- 响应式布局(适配不同屏幕)

#### Scenario: 生成PDF存档报告
**Given** 测试结果数据
**When** 生成PDF报告
**Then** 应包含:
- 完整的文本和表格内容
- 高清图表(300 DPI)
- 页眉页脚和页码
- 目录和书签

---

