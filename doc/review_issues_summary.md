# DMGENet 论文-代码核实报告 & 审稿修改方案

> 生成日期: 2026-03-30
> 修改截止日期: 2026-04-16
> 决定: Major Revision

---

## 第一部分：论文与代码不一致之处（Paper vs Code）

### 🔴 严重不一致（必须修正）

#### 1. 数据归一化方式不一致
- **论文 Eq.22**: 使用 z-score 归一化 `x_norm = (x - μ) / σ`
- **代码** (`data/dataloader_Beijing_12.py:61`): 实际使用 **min-max 归一化** `data_inverse = data * (max - min) + min`
- **影响**: 论文描述与实际实验不符，属于方法论错误
- **修改建议**: 论文 Eq.22 应改为 min-max 公式，或将代码改为 z-score 并重新跑实验

#### 2. 历史窗口长度不一致
- **论文 Table 4 & Section 5.3**: Historical window length = **24**
- **代码** (`exp_base_model.py:232`): `T_in = 72`，`seq_len = 72`
- **影响**: 实验设置描述与实际代码完全不符
- **修改建议**: 论文 Table 4 中应更正为 72，并在 Section 5.3 中说明 72 步对应 72 小时（3天）的历史窗口

#### 3. TCN kernel size 不一致
- **论文 Table 4**: TCN kernel size = **4**
- **代码** (`_Support/TemporalConvNet.py:50`): `kernel_size=2`（默认值），且 `gated_TCN` 创建 TCN 时未传入 kernel_size 参数
- **影响**: 论文报告的超参数与实际代码不符
- **修改建议**: 将代码改为 `kernel_size=4` 并重新实验，或将论文 Table 4 修正为 2

#### 4. GAT 多头聚合方式不一致
- **论文 Eq.13**: 描述为 K 个注意力头的输出**取平均** `1/K Σ`
- **代码** (`GNN/GATLAyer.py:74`): 实际使用**拼接(concatenation)** `torch.cat([att(x, adj) for att in self.attentions], dim=-1)`，再通过 `out_att` 降维
- **影响**: 论文数学公式与代码实现不符
- **修改建议**: 将论文 Eq.13 改为描述 concatenation + output projection（这是更标准的 GAT 实现）

#### 5. GCN 层数不一致
- **论文 Table 4**: Number of GCN layers = **1**
- **代码** (`GNN/GCNLayer.py:62-73`): GCNLayer 包含 `conv1 → ReLU → conv2`，即**2层** GraphConv
- **影响**: 论文配置表与代码实现不符
- **修改建议**: 将 Table 4 修正为 "Number of GCN layers = 2"，或简化代码为 1 层

#### 6. 图标记符号不一致（Reviewer #1 已指出）
- **论文正文 Section 4.4.1**: 使用 D, N, **S**, **F** (S=分布相似, F=功能相似)
- **论文 Figure 4, Figure 5**: 使用 D, N, **C**, **S** (C 和 S 与正文含义不同)
- **代码**: 使用 D, N, S, **POI**
- **修改建议**: 统一全文为 D, N, S, F，图中标注同步修正，代码中的 POI 变量名不影响结果但应在论文中保持一致

### 🟡 中等不一致（建议修正）

#### 7. ASTAM 实现与论文公式差异
- **论文 Eq.17-20**:
  - Q = E_G × W_Q (查询矩阵)
  - K_T = H_T × W_Kt, K_S = H_S × W_Ks (时间和空间键矩阵)
  - A_T = Q(K_T)^T / **D** (包含缩放因子 D)
- **代码** (`model/model_1.py:119-132`):
  - 查询 Q 通过 Conv2d 从 node_embedding 计算（基本一致）
  - **缺少 Key 的线性变换**: x_t 和 x_s 直接用于计算注意力，没有 W_Kt 和 W_Ks 变换
  - **缺少缩放因子 /D**: 注意力分数没有除以 √D
  - 最终 softmax 只在 2 个标量上（temporal vs spatial），而非论文暗示的矩阵注意力
- **修改建议**:
  - 方案A（改代码）: 添加 W_Kt, W_Ks 变换和缩放因子，重新实验
  - 方案B（改论文）: 将 Eq.18-20 简化为代码的实际实现方式，即直接点积计算标量权重

#### 8. 分布相似图的概率分布计算
- **论文 Eq.4-7**: "Let P_i and P_j denote the **probability distributions** of pollutant concentrations"
- **代码** (`_Support/Graph_Construction_Beijing_12.py:141`): 直接将原始数据值传入 `jensen_shannon_distance`，**未转换为概率分布**
- **影响**: `scipy.stats.entropy` 期望输入为概率分布（和为1），原始浓度值不满足此条件
- **修改建议**: 在计算 JS 散度前，先将数据通过直方图离散化为概率分布

---

## 第二部分：代码质量与优化建议

### 🔴 需要修复的问题

| # | 问题 | 位置 | 建议 |
|---|------|------|------|
| 1 | `F.tanh` 和 `F.sigmoid` 已弃用 | `model/model_1.py:45,47` | 改为 `torch.tanh()` 和 `torch.sigmoid()` |
| 2 | `nn.Parameter(...).cuda()` 双重 cuda | `model/model_1.py:101` | 删除 `.cuda()`，让 `model.cuda()` 统一管理设备 |
| 3 | `squeeze()` 无 dim 参数 | `model/model_1.py:120,124` | 指定 `squeeze(dim=-1).squeeze(dim=-1)` 防止 batch=1 时维度错误 |
| 4 | 验证时缺少 `torch.no_grad()` | `exp_base_model.py:84-98` | 在 `val()` 中添加 `with torch.no_grad():` 减少显存占用 |
| 5 | Replay buffer 存储 None 值 | `RLMC_final/RLMC.py:126-137` → `train_RLMC_final.py:116-124` | 当 `done=True` 时 `next_observation=None`，push 到 buffer 会导致后续采样崩溃 |
| 6 | 硬编码 `.cuda()` | 全局多处 | 应使用 `device` 参数，提高代码可移植性 |

### 🟡 建议优化

| # | 优化点 | 说明 |
|---|--------|------|
| 1 | 添加学习率调度器到 RLMC | 当前 DDPG 训练使用固定学习率，可添加调度器提升收敛稳定性 |
| 2 | 添加梯度裁剪到 Actor | Critic 已有 `clip_grad_norm_`，Actor 没有 |
| 3 | 添加模型参数量统计 | 审稿人要求计算效率分析，需要输出参数量和 FLOPs |
| 4 | 添加训练/推理时间记录 | 审稿人要求的计算成本分析 |
| 5 | 添加随机种子到 CUDA | `torch.cuda.manual_seed(seed)` 和 `torch.cuda.manual_seed_all(seed)` |
| 6 | 距离图 σ 值硬编码 | `Graph_Construction_Beijing_12.py:83` 中 σ=16 硬编码，应参数化 |

---

## 第三部分：审稿意见逐条分析与修改方案

### 共性问题汇总

| 问题类别 | R1 | R2 | R3 | 优先级 | 工作量 |
|----------|:---:|:---:|:---:|--------|--------|
| 数据集过旧/泛化性 | ✓ | ✓ | - | **极高** | 大（需新数据集实验） |
| 缺少/弱基线对比 | ✓ | ✓ | ✓ | **极高** | 中（需跑新基线） |
| 消融实验不足 | - | ✓ | - | **高** | 小（已有，需强化展示） |
| 硬件/可复现性细节 | ✓ | ✓ | - | **高** | 小（纯文字补充） |
| 语言润色 | - | ✓ | ✓ | 中 | 中 |
| 统计显著性检验 | - | ✓ | - | 中 | 小（已有多次运行数据） |
| 计算成本分析 | - | ✓ | - | 中 | 小（需补充代码+文字） |
| 讨论深度不足 | - | - | ✓ | 中 | 中（纯文字） |
| 结论过于笼统 | - | - | ✓ | 中 | 小（纯文字） |
| 缺少局限性章节 | - | - | ✓ | 中 | 小（纯文字） |
| 图表质量 | - | - | ✓ | 中 | 中 |
| 缩写格式规范 | - | - | ✓ | **高** | 小（纯格式） |
| 文献综述扩展 | ✓ | ✓ | ✓ | **高** | 中（纯文字） |

---

### Reviewer #1 逐条回应

#### R1-1. 数据集过旧 (2013-2017)
- **状态**: 需要新实验
- **论文修改**: 增加一个当代数据集的验证实验（建议：2020-2023年的中国空气质量数据，如全国城市空气质量数据集）
- **代码修改**:
  - 新增 dataloader 适配新数据集
  - 新增该数据集的图构建代码
  - 如果无法获取新数据集，至少在 Limitations 中讨论数据时效性，并在 Future Work 中承诺后续验证
- **工作量**: ★★★★ (最耗时的修改项)

#### R1-2. 预测时长仅 6 小时
- **状态**: **代码已支持** 12h 和 24h (`train_RLMC_final.py:466` 已有 `[6, 12, 24]`)
- **论文修改**: 将 Table 2 扩展为包含 1h, 3h, 6h, 12h, 24h 的完整结果
- **代码修改**: 确保 `exp_base_model.py` 中 `T_out in [1, 3, 6, 12, 24]` 已完成训练
- **工作量**: ★★ (主要是跑实验和整理数据)

#### R1-3. 缺少基线对比
- **论文修改**: 在 Related Works 和 Baselines 中添加审稿人指定的 4 篇论文，并尽可能复现其中 1-2 个作为新基线
- **代码修改**: 如果复现，需添加新的基线模型代码
- **工作量**: ★★★

#### R1-4. 图标记符号不一致
- **已确认**: Figure 4, 5 用了 C, S 而正文用 S, F
- **论文修改**: 统一所有图表和正文的符号为 D, N, S, F
- **工作量**: ★

#### R1-5. 分布漂移未讨论
- **论文修改**:
  - 在 Section 3 或 4 中添加关于非平稳性的讨论
  - 可以添加一个简单的 ADF 检验或 KPSS 检验来定量说明数据的非平稳性
  - 讨论 TCN 和动态图构建如何隐式处理分布漂移
- **代码修改**: 添加一个 `stationarity_test.py` 脚本，输出 ADF 检验结果
- **工作量**: ★★

#### R1-6. MDP 状态的 24 步窗口限制（实际为 72 步）
- **论文修改**:
  - 首先更正 T=24 为 T=72（见第一部分不一致 #2）
  - 讨论为何 72 步（3天）的窗口足以捕获空气质量的主要时间依赖
  - 可引用空气质量文献中的时间自相关分析来支撑
  - 承认长程依赖可能超过此窗口，列入 Limitations
- **工作量**: ★

#### R1-7. 超参数选择缺乏优化
- **论文修改**:
  - 添加关键超参数的敏感性分析（如 TCN kernel size, dilation rates, hidden units）
  - 可以添加一个超参数敏感性表格或图
- **代码修改**: 编写超参数搜索脚本，至少对 2-3 个关键参数做 grid search
- **工作量**: ★★★

#### R1-8. RL 集成 vs 多任务学习的论证
- **论文修改**: 在 Introduction 或 Section 4.4 开头添加段落，解释：
  - 多任务学习共享特征但固定架构，无法动态调整
  - RL 集成允许根据实时反馈动态调整权重
  - 引用相关 ensemble vs multi-task 文献
- **工作量**: ★

---

### Reviewer #2 逐条回应

#### R2-1. 技术新颖性论证不足
- **论文修改**: 在 Section 1（贡献列表之前）添加段落，明确区分本文与现有方法：
  - 区别于 MSTGAN: 本文使用 4 种图而非 2 种
  - 区别于 STCN: 本文引入 RL 动态集成而非固定权重
  - 区别于 ASTGCN: 本文的 ASTAM 是节点级自适应而非全局注意力
- **工作量**: ★★

#### R2-2. 理论基础薄弱
- **论文修改**:
  - 为 Gated TCN 的门控机制添加信息论解释
  - 为 HGLM 的 gate fusion 添加互补性分析
  - 为 DDPG 的收敛性添加理论讨论
- **工作量**: ★★

#### R2-3. SOTA 对比不完整
- 同 R1-3
- **额外**: 添加审稿人推荐的 7 篇论文到 Related Works

#### R2-4. 结论夸大
- **论文修改**:
  - Abstract 中的 "robustness" 和 "generalizability" 改为 "improved performance"
  - Conclusion 中限定声明范围（如 "on the Beijing dataset"）
- **工作量**: ★

#### R2-5. 语言润色
- **论文修改**: 全文语言润色
- **工作量**: ★★

#### R2-6. 泛化性
- 同 R1-1（添加新数据集或详细讨论限制）

#### R2-7. 消融实验
- **当前状态**: 论文已有 Section 6.3（图结构消融）和 Section 6.4（模块消融），Table 5, 6, Figure 10
- **论文修改**:
  - 使消融实验更突出，添加更详细的分析文字
  - 考虑添加交叉消融（如同时移除两个组件）
- **工作量**: ★

#### R2-8. 计算成本分析
- **论文修改**: 添加 Table，包含：
  - 参数量（Parameters）
  - 训练时间（Training time per epoch）
  - 推理时间（Inference time per sample）
  - 显存占用（GPU memory）
- **代码修改**:
  ```python
  # 添加到 exp_base_model.py
  from thop import profile
  flops, params = profile(model, inputs=(sample_input,))
  ```
- **工作量**: ★★

#### R2-9. 可复现性
- **论文修改**: 添加 Implementation Details 小节：
  - Hardware: GPU 型号, CPU, RAM
  - Software: PyTorch 版本, CUDA 版本, Python 版本
  - Training: batch size=64, optimizer=Adam, lr=0.001, early stopping patience=7
  - 代码仓库链接（已有 GitHub 链接）
- **工作量**: ★

#### R2-10. 统计显著性
- **当前状态**: 代码已支持 10 次重复实验 (`train_RLMC_final.py:464` `repeat_times=10`)
- **论文修改**:
  - 在 Table 2 中添加 mean±std
  - 进行配对 t 检验或 Wilcoxon 检验
- **代码修改**: 在 `calculating_errors.py` 中添加统计检验函数
- **工作量**: ★★

#### R2-11. 文献综述扩展
- 添加审稿人推荐的 7 篇论文（部分与空气质量预测无直接关系，如绿色数据中心、区块链作物监测，可在 broader context 中简要提及）
- **工作量**: ★★

#### R2-12. 鲁棒性测试
- **论文修改**: 添加噪声鲁棒性实验
- **代码修改**: 在测试数据中注入不同水平的高斯噪声（如 5%, 10%, 15%），评估模型性能衰减
- **工作量**: ★★

---

### Reviewer #3 逐条回应

#### R3-1. 结论过于笼统
- **论文修改**:
  - 添加具体数字："DMGENet achieves RMSE of 16.48 for 1-hour forecasting, representing a 3.35% improvement over the best baseline"
  - 明确列出各预测时长的改进比例
- **工作量**: ★

#### R3-2. 缺少局限性章节
- **论文修改**: 在 Conclusion 前添加 "7. Limitations" 章节：
  - 数据集局限：仅在北京验证
  - 时间范围：2013-2017
  - 目标变量：仅 PM2.5
  - 计算开销：4 个独立模型 + RL agent
  - MDP 窗口限制
- **工作量**: ★

#### R3-3. 讨论深度不足
- **论文修改**: 在 Section 6 各子节中添加：
  - 为什么 DMGENet 在长时预测中优势更大？（多图互补信息在长时更关键）
  - 为什么功能相似图贡献最小？（POI 数据粒度可能不够）
  - RL 权重分布的可视化分析（已有 weights.csv）
- **工作量**: ★★

#### R3-4. 图表质量
- **论文修改**:
  - 提高 Figure 6-9 的分辨率
  - 为所有图添加详细的 caption
  - 表格统一格式，加粗最优值
  - 添加图例说明
- **工作量**: ★★

#### R3-5. 基线对比不足
- 同 R1-3, R2-3

#### R3-6. 评价指标不足
- **论文修改**: 除 RMSE, MAE, IA 外，添加：
  - R² (代码已计算但论文未报告)
  - MAPE 或 SMAPE
- **代码修改**: 已有相关函数，只需在结果表中添加
- **工作量**: ★

#### R3-7. 语言润色
- 同 R2-5

#### R3-8. 文献综述扩展
- 添加审稿人推荐的 5 篇论文（注意：部分论文与空气质量不直接相关，如区块链作物监测。应诚实回应其相关性，仅在合理时引用）
- **工作量**: ★

#### R3-9. 缩写格式规范（EAAI 期刊硬性要求）
- **标题**: 移除所有缩写
  - 当前: "A Reinforcement Learning-Based Spatiotemporal Dynamic Multi-Graph Ensemble Framework for Multi-Station Air Quality Prediction"
  - 已无缩写 ✅
- **关键词**: 移除所有缩写，添加 "Artificial Intelligence" 和 "Air Quality Prediction"
  - 当前: 已有 "Air quality prediction, Deep learning, Reinforcement ensemble learning, Graph neural network, Adaptive attention mechanism"
  - 修改: 确保没有缩写，添加 "Artificial intelligence application"
- **摘要**: 所有缩写首次使用时展开
  - DMGENet → Dynamic Multi-Graph Ensemble Neural Network (DMGENet)
  - TCN → Temporal Convolutional Network (TCN)
  - HGLM → Hybrid Graph Learning Module (HGLM)
  - ASTAM → Adaptive Spatiotemporal Attention Mechanism (ASTAM)
  - DDPG → Deep Deterministic Policy Gradient (DDPG)
  - RMSE, MAE, GC-LSTM, MSTGAN 等都需要展开
- **工作量**: ★

#### R3-10. EM 系统一致性
- 确保提交系统中的标题、关键词、摘要与手稿完全一致
- **工作量**: ★

---

## 第四部分：优先级排序与实施路线图

### Phase 1: 紧急修复（第1-3天）— 代码修正 & 实验准备

1. **修正代码bug** (0.5天)
   - 修复 `F.tanh`/`F.sigmoid` 弃用警告
   - 修复 node_embedding 双重 cuda
   - 修复 squeeze 无 dim
   - 添加 val() 中的 no_grad
   - 修复 replay buffer None 值问题

2. **确认并统一论文-代码不一致** (0.5天)
   - 确定 Eq.22 归一化方式（改论文描述 → min-max）
   - 确定窗口长度（改论文 → 72）
   - 确定 kernel_size（改论文 → 2，或改代码 → 4 并重跑）
   - 确定 GAT 聚合方式（改论文公式 → concatenation）
   - 确定 GCN 层数（改论文 → 2）

3. **启动新数据集实验** (2天)
   - 准备当代数据集（如 2020-2023 中国城市空气质量数据）
   - 编写数据预处理和图构建代码
   - 启动 1h, 3h, 6h, 12h, 24h 全时长实验

### Phase 2: 补充实验（第4-8天）

4. **扩展预测时长** - 确保 12h 和 24h 结果完整 (1天)
5. **超参数敏感性分析** (1天)
6. **噪声鲁棒性实验** (1天)
7. **计算效率分析** - 参数量、训练/推理时间 (0.5天)
8. **统计显著性检验** - 利用已有 10 次重复实验数据 (0.5天)
9. **新基线模型** - 至少复现 1-2 个审稿人推荐的方法 (2天)

### Phase 3: 论文修改（第9-14天）

10. **统一符号标记** (0.5天)
11. **修正公式和配置表** (0.5天)
12. **扩展 Related Works** - 添加全部 16 篇推荐文献 (1天)
13. **添加 Limitations 章节** (0.5天)
14. **深化 Discussion** (1天)
15. **改写 Conclusion** - 添加具体数字 (0.5天)
16. **格式修正** - 缩写展开、图表质量提升 (1天)
17. **语言润色** (1天)

### Phase 4: 最终检查（第15-17天）

18. **Response Letter 撰写** (1天)
19. **全文校对** (1天)
20. **EM 系统提交准备** (0.5天)

---

## 附录：代码修改清单（可直接执行）

### A. 立即修复项

```python
# 1. model/model_1.py:45,47 - 替换弃用函数
# 旧: TCN1_output = F.tanh(self.TCN1(x))
# 新: TCN1_output = torch.tanh(self.TCN1(x))
# 旧: TCN2_output = F.sigmoid(self.TCN2(x))
# 新: TCN2_output = torch.sigmoid(self.TCN2(x))

# 2. model/model_1.py:101 - 移除双重 cuda
# 旧: self.node_embedding = nn.Parameter(torch.randn(num_nodes, apt_size).cuda(), requires_grad=True).cuda()
# 新: self.node_embedding = nn.Parameter(torch.randn(num_nodes, apt_size), requires_grad=True)

# 3. model/model_1.py:120,124 - 明确 squeeze 维度
# 旧: ...squeeze()
# 新: ...squeeze(-1).squeeze(-1)

# 4. exp_base_model.py:84 - 添加 no_grad
# 在 val() 方法的 for 循环外层添加:
# with torch.no_grad():
```

### B. 新增代码需求

| 文件 | 用途 |
|------|------|
| `utils/compute_efficiency.py` | 计算参数量、FLOPs、推理时间 |
| `utils/statistical_test.py` | 配对 t 检验、Wilcoxon 检验 |
| `utils/noise_robustness.py` | 噪声鲁棒性实验 |
| `utils/hyperparameter_sensitivity.py` | 超参数敏感性分析 |
| `utils/stationarity_test.py` | ADF/KPSS 平稳性检验 |
| `data/dataloader_new_dataset.py` | 新数据集加载器 |
| `_Support/Graph_Construction_new.py` | 新数据集图构建 |
