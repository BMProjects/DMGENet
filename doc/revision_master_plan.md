# DMGENet 修稿总体执行方案（综合版）

> 综合来源：代码审查报告 + `revision_priority_and_rebuttal.html` + `review_issues_summary.md`
> 生成日期：2026-03-30 | 修改截止：2026-04-16（共 17 天）
> 期刊决定：Major Revision

---

## 核心原则

1. **可信度优先**：先修会让审稿人质疑方法学有效性的问题（测试集泄漏、论文-代码不一致）
2. **最小改动**：在保持方法先进性的前提下，尽量修改论文描述而非重做实验
3. **证据驱动**：每个修改都必须有可验证的代码/数据支撑

---

## 一、P0：必须立即修复的代码问题

> 这些问题直接影响实验结果的可信度，不修则 rebuttal 无法自圆其说。

### P0-1. 🔴 测试集泄漏（最严重）

**问题描述**：RLMC 训练中存在两层测试集泄漏：
- `train_RLMC_final.py:159` — 基于 **test_loss** 选择最佳 episode 的 Actor
- `train_RLMC_final.py:593` — 基于 **test MAE** 选择最佳 run

**修复方案**：

```
文件: train_RLMC_final.py

Episode 级别修复 (Exp.train 方法):
- 新增: 在 train() 中同时计算 val_loss = self.compute_test_loss(self.train_X, ...)
  （注意：RL 的 train 数据来源于原 val 集，所以需要额外划分一个 RL-validation split，
  或者使用 episode_reward 作为选择依据）
- 修改: if val_loss < best_val_loss: 保存模型（替代原 test_loss 判断）
- 保留: test_loss 仅作为日志记录，不参与模型选择

Run 级别修复 (main 函数):
- 修改: 使用训练过程中的最佳 val_loss（或最终 episode_reward）选择最佳 run
- 修改: test() 仅在最终选定的 run 上执行一次
```

**验证方法**：修复后对比修复前后的测试结果，确认性能差异在合理范围内（若差异过大，说明原结果依赖泄漏）。

**对论文的影响**：
- 如果修复后性能下降显著：需要更新 Table 2, 5, 6, 7 中的所有数字
- 如果性能基本不变：在 rebuttal 中说明 "修正了评估协议，结果保持一致"
- Section 5（Experimental setup）中需明确描述三阶段评估协议

### P0-2. 🔴 Replay Buffer 存储 None 值

**问题描述**：`RLMC_env.step()` 在 `done=True` 时返回 `obs=None, err=None`，被 push 到 buffer。后续 `np.array(None)` 会产生 object 类型数组，采样时可能崩溃或产生错误梯度。

**修复方案**：
```python
# 文件: train_RLMC_final.py, Exp.train() 方法中
# 在 self.buffer.push 之前添加条件判断:

if not done:
    self.buffer.push(
        observation, error, action, reward,
        next_observation, next_error, float(done)
    )
# 或者: 在 RLMC_env.step() 中，done=True 时返回零值张量而非 None
```

### P0-3. 🔴 论文与代码设置统一

以下不一致需要 **逐项确认并统一**，原则是"改论文描述，不重跑实验"：

| # | 不一致项 | 论文描述 | 代码实际值 | 决策建议 |
|---|---------|---------|-----------|---------|
| a | 归一化方式 | z-score (Eq.22) | min-max | **改论文** → min-max 公式 |
| b | 历史窗口 | T=24 (Table 4, §5.3) | T_in=72 | **改论文** → 72 |
| c | TCN kernel_size | 4 (Table 4) | 默认2 (未传参) | **需验证**：检查实际训练是否传了参数 |
| d | GAT 多头聚合 | 平均 (Eq.13) | concat+projection | **改论文** → 描述 concat |
| e | GCN 层数 | 1 (Table 4) | 2 (conv1→relu→conv2) | **改论文** → 2 |
| f | 图符号 D,N,S,F vs C,S | 正文/图不一致 | D,N,S,POI | **改论文** → 统一 D,N,S,F |
| g | ASTAM 公式 | 有 W_Kt, W_Ks, /√D | 无 key 变换无缩放 | **改论文** → 简化公式匹配代码 |

**关于 (c) TCN kernel_size 已验证**：
- `gated_TCN` 创建 `TemporalConvNet(input_size, num_channels)` 时**未传入 kernel_size**
- `TemporalConvNet.__init__` 默认值为 `kernel_size=2`
- ✅ 确认：实验实际使用 `kernel_size=2`，论文 Table 4 声称 4 是错误的
- **决策**：改论文 Table 4 → `kernel_size=2`（零成本）
- 注意：dilation rates `[1,2,4,8]` 是正确的（代码 `2**i`），与 Table 4 一致

### P0-4. 🔴 格式合规（编辑部硬性要求）

- 标题：当前已无缩写 ✅
- 关键词：需确认无缩写，需添加 "artificial intelligence" 和 "air quality forecasting application"
- 摘要：所有缩写首次出现时展开定义（DMGENet, TCN, HGLM, ASTAM, DDPG, RMSE, MAE, GC-LSTM, MSTGAN）
- EM 提交系统中的标题/关键词/摘要需与手稿一致

---

## 二、P0 代码修复：具体执行步骤

### Step 1: 修复弃用警告和设备管理

```python
# 文件: model/model_1.py

# 行 45: F.tanh → torch.tanh
# 行 47: F.sigmoid → torch.sigmoid
# 行 101: 移除双重 .cuda()
#   旧: nn.Parameter(torch.randn(num_nodes, apt_size).cuda(), requires_grad=True).cuda()
#   新: nn.Parameter(torch.randn(num_nodes, apt_size), requires_grad=True)
# 行 120, 124: squeeze() → squeeze(-1).squeeze(-1)
```

### Step 2: 修复验证流程缺少 no_grad

```python
# 文件: exp_base_model.py, val() 方法

# 在 for 循环外添加:
with torch.no_grad():
    for i, (features, target) in enumerate(self.val_loader):
        ...
    for i, (features, target) in enumerate(self.test_loader):
        ...
```

### Step 3: 修复测试集泄漏

```python
# 文件: train_RLMC_final.py, Exp.train() 方法

# 1. 将模型选择改为基于 validation loss
#    由于 RL 的 train 数据来自原 val 集，可以用 episode_reward 作为模型选择依据
#    或者：将 val 集再分出 80%做 RL 训练，20%做 RL 验证

# 最简方案：使用 episode_reward 选择
best_reward = float("-inf")
...
if episode_reward > best_reward:
    best_reward = episode_reward
    # 保存模型

# test_loss 仅作为日志，不参与选择

# 2. main() 中 run 级别选择也改为基于 validation 指标
```

### Step 4: 修复 replay buffer None 问题

```python
# 文件: RLMC_final/RLMC.py, RLMC_env.step()

# 方案：done 时返回零值代替 None
def step(self, action):
    reward = self._reward(action)
    self.current_step += 1
    done = self.current_step >= len(self.data_x)
    if not done:
        obs, err = self._get_state()
    else:
        # 返回零值占位，而非 None
        obs = np.zeros_like(self.data_x[0])
        err = np.zeros_like(self.data_error[0])
    return obs, err, reward, done, {}
```

---

## 三、P1：高收益补强实验与论文修改

### P1-1. 扩展预测时长至 1h/6h/12h/24h

**当前状态**：代码已支持（`train_RLMC_final.py:466` 有 `[6, 12, 24]`），`exp_base_model.py:249` 有 `[1, 6, 12, 24]`

**需要做的**：
1. 确认所有 horizon 的基础模型已训练完成
2. 确认 RLMC 在 1h/6h/12h/24h 上都有结果
3. 论文 Table 2 扩展为 1h/6h/12h/24h（当前仅有 1-6h）
4. 补充讨论：随预测步长增加的误差累积规律

### P1-2. 补充更强的基线模型

**三位审稿人共同关注**。策略：

| 优先级 | 方法 | 理由 | 实现方式 |
|--------|------|------|---------|
| 必须补 | 更新 Related Works | 所有审稿人都要求 | 添加 ~16 篇推荐文献引用 |
| 强烈建议 | 1-2 个近两年 SOTA | 回应"baseline 过时" | 找公开实现复现，或引用其公开结果 |
| 已有可用 | 集成方法对比 | 已有 Table 7 | 已有 Average/GWO/PSO/GA vs RLMC ✅ |

**关于审稿人推荐的特定文献**：
- R1 推荐的 4 篇：空气质量预测直接相关，应引用并在可能时对比
- R2 推荐的 7 篇：部分高度相关（CNN-LSTM, FL-BGRU），部分间接相关（绿色数据中心、温度波分析），均应引用
- R3 推荐的 5 篇：Graph Multi-Attention + Bayesian HPO 和 STGNN-TCN 高度相关，其余（区块链、机器视觉）关联较弱，在 broader context 中简述

### P1-3. 统计显著性

**当前状态**：已有 10 次重复运行 (`repeat_times=10`)，已保存 `all_runs_metrics.csv` 和 `summary_metrics.csv`

**需要做的**：
1. 基础模型也需多次运行（当前仅 1 次）→ 至少 5 次随机种子
2. 所有结果表改为 **mean ± std** 格式
3. 添加配对 t-检验或 Wilcoxon 检验（DMGENet vs 最强基线）
4. 代码：编写 `utils/statistical_test.py`

### P1-4. 可复现性信息补充

**纯文字修改，无代码成本**：
- GPU 型号、CPU、RAM
- PyTorch 版本、CUDA 版本、Python 版本
- Batch size = 64, Optimizer = Adam, lr = 0.001
- Early stopping: patience = 7
- Learning rate decay: 前5个 epoch 不衰减，之后 ×0.8/epoch
- 随机种子: 2026 (基础模型), 0-9 (RLMC 10 runs)
- 训练时长统计（已有 epoch_time 记录）

### P1-5. 计算效率分析

**代码修改**：
```python
# 新增 utils/compute_efficiency.py
from thop import profile, clever_format
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input_sample, n_runs=100):
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(10):
            model(input_sample)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_runs):
            model(input_sample)
        torch.cuda.synchronize()
        end = time.time()
    return (end - start) / n_runs
```

**论文修改**：添加一个表格对比各模型的参数量、训练时间/epoch、推理时间/sample

### P1-6. Discussion / Limitations 加强

**新增 Limitations 章节**（在 Conclusion 之前）：
1. 数据集时效性（2013-2017）
2. 仅验证 PM2.5 单一指标
3. 图构建依赖先验知识（距离阈值、邻居半径等）
4. DDPG 的 MDP 近似假设
5. 四模型并行训练的计算开销

**Discussion 深化**：
1. 为什么邻居图和功能相似图贡献差异大？（从 Table 5, 6 数据分析）
2. RL 权重随时间的动态变化可视化（利用已有 weights.csv）
3. 不同站点的预测难度差异分析

---

## 四、P2：有时间再做的增强实验

### P2-1. 新数据集泛化验证

- **最佳方案**：获取 2020-2023 年中国城市空气质量数据（如全国环保监测数据）
- **备选方案**：使用其他城市的公开数据集（如上海、印度德里等）
- **最低方案**：在 Limitations 中诚实承认，削弱 "practical utility" 表述

### P2-2. 噪声鲁棒性实验

```python
# utils/noise_robustness.py
def add_gaussian_noise(data, noise_level):
    noise = np.random.normal(0, noise_level * np.std(data), data.shape)
    return data + noise

# 在不同噪声水平下 (5%, 10%, 15%, 20%) 评估模型
```

### P2-3. 非平稳性分析

```python
# utils/stationarity_test.py
from statsmodels.tsa.stattools import adfuller, kpss

def test_stationarity(series):
    adf_result = adfuller(series)
    kpss_result = kpss(series, regression='c')
    return adf_result, kpss_result
```

### P2-4. 超参数敏感性分析

对以下关键超参数进行 grid search，每组跑 3 次取平均：
- TCN hidden units: [32, 64, 128]
- GAT attention heads: [2, 4, 8]
- Node embedding size: [5, 10, 20]
- Number of spatiotemporal blocks: [1, 2, 3]
- 图构建阈值：距离 σ_D, 分布 σ_S, 邻居半径 R, POI 阈值

---

## 五、执行时间表

### 第一轮：修复可信度（Day 1-5）

| 天 | 任务 | 输出 |
|----|------|------|
| D1 | 修复代码 bug（P0-1~P0-3 代码部分）| 干净的代码提交 |
| D2 | 修复测试集泄漏 → 重新训练 RLMC（至少 1h, 6h）| 新的 RLMC 结果 |
| D3 | 继续 RLMC 训练 (12h, 24h) + 统一论文描述 | 所有 horizon 结果 |
| D4 | 对比新旧结果，确认性能无大幅下降 | 验证报告 |
| D5 | 更新论文所有表格和公式 (P0-3) + 格式修正 (P0-4) | 论文初修版 |

### 第二轮：补强证据（Day 6-10）

| 天 | 任务 | 输出 |
|----|------|------|
| D6 | 基础模型多次运行（5种子）+ 统计检验 | mean±std 数据 |
| D7 | 新基线复现（如有公开代码）| 新基线结果 |
| D8 | 计算效率分析 + 可复现性信息 | 效率表格 |
| D9 | Discussion 深化 + Limitations 章节 | 论文章节 |
| D10 | Related Works 扩展（添加~16篇引用）| 论文章节 |

### 第三轮：润色提交（Day 11-17）

| 天 | 任务 | 输出 |
|----|------|------|
| D11-12 | 全文语言润色 + 图表质量提升 | 润色版论文 |
| D13-14 | 超参数敏感性（如有时间）| 附加实验 |
| D15 | Response Letter 撰写 | 逐条回复 |
| D16 | 全文最终检查 + EM 系统准备 | 终稿 |
| D17 | 提交缓冲 | - |

---

## 六、修复后的验证检查清单

### 代码修复验证

- [ ] `F.tanh` / `F.sigmoid` 弃用警告消除
- [ ] `node_embedding` 无双重 `.cuda()`
- [ ] `squeeze` 指定 dim 参数
- [ ] `val()` 包含 `torch.no_grad()`
- [ ] Replay buffer 不再存储 None
- [ ] 模型选择基于 validation（非 test）
- [ ] Run 选择基于 validation（非 test）

### 论文-代码一致性验证

- [ ] Eq.22 归一化公式与 `dataloader_Beijing_12.py` 一致
- [ ] Table 4 窗口长度与 `exp_base_model.py` T_in 一致
- [ ] Table 4 kernel_size 与 `TemporalConvNet` 实际参数一致
- [ ] Eq.13 GAT 聚合描述与 `GATLAyer.py` 实现一致
- [ ] Table 4 GCN 层数与 `GCNLayer.py` 实现一致
- [ ] 全文图符号 D/N/S/F 统一
- [ ] ASTAM 公式与 `model_1.py` SpatioTemporal_block 实现一致
- [ ] 分布相似图描述与 `Graph_Construction_Beijing_12.py` 一致

### 实验结果验证

- [ ] 修复测试集泄漏后，结果是否仍优于基线
- [ ] 所有 horizon (1h/6h/12h/24h) 结果完整
- [ ] 消融实验结果与修复后代码一致
- [ ] mean±std 统计数据完整

### 论文格式验证

- [ ] 标题无缩写
- [ ] 关键词无缩写，含 "artificial intelligence"
- [ ] 摘要所有缩写首次定义
- [ ] EM 系统内容与手稿一致

---

## 七、关键决策点

以下决策需要尽早做出，因为它们决定了后续工作的方向：

### 决策 1：TCN kernel_size
- 选项 A：改论文 Table 4 为 2（零成本）
- 选项 B：改代码为 4 + 重跑所有实验（高成本，可能结果更好）
- **建议**：先验证代码实际使用的是多少，如果确实是 2 则选 A

### 决策 2：ASTAM 公式
- 选项 A：简化论文公式匹配代码（保留当前性能）
- 选项 B：按论文公式修改代码 + 添加 key 变换和缩放 + 重跑（风险：性能可能变化）
- **建议**：选 A，因为当前性能已经优于基线

### 决策 3：分布相似图的概率分布
- 选项 A：修改论文描述，说明使用原始值的 JS 散度（技术上不准确但改动小）
- 选项 B：代码中添加直方图离散化步骤 + 重跑（更严谨）
- **建议**：选 B 如果时间允许，否则选 A 并在 limitations 中提及

### 决策 4：新数据集
- 选项 A：不添加，在 Limitations 中充分讨论（最低成本）
- 选项 B：添加一个较新的公开数据集（高成本，高收益）
- **建议**：视时间决定，但 R1 明确要求，不做可能被拒

### 决策 5：测试集泄漏修复后性能
- 如果性能下降 < 3%：直接使用新结果
- 如果性能下降 3-10%：检查是否有其他优化空间（如 RL 训练轮数增加）
- 如果性能下降 > 10%：需要重新审视 RL 方法的有效性，考虑调整 RL 超参数

---

## 八、Rebuttal 写作要点

### 结构模板
每条意见按以下结构回复：
1. **感谢** + **承认/澄清**
2. **已做修改**（具体到 Section/Table/Figure 编号）
3. **残余限制**（诚实说明无法完全解决的部分）

### 关键注意事项
- ❌ 不说 "the reviewer misunderstood"
- ❌ 不把所有问题推到 "future work"
- ❌ 不说 "this does not affect the conclusion" 除非已验证
- ✅ 测试集泄漏问题必须主动承认并说明已修复
- ✅ 论文-代码不一致必须逐项确认修正
- ✅ 所有新结果都标明随机种子、硬件、训练协议
