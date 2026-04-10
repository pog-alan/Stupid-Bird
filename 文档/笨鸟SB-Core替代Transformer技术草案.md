# 笨鸟 SB-Core 替代 Transformer 技术草案

## 1. 目标定义

`SB-Core` 的目标不是在应用层“辅助”Transformer，而是在语言模型主干层替代它。

这意味着第一阶段必须满足以下条件：

1. 主干网络不使用 `self-attention`。
2. 推理不依赖 `KV cache`。
3. 训练目标仍然是标准 `next-token prediction`。
4. 在相同参数量、相同训练 token、相同数据条件下，至少在一项关键指标上优于同规模 Transformer：
   - 长上下文效率
   - 长期记忆稳定性
   - 持续学习能力
   - 外部记忆接入成本

## 2. 非目标

第一阶段明确不做：

- 不直接追求开放域聊天效果
- 不直接追求超过大型闭源模型
- 不把 `SB-RAG` 当作主干替代证明
- 不把结构化推理器当作语言主干
- 不以“可解释性”替代“语言建模能力”

## 3. 总体架构

`SB-Core` 建议拆成 5 层。

### 3.1 Token Embedding 层

输入 token `x_t` 被映射为基础表示 `e_t`：

```text
e_t = Embedding(x_t) + PositionSignal(t)
```

这里的位置编码不要求是 Transformer 风格，可以使用：

- 可学习时间嵌入
- 连续时间刻度
- 相对时间差
- 多时间尺度门控信号

### 3.2 工作状态核心层

这是替代 Transformer 的真正主干。

核心状态更新形式：

```text
h_t^l = Core_l(e_t, h_{t-1}^l, h_t^{l-1}, r_t^l)
```

其中：

- `h_t^l`：第 `l` 层在时刻 `t` 的工作状态
- `h_{t-1}^l`：上一时刻同层状态
- `h_t^{l-1}`：当前时刻下层输出
- `r_t^l`：记忆读取结果

推荐先做门控残差形式：

```text
u_t^l = W_u [h_{t-1}^l ; h_t^{l-1} ; r_t^l ; e_t]
g_t^l = sigmoid(W_g u_t^l)
c_t^l = phi(W_c u_t^l)
h_t^l = (1 - g_t^l) * h_{t-1}^l + g_t^l * c_t^l
```

第一版应强调：

- 流式更新
- 多时间尺度
- 有界状态
- 可并堆多层

### 3.3 稀疏路由层

路由器根据当前状态只激活少量记忆槽：

```text
q_t^l = Router_l(h_t^l)
s_t = score(q_t^l, K)
I_t = topk(s_t, k)
```

其中：

- `K`：记忆键集合
- `I_t`：当前时刻被激活的槽位索引

这一层的关键不是“更大”，而是“更稀疏、更稳定”。

### 3.4 外部记忆层

外部记忆不再是 RAG 应用层里的文档库，而是模型主干的一部分：

```text
r_t = Read(M, I_t)
z_t = Fuse(h_t, r_t)
w_t = WriteGate(z_t)
M <- Update(M, I_t, z_t, w_t)
```

建议把记忆分为 3 类：

1. `working memory`
   - 近程工作状态
   - 快速更新
   - 易遗忘
2. `episodic memory`
   - 长程事件片段
   - 中速更新
   - 可检索
3. `semantic memory`
   - 稳定概念槽
   - 慢速更新
   - 需高阈值写入

### 3.5 输出头

输出仍然走标准语言建模：

```text
logits_t = W_o z_t
p(x_{t+1}) = softmax(logits_t)
```

这一步必须保持和 Transformer 可比较。

## 4. 训练目标

第一阶段不要发明太多主目标，先保持标准语言建模，再逐步加辅助损失。

### 4.1 主损失

```text
L_lm = - sum_t log p(x_{t+1} | x_{\le t})
```

### 4.2 路由稀疏损失

约束每步只激活少量槽位：

```text
L_sparse = sum_t ||a_t||_1
```

### 4.3 负载均衡损失

避免所有 token 都挤到同一批槽位：

```text
L_balance = variance(slot_usage_distribution)
```

### 4.4 写入稳定损失

避免记忆被高频噪声污染：

```text
L_write = ||M_t - stopgrad(M_{t-1})|| * unstable_write_gate
```

### 4.5 总损失

```text
L_total = L_lm + lambda_1 L_sparse + lambda_2 L_balance + lambda_3 L_write
```

第一版建议：

- 主损失占绝对主导
- 其他项只做轻量约束

## 5. 构建路线

### 5.1 P0：玩具任务验证

目标：证明架构能收敛、能记住、能路由。

任务：

- copy
- reverse
- passkey retrieval
- long-range matching

成功标准：

- loss 稳定下降
- 在长序列下性能不崩
- 路由分布不塌缩

### 5.2 P1：小型语言模型

目标：证明 `SB-Core` 能做标准语言建模。

设置：

- 20M 到 80M 参数
- 单语或小规模多语数据
- 与同规模 tiny Transformer 公平对照

指标：

- train loss
- val loss
- perplexity
- 每 token 推理耗时
- 长度扩展下的显存占用

### 5.3 P2：长上下文专项

目标：验证 `SB-Core` 替代 Transformer 的关键卖点。

任务：

- needle in haystack
- long-context recall
- event chain replay
- multi-turn fact update

指标：

- 序列长度扩展时的精度下降曲线
- 推理时间增长曲线
- 内存使用增长曲线

### 5.4 P3：接回 SB-RAG 与 SB-Reason

目标：把语言主干和现有笨鸟系统重新接上。

顺序建议：

1. `SB-Core`
2. `SB-Core + external semantic memory`
3. `SB-Core + SB-RAG`
4. `SB-Core + structure reasoner`

## 6. 公平测试规则

如果你想说“替代 Transformer”，测试必须公平。

固定以下条件：

- 相同 tokenizer
- 相同训练 token 数
- 相同参数量级
- 相同 batch 大小范围
- 相同优化器
- 相同学习率调度
- 相同数据清洗方式
- 相同硬件或等价硬件预算

不能用更少数据、不同参数量、不同任务去做“替代”结论。

## 7. 推荐仓库结构

建议把 `SB-Core` 作为独立实验线放进当前项目，不直接污染 `v0.1` 应用层：

```text
sb/
  core_lm.py
  core_lm_torch.py
  core_lm_data.py
  memory_bank.py
  router.py
  train_lm.py
  eval_long_context.py
```

它们与现有模块的关系：

- `sb/reasoner.py`：应用层解释引擎
- `sb/rag_pipeline.py`：知识增强层
- `sb/server.py`：服务层
- `sb/core_lm.py`：未来替代 Transformer 的主干实验线
- `sb/core_lm_torch.py`：当前可训练的 `SB-Core-Mini` PyTorch 原型
- `sb/core_lm_data.py`：`copy / passkey` 玩具任务数据生成器

当前仓库已经提供了一个最小训练示例：

- `examples/v02_sb_core_toy_train.py`

这份示例不是最终训练器，而是为了验证：

1. 架构可以前向运行
2. loss 可以下降
3. 路由与工作记忆统计可以被观测

## 8. 第一版实验计划

### 阶段 A：先证明可训练

最低标准：

- 训练不发散
- 路由不完全塌缩
- 记忆槽有可解释使用分布

### 阶段 B：再证明可比较

最低标准：

- 在 tiny LM 上 loss 接近 Transformer
- 在长程任务上至少一项优于 Transformer

### 阶段 C：最后证明可扩展

最低标准：

- 可以接外部记忆
- 可以做持续学习
- 可以与 SB-RAG 合并

## 9. 推荐的第一版判断标准

不要把第一版成功标准定成“全面超越 Transformer”，而应定成：

1. `SB-Core` 能稳定完成标准 next-token prediction。
2. 在长上下文任务上具有明确架构优势。
3. 外部记忆接入比 Transformer 更自然。
4. 在持续学习场景中遗忘更可控。

## 10. 结论

如果只从架构层取代 Transformer，正确路径不是先做聊天，而是先做：

`流式状态主干 + 稀疏路由 + 外部记忆 + 标准语言建模头`

这条线成功后，现有的 `SB-RAG`、结构化推理和持续学习系统才能真正变成“主干之上的能力层”，而不只是外挂层。
