# 笨鸟 SB-Core 长上下文实验论文草稿

## 摘要

本文提出一种尝试在语言模型主干层替代 Transformer 的新架构 `SB-Core`。与标准自注意力主干不同，`SB-Core` 以流式递归状态、稀疏记忆路由和显式工作记忆写入为核心，不依赖 `self-attention`，也不使用 `KV cache`。为了避免在概念层空转，本文同时给出一个可训练的 `SB-Core-Mini` PyTorch 原型，以及一套与 tiny Transformer 公平对照的长上下文合成基准。实验聚焦 `passkey retrieval` 与 `needle in haystack` 两类任务。结果显示，`SB-Core-Mini` 已经能够在训练长度内稳定收敛，但在训练外长上下文泛化上仍然显著不足；不过，它在当前实验中保留了一点弱但可测的残余召回信号，说明“递归状态 + 稀疏记忆”这条路线值得继续推进。本文的结论不是“SB 已经替代 Transformer”，而是“SB 已形成一条可以被反复验证、逐步强化的替代路线”。

**关键词**：笨鸟，SB-Core，长上下文，Transformer 替代，外部记忆，稀疏路由

## 1. 研究问题

Transformer 的优势非常明确：训练稳定、并行效率高、生态成熟、下一个 token 预测范式清晰。但它也存在两个长期被讨论的结构性问题：

- 长上下文成本随着序列长度快速上升。
- 历史信息主要寄存在上下文窗口或 `KV cache` 中，长期记忆与持续学习并不自然。

因此，本工作要回答的问题不是“能否做一个更复杂的 RAG 层”，而是：

**是否可以构建一个不依赖自注意力主干、以流式状态和稀疏记忆为核心、仍然能够执行标准语言建模的架构。**

这就是 `SB-Core` 的目标。

## 2. 架构假设

`SB-Core` 只在主干层挑战 Transformer，不在应用层偷换问题。

第一阶段固定遵守以下约束：

1. 主干网络不使用 `self-attention`。
2. 推理过程不依赖 `KV cache`。
3. 训练目标仍是标准 `next-token prediction`。
4. 对比对象是同量级 tiny Transformer，而不是故意做弱基线。

## 3. SB-Core-Mini 原型

当前原型的实现位于：

- [core_lm_torch.py](d:/SB/sb/core_lm_torch.py)
- [core_lm_data.py](d:/SB/sb/core_lm_data.py)
- [transformer_baseline.py](d:/SB/sb/transformer_baseline.py)
- [v02_long_context_compare.py](d:/SB/examples/v02_long_context_compare.py)

### 3.1 主干结构

`SB-Core-Mini` 的每一层由三部分组成：

1. `SBMemoryRouter`
   根据当前状态与上一时刻状态生成查询，只从工作记忆与语义记忆中激活 `top-k` 个槽位。

2. `SBRecurrentCell`
   使用门控递归更新当前隐藏状态，并通过残差与层归一化稳定训练。

3. `SBMemoryWriter`
   将当前层输出按受控写门写回工作记忆槽位。

整体计算链可以写成：

```text
query_t = Router([x_t, h_{t-1}])
r_t = Read(M_work, M_semantic, query_t)
h_t = RecurrentCell(x_t, h_{t-1}, r_t)
M_work <- Write(h_t)
logits_t = LMHead(h_t)
```

### 3.2 记忆结构

当前原型只显式实现了两类记忆：

- `working memory`
  面向当前序列局部状态，快速写入，容易覆盖。

- `semantic memory`
  面向跨样本共享的稳定参数槽，训练期可学习。

`episodic memory` 在本文原型中尚未实现，这正是未来需要补齐的关键点之一。

## 4. 实验设计

### 4.1 对照原则

为了尽量保证“比的是架构，不是别的条件”，本实验固定：

- 相同词表规模：`vocab_size = 40`
- 相近参数量：
  - `SB-Core-Mini`：`258,338`
  - `tiny Transformer`：`246,912`
- 相同训练步数：`220`
- 相同训练批大小：`48`
- 相同评测批大小：`64`
- 相同优化器：`AdamW`
- 相同学习率：`6e-3`

### 4.2 任务设置

本文使用两类合成长上下文任务。

#### 任务 A：Passkey Retrieval

模型需要在较早位置见到一段短 key，并在较晚位置接收到查询信号后准确复现这段 key。

- 训练长度：`22`
- 长上下文测试长度：`60`

#### 任务 B：Needle in Haystack

模型需要在长随机序列中记住一段 `key -> value` 记录，并在查询 key 后产出对应 value。

- 训练长度：`28`
- 长上下文测试长度：`76`

### 4.3 评测指标

为避免总体 token loss 被大量 filler token 稀释，本文只对真正需要回忆的输出段计算指标：

- `token accuracy`
- `exact match`

也就是说，模型前面生成得再漂亮，只要关键值段错了，评测就不会给高分。

## 5. 实验结果

实验脚本：

```powershell
python -m examples.v02_long_context_compare
```

本次在当前 CUDA 环境中得到的结果如下。

| 任务 | 模型 | 训练长度内 exact match | 长上下文 exact match | 长上下文 token acc |
| --- | --- | ---: | ---: | ---: |
| passkey retrieval | SB-Core-Mini | 1.0000 | 0.0000 | 0.0195 |
| passkey retrieval | tiny Transformer | 1.0000 | 0.0000 | 0.0000 |
| needle in haystack | SB-Core-Mini | 1.0000 | 0.0000 | 0.0205 |
| needle in haystack | tiny Transformer | 1.0000 | 0.0000 | 0.0010 |

对应的训练损失与评测耗时如下。

| 任务 | 模型 | 训练损失 | 平均评测耗时（ms / batch） |
| --- | --- | ---: | ---: |
| passkey retrieval | SB-Core-Mini | 0.00197 | 190.95 |
| passkey retrieval | tiny Transformer | 0.00078 | 1.64 |
| needle in haystack | SB-Core-Mini | 0.00185 | 236.76 |
| needle in haystack | tiny Transformer | 0.00092 | 1.89 |

## 6. 结果分析

### 6.1 已经成立的部分

这次实验至少证明了三件事：

1. `SB-Core-Mini` 可以在标准 `next-token prediction` 框架下训练并收敛。
2. 它不依赖自注意力，也不依赖 `KV cache`，因此主干替代问题已经被真正落到代码里。
3. 它在长上下文失效时仍留下了少量残余召回信号，说明工作记忆和稀疏路由并非完全无效。

### 6.2 还没有成立的部分

如果标准定成“从架构上替代 Transformer”，当前版本显然还不够：

- 两个任务在训练外长度上的 `exact match` 都还是 `0.0`
- `SB-Core` 的推理速度明显慢于 tiny Transformer
- 当前工作记忆更像局部缓存，而不像真正可泛化的长程记忆机制

换句话说，现阶段 `SB-Core` 还只是“有方向感的可训练原型”，不是“已证明有效的替代主干”。

### 6.3 为什么会这样

当前失败大概率来自以下结构缺口：

1. **缺少显式 episodic memory**
   现在只有工作记忆和静态语义记忆，缺少真正面向跨长距离检索的事件级记忆层。

2. **写入策略过于简单**
   当前写入槽位主要按步数轮转，容易发生覆盖，而不是按内容重要性写入。

3. **训练课程过短**
   模型只在短长度上训练，几乎没有理由自行学会稳定长度外泛化。

4. **路由监督不足**
   当前模型只靠语言建模损失间接学路由，没有专门约束“该读哪一类记忆”。

## 7. 下一阶段路线

如果要继续推进，而不是停在“看起来很新”，更合理的顺序是：

### 7.1 课程训练

把训练长度从固定短序列改成逐步增长，例如：

```text
22 -> 40 -> 64 -> 96
```

让模型在记忆跨度上逐步适应，而不是一次性跨大坎。

### 7.2 内容寻址写入

把当前按时间轮转的写入方式，升级成“内容相似度 + 重要性门控 + 写保护”联合决策。

### 7.3 增加 episodic memory

加入事件片段级别的中速记忆层，让模型能把“可查询事实”与“即时工作状态”分开存。

### 7.4 路由辅助损失

除了 `next-token loss`，加入：

- 路由稀疏损失
- 槽位负载均衡损失
- 关键记忆命中监督

### 7.5 再做公平对照

完成上述改动后，再重复本文同一套基准，不改任务、不改指标，只比较结果是否改善。

## 8. 结论

本文给出的不是一篇“宣布替代 Transformer 已成功”的论文，而是一篇把这条路线从概念变成工程对象的阶段性论文。

当前最重要的结论有两个：

1. `SB-Core` 已经具备被严肃测试的最小条件，不再只是一个抽象想法。
2. 当前结果还远不足以宣称替代 Transformer，但已经显示出一个值得继续优化的架构雏形。

如果后续能够补齐课程训练、episodic memory、内容寻址写入和路由监督，`SB-Core` 才有机会从“新奇结构”推进为“真正的主干替代候选”。

## 附录：当前相关文件

- [笨鸟SB-Core替代Transformer技术草案.md](d:/SB/文档/笨鸟SB-Core替代Transformer技术草案.md)
- [v02_sb_core_toy_train.py](d:/SB/examples/v02_sb_core_toy_train.py)
- [v02_long_context_compare.py](d:/SB/examples/v02_long_context_compare.py)
- [core_lm_torch.py](d:/SB/sb/core_lm_torch.py)
- [transformer_baseline.py](d:/SB/sb/transformer_baseline.py)
