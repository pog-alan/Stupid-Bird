# Stupid Bird

`Stupid Bird (SB)` 是一个分成两条线推进的实验项目：

- `SB v0.1`：受限场景理解、结构化推理、RAG 与服务接口
- `SB-Core`：实验线，目标是用“存储融合”逐步替代 Transformer 主干计算

当前研发重点是 `SB-Core`。默认约定下，实验改动只针对实验线，不影响 `v0.1` 主线。

## 分支约定

- `main`：语言模型线，包含 `SB v0.1`、`SB-Core`、RAG、长上下文实验、真实语料训练与验证。
- `segSB`：图像分割与视觉推理线，包含 `SB-Visual`、v03 视觉训练脚本、rural/urban 真实图片数据流程。
- 后续默认规则：语言模型、长上下文、非 attention 召回网络改 `main`；图像分割、视觉场景理解改 `segSB`。

## 当前实验线能力

`SB-Core` 目前已经具备这些核心机制：

- 递归状态主干，不使用 `self-attention`
- `working / episodic / semantic` 三类存储
- `entity / relation / event` 三支路 signal 多层抽象
- `key-centric replay query builder`
- 短键保护槽，用于优化 `passkey / delayed recall`
- 分层上下文机制的首轮落地：
  - 复杂映射关系：`1-1 / 1-n / n-1`
  - 记忆合并：`full_merge / partial_merge / multi_merge`
  - 多步逐渐遗忘：`keep / cool / fade / archive / prune`
- 运行时 GPU 自检与记录

## 关键文件

- `sb/core_lm_torch.py`
  `SB-Core-Mini` 的实验主干、记忆路由、episodic replay、短键保护槽和遗忘机制
- `sb/hierarchical_context.py`
  分层上下文、记忆合并、任务回放、逐步遗忘的设计规格
- `examples/v02_sb_core_toy_train.py`
  实验线最小训练脚本，会输出 GPU 使用情况
- `examples/v02_long_context_compare.py`
  `SB-Core` 与 tiny Transformer 的长上下文对比脚本，会输出 GPU 使用情况

## 快速开始

安装依赖后，在仓库根目录运行：

```powershell
python -m compileall sb examples
python -m examples.v02_sb_core_toy_train
python -m examples.v02_long_context_compare
```

训练脚本会在启动时输出类似下面的运行信息：

```json
{
  "runtime": {
    "requested_device": "cuda",
    "cuda_available": true,
    "gpu_used": true,
    "device_name": "NVIDIA RTX A5000"
  }
}
```

如果没有可用 CUDA，会自动回退到 CPU。

## 当前结论

这条实验线已经证明了几件事：

- `SB-Core` 可以稳定训练
- 运行时可以明确记录是否使用了 GPU
- 短键保护槽 + key-centric replay 能在 `passkey / delayed recall` 任务上形成有效召回
- 无限上下文相关机制已经开始从设计层进入训练主干

同时也要诚实说明：

- `SB-Core` 还没有在整体上取代 Transformer
- 长上下文泛化仍然不稳定
- 当前实现更像“机制验证原型”，还不是高吞吐、高稳定的最终架构

## 仓库说明

- 历史说明文档已从仓库中移除，只保留本 `README`
- `文档/` 目录下保留的是运行所需的 JSON 配置和数据资产，不是说明文档
- `LICENSE` 采用 `MIT`

## SB State Cache 设计

`SB State Cache` 是实验线用来替代传统 `KV cache` 的增量运行机制。它不保存每层 attention 的历史 `key/value`，而是保存 `SB-Core` 已经融合后的连续记忆状态 `SBCoreMemoryState`。

核心原则：

- 不启用 `self-attention`，不依赖 Transformer 式 `KV cache`
- 缓存对象是 `working / episodic / episodic_key / summary / scene / schema buffer` 等 SB 记忆状态
- 输入按 segment 推进，每段只对新增 signal 做状态更新
- 如果前端传入完整 prompt，缓存会检测上一次 prompt 是否为当前 prompt 的前缀；若匹配，则只计算新增 suffix
- 缓存按 session 隔离，使用 LRU 上限控制，避免无限占用显存或内存
- 缓存元数据记录 `computed_tokens / reused_tokens / cache_hit / reset_reason / schema alignment`，方便后续评测

关键文件：

- `sb/state_cache.py`
  `SBCoreStateCache`、`SBStateCacheConfig` 和 `SBStateCacheForwardResult` 的实现
- `examples/v02_state_cache_smoke.py`
  最小验证：第一次计算完整 prompt，第二次复用前缀状态，仅计算新增 token

最小验证命令：

```powershell
python -m examples.v02_state_cache_smoke
```

这条路线的目标不是把 `KV cache` 移植进 SB，而是形成 SB 自己的 `state cache`：把已经完成的存储融合结果作为下一段计算的起点，从而让长上下文更像“持续记忆推进”，而不是重复扫描历史 token。

## SB-Core 当前收窄目标：非 attention 长程召回网络

为了让实验线更可验证，`SB-Core` 的近期目标暂时不再表述为“完整替代 Transformer”，而是收窄为：

**在不使用 self-attention、不使用 Transformer KV cache 的前提下，构建一个依靠稀疏记忆读写与状态缓存完成长程召回的序列网络。**

这意味着当前核心不是开放式生成能力，而是长程信息恢复能力。优先验证任务包括：

- `passkey_retrieval`：前文给出 key/value，远距离查询时恢复 value
- `delayed_recall`：信号延迟若干步后被再次请求
- `needle_in_haystack`：在长噪声序列中召回指定 needle
- `segmented_recall`：长序列被切成多个 segment 后，通过 `memory_state` / `state cache` 保持召回能力

### 数学对象

输入序列为 `x_1:T`，其中 `x_t in V`。模型逐步处理序列，不构造 `T x T` token attention 矩阵。

完整连续状态定义为：

```text
S_t = ({h_t^l}_{l=1..L}, {M_t^b}_{b in B}, B_t^summary, B_t^scene, c_t, meta_t)
M_t^b = (K_t^b, V_t^b, strength_t^b, age_t^b, usage_t^b, schema_mass_t^b)
B = {working, episodic, key, summary, scene}
```

其中：

- `h_t^l` 是第 `l` 层递归隐藏状态
- `M^working` 负责短期局部状态
- `M^episodic` 负责跨段事件片段
- `M^key` 负责 key-centric replay，专门服务 passkey / delayed recall
- `M^summary` 负责分层摘要
- `M^scene` 负责更稳定的场景级或任务级记忆
- `c_t` 是动态 schema 分布，用来约束写入、召回和摘要链路

### 非 attention 约束

SB-Core 当前严格禁止以下操作：

- 不构造 `A = softmax(QK^T / sqrt(d))`
- 不做任意 token 对任意 token 的全局 self-attention
- 不保存 Transformer 式每层历史 `K/V`
- 不把 `KV cache` 作为长上下文能力来源

允许的缓存只有 `SB State Cache`：

```text
Cache(session) = (prefix_digest, token_count, S_t, metadata)
```

它缓存的是融合后的 `S_t`，不是 attention key/value 历史。

### 递推计算

输入嵌入与 signal 抽象：

```text
e_t = E[x_t]
a_t^0 = P_e e_t
a_t^{m+1}, c_t^{m+1}, stop_t^m = A_m(a_t^m, c_t^m)
a_t = a_t^{m*}, where m* = min{m | stop_t^m >= tau_stop}
```

记忆召回：

```text
q_t^{l,b} = Q_b[h_{t-1}^l; z_t^{l-1}; a_t; c_t]
score_i^{l,b} =
  sim(q_t^{l,b}, K_{t-1,i}^b) / tau_b
  + alpha_b strength_{t-1,i}^b
  - beta_b age_{t-1,i}^b
  + gamma_b schema_align(c_t, schema_i^b)
I_t^{l,b} = TopK(score^{l,b}, k)
r_t^{l,b} = sum_{i in I_t^{l,b}} softmax(score_i^{l,b}) V_{t-1,i}^b
```

递归融合：

```text
r_t^l = Fuse_b({r_t^{l,b}}_{b in B})
u_t^l = [h_{t-1}^l; z_t^{l-1}; a_t; r_t^l; c_t]
g_t^l = sigmoid(W_g^l u_t^l)
candidate_t^l = phi(W_c^l u_t^l)
h_t^l = (1 - g_t^l) * h_{t-1}^l + g_t^l * candidate_t^l
z_t^l = Norm(W_z^l[h_t^l; r_t^l; a_t])
```

记忆写入：

```text
write_t^b = sigmoid(W_w^b[z_t^L; a_t; c_t])
hat_k_t^b = K_b[z_t^L; a_t; c_t]
hat_v_t^b = V_b[z_t^L; a_t; c_t]
j_t^b = argmin_i overwrite_cost_i^b
overwrite_cost_i^b = strength_i^b + protection_i^b - eta_b age_i^b
K_{t,j}^b = (1-write_t^b)K_{t-1,j}^b + write_t^b hat_k_t^b
V_{t,j}^b = (1-write_t^b)V_{t-1,j}^b + write_t^b hat_v_t^b
strength_t^b = rho_b strength_{t-1}^b + write_t^b one_hot(j_t^b)
```

key-centric replay：

```text
q_t^replay = R(h_t^L, a_t, c_t, key_focus_t, delay_t)
I_t^K = TopK(sim(q_t^replay, K_{t-1}^K) + key_usage_bonus - age_penalty, k_K)
r_t^replay = Read(M_{t-1}^K, I_t^K)
z_t^L <- z_t^L + lambda_replay Gate(z_t^L, r_t^replay) * r_t^replay
```

输出与训练目标：

```text
logits_t = W_o z_t^L
p_theta(y_t | x_1:t, S_0) = softmax(logits_t)

L =
  CE(y_t, p_theta)
  + lambda_recall L_recall
  + lambda_route L_sparse_route
  + lambda_schema L_schema_align
  + lambda_write L_write_stability
  + lambda_forget L_forgetting_control
```

### 复杂度目标

因为不构造 `T x T` attention，SB-Core 的目标不是 `O(T^2 d)`。

若每步对全部记忆槽打分：

```text
O(T * L * sum_b N_b * d)
```

若后续把记忆检索换成 ANN、分桶索引或学习型稀疏索引，目标变成：

```text
O(T * L * |B| * k * d + index_cost)
```

状态空间为：

```text
O(L * sum_b N_b * d)
```

其中历史 token 不以 KV cache 形式线性累积，长程信息必须被写入、合并、摘要、遗忘和重放。

### 验收标准

当前阶段验收不看“是否像大语言模型一样聊天”，只看长程召回闭环：

- 配置层 `use_attention=False` 且 `use_kv_cache=False`，开启即报错
- `passkey_retrieval / delayed_recall / needle_in_haystack` 至少一项 `carry on > carry off`
- `summary_schema_alignment_mean` 和 `scene_schema_alignment_mean` 在长上下文阶段稳定大于 0
- `SB State Cache` 命中时 `reused_tokens` 增长，`computed_tokens` 只覆盖新增 suffix
- 实验报告必须输出召回准确率、精确匹配率、cache 命中率、schema 链路激活率

形式化规格入口：

```powershell
python -m examples.v02_core_recall_math_smoke
```

## 真实语料训练与验证流程

当前真实语料链路使用三类公开中文数据：

- `Wikipedia zh`：作为 foundation 真实连续文本
- `CLUE afqmc / tnews / cmnli`：作为 structured 真实任务文本
- `LongBench passage_retrieval_zh / multifieldqa_zh / dureader`：作为 long_context 长程召回与问答验证文本

下载并生成本地 manifest：

```powershell
python -m examples.v02_prepare_datasets
```

把真实语料整理成 `foundation / structured / long_context` 三阶段训练与验证集：

```powershell
python -m examples.v02_prepare_text_corpus `
  --profile-path configs/sb_core_stage3_data_profile.json `
  --manifest-path data/manifest.json
```

最小真实语料训练与验证 smoke：

```powershell
python -m examples.v02_text_curriculum_train `
  --prepared-corpus-manifest data/processed/text_corpus_stage3_baseline/corpus_manifest.json `
  --foundation-steps 2 `
  --structured-steps 2 `
  --long-steps 2 `
  --foundation-seq-len 64 `
  --structured-seq-len 80 `
  --long-seq-len 128 `
  --foundation-batch-size 1 `
  --structured-batch-size 1 `
  --long-batch-size 1 `
  --d-model 32 `
  --state-dim 32 `
  --num-layers 1 `
  --semantic-slots 8 `
  --working-slots 8 `
  --router-top-k 2 `
  --stage-val-batches 1 `
  --eval-every 6 `
  --eval-max-samples 1 `
  --eval-tasks passage_retrieval_zh `
  --eval-carry-policy uniform `
  --longbench-answer-ratio 0.75 `
  --carry-memory-stages long_context `
  --checkpoint-dir data/processed/checkpoints/text_curriculum_real_smoke `
  --experiment-tag real_corpus_recall_smoke `
  --output-path data/processed/experiments/text_curriculum_real_smoke.json
```

carry 对照验证：

```powershell
python -m examples.v02_longbench_local_eval `
  --checkpoint data/processed/checkpoints/text_curriculum_real_smoke/last.pt `
  --tasks passage_retrieval_zh `
  --max-samples 1 `
  --carry-policy uniform `
  --output-path data/processed/experiments/longbench_real_smoke_carry_on.json

python -m examples.v02_longbench_local_eval `
  --checkpoint data/processed/checkpoints/text_curriculum_real_smoke/last.pt `
  --tasks passage_retrieval_zh `
  --max-samples 1 `
  --carry-policy uniform `
  --no-carry-memory `
  --output-path data/processed/experiments/longbench_real_smoke_carry_off.json
```

当前一次真实语料 smoke 的结果：

- 训练数据：`foundation=949`，`structured=2918`，`long_context=365`
- 验证数据：`foundation=50`，`structured=154`，`long_context=19`
- 运行设备：`NVIDIA RTX A5000`，`gpu_used=true`
- `long_context` 训练阶段：`summary_schema_alignment_mean=0.0358`，`scene_schema_alignment_mean=0.0597`，`schema_chain_activated=true`
- LongBench carry on：`mean_answer_loss=8.1782`，`schema_chain_activated=true`，`schema_chain_activated_ratio=0.9744`
- LongBench carry off：`mean_answer_loss=8.6178`，`schema_chain_activated=false`，`schema_chain_activated_ratio=0.0`

注意：这只是 6 step 的 smoke，不代表模型已经学会真实长程问答；它验证的是完整链路可以在真实语料上训练、验证、保存 checkpoint，并且 carry memory 会实际激活 summary/scene 召回链。

## Chinese-C4 训练语料

Chinese-C4 使用 [shjwudp/chinese-c4](https://huggingface.co/datasets/shjwudp/chinese-c4)，数据格式为 `.jsonl.zst` 分片。第一次使用前需要安装 zstd 解压依赖：

```powershell
python -m pip install zstandard
```

下载一个可控样本并写入 `data/manifest.json`：

```powershell
python -m examples.v02_prepare_chinese_c4 `
  --max-rows 5000 `
  --max-shards 2 `
  --min-chars 80 `
  --manifest-path data/manifest.json
```

重新构建带 Chinese-C4 的文本训练集：

```powershell
python -m examples.v02_prepare_text_corpus `
  --profile-path configs/sb_core_chinese_c4_data_profile.json `
  --manifest-path data/manifest.json
```

Chinese-C4 smoke 训练：

```powershell
python -m examples.v02_text_curriculum_train `
  --prepared-corpus-manifest data/processed/text_corpus_chinese_c4_baseline/corpus_manifest.json `
  --foundation-steps 1 `
  --structured-steps 1 `
  --long-steps 1 `
  --foundation-seq-len 64 `
  --structured-seq-len 80 `
  --long-seq-len 128 `
  --foundation-batch-size 1 `
  --structured-batch-size 1 `
  --long-batch-size 1 `
  --d-model 32 `
  --state-dim 32 `
  --num-layers 1 `
  --semantic-slots 8 `
  --working-slots 8 `
  --router-top-k 2 `
  --stage-val-batches 1 `
  --longbench-answer-ratio 0.75 `
  --carry-memory-stages long_context `
  --checkpoint-dir data/processed/checkpoints/text_curriculum_chinese_c4_smoke `
  --experiment-tag chinese_c4_recall_smoke `
  --output-path data/processed/experiments/text_curriculum_chinese_c4_smoke.json
```

当前已下载 Chinese-C4 样本：

- 数据源：`shjwudp/chinese-c4`
- 本地样本：`data/raw/chinese_c4/chinese_c4_sample_5000.jsonl`
- 样本行数：`5000`
- 样本大小：约 `12.42 MB`
- 已并入 foundation 训练集：`foundation train=3798`，`foundation val=200`
- Chinese-C4 smoke：`gpu_used=true`，checkpoint 位于 `data/processed/checkpoints/text_curriculum_chinese_c4_smoke/last.pt`
