# Stupid Bird

`Stupid Bird (SB)` 是一个分成两条线推进的实验项目：

- `SB v0.1`：受限场景理解、结构化推理、RAG 与服务接口
- `SB-Core`：实验线，目标是用“存储融合”逐步替代 Transformer 主干计算

当前研发重点是 `SB-Core`。默认约定下，实验改动只针对实验线，不影响 `v0.1` 主线。

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
