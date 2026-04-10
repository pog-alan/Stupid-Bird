# 笨鸟 SB-RAG 设计

## 定位

`SB-RAG` 是笨鸟面向 LLM 的第一版成型检索增强架构。它不是单纯把文本切块后做向量召回，而是把笨鸟已有的结构化场景理解能力接到检索链路里，形成：

- 结构化理解
- 记忆召回
- 文档召回
- 证据编排
- 回答草案
- LLM 上下文包
- 外部模型生成

## 核心流程

1. 输入文本先进入 `SBV01Engine`，抽取对象、属性、关系、事件和场景候选。
2. 结构化结果用于扩展查询词，驱动两类召回：
   - 向量记忆召回：概念空间、属性空间、关系空间、场景空间
   - 文档分块召回：外部网页、手动知识文档、后台采集结果
3. 检索结果进入 `SBRAGPipeline`，按“向量分 + 词面奖励”重排。
4. 管线同时产出四类结果：
   - `analysis`：结构化分析
   - `answer_draft`：不依赖外部 LLM 的 grounded 回答草案
   - `llm_packet`：可直接发送给外部 LLM 的消息包
   - `llm_response`：外部模型生成结果

## 关键模块

- `sb/rag_store.py`
  - 文档知识库存储
  - 文档去重
  - 分块
  - 爬虫结果入库
- `sb/rag_pipeline.py`
  - 混合召回
  - 查询扩展
  - 文档重排
  - LLM 包构造
- `sb/rag_answer.py`
  - 回答草案
  - grounded facts
  - uncertainties
  - citations
- `sb/llm_client.py`
  - OpenAI 兼容接口客户端
  - 环境变量读取 API Key
  - 外部生成返回解析
- `sb/server.py`
  - `/rag_ingest`
  - `/rag_query`
  - `/llm_context`
  - `/generate`

## 当前接口

### `POST /rag_ingest`

写入手动知识文档。

### `POST /rag_query`

执行完整 SB-RAG 查询，返回：

- `analysis`
- `retrieved_memories`
- `retrieved_documents`
- `answer_draft`
- `llm_packet`

### `POST /llm_context`

执行完整 SB-RAG 召回，但只返回适合喂给外部 LLM 的上下文包，同时附带 `answer_draft` 供前端预览。

### `POST /generate`

执行完整 SB-RAG 召回后，再调用外部 LLM 生成最终答案，返回：

- `analysis`
- `llm_packet`
- `llm_response`

## LLM 接入方式

当前生成层按“OpenAI 兼容接口”设计，默认使用：

- `llm.base_url`
- `llm.chat_path`
- `llm.model`
- `llm.api_key_env`

当前默认值已经按 MiMo 官方接口收紧为：

- `base_url = https://api.xiaomimimo.com/v1`
- `chat_path = /chat/completions`
- `model = mimo-v2-pro`
- `api_key_env = MIMO_API_KEY`

其中 API Key 只从环境变量读取，不写入仓库配置。

推荐配置方式：

```powershell
$env:MIMO_API_KEY="你的新密钥"
python -m examples.v01_server
```

配置文件中的 `llm` 段只保留非敏感信息，例如：

```json
{
  "llm": {
    "enabled": true,
    "provider": "openai_compatible",
    "base_url": "https://api.xiaomimimo.com/v1",
    "chat_path": "/chat/completions",
    "model": "mimo-v2-pro",
    "api_key_env": "MIMO_API_KEY"
  }
}
```

服务端 `GET /status` 现在还会返回：

- `llm_endpoint`
- `llm_api_key_env`
- `llm_api_key_present`
- `llm_ready`

这样可以快速判断当前进程是否真的读到了运行环境里的 key。

## 与普通 RAG 的区别

普通 RAG 常见路径是：

`原始文本 -> 向量检索 -> LLM`

笨鸟 SB-RAG 是：

`原始文本 -> 结构化理解 -> 结构引导检索 -> 记忆/文档混合召回 -> grounded 回答/LLM 包 -> 外部模型生成`

因此它更适合：

- 对象-属性-关系-事件型场景理解
- 可解释输出
- 主动提问
- 持续学习后的增量知识接入

## 当前边界

这版 SB-RAG 仍然是 `v0.1` 风格的受限领域实现，因此边界仍然有效：

- 适合短场景描述
- 适合受限领域知识
- 适合局部异常、堆放、污染、位置关系场景
- 还不适合开放域百科问答
- 还不是通用聊天模型本体

## 建议使用方式

1. 后台用 `/rag_ingest` 或爬虫持续补充文档知识库。
2. 前台查询优先走 `/rag_query`，直接拿 `answer_draft` 和证据。
3. 需要高质量自然语言输出时，再走 `/generate` 调用外部模型。
