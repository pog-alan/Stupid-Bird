from .auto_crawl import CrawlSource, Crawler, CrawlerConfig
from .core import Hypothesis, SBNetwork, Signal, SignalTemplate, Space, TransformRule
from .dialog import DialogStore
from .embedding_backends import (
    EMBEDDING_PRESETS,
    EmbeddingEncoder,
    HTTPEmbeddingEncoder,
    SentenceTransformerEncoder,
    create_embedding_encoder,
    resolve_embedding_backend_config,
)
from .extractor import ExtractedCandidate, SimpleExtractor
from .hierarchical_context import (
    ForgetStage,
    ForgettingPolicy,
    HierarchicalContextConfig,
    HierarchicalContextSpec,
    MappingCardinality,
    MappingRelation,
    MemoryLevel,
    MemoryNode,
    MergeMode,
    MergePlan,
    MergePolicy,
    ForgettingStep,
    ReplayQuery,
    ReplaySegment,
    SummaryLevelConfig,
)
from .ingest import Ingestor, KnowledgeStore, apply_stable_entries_to_config
from .llm_bridge import build_llm_context, retrieve_for_llm
from .llm_client import LLMConfig, OpenAICompatibleLLMClient, create_llm_client, load_llm_config
from .llm_runtime import SBLLMRuntime, SBLLMRuntimeConfig
from .core_lm import SBCoreConfig, SBCoreModelSpec
from .eval_long_context import (
    LongContextEvaluationSuite,
    LongContextMeasurement,
    LongContextScenario,
    LongContextTask,
)
from .memory_bank import MemoryBankConfig, MemorySlot, MemoryUpdatePlan, SparseMemoryBankSpec
from .ontology import SBV01Ontology, load_default_ontology
from .proactive import ProactiveQuestion, append_questions_to_payload, propose_questions
from .quality_gate import CandidateDecision, QualityGate, QualityGateConfig
from .rag_answer import build_grounded_answer
from .rag_pipeline import SBRAGConfig, SBRAGPipeline
from .rag_store import DocumentChunk, RAGKnowledgeBase, SourceDocument
from .reasoner import SBV01Engine
from .router import RouterConfig, RoutingDecision, SparseRouterSpec
from .server import run_server
from .train_lm import CurriculumStage, ExperimentStage, LossWeights, SBCoreTrainingPlan, TrainLMConfig
from .vector_memory import HashedVectorEncoder, MemoryRecord, VectorHit, VectorMemoryIndex

DEFAULT_TASKS = ("passage_retrieval_zh", "multifieldqa_zh", "dureader")


def evaluate_longbench_local(*args, **kwargs):
    from .longbench_local_eval import evaluate_longbench_local as _evaluate_longbench_local

    return _evaluate_longbench_local(*args, **kwargs)


def score_answer_continuation(*args, **kwargs):
    from .longbench_local_eval import score_answer_continuation as _score_answer_continuation

    return _score_answer_continuation(*args, **kwargs)


_TORCH_AVAILABLE = True
try:
    from .core_lm_data import (
        ToySequenceBatch,
        ToyTaskVocab,
        decode_tokens,
        sample_copy_batch,
        sample_needle_in_haystack_batch,
        sample_passkey_batch,
    )
    from .core_lm_torch import (
        SBCoreMiniLM,
        SBCoreMemoryState,
        SBCoreMiniTorchConfig,
        SBRuntimeGates,
        next_token_loss,
        runtime_device_report,
        staged_runtime_gates,
    )
    from .transformer_baseline import TinyTransformerConfig, TinyTransformerLM
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _TORCH_AVAILABLE = False

_TEXT_CORPUS_AVAILABLE = True
try:
    from .text_corpus import (
        CharTokenizer,
        StageCorpusPaths,
        SubwordTokenizer,
        TextBatch,
        TextCorpusPreparationConfig,
        build_char_tokenizer,
        build_subword_tokenizer,
        build_stage_texts,
        load_char_tokenizer,
        load_longbench_rows,
        load_prepared_corpus_paths,
        load_stage_corpus,
        load_text_tokenizer,
        prepare_local_text_corpus,
        sample_longbench_answer_batch,
        sample_stage_batch,
        sample_text_batch,
        summarize_stage_corpus,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"pyarrow", "torch"}:
        raise
    _TEXT_CORPUS_AVAILABLE = False

__all__ = [
    "CandidateDecision",
    "CrawlSource",
    "Crawler",
    "CrawlerConfig",
    "CurriculumStage",
    "DialogStore",
    "DEFAULT_TASKS",
    "DocumentChunk",
    "EMBEDDING_PRESETS",
    "EmbeddingEncoder",
    "ExperimentStage",
    "ExtractedCandidate",
    "ForgetStage",
    "ForgettingStep",
    "ForgettingPolicy",
    "HTTPEmbeddingEncoder",
    "HashedVectorEncoder",
    "HierarchicalContextConfig",
    "HierarchicalContextSpec",
    "Hypothesis",
    "Ingestor",
    "KnowledgeStore",
    "LLMConfig",
    "LongContextEvaluationSuite",
    "LongContextMeasurement",
    "LongContextScenario",
    "LongContextTask",
    "LossWeights",
    "MappingCardinality",
    "MappingRelation",
    "MemoryBankConfig",
    "MemoryLevel",
    "MemoryNode",
    "MemoryRecord",
    "MemorySlot",
    "MemoryUpdatePlan",
    "MergePolicy",
    "MergeMode",
    "MergePlan",
    "OpenAICompatibleLLMClient",
    "ProactiveQuestion",
    "QualityGate",
    "QualityGateConfig",
    "RAGKnowledgeBase",
    "ReplayQuery",
    "ReplaySegment",
    "RouterConfig",
    "RoutingDecision",
    "SBLLMRuntime",
    "SBLLMRuntimeConfig",
    "SBCoreConfig",
    "SBCoreModelSpec",
    "SBCoreTrainingPlan",
    "SBRAGConfig",
    "SBRAGPipeline",
    "SBNetwork",
    "SBV01Engine",
    "SBV01Ontology",
    "SentenceTransformerEncoder",
    "Signal",
    "SignalTemplate",
    "SimpleExtractor",
    "SourceDocument",
    "Space",
    "SparseMemoryBankSpec",
    "SparseRouterSpec",
    "SummaryLevelConfig",
    "TrainLMConfig",
    "TransformRule",
    "VectorHit",
    "VectorMemoryIndex",
    "append_questions_to_payload",
    "apply_stable_entries_to_config",
    "build_grounded_answer",
    "build_llm_context",
    "create_embedding_encoder",
    "create_llm_client",
    "evaluate_longbench_local",
    "load_default_ontology",
    "load_llm_config",
    "propose_questions",
    "resolve_embedding_backend_config",
    "retrieve_for_llm",
    "run_server",
    "score_answer_continuation",
]

if _TORCH_AVAILABLE:
    from .signal_schema import DynamicSchemaConfig, DynamicSchemaOperator
    __all__.extend(
        [
            "DynamicSchemaConfig",
            "DynamicSchemaOperator",
            "SBCoreMiniLM",
            "SBCoreMemoryState",
            "SBCoreMiniTorchConfig",
            "SBRuntimeGates",
            "TinyTransformerConfig",
            "TinyTransformerLM",
            "ToySequenceBatch",
            "ToyTaskVocab",
            "decode_tokens",
            "next_token_loss",
            "runtime_device_report",
            "sample_copy_batch",
            "sample_needle_in_haystack_batch",
            "sample_passkey_batch",
            "staged_runtime_gates",
        ]
    )

if _TEXT_CORPUS_AVAILABLE:
    __all__.extend(
        [
            "CharTokenizer",
            "StageCorpusPaths",
            "SubwordTokenizer",
            "TextBatch",
            "TextCorpusPreparationConfig",
            "build_char_tokenizer",
            "build_subword_tokenizer",
            "build_stage_texts",
            "load_char_tokenizer",
            "load_longbench_rows",
            "load_prepared_corpus_paths",
            "load_stage_corpus",
            "load_text_tokenizer",
            "prepare_local_text_corpus",
            "sample_longbench_answer_batch",
            "sample_stage_batch",
            "sample_text_batch",
            "summarize_stage_corpus",
        ]
    )
