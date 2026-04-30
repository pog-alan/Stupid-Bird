from .acmm import (
    ACMMConfig,
    ACMMEmotionVector,
    ACMM_MEMORY_TYPES,
    ACMMStepResult,
    AffectiveCausalMemoryModel,
    CausalGraph,
    CausalRule,
    GateDecision,
    LayeredMemoryStore,
    MemoryItem as ACMMemoryItem,
    ObjectState,
    RelationState,
    RetrievedMemory,
    WorldState,
    acmm_loss_spec,
    acmm_model_spec,
    compute_gates,
)
from .acmm_formal import (
    ACMMFormalSpec,
    FormalMap,
    FormalMetric,
    FormalObject,
    FormalSection,
    acmm_formal_markdown,
    acmm_formal_spec_dict,
    build_acmm_formal_spec,
)
from .acmm_text import (
    DEFAULT_TEXT_SIGNAL_RULES,
    TextObservation,
    TextSignalRule,
    build_text_observation,
    find_manifest_dataset_path,
    iter_chinese_c4_texts,
)
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
from .emotion_feedback import (
    EmotionAction,
    EmotionFeedback,
    EmotionFeedbackConfig,
    EmotionSupervisionConfig,
    EmotionSupervisionTarget,
    EmotionVector,
    MachineEmotionFeedbackEngine,
    append_emotion_feedback_to_payload,
    append_emotion_supervision_to_payload,
    build_emotion_feedback,
    build_emotion_supervision,
    emotion_supervision_loss_spec,
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
from .core_lm import (
    CORE_OBJECTIVE_NON_ATTENTION_RECALL,
    DEFAULT_RECALL_BANKS,
    SBCoreConfig,
    SBCoreModelSpec,
    SBRecallBankSpec,
    SBRecallMathSpec,
)
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
    "ACMMConfig",
    "ACMMEmotionVector",
    "ACMM_MEMORY_TYPES",
    "ACMMStepResult",
    "ACMMemoryItem",
    "ACMMFormalSpec",
    "AffectiveCausalMemoryModel",
    "CandidateDecision",
    "CORE_OBJECTIVE_NON_ATTENTION_RECALL",
    "CrawlSource",
    "Crawler",
    "CrawlerConfig",
    "CausalGraph",
    "CausalRule",
    "DEFAULT_RECALL_BANKS",
    "DEFAULT_TEXT_SIGNAL_RULES",
    "CurriculumStage",
    "DialogStore",
    "DEFAULT_TASKS",
    "DocumentChunk",
    "EMBEDDING_PRESETS",
    "EmbeddingEncoder",
    "EmotionAction",
    "EmotionFeedback",
    "EmotionFeedbackConfig",
    "EmotionSupervisionConfig",
    "EmotionSupervisionTarget",
    "EmotionVector",
    "ExperimentStage",
    "ExtractedCandidate",
    "ForgetStage",
    "ForgettingStep",
    "ForgettingPolicy",
    "FormalMap",
    "FormalMetric",
    "FormalObject",
    "FormalSection",
    "GateDecision",
    "HTTPEmbeddingEncoder",
    "HashedVectorEncoder",
    "HierarchicalContextConfig",
    "HierarchicalContextSpec",
    "Hypothesis",
    "Ingestor",
    "KnowledgeStore",
    "LayeredMemoryStore",
    "LLMConfig",
    "LongContextEvaluationSuite",
    "LongContextMeasurement",
    "LongContextScenario",
    "LongContextTask",
    "LossWeights",
    "MappingCardinality",
    "MappingRelation",
    "MachineEmotionFeedbackEngine",
    "MemoryBankConfig",
    "MemoryLevel",
    "MemoryNode",
    "MemoryRecord",
    "MemorySlot",
    "MemoryUpdatePlan",
    "MergePolicy",
    "MergeMode",
    "MergePlan",
    "ObjectState",
    "OpenAICompatibleLLMClient",
    "ProactiveQuestion",
    "QualityGate",
    "QualityGateConfig",
    "RAGKnowledgeBase",
    "ReplayQuery",
    "ReplaySegment",
    "RelationState",
    "RetrievedMemory",
    "RouterConfig",
    "RoutingDecision",
    "SBLLMRuntime",
    "SBLLMRuntimeConfig",
    "SBCoreConfig",
    "SBCoreModelSpec",
    "SBRecallBankSpec",
    "SBRecallMathSpec",
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
    "TextObservation",
    "TextSignalRule",
    "TrainLMConfig",
    "TransformRule",
    "VectorHit",
    "VectorMemoryIndex",
    "WorldState",
    "acmm_formal_markdown",
    "acmm_formal_spec_dict",
    "acmm_loss_spec",
    "acmm_model_spec",
    "append_questions_to_payload",
    "append_emotion_feedback_to_payload",
    "append_emotion_supervision_to_payload",
    "apply_stable_entries_to_config",
    "build_emotion_feedback",
    "build_emotion_supervision",
    "build_acmm_formal_spec",
    "build_grounded_answer",
    "build_llm_context",
    "build_text_observation",
    "create_embedding_encoder",
    "create_llm_client",
    "compute_gates",
    "emotion_supervision_loss_spec",
    "evaluate_longbench_local",
    "find_manifest_dataset_path",
    "iter_chinese_c4_texts",
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
    from .state_cache import (
        SBCachedState,
        SBCoreStateCache,
        SBStateCacheConfig,
        SBStateCacheForwardResult,
    )
    __all__.extend(
        [
            "DynamicSchemaConfig",
            "DynamicSchemaOperator",
            "SBCachedState",
            "SBCoreMiniLM",
            "SBCoreMemoryState",
            "SBCoreMiniTorchConfig",
            "SBCoreStateCache",
            "SBStateCacheConfig",
            "SBStateCacheForwardResult",
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
