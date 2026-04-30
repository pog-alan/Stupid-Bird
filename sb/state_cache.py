from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

import torch
from torch import Tensor

from .core_lm_torch import SBCoreMemoryState


def _now() -> float:
    return time.time()


def _flatten_single_sequence(input_ids: Tensor | Sequence[int]) -> tuple[int, ...]:
    if isinstance(input_ids, Tensor):
        detached = input_ids.detach().cpu()
        if detached.ndim == 0:
            return (int(detached.item()),)
        if detached.ndim == 1:
            return tuple(int(item) for item in detached.tolist())
        if detached.ndim == 2:
            if detached.shape[0] != 1:
                raise ValueError("SB State Cache 当前只支持单会话 batch_size=1 的前缀复用。")
            return tuple(int(item) for item in detached[0].tolist())
        raise ValueError("input_ids 必须是一维或 batch_size=1 的二维张量。")
    return tuple(int(item) for item in input_ids)


def _tokens_to_tensor(tokens: Sequence[int], *, like: Tensor) -> Tensor:
    return torch.tensor([list(tokens)], dtype=like.dtype, device=like.device)


def _digest_tokens(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _summarize_aux(aux: Mapping[str, Any] | None, keys: Sequence[str]) -> Dict[str, float]:
    if not aux:
        return {}
    summary: Dict[str, float] = {}
    for key in keys:
        value = aux.get(key)
        if isinstance(value, (int, float)):
            summary[key] = float(value)
    return summary


@dataclass(frozen=True)
class SBStateCacheConfig:
    """SB-Core 的段级状态缓存配置。

    这个缓存保存的是 SB 融合后的连续记忆状态，不保存 Transformer attention K/V。
    """

    max_sessions: int = 16
    token_history_limit: int = 8192
    detach_on_store: bool = True
    move_to_cpu: bool = False
    max_idle_seconds: float = 0.0
    aux_metric_keys: tuple[str, ...] = (
        "episodic_replay_schema_alignment_mean",
        "summary_schema_alignment_mean",
        "scene_schema_alignment_mean",
        "episodic_key_read_score_mean",
        "summary_read_score_mean",
        "scene_read_score_mean",
    )

    def validate(self) -> None:
        if self.max_sessions <= 0:
            raise ValueError("max_sessions 必须大于 0。")
        if self.token_history_limit <= 0:
            raise ValueError("token_history_limit 必须大于 0。")
        if self.max_idle_seconds < 0:
            raise ValueError("max_idle_seconds 不能小于 0。")


@dataclass
class SBCachedState:
    session_id: str
    state: SBCoreMemoryState
    token_count: int = 0
    segment_count: int = 0
    token_history: tuple[int, ...] = field(default_factory=tuple)
    history_truncated: bool = False
    token_digest: str = ""
    stage_name: str = ""
    last_aux: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    cache_hits: int = 0
    cache_misses: int = 0

    def as_metadata(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "token_count": int(self.token_count),
            "segment_count": int(self.segment_count),
            "history_tokens_retained": int(len(self.token_history)),
            "history_truncated": bool(self.history_truncated),
            "token_digest": self.token_digest,
            "stage_name": self.stage_name,
            "last_aux": dict(self.last_aux),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
        }


@dataclass(frozen=True)
class SBStateCacheForwardResult:
    session_id: str
    output: Dict[str, Any] | None
    state: SBCoreMemoryState
    computed_tokens: int
    reused_tokens: int
    cache_hit: bool
    reset_reason: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)


class SBCoreStateCache:
    """面向 SB-Core 的状态缓存。

    设计目标：
    - append-only 输入：把上一段输出的 `SBCoreMemoryState` 直接喂给下一段。
    - full prompt 输入：如果新 prompt 以前一次 prompt 为前缀，只计算新增 suffix。
    - 有界保存：按会话 LRU 淘汰，token 历史只保留有限窗口。
    """

    def __init__(self, config: SBStateCacheConfig | None = None) -> None:
        self.config = config or SBStateCacheConfig()
        self.config.validate()
        self._entries: "OrderedDict[str, SBCachedState]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._resets = 0

    def __len__(self) -> int:
        return len(self._entries)

    def reset(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._entries.clear()
            self._resets += 1
            return
        if session_id in self._entries:
            del self._entries[session_id]
            self._resets += 1

    def get(self, session_id: str, *, device: torch.device | str | None = None) -> SBCachedState | None:
        self._expire_idle()
        entry = self._entries.get(session_id)
        if entry is None:
            self._misses += 1
            return None
        self._entries.move_to_end(session_id)
        if device is not None:
            entry.state = entry.state.moved_to(device)
        entry.cache_hits += 1
        self._hits += 1
        return entry

    def put(
        self,
        session_id: str,
        state: SBCoreMemoryState,
        *,
        appended_tokens: Sequence[int] | Tensor = (),
        stage_name: str = "",
        aux: Mapping[str, Any] | None = None,
    ) -> SBCachedState:
        tokens = _flatten_single_sequence(appended_tokens) if len(appended_tokens) else ()
        previous = self._entries.get(session_id)
        if previous is None:
            token_history = tokens[-self.config.token_history_limit :]
            token_count = len(tokens)
            segment_count = 1 if tokens else 0
            history_truncated = len(tokens) > self.config.token_history_limit
            cache_hits = 0
            cache_misses = 0
            created_at = _now()
        else:
            token_history = (previous.token_history + tokens)[-self.config.token_history_limit :]
            token_count = previous.token_count + len(tokens)
            segment_count = previous.segment_count + (1 if tokens else 0)
            history_truncated = (
                previous.history_truncated
                or len(previous.token_history) + len(tokens) > self.config.token_history_limit
            )
            cache_hits = previous.cache_hits
            cache_misses = previous.cache_misses
            created_at = previous.created_at

        stored_state = state.detached() if self.config.detach_on_store else state
        if self.config.move_to_cpu:
            stored_state = stored_state.moved_to("cpu")
        entry = SBCachedState(
            session_id=session_id,
            state=stored_state,
            token_count=token_count,
            segment_count=segment_count,
            token_history=tuple(token_history),
            history_truncated=history_truncated,
            token_digest=_digest_tokens(token_history),
            stage_name=stage_name,
            last_aux=_summarize_aux(aux, self.config.aux_metric_keys),
            created_at=created_at,
            updated_at=_now(),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )
        self._entries[session_id] = entry
        self._entries.move_to_end(session_id)
        self._evict_lru()
        return entry

    def append(
        self,
        model: Any,
        input_ids: Tensor,
        *,
        session_id: str = "default",
        stage_name: str = "",
        reset: bool = False,
        return_aux: bool = True,
    ) -> SBStateCacheForwardResult:
        if reset:
            self.reset(session_id)
        entry = self.get(session_id, device=input_ids.device)
        tokens = _flatten_single_sequence(input_ids)
        output = model(
            input_ids,
            memory_state=entry.state if entry is not None else None,
            return_state=True,
            return_aux=return_aux,
        )
        state = output["state"]
        next_entry = self.put(
            session_id,
            state,
            appended_tokens=tokens,
            stage_name=stage_name,
            aux=output.get("aux") if isinstance(output, Mapping) else None,
        )
        return SBStateCacheForwardResult(
            session_id=session_id,
            output=output,
            state=next_entry.state,
            computed_tokens=len(tokens),
            reused_tokens=max(next_entry.token_count - len(tokens), 0),
            cache_hit=entry is not None,
            reset_reason="manual_reset" if reset else "",
            metadata=next_entry.as_metadata(),
        )

    def advance_from_prompt(
        self,
        model: Any,
        input_ids: Tensor,
        *,
        session_id: str = "default",
        stage_name: str = "",
        return_aux: bool = True,
        allow_prefix_reuse: bool = True,
    ) -> SBStateCacheForwardResult:
        tokens = _flatten_single_sequence(input_ids)
        entry = self.get(session_id, device=input_ids.device)
        reset_reason = ""
        reused_tokens = 0

        if allow_prefix_reuse and entry is not None and not entry.history_truncated:
            cached_tokens = entry.token_history
            if len(tokens) >= len(cached_tokens) and tuple(tokens[: len(cached_tokens)]) == cached_tokens:
                suffix = tokens[len(cached_tokens) :]
                reused_tokens = len(cached_tokens)
                if not suffix:
                    return SBStateCacheForwardResult(
                        session_id=session_id,
                        output=None,
                        state=entry.state,
                        computed_tokens=0,
                        reused_tokens=reused_tokens,
                        cache_hit=True,
                        metadata=entry.as_metadata(),
                    )
                return self._forward_suffix(
                    model,
                    suffix,
                    like=input_ids,
                    session_id=session_id,
                    stage_name=stage_name,
                    previous=entry,
                    reused_tokens=reused_tokens,
                    return_aux=return_aux,
                )
            reset_reason = "prefix_mismatch"

        if entry is None:
            reset_reason = "cache_miss"
        elif entry.history_truncated:
            reset_reason = "history_truncated"

        if reset_reason:
            self.reset(session_id)
        output = model(
            input_ids,
            memory_state=None,
            return_state=True,
            return_aux=return_aux,
        )
        next_entry = self.put(
            session_id,
            output["state"],
            appended_tokens=tokens,
            stage_name=stage_name,
            aux=output.get("aux") if isinstance(output, Mapping) else None,
        )
        return SBStateCacheForwardResult(
            session_id=session_id,
            output=output,
            state=next_entry.state,
            computed_tokens=len(tokens),
            reused_tokens=0,
            cache_hit=False,
            reset_reason=reset_reason,
            metadata=next_entry.as_metadata(),
        )

    def stats(self) -> Dict[str, object]:
        return {
            "sessions": len(self._entries),
            "hits": int(self._hits),
            "misses": int(self._misses),
            "resets": int(self._resets),
            "entries": [entry.as_metadata() for entry in self._entries.values()],
        }

    def _forward_suffix(
        self,
        model: Any,
        suffix: Sequence[int],
        *,
        like: Tensor,
        session_id: str,
        stage_name: str,
        previous: SBCachedState,
        reused_tokens: int,
        return_aux: bool,
    ) -> SBStateCacheForwardResult:
        suffix_ids = _tokens_to_tensor(suffix, like=like)
        output = model(
            suffix_ids,
            memory_state=previous.state,
            return_state=True,
            return_aux=return_aux,
        )
        next_entry = self.put(
            session_id,
            output["state"],
            appended_tokens=suffix,
            stage_name=stage_name,
            aux=output.get("aux") if isinstance(output, Mapping) else None,
        )
        return SBStateCacheForwardResult(
            session_id=session_id,
            output=output,
            state=next_entry.state,
            computed_tokens=len(suffix),
            reused_tokens=reused_tokens,
            cache_hit=True,
            metadata=next_entry.as_metadata(),
        )

    def _evict_lru(self) -> None:
        while len(self._entries) > self.config.max_sessions:
            self._entries.popitem(last=False)

    def _expire_idle(self) -> None:
        if self.config.max_idle_seconds <= 0:
            return
        cutoff = _now() - self.config.max_idle_seconds
        expired = [session_id for session_id, entry in self._entries.items() if entry.updated_at < cutoff]
        for session_id in expired:
            del self._entries[session_id]
