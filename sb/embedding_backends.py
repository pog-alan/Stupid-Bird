from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence
from urllib import request


class EmbeddingEncoder:
    backend_name = "base"

    def encode(self, text: str) -> List[float]:
        return self.encode_query(text)

    def encode_many(self, texts: Sequence[str]) -> List[List[float]]:
        return self.encode_documents(texts)

    def encode_query(self, text: str) -> List[float]:
        return self.encode_documents([text])[0]

    def encode_documents(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


class HashedVectorEncoder(EmbeddingEncoder):
    backend_name = "hashed"

    def __init__(self, dimensions: int = 192) -> None:
        self.dimensions = dimensions

    def encode_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._encode_single(text) for text in texts]

    def _encode_single(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = _tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            index = hash(token) % self.dimensions
            vector[index] += 1.0
        return _normalize(vector)


@dataclass
class HTTPEmbeddingEncoder(EmbeddingEncoder):
    endpoint: str
    timeout_seconds: int = 15
    model: str | None = None
    api_key: str | None = None
    input_key: str = "texts"
    backend_name: str = "http"

    def encode_documents(self, texts: Sequence[str]) -> List[List[float]]:
        payload = {self.input_key: list(texts)}
        if self.model:
            payload["model"] = self.model

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
        vectors = _extract_vectors(raw)
        return [_normalize([float(value) for value in vector]) for vector in vectors]


class SentenceTransformerEncoder(EmbeddingEncoder):
    backend_name = "sentence_transformers"

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: str | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        query_instruction: str = "",
        passage_instruction: str = "",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.query_instruction = query_instruction.strip()
        self.passage_instruction = passage_instruction.strip()
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
        )

    def encode_query(self, text: str) -> List[float]:
        vectors = self.model.encode(
            [self._prepare_query(text)],
            batch_size=1,
            normalize_embeddings=self.normalize_embeddings,
        )
        return [float(value) for value in vectors[0]]

    def encode_documents(self, texts: Sequence[str]) -> List[List[float]]:
        prepared = [self._prepare_document(text) for text in texts]
        vectors = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
        )
        return [[float(value) for value in vector] for vector in vectors]

    def _prepare_query(self, text: str) -> str:
        if not self.query_instruction:
            return text
        return f"{self.query_instruction}{text}"

    def _prepare_document(self, text: str) -> str:
        if not self.passage_instruction:
            return text
        return f"{self.passage_instruction}{text}"


EMBEDDING_PRESETS: Dict[str, Dict[str, Any]] = {
    "bge-small-zh-v1.5": {
        "type": "sentence_transformers",
        "model_name": "BAAI/bge-small-zh-v1.5",
        "query_instruction": "为这个句子生成表示以用于检索相关文章：",
        "passage_instruction": "",
        "batch_size": 32,
        "normalize_embeddings": True,
        "fallback_to_hash": True,
    },
    "bge-base-zh-v1.5": {
        "type": "sentence_transformers",
        "model_name": "BAAI/bge-base-zh-v1.5",
        "query_instruction": "为这个句子生成表示以用于检索相关文章：",
        "passage_instruction": "",
        "batch_size": 16,
        "normalize_embeddings": True,
        "fallback_to_hash": True,
    },
}


def create_embedding_encoder(config: Mapping[str, object] | None) -> EmbeddingEncoder:
    if not config:
        return HashedVectorEncoder()

    resolved = resolve_embedding_backend_config(config)
    backend_type = str(resolved.get("type", "hashed")).lower()
    fallback_to_hash = bool(resolved.get("fallback_to_hash", True))

    if backend_type == "hashed":
        return HashedVectorEncoder(dimensions=int(resolved.get("dimensions", 192)))

    if backend_type == "http":
        try:
            return HTTPEmbeddingEncoder(
                endpoint=str(resolved["endpoint"]),
                timeout_seconds=int(resolved.get("timeout_seconds", 15)),
                model=str(resolved["model"]) if resolved.get("model") else None,
                api_key=str(resolved["api_key"]) if resolved.get("api_key") else None,
                input_key=str(resolved.get("input_key", "texts")),
            )
        except Exception:
            if fallback_to_hash:
                return HashedVectorEncoder(dimensions=int(resolved.get("dimensions", 192)))
            raise

    if backend_type == "sentence_transformers":
        try:
            return SentenceTransformerEncoder(
                model_name=str(resolved.get("model_name", "BAAI/bge-small-zh-v1.5")),
                batch_size=int(resolved.get("batch_size", 32)),
                normalize_embeddings=bool(resolved.get("normalize_embeddings", True)),
                device=str(resolved["device"]) if resolved.get("device") else None,
                cache_folder=str(resolved["cache_folder"]) if resolved.get("cache_folder") else None,
                trust_remote_code=bool(resolved.get("trust_remote_code", False)),
                query_instruction=str(resolved.get("query_instruction", "")),
                passage_instruction=str(resolved.get("passage_instruction", "")),
            )
        except Exception:
            if fallback_to_hash:
                return HashedVectorEncoder(dimensions=int(resolved.get("dimensions", 192)))
            raise

    return HashedVectorEncoder(dimensions=int(resolved.get("dimensions", 192)))


def resolve_embedding_backend_config(config: Mapping[str, object]) -> Dict[str, Any]:
    preset_name = str(config.get("preset", "")).strip()
    if preset_name and preset_name in EMBEDDING_PRESETS:
        resolved = dict(EMBEDDING_PRESETS[preset_name])
        resolved.update(config)
        return resolved
    return dict(config)


def _extract_vectors(payload: Mapping[str, object]) -> List[List[float]]:
    if isinstance(payload.get("vectors"), list):
        return payload["vectors"]  # type: ignore[return-value]
    if isinstance(payload.get("embeddings"), list):
        return payload["embeddings"]  # type: ignore[return-value]
    data = payload.get("data")
    if isinstance(data, list):
        vectors: List[List[float]] = []
        for item in data:
            if isinstance(item, Mapping) and isinstance(item.get("embedding"), list):
                vectors.append(item["embedding"])  # type: ignore[arg-type]
        if vectors:
            return vectors
    raise ValueError("Embedding response does not contain vectors")


def _normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]


def _tokenize(text: str) -> List[str]:
    normalized = text.strip().lower()
    if not normalized:
        return []
    tokens: List[str] = []
    for char in normalized:
        if not char.isspace():
            tokens.append(char)
    for index in range(len(normalized) - 1):
        pair = normalized[index : index + 2].strip()
        if len(pair) == 2 and not pair.isspace():
            tokens.append(pair)
    return tokens
