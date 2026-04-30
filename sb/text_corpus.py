from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import pyarrow.parquet as pq
import torch

_TOKENIZERS_AVAILABLE = True
try:
    from tokenizers import Tokenizer
    from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers
except ModuleNotFoundError:
    _TOKENIZERS_AVAILABLE = False


@dataclass(frozen=True)
class StageCorpusPaths:
    foundation_path: str
    structured_path: str
    long_context_path: str
    tokenizer_path: str
    foundation_val_path: str = ""
    structured_val_path: str = ""
    long_context_val_path: str = ""
    manifest_path: str = ""


@dataclass(frozen=True)
class TextCorpusPreparationConfig:
    profile_name: str = "default"
    wikipedia_limit: int = 3000
    clue_limit_per_subset: int = 4000
    longbench_limit_per_task: int = 200
    max_vocab_size: int = 4096
    min_freq: int = 2
    tokenizer_kind: str = "char"
    validation_ratio: float = 0.05
    seed: int = 23


@dataclass(frozen=True)
class CharTokenizer:
    pad_token: str
    bos_token: str
    eos_token: str
    unk_token: str
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(char, self.unk_id) for char in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: Sequence[int], *, skip_special: bool = True) -> str:
        special = {self.pad_token, self.bos_token, self.eos_token}
        chars: List[str] = []
        for token_id in token_ids:
            if 0 <= int(token_id) < len(self.itos):
                token = self.itos[int(token_id)]
            else:
                token = self.unk_token
            if skip_special and token in special:
                continue
            chars.append(token)
        return "".join(chars)

    def to_dict(self) -> Dict[str, object]:
        return {
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "itos": self.itos,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "CharTokenizer":
        itos = list(payload["itos"])
        stoi = {token: index for index, token in enumerate(itos)}
        return cls(
            pad_token=str(payload["pad_token"]),
            bos_token=str(payload["bos_token"]),
            eos_token=str(payload["eos_token"]),
            unk_token=str(payload["unk_token"]),
            stoi=stoi,
            itos=itos,
        )


class SubwordTokenizer:
    def __init__(
        self,
        tokenizer: "Tokenizer",
        *,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.stoi = {token: int(token_id) for token, token_id in tokenizer.get_vocab().items()}

    @property
    def pad_id(self) -> int:
        return int(self.stoi[self.pad_token])

    @property
    def bos_id(self) -> int:
        return int(self.stoi[self.bos_token])

    @property
    def eos_id(self) -> int:
        return int(self.stoi[self.eos_token])

    @property
    def unk_id(self) -> int:
        return int(self.stoi[self.unk_token])

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = list(self.tokenizer.encode(text).ids)
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: Sequence[int], *, skip_special: bool = True) -> str:
        ids = [int(token_id) for token_id in token_ids]
        if skip_special:
            special_ids = {self.pad_id, self.bos_id, self.eos_id}
            ids = [token_id for token_id in ids if token_id not in special_ids]
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)


TextTokenizer = CharTokenizer | SubwordTokenizer


@dataclass(frozen=True)
class TextBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    focus_mask: torch.Tensor | None = None


def _load_manifest(manifest_path: str | Path) -> Dict[str, object]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            rows.append(json.loads(line))
    return rows


def _read_parquet_rows(path: Path, limit: int | None = None) -> List[Dict[str, object]]:
    table = pq.read_table(path)
    if limit is not None:
        table = table.slice(0, limit)
    return table.to_pylist()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def _format_clue_afqmc(row: Dict[str, object]) -> str:
    return (
        "task: sentence_pair_matching\n"
        f"sentence_1: {row['sentence1']}\n"
        f"sentence_2: {row['sentence2']}\n"
        f"label: {row['label']}"
    )


def _format_clue_tnews(row: Dict[str, object]) -> str:
    return f"task: news_classification\ntext: {row['sentence']}\nlabel: {row['label']}"


def _format_clue_cmnli(row: Dict[str, object]) -> str:
    return (
        "task: natural_language_inference\n"
        f"premise: {row['sentence1']}\n"
        f"hypothesis: {row['sentence2']}\n"
        f"label: {row['label']}"
    )


def _format_longbench(task_name: str, row: Dict[str, object]) -> str:
    answers = row.get("answers") or []
    first_answer = answers[0] if answers else ""
    return (
        f"task: {task_name}\n"
        f"question: {row.get('input', '')}\n"
        f"context: {row.get('context', '')}\n"
        f"reference_answer: {first_answer}"
    )


def _format_wikipedia(row: Dict[str, object]) -> str:
    return f"title: {row.get('title', '')}\ntext: {row.get('text', '')}"


def _format_chinese_c4(row: Dict[str, object]) -> str:
    return str(row.get("text", ""))


def load_longbench_rows(
    manifest_path: str | Path,
    *,
    longbench_limit_per_task: int = 200,
) -> List[Dict[str, object]]:
    manifest = _load_manifest(manifest_path)
    datasets = {entry["name"]: entry for entry in manifest["datasets"]}
    longbench_entry = datasets["longbench"]
    rows: List[Dict[str, object]] = []
    for task in longbench_entry["tasks"]:
        task_name = task["task"]
        task_path = Path(task["path"])
        for row in _read_jsonl(task_path, limit=longbench_limit_per_task):
            answers = row.get("answers") or []
            if not answers:
                continue
            rows.append(
                {
                    "task": task_name,
                    "input": str(row.get("input", "")),
                    "context": str(row.get("context", "")),
                    "answer": str(answers[0]),
                }
            )
    return rows


def build_stage_texts(
    manifest_path: str | Path,
    *,
    wikipedia_limit: int = 3000,
    clue_limit_per_subset: int = 4000,
    longbench_limit_per_task: int = 200,
) -> Dict[str, List[str]]:
    manifest = _load_manifest(manifest_path)
    datasets = {entry["name"]: entry for entry in manifest["datasets"]}

    foundation_texts: List[str] = []
    structured_texts: List[str] = []
    long_context_texts: List[str] = []

    if "wikipedia_zh_sample" in datasets:
        wikipedia_path = Path(datasets["wikipedia_zh_sample"]["path"])
        for row in _read_jsonl(wikipedia_path, limit=wikipedia_limit):
            text = _normalize_text(_format_wikipedia(row))
            if len(text) >= 40:
                foundation_texts.append(text)

    if "chinese_c4_sample" in datasets:
        chinese_c4_path = Path(datasets["chinese_c4_sample"]["path"])
        for row in _read_jsonl(chinese_c4_path, limit=wikipedia_limit):
            text = _normalize_text(_format_chinese_c4(row))
            if len(text) >= 40:
                foundation_texts.append(text)

    clue_entry = datasets["clue"]
    clue_formatters = {
        "afqmc": _format_clue_afqmc,
        "tnews": _format_clue_tnews,
        "cmnli": _format_clue_cmnli,
    }
    for subset in clue_entry["subsets"]:
        subset_name = subset["name"]
        train_path = Path(subset["splits"]["train"]["path"])
        formatter = clue_formatters[subset_name]
        for row in _read_parquet_rows(train_path, limit=clue_limit_per_subset):
            text = _normalize_text(formatter(row))
            if len(text) >= 12:
                structured_texts.append(text)

    longbench_entry = datasets["longbench"]
    for task in longbench_entry["tasks"]:
        task_name = task["task"]
        task_path = Path(task["path"])
        for row in _read_jsonl(task_path, limit=longbench_limit_per_task):
            text = _normalize_text(_format_longbench(task_name, row))
            if len(text) >= 80:
                long_context_texts.append(text)

    return {
        "foundation": foundation_texts,
        "structured": structured_texts,
        "long_context": long_context_texts,
    }


def build_char_tokenizer(
    texts: Sequence[str],
    *,
    max_vocab_size: int = 4096,
    min_freq: int = 2,
) -> CharTokenizer:
    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(text)
    vocab = specials[:]
    for char, freq in counter.most_common():
        if freq < min_freq:
            continue
        if char in vocab:
            continue
        vocab.append(char)
        if len(vocab) >= max_vocab_size:
            break
    stoi = {token: index for index, token in enumerate(vocab)}
    return CharTokenizer(
        pad_token=specials[0],
        bos_token=specials[1],
        eos_token=specials[2],
        unk_token=specials[3],
        stoi=stoi,
        itos=vocab,
    )


def build_subword_tokenizer(
    texts: Sequence[str],
    *,
    tokenizer_json_path: str | Path,
    vocab_size: int = 4096,
    min_freq: int = 2,
) -> SubwordTokenizer:
    if not _TOKENIZERS_AVAILABLE:
        raise RuntimeError("tokenizers is not installed; cannot build a subword tokenizer.")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer_path = Path(tokenizer_json_path)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return SubwordTokenizer(tokenizer)


def _write_tokenizer_manifest(*, path: Path, tokenizer_kind: str, payload: Dict[str, object]) -> None:
    path.write_text(
        json.dumps({"kind": tokenizer_kind, "payload": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_stage_rows(path: Path, texts: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def _split_stage_texts(
    texts: Sequence[str],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[List[str], List[str]]:
    rows = list(texts)
    if not rows:
        return [], []
    if validation_ratio <= 0.0:
        return rows, []
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(rows) * validation_ratio)))
    val_indices = set(indices[:val_count])
    train_rows = [rows[index] for index in range(len(rows)) if index not in val_indices]
    val_rows = [rows[index] for index in range(len(rows)) if index in val_indices]
    if not train_rows:
        train_rows, val_rows = val_rows, []
    return train_rows, val_rows


def _length_stats(texts: Sequence[str], tokenizer: TextTokenizer) -> Dict[str, float | int]:
    if not texts:
        return {
            "rows": 0,
            "char_mean": 0.0,
            "char_max": 0,
            "token_mean": 0.0,
            "token_max": 0,
        }
    char_lengths = [len(text) for text in texts]
    token_lengths = [len(tokenizer.encode(text)) for text in texts]
    return {
        "rows": len(texts),
        "char_mean": round(sum(char_lengths) / len(char_lengths), 2),
        "char_max": max(char_lengths),
        "token_mean": round(sum(token_lengths) / len(token_lengths), 2),
        "token_max": max(token_lengths),
    }


def prepare_local_text_corpus(
    manifest_path: str | Path,
    *,
    output_dir: str | Path = "data/processed/text_corpus",
    wikipedia_limit: int = 3000,
    clue_limit_per_subset: int = 4000,
    longbench_limit_per_task: int = 200,
    max_vocab_size: int = 4096,
    min_freq: int = 2,
    tokenizer_kind: str = "char",
    validation_ratio: float = 0.05,
    seed: int = 23,
    profile_name: str = "default",
) -> StageCorpusPaths:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    stage_texts = build_stage_texts(
        manifest_path,
        wikipedia_limit=wikipedia_limit,
        clue_limit_per_subset=clue_limit_per_subset,
        longbench_limit_per_task=longbench_limit_per_task,
    )

    combined_texts = stage_texts["foundation"] + stage_texts["structured"] + stage_texts["long_context"]
    tokenizer_manifest_path = output_root / "tokenizer_manifest.json"

    if tokenizer_kind == "char":
        tokenizer = build_char_tokenizer(
            combined_texts,
            max_vocab_size=max_vocab_size,
            min_freq=min_freq,
        )
        _write_tokenizer_manifest(
            path=tokenizer_manifest_path,
            tokenizer_kind="char",
            payload=tokenizer.to_dict(),
        )
    elif tokenizer_kind == "subword":
        tokenizer_json_path = output_root / "tokenizer_subword.json"
        tokenizer = build_subword_tokenizer(
            combined_texts,
            tokenizer_json_path=tokenizer_json_path,
            vocab_size=max_vocab_size,
            min_freq=min_freq,
        )
        _write_tokenizer_manifest(
            path=tokenizer_manifest_path,
            tokenizer_kind="subword",
            payload={"tokenizer_json_path": str(tokenizer_json_path.resolve())},
        )
    else:
        raise ValueError(f"unsupported tokenizer_kind: {tokenizer_kind}")

    stage_splits: Dict[str, tuple[List[str], List[str]]] = {}
    stage_seeds = {
        "foundation": seed,
        "structured": seed + 101,
        "long_context": seed + 202,
    }
    for stage_name, texts in stage_texts.items():
        stage_splits[stage_name] = _split_stage_texts(
            texts,
            validation_ratio=validation_ratio,
            seed=stage_seeds.get(stage_name, seed),
        )

    foundation_path = output_root / "foundation.jsonl"
    foundation_val_path = output_root / "foundation_val.jsonl"
    structured_path = output_root / "structured.jsonl"
    structured_val_path = output_root / "structured_val.jsonl"
    long_context_path = output_root / "long_context.jsonl"
    long_context_val_path = output_root / "long_context_val.jsonl"

    _write_stage_rows(foundation_path, stage_splits["foundation"][0])
    _write_stage_rows(foundation_val_path, stage_splits["foundation"][1])
    _write_stage_rows(structured_path, stage_splits["structured"][0])
    _write_stage_rows(structured_val_path, stage_splits["structured"][1])
    _write_stage_rows(long_context_path, stage_splits["long_context"][0])
    _write_stage_rows(long_context_val_path, stage_splits["long_context"][1])

    corpus_manifest_path = output_root / "corpus_manifest.json"
    corpus_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": asdict(
            TextCorpusPreparationConfig(
                profile_name=profile_name,
                wikipedia_limit=wikipedia_limit,
                clue_limit_per_subset=clue_limit_per_subset,
                longbench_limit_per_task=longbench_limit_per_task,
                max_vocab_size=max_vocab_size,
                min_freq=min_freq,
                tokenizer_kind=tokenizer_kind,
                validation_ratio=validation_ratio,
                seed=seed,
            )
        ),
        "paths": {
            "foundation_path": str(foundation_path.resolve()),
            "foundation_val_path": str(foundation_val_path.resolve()),
            "structured_path": str(structured_path.resolve()),
            "structured_val_path": str(structured_val_path.resolve()),
            "long_context_path": str(long_context_path.resolve()),
            "long_context_val_path": str(long_context_val_path.resolve()),
            "tokenizer_path": str(tokenizer_manifest_path.resolve()),
        },
        "stats": {
            "foundation": {
                "train": _length_stats(stage_splits["foundation"][0], tokenizer),
                "validation": _length_stats(stage_splits["foundation"][1], tokenizer),
            },
            "structured": {
                "train": _length_stats(stage_splits["structured"][0], tokenizer),
                "validation": _length_stats(stage_splits["structured"][1], tokenizer),
            },
            "long_context": {
                "train": _length_stats(stage_splits["long_context"][0], tokenizer),
                "validation": _length_stats(stage_splits["long_context"][1], tokenizer),
            },
        },
    }
    corpus_manifest_path.write_text(json.dumps(corpus_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return StageCorpusPaths(
        foundation_path=str(foundation_path.resolve()),
        structured_path=str(structured_path.resolve()),
        long_context_path=str(long_context_path.resolve()),
        tokenizer_path=str(tokenizer_manifest_path.resolve()),
        foundation_val_path=str(foundation_val_path.resolve()),
        structured_val_path=str(structured_val_path.resolve()),
        long_context_val_path=str(long_context_val_path.resolve()),
        manifest_path=str(corpus_manifest_path.resolve()),
    )


def load_stage_corpus(path: str | Path) -> List[str]:
    rows = _read_jsonl(Path(path))
    return [str(row["text"]) for row in rows if str(row.get("text", "")).strip()]


def load_char_tokenizer(path: str | Path) -> CharTokenizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "kind" in payload:
        payload = payload["payload"]
    return CharTokenizer.from_dict(payload)


def load_text_tokenizer(path: str | Path) -> TextTokenizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "kind" not in payload:
        return CharTokenizer.from_dict(payload)

    kind = str(payload["kind"])
    body = dict(payload["payload"])
    if kind == "char":
        return CharTokenizer.from_dict(body)
    if kind == "subword":
        if not _TOKENIZERS_AVAILABLE:
            raise RuntimeError("tokenizers is not installed; cannot load a subword tokenizer.")
        tokenizer = Tokenizer.from_file(str(body["tokenizer_json_path"]))
        return SubwordTokenizer(tokenizer)
    raise ValueError(f"unsupported tokenizer kind: {kind}")


def load_prepared_corpus_paths(path: str | Path) -> StageCorpusPaths:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    paths = dict(payload.get("paths", {}))
    return StageCorpusPaths(
        foundation_path=str(paths["foundation_path"]),
        structured_path=str(paths["structured_path"]),
        long_context_path=str(paths["long_context_path"]),
        tokenizer_path=str(paths["tokenizer_path"]),
        foundation_val_path=str(paths.get("foundation_val_path", "")),
        structured_val_path=str(paths.get("structured_val_path", "")),
        long_context_val_path=str(paths.get("long_context_val_path", "")),
        manifest_path=str(Path(path).resolve()),
    )


def sample_text_batch(
    texts: Sequence[str],
    tokenizer: TextTokenizer,
    *,
    batch_size: int,
    seq_len: int,
    device: str | torch.device,
    rng: random.Random | None = None,
) -> TextBatch:
    if not texts:
        raise ValueError("texts must not be empty.")

    random_gen = rng or random
    encoded_sequences = [tokenizer.encode(text) for text in texts]
    window_len = seq_len + 1
    batch_tokens: List[List[int]] = []

    for _ in range(batch_size):
        sequence = encoded_sequences[random_gen.randrange(len(encoded_sequences))]
        if len(sequence) >= window_len:
            start = random_gen.randrange(0, len(sequence) - window_len + 1)
            window = sequence[start:start + window_len]
        else:
            padding = [tokenizer.pad_id] * (window_len - len(sequence))
            window = sequence + padding
        batch_tokens.append(window)

    batch_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device)
    return TextBatch(
        input_ids=batch_tensor[:, :-1],
        target_ids=batch_tensor[:, 1:],
        focus_mask=torch.ones_like(batch_tensor[:, 1:], dtype=torch.bool),
    )


def sample_longbench_answer_batch(
    rows: Sequence[Dict[str, object]],
    tokenizer: TextTokenizer,
    *,
    batch_size: int,
    seq_len: int,
    device: str | torch.device,
    rng: random.Random | None = None,
) -> TextBatch:
    if not rows:
        raise ValueError("longbench rows must not be empty.")

    random_gen = rng or random
    window_len = seq_len + 1
    batch_inputs: List[List[int]] = []
    batch_targets: List[List[int]] = []
    batch_focus_masks: List[List[bool]] = []

    for _ in range(batch_size):
        row = rows[random_gen.randrange(len(rows))]
        prompt = _format_longbench(
            str(row["task"]),
            {
                "input": row["input"],
                "context": row["context"],
                "answers": [],
            },
        ) + "\nanswer:"
        answer = str(row["answer"])

        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        answer_ids = tokenizer.encode(answer, add_bos=False, add_eos=True)
        full_ids = prompt_ids + answer_ids
        start_index = max(0, len(full_ids) - window_len)
        kept_ids = full_ids[start_index:]

        input_ids = kept_ids[:-1]
        target_ids = kept_ids[1:]
        target_positions = list(range(start_index + 1, start_index + len(kept_ids)))
        focus_mask = [position >= len(prompt_ids) for position in target_positions]

        pad_len = seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [tokenizer.pad_id] * pad_len
            target_ids = target_ids + [tokenizer.pad_id] * pad_len
            focus_mask = focus_mask + [False] * pad_len

        batch_inputs.append(input_ids[:seq_len])
        batch_targets.append(target_ids[:seq_len])
        batch_focus_masks.append(focus_mask[:seq_len])

    return TextBatch(
        input_ids=torch.tensor(batch_inputs, dtype=torch.long, device=device),
        target_ids=torch.tensor(batch_targets, dtype=torch.long, device=device),
        focus_mask=torch.tensor(batch_focus_masks, dtype=torch.bool, device=device),
    )


def sample_stage_batch(
    stage_corpora: Dict[str, Sequence[str]],
    tokenizer: TextTokenizer,
    *,
    stage_name: str,
    batch_size: int,
    seq_len: int,
    device: str | torch.device,
    rng: random.Random | None = None,
) -> TextBatch:
    return sample_text_batch(
        stage_corpora[stage_name],
        tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        rng=rng,
    )


def summarize_stage_corpus(paths: StageCorpusPaths | Dict[str, str]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    path_items = asdict(paths).items() if isinstance(paths, StageCorpusPaths) else dict(paths).items()
    for stage_name, path_str in path_items:
        if not path_str:
            continue
        path = Path(path_str)
        if path.suffix == ".jsonl":
            rows = _read_jsonl(path)
            summary[stage_name] = {
                "path": str(path.resolve()),
                "rows": len(rows),
            }
        elif path.name == "corpus_manifest.json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary[stage_name] = {
                "path": str(path.resolve()),
                "profile": payload.get("profile", {}),
                "stats": payload.get("stats", {}),
            }
        else:
            tokenizer = load_text_tokenizer(path)
            kind = "subword" if isinstance(tokenizer, SubwordTokenizer) else "char"
            summary[stage_name] = {
                "path": str(path.resolve()),
                "vocab_size": tokenizer.vocab_size,
                "kind": kind,
            }
    return summary
