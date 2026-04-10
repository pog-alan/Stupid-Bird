from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass(frozen=True)
class ToyTaskVocab:
    pad_token: int = 0
    bos_token: int = 1
    sep_token: int = 2
    query_token: int = 3
    eos_token: int = 4
    value_start: int = 5


@dataclass(frozen=True)
class ToySequenceBatch:
    tokens: torch.Tensor
    focus_mask: torch.Tensor
    task_name: str


def _validate_vocab(vocab_size: int, vocab: ToyTaskVocab) -> None:
    if vocab_size <= vocab.value_start:
        raise ValueError("vocab_size must exceed reserved tokens.")


def _sample_values(
    batch_size: int,
    length: int,
    low: int,
    high: int,
    device: str | torch.device,
    exclude: torch.Tensor | None = None,
) -> torch.Tensor:
    if length <= 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=device)

    values = torch.randint(low=low, high=high, size=(batch_size, length), device=device)
    if exclude is None or exclude.numel() == 0:
        return values

    exclude = exclude.to(device=device, dtype=torch.long)
    if exclude.dim() == 1:
        exclude = exclude.unsqueeze(0).expand(batch_size, -1)

    conflict = (values.unsqueeze(-1) == exclude.unsqueeze(1)).any(dim=-1)
    attempts = 0
    while conflict.any():
        replacement = torch.randint(low=low, high=high, size=values.shape, device=device)
        values = torch.where(conflict, replacement, values)
        conflict = (values.unsqueeze(-1) == exclude.unsqueeze(1)).any(dim=-1)
        attempts += 1
        if attempts > 8:
            break
    return values


def _focus_mask(batch_size: int, target_length: int, start: int, width: int, device: str | torch.device) -> torch.Tensor:
    mask = torch.zeros((batch_size, target_length), dtype=torch.bool, device=device)
    if width > 0:
        mask[:, start:start + width] = True
    return mask


def sample_copy_batch(
    batch_size: int,
    segment_len: int,
    vocab_size: int,
    device: str | torch.device = "cpu",
    vocab: ToyTaskVocab | None = None,
) -> torch.Tensor:
    vocab = vocab or ToyTaskVocab()
    _validate_vocab(vocab_size, vocab)

    values = _sample_values(
        batch_size=batch_size,
        length=segment_len,
        low=vocab.value_start,
        high=vocab_size,
        device=device,
    )
    bos = torch.full((batch_size, 1), vocab.bos_token, dtype=torch.long, device=device)
    sep = torch.full((batch_size, 1), vocab.sep_token, dtype=torch.long, device=device)
    eos = torch.full((batch_size, 1), vocab.eos_token, dtype=torch.long, device=device)
    return torch.cat([bos, values, sep, values, eos], dim=1)


def sample_passkey_batch(
    batch_size: int,
    prefix_len: int,
    filler_len: int,
    key_length: int,
    vocab_size: int,
    device: str | torch.device = "cpu",
    vocab: ToyTaskVocab | None = None,
    return_metadata: bool = False,
) -> torch.Tensor | ToySequenceBatch:
    vocab = vocab or ToyTaskVocab()
    _validate_vocab(vocab_size, vocab)

    key = _sample_values(
        batch_size=batch_size,
        length=key_length,
        low=vocab.value_start,
        high=vocab_size,
        device=device,
    )
    prefix = _sample_values(
        batch_size=batch_size,
        length=prefix_len,
        low=vocab.value_start,
        high=vocab_size,
        device=device,
        exclude=key,
    )
    filler = _sample_values(
        batch_size=batch_size,
        length=filler_len,
        low=vocab.value_start,
        high=vocab_size,
        device=device,
        exclude=key,
    )
    bos = torch.full((batch_size, 1), vocab.bos_token, dtype=torch.long, device=device)
    sep = torch.full((batch_size, 1), vocab.sep_token, dtype=torch.long, device=device)
    query = torch.full((batch_size, 1), vocab.query_token, dtype=torch.long, device=device)
    eos = torch.full((batch_size, 1), vocab.eos_token, dtype=torch.long, device=device)
    sequence = torch.cat([bos, prefix, sep, key, filler, query, key, eos], dim=1)

    if not return_metadata:
        return sequence

    query_index = 1 + prefix_len + 1 + key_length + filler_len
    mask = _focus_mask(
        batch_size=batch_size,
        target_length=sequence.shape[1] - 1,
        start=query_index,
        width=key_length,
        device=device,
    )
    return ToySequenceBatch(tokens=sequence, focus_mask=mask, task_name="passkey_retrieval")


def sample_needle_in_haystack_batch(
    batch_size: int,
    prefix_len: int,
    suffix_len: int,
    key_length: int,
    value_length: int,
    vocab_size: int,
    device: str | torch.device = "cpu",
    vocab: ToyTaskVocab | None = None,
    return_metadata: bool = False,
) -> torch.Tensor | ToySequenceBatch:
    vocab = vocab or ToyTaskVocab()
    _validate_vocab(vocab_size, vocab)

    key = _sample_values(
        batch_size=batch_size,
        length=key_length,
        low=vocab.value_start,
        high=vocab_size,
        device=device,
    )
    value = _sample_values(
        batch_size=batch_size,
        length=value_length,
        low=vocab.value_start,
        high=vocab_size,
        device=device,
        exclude=key,
    )
    record = torch.cat([key, value], dim=1)
    prefix = _sample_values(
        batch_size=batch_size,
        length=prefix_len,
        low=vocab.value_start,
        high=vocab_size,
        device=device,
        exclude=record,
    )
    suffix = _sample_values(
        batch_size=batch_size,
        length=suffix_len,
        low=vocab.value_start,
        high=vocab_size,
        device=device,
        exclude=record,
    )
    bos = torch.full((batch_size, 1), vocab.bos_token, dtype=torch.long, device=device)
    sep = torch.full((batch_size, 1), vocab.sep_token, dtype=torch.long, device=device)
    query = torch.full((batch_size, 1), vocab.query_token, dtype=torch.long, device=device)
    eos = torch.full((batch_size, 1), vocab.eos_token, dtype=torch.long, device=device)
    sequence = torch.cat([bos, prefix, sep, key, value, suffix, query, key, value, eos], dim=1)

    if not return_metadata:
        return sequence

    query_index = 1 + prefix_len + 1 + key_length + value_length + suffix_len
    mask = _focus_mask(
        batch_size=batch_size,
        target_length=sequence.shape[1] - 1,
        start=query_index + key_length,
        width=value_length,
        device=device,
    )
    return ToySequenceBatch(tokens=sequence, focus_mask=mask, task_name="needle_in_haystack")


def decode_tokens(tokens: torch.Tensor | List[int]) -> str:
    values = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
    return " ".join(str(item) for item in values)
