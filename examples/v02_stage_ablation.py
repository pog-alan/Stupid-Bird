from __future__ import annotations

import json

import torch

from examples.v02_long_context_compare import build_sb_model, build_tasks, evaluate_model, train_model
from sb.core_lm_torch import runtime_device_report


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 40
    task = next(item for item in build_tasks(vocab_size=vocab_size) if item.name == "passkey_retrieval")

    report = {
        "runtime": runtime_device_report(device),
        "task": task.name,
        "train_steps": 16,
        "train_batch_size": 4,
        "eval_batch_size": 4,
        "modes": [],
    }

    for mode_name, staged in [("full_on", False), ("staged", True)]:
        torch.manual_seed(task.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task.seed)
        model = build_sb_model(vocab_size=vocab_size, max_seq_len=160, device=device)
        train_loss = train_model(
            model,
            task.train_batch,
            device=device,
            train_steps=16,
            batch_size=4,
            lr=6e-3,
            use_staged_gates=staged,
        )
        in_token, in_exact, _ = evaluate_model(
            model,
            task.eval_batch,
            device=device,
            batch_size=4,
            repeats=1,
        )
        long_token, long_exact, _ = evaluate_model(
            model,
            task.long_eval_batch,
            device=device,
            batch_size=4,
            repeats=1,
        )
        report["modes"].append(
            {
                "mode": mode_name,
                "train_loss": train_loss,
                "in_distribution_token_acc": in_token,
                "in_distribution_exact_match": in_exact,
                "long_context_token_acc": long_token,
                "long_context_exact_match": long_exact,
            }
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
