from __future__ import annotations

import json

import torch

from sb.core_lm_data import decode_tokens, sample_copy_batch
from sb.core_lm_torch import (
    SBCoreMiniLM,
    SBCoreMiniTorchConfig,
    SBRuntimeGates,
    next_token_loss,
    runtime_device_report,
    staged_runtime_gates,
)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)
    print(json.dumps({"runtime": runtime_device_report(device)}, ensure_ascii=False))

    config = SBCoreMiniTorchConfig(
        vocab_size=16,
        d_model=64,
        state_dim=96,
        num_layers=2,
        semantic_memory_slots=24,
        working_memory_slots=8,
        router_top_k=4,
        dropout=0.0,
        max_seq_len=64,
    )
    model = SBCoreMiniLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-3, weight_decay=0.01)
    warmup_steps = 24

    losses = []
    for step in range(160):
        stage_name, stage_gates = staged_runtime_gates(step_index=step, total_steps=160)
        model.set_runtime_gates(stage_gates)
        lr_scale = min(1.0, float(step + 1) / float(warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = 7e-3 * lr_scale
        sequence = sample_copy_batch(batch_size=48, segment_len=4, vocab_size=config.vocab_size, device=device)
        inputs = sequence[:, :-1]
        targets = sequence[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        forward = model(inputs)
        logits = forward["logits"]
        loss = next_token_loss(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        if step in {0, 19, 39, 79, 119, 159}:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "stage": stage_name,
                        "runtime_gates": model.get_runtime_gates(),
                        "loss": round(losses[-1], 4),
                        "aux": forward["aux"],
                    },
                    ensure_ascii=False,
                )
            )

    eval_seq = sample_copy_batch(batch_size=1, segment_len=4, vocab_size=config.vocab_size, device=device)
    eval_inputs = eval_seq[:, :-1]
    eval_targets = eval_seq[:, 1:]
    with torch.no_grad():
        model.set_runtime_gates(SBRuntimeGates())
        eval_forward = model(eval_inputs)
        predictions = eval_forward["logits"].argmax(dim=-1)
        eval_loss = next_token_loss(eval_forward["logits"], eval_targets)
        eval_accuracy = float((predictions == eval_targets).float().mean().cpu())

    print("=== Eval Copy Sequence ===")
    print("input :", decode_tokens(eval_inputs[0].cpu()))
    print("target:", decode_tokens(eval_targets[0].cpu()))
    print("pred  :", decode_tokens(predictions[0].cpu()))
    print("eval_loss:", round(float(eval_loss.cpu()), 4))
    print("eval_token_acc:", round(eval_accuracy, 4))
    print("initial_loss:", round(losses[0], 4))
    print("final_loss:", round(losses[-1], 4))


if __name__ == "__main__":
    main()
