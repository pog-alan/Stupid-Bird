from __future__ import annotations

import json

from sb.core_lm import SBCoreConfig, SBCoreModelSpec
from sb.eval_long_context import LongContextEvaluationSuite
from sb.train_lm import SBCoreTrainingPlan


def main() -> None:
    spec = SBCoreModelSpec(
        SBCoreConfig(
            vocab_size=32000,
            d_model=512,
            num_layers=8,
            state_dim=1024,
            memory_slots=2048,
            router_top_k=8,
        )
    )
    plan = SBCoreTrainingPlan()
    suite = LongContextEvaluationSuite()

    payload = {
        "recurrent_update": spec.recurrent_update_equation(),
        "memory_flow": spec.memory_flow_equation(),
        "output": spec.output_equation(),
        "stages": [
            {
                "stage_id": stage.stage_id,
                "goal": stage.goal,
                "tasks": stage.tasks,
                "acceptance": stage.acceptance,
            }
            for stage in plan.build_stages()
        ],
        "metrics": plan.tracked_metrics(),
        "long_context_tasks": [
            {
                "name": task.name,
                "description": task.description,
                "primary_metric": task.primary_metric,
            }
            for task in suite.default_tasks()
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
