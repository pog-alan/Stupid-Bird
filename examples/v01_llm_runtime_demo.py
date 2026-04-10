from __future__ import annotations

import json

from sb import DialogStore, SBV01Engine


def main() -> None:
    engine = SBV01Engine.from_default_config()
    dialog = DialogStore()
    session_id = "demo-session"

    dialog.update(session_id, "院子角落堆着一些袋子。", [])
    dialog.update(session_id, "旁边还有一个翻倒的垃圾桶。", [])
    current_text = "地面有些散乱，这种情况更像什么场景？"
    context_text = dialog.build_context_text(session_id, current_text, max_turns=3)

    analysis = engine.analyze(context_text)
    history_summary = dialog.build_history_summary(session_id, max_turns=3)
    dialog_state = {
        "session_id": session_id,
        "turns": len(dialog.recent_turns(session_id, max_turns=10)),
        "history_summary": history_summary,
    }
    packet = engine.build_llm_payload(
        context_text,
        analysis,
        dialog_state=dialog_state,
        history_summary=history_summary,
    )

    print("=== LLM Runtime Packet ===")
    print(json.dumps(packet, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
