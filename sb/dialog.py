from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class DialogTurn:
    text: str
    timestamp: str


@dataclass
class DialogState:
    session_id: str
    turns: List[DialogTurn] = field(default_factory=list)
    last_questions: List[str] = field(default_factory=list)
    updated_at: str = ""

    def add_turn(self, text: str) -> None:
        self.turns.append(DialogTurn(text=text, timestamp=_now()))
        self.updated_at = _now()

    def set_questions(self, questions: List[str]) -> None:
        self.last_questions = questions
        self.updated_at = _now()


class DialogStore:
    def __init__(self, max_sessions: int = 200) -> None:
        self._states: Dict[str, DialogState] = {}
        self._max_sessions = max_sessions

    def get(self, session_id: str) -> DialogState:
        if session_id not in self._states:
            if len(self._states) >= self._max_sessions:
                self._states.pop(next(iter(self._states)))
            self._states[session_id] = DialogState(session_id=session_id, updated_at=_now())
        return self._states[session_id]

    def update(self, session_id: str, text: str, questions: List[str]) -> DialogState:
        state = self.get(session_id)
        state.add_turn(text)
        state.set_questions(questions)
        return state

    def build_context_text(self, session_id: str, new_text: str, max_turns: int = 2) -> str:
        state = self.get(session_id)
        if not state.turns:
            return new_text
        recent = [turn.text for turn in state.turns[-max_turns:]]
        recent.append(new_text)
        return "。".join(item.strip("。") for item in recent if item)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
