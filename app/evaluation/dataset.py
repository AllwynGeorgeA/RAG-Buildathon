"""Loads the golden evaluation dataset (app/evaluation/test_cases.json)."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class EvalCase(BaseModel):
    id: str
    category: str
    question: str
    expected_behavior: str
    required_sources: list[str] = []
    must_refuse: bool = False
    expect_relevant: bool = True
    notes: str = ""


def load_test_cases(path: str | Path | None = None) -> list[EvalCase]:
    default_path = Path(__file__).parent / "test_cases.json"
    target = Path(path) if path else default_path
    data = json.loads(target.read_text(encoding="utf-8"))
    return [EvalCase.model_validate(item) for item in data]
