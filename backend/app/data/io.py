from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def write_jsonl(path: Path, rows: list[BaseModel], *, sort_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: str(getattr(row, sort_key)))
    text = "".join(
        json.dumps(row.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) + "\n"
        for row in sorted_rows
    )
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    rows: list[ModelT] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: invalid row: {exc}") from exc
    return rows
