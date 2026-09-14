"""Deterministic profile and report serialization."""

import json
import os
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel


def write_jsonl(records: Iterable[BaseModel], path: Path) -> None:
    """Atomically replace a JSONL artifact with deterministic model serialization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for record in records:
                stream.write(record.model_dump_json(exclude_none=False) + "\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_json(record: BaseModel, path: Path) -> None:
    """Atomically replace one formatted JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(
                record.model_dump(mode="json"),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
