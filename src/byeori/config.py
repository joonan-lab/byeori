from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    state_dir: Path
    openalex_api_key: str | None
    aws_region: str
    aws_bucket: str | None
    aws_table: str | None
    aws_ingest_function: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(os.environ.get("AWS_KIRO_WIKI_ROOT", PROJECT_ROOT)).resolve()
        return cls(
            root=root,
            data_dir=root / "data",
            state_dir=root / "state",
            openalex_api_key=os.environ.get("OPENALEX_API_KEY") or None,
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            aws_bucket=os.environ.get("AWS_KIRO_WIKI_BUCKET") or None,
            aws_table=os.environ.get("AWS_KIRO_WIKI_TABLE") or None,
            aws_ingest_function=os.environ.get("AWS_KIRO_WIKI_INGEST_FUNCTION") or None,
        )

    def ensure_directories(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
