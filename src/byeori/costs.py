"""Token prices and a local ledger for measuring what each paper costs and how long it takes.

Prices are Anthropic first-party list prices per one million tokens (cached 2026-06-24).
Amazon Bedrock bills separately and its rates were not machine-readable when this was written,
so every figure derived here is an estimate until Cost Explorer confirms it. See
docs/COST-ESTIMATE.md.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PRICES_USD_PER_MILLION: dict[str, dict[str, float]] = {
    # key fragment -> input, output, cache write, cache read
    "claude-fable-5-1": {"input": 10.0, "output": 50.0, "cache_write": 12.50, "cache_read": 0.25},
    "claude-fable-5": {"input": 10.0, "output": 50.0, "cache_write": 12.50, "cache_read": 1.00},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_write": 1.25, "cache_read": 0.10},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0, "cache_write": 2.50, "cache_read": 0.20},
    # Before "claude-opus-5": the lookup takes the first fragment the id contains, and that one is a
    # substring of this id, so the other order priced Opus 5.5 at Opus 5's rates. These are the
    # Bedrock rates for the global profile in the lab's region, read from the AWS Price List API
    # (AmazonBedrockFoundationModels, APN2_*_global_standard) on 2026-09-23.
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_write": 5.00, "cache_read": 0.20},
    "claude-opus-5": {"input": 5.0, "output": 25.0, "cache_write": 6.25, "cache_read": 0.50},
}
OPENALEX_FULLTEXT_USD = 0.02  # PDF plus GROBID XML against the free key's daily budget


def price_table(model_id: str) -> dict[str, float] | None:
    for fragment, prices in PRICES_USD_PER_MILLION.items():
        if fragment in model_id:
            return prices
    return None


def estimate_draft_usd(model_id: str, usage: dict[str, Any]) -> float | None:
    prices = price_table(model_id)
    if prices is None:
        return None
    cache_read = int(usage.get("cacheReadInputTokens", 0))
    cache_write = int(usage.get("cacheWriteInputTokens", 0))
    # Converse inputTokens already excludes cache reads and writes. See AWS:
    # https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
    plain_input = int(usage.get("inputTokens", 0))
    total = (
        plain_input * prices["input"] + cache_write * prices["cache_write"]
        + cache_read * prices["cache_read"] + int(usage.get("outputTokens", 0)) * prices["output"]
    )
    return round(total / 1_000_000, 5)


def ledger_path(state_dir: Path) -> Path:
    return state_dir / "cost-ledger.jsonl"


def record(state_dir: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """Append one timed, costed step to the local ledger and return the stored entry."""
    stored = {"recorded_at": datetime.now(UTC).replace(microsecond=0).isoformat(), **entry}
    path = ledger_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stored, ensure_ascii=False) + "\n")
    return stored


def summarize(state_dir: Path) -> dict[str, Any]:
    path = ledger_path(state_dir)
    if not path.exists():
        return {"entries": 0, "estimated_usd": 0.0, "seconds": 0.0, "by_step": {}}
    by_step: dict[str, dict[str, float]] = {}
    total_usd = 0.0
    total_seconds = 0.0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        count += 1
        step = by_step.setdefault(entry["step"], {"count": 0, "estimated_usd": 0.0, "seconds": 0.0,
                                                  "input_tokens": 0, "output_tokens": 0})
        step["count"] += 1
        step["estimated_usd"] += entry.get("estimated_usd") or 0.0
        step["seconds"] += entry.get("seconds") or 0.0
        step["input_tokens"] += entry.get("input_tokens") or 0
        step["output_tokens"] += entry.get("output_tokens") or 0
        total_usd += entry.get("estimated_usd") or 0.0
        total_seconds += entry.get("seconds") or 0.0
    for step in by_step.values():
        step["estimated_usd"] = round(step["estimated_usd"], 4)
        step["seconds"] = round(step["seconds"], 1)
    return {"entries": count, "estimated_usd": round(total_usd, 4), "seconds": round(total_seconds, 1),
            "by_step": by_step, "ledger": str(path)}
