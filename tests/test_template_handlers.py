"""Every handler string in the templates resolves to a function in the package (or in src/index.py)."""
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

# Collect handlers as (template_name, handler) tuples
HANDLERS = []

# Collect from template.yaml
for match in re.finditer(r"Handler: *([A-Za-z0-9_.]+)", (ROOT / "infra/template.yaml").read_text(encoding="utf-8")):
    HANDLERS.append(("template.yaml", match.group(1)))

# Collect from lab-template.json and jev-eval-template.json
for name in ("infra/lab-template.json", "infra/jev-eval-template.json"):
    document = json.loads((ROOT / name).read_text(encoding="utf-8"))
    for resource in document["Resources"].values():
        handler = resource.get("Properties", {}).get("Handler")
        if isinstance(handler, str):
            HANDLERS.append((name, handler))

# Remove duplicates while preserving order
seen = set()
unique_handlers = []
for item in HANDLERS:
    if item not in seen:
        seen.add(item)
        unique_handlers.append(item)
HANDLERS = unique_handlers


@pytest.mark.parametrize("template_name,handler", HANDLERS, ids=lambda x: f"{x[0]}:{x[1]}")
def test_handler_resolves(template_name, handler, monkeypatch):
    module_name, function_name = handler.rsplit(".", 1)
    monkeypatch.syspath_prepend(str(ROOT / "src"))

    # Mock environment variables needed by byeori modules
    monkeypatch.setenv("BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("TABLE_NAME", "test-table")
    monkeypatch.setenv("OPENALEX_API_KEY_PARAMETER", "/test/path")

    # Special case: Jev template's index.handler means byeori.jev_eval.handler
    if module_name == "index" and template_name == "infra/jev-eval-template.json":
        module = importlib.import_module("byeori.jev_eval")
    elif module_name == "index":
        module = importlib.import_module("index")
    else:
        module = importlib.import_module(module_name)

    assert callable(getattr(module, function_name))
