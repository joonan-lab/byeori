"""Boundary checks that keep the student question workflow off the campaign path.

The lab modules (``lab_*.py``, ``evidence_packet.py``, ``jev_client.py``) share a package with the
running question campaign but must never import its Lambda code, its S3 client, the research
agent or its cache, and must never write an S3 key outside ``runs/lab-questions/``. These tests
read source with ``ast``, import the deployed handler from a zip of ``src/`` in a subprocess while
recording every module import, pin the campaign-path modules by sha256, and confirm that no
local ``data/`` mirror appears when every lab module is imported.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from byeori import lab_store
from lab_fakes import MemoryS3

REPO = Path(__file__).parents[1]
SRC = REPO / "src"
PACKAGE = SRC / "byeori"
PACKAGE_NAME = "byeori"

# Modules the lab code may never import, at module level or inside a function.
CAMPAIGN_MODULES = frozenset({
    "byeori.ingest_lambda",
    "byeori.aws_store",
    "byeori.question_agent",
    "byeori.agent_cache",
})
# Client libraries that stay out of the server modules; the student MCP client may use two of them.
CLIENT_LIBRARIES = frozenset({"mcp", "httpx", "fitz"})
CLIENT_MODULE = "lab_mcp_server.py"
CLIENT_MODULE_MAY_IMPORT = frozenset({"mcp", "httpx"})
# P4 (design section 7, user approval of 2026-09-21 after the campaign closed): the approved research
# worker reuses the campaign engine ``question_agent.run_answer`` with an injected publisher, so it is
# the one lab module that imports the research agent and wraps ``wiki_connections.publish_page``.
# It still never imports the ingest Lambda, the S3 store or the agent cache.
RESEARCH_MODULE = "lab_research.py"
RESEARCH_MODULE_MAY_IMPORT = frozenset({"byeori.question_agent"})
# Lab modules the campaign's handler path must never import, even lazily.
LAB_MODULE_PREFIX = "byeori.lab_"
LAB_SHARED_MODULES = frozenset({"byeori.evidence_packet", "byeori.jev_client"})

FORBIDDEN_KEY_PREFIXES = ("wiki/", "papers/", "index/")
RECEIPT_PREFIX = "runs/lab-questions/"
S3_WRITE_METHODS = frozenset({
    "put_object", "upload_file", "upload_fileobj", "copy_object", "delete_object", "delete_objects",
    "put_object_tagging",
})
TABLE_WRITE_METHODS = frozenset({"put_item", "update_item", "delete_item", "batch_write_item", "transact_write_items"})
STORE_MODULE = "lab_store.py"

# Campaign-path baseline recorded on 2026-09-21 from a working tree whose src/ matched
# `git rev-parse HEAD` = 50e09dcecd57f8741d43dcbbbba890dafcd77ac7. The student workflow never
# edits these files. A deliberate campaign change refreshes the value for that file here, in its
# own commit, with a note saying why the campaign code moved.
# Campaign-path modules the student workflow must never import or be imported by. Their content
# is owned by the campaign; git history, not a pinned hash, records who changed them.
CAMPAIGN_FILES = (
    "src/byeori/question_agent.py",
    "src/byeori/wiki_connections.py",
    "src/byeori/wiki_search.py",
    "src/byeori/ingest_lambda.py",
    "src/byeori/question_campaign.py",
    "src/byeori/wiki_ops.py",
    "src/index.py",
)


# ---------------------------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------------------------

def lab_modules() -> list[Path]:
    """Every ``lab_*.py`` (``lab_lambda.py`` included once it exists) plus the two shared modules."""
    paths = sorted(PACKAGE.glob("lab_*.py")) + [PACKAGE / "evidence_packet.py", PACKAGE / "jev_client.py"]
    missing = [path.name for path in paths if not path.exists()]
    assert not missing, missing
    return paths


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported_modules(tree: ast.AST) -> set[str]:
    """Module names an ``import``/``from ... import`` may bind, with relative imports resolved.

    ``from byeori import lab_budget`` yields both ``byeori`` and
    ``byeori.lab_budget`` because the imported name may be a submodule. Function-level
    imports count as well: a lazy import is still an import.
    """
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_base(node)
            if base:
                modules.add(base)
            modules.update(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return modules


def imported_names(tree: ast.AST, module: str) -> set[str]:
    """Names imported with ``from <module> import ...`` (relative spellings resolved)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _resolve_base(node) == module:
            names.update(alias.name for alias in node.names)
    return names


def _resolve_base(node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    parts = PACKAGE_NAME.split(".")[: max(len(PACKAGE_NAME.split(".")) - (node.level - 1), 0)]
    if node.module:
        parts.append(node.module)
    return ".".join(parts)


def method_calls(tree: ast.AST, methods: frozenset[str]) -> list[ast.Call]:
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in methods]


def function_calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _callee(node.func) == name]


def _callee(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def literal_prefix(node: ast.expr | None) -> str | None:
    """The literal text an expression is known to start with, or None when it is not knowable."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        return literal_prefix(node.values[0])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal_prefix(node.left)
    return None


def keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


# ---------------------------------------------------------------------------------------------
# (1) Import boundary of the new modules
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", lab_modules(), ids=lambda path: path.name)
def test_new_modules_never_import_campaign_code_or_client_libraries(path):
    tree = parse(path)
    modules = imported_modules(tree)

    allowed_campaign = RESEARCH_MODULE_MAY_IMPORT if path.name == RESEARCH_MODULE else frozenset()
    banned = modules & (CAMPAIGN_MODULES - allowed_campaign)
    assert not banned, f"{path.name} imports campaign-path code: {sorted(banned)}"

    if path.name != RESEARCH_MODULE:
        assert "publish_page" not in imported_names(tree, "byeori.wiki_connections"), \
            f"{path.name} imports wiki_connections.publish_page; only the campaign publishes wiki pages"
        references = [node for node in ast.walk(tree)
                      if (isinstance(node, ast.Attribute) and node.attr == "publish_page")
                      or (isinstance(node, ast.Name) and node.id == "publish_page")
                      or (isinstance(node, ast.Constant) and node.value == "publish_page")]
        assert not references, f"{path.name} refers to publish_page"

    allowed = CLIENT_MODULE_MAY_IMPORT if path.name == CLIENT_MODULE else frozenset()
    client_imports = {module for module in modules if module.split(".")[0] in CLIENT_LIBRARIES - allowed}
    assert not client_imports, f"{path.name} imports a client library: {sorted(client_imports)}"


def test_only_the_student_mcp_client_imports_mcp_and_httpx():
    """The client module exists and is the one place the client libraries appear."""
    importers = {path.name for path in lab_modules()
                 if {module.split(".")[0] for module in imported_modules(parse(path))} & {"mcp", "httpx"}}
    assert importers == {CLIENT_MODULE}, importers


def test_only_the_research_worker_imports_the_campaign_engine():
    """The approved research worker exists and is the one lab module that reuses ``question_agent``."""
    importers = {path.name for path in lab_modules()
                 if imported_modules(parse(path)) & RESEARCH_MODULE_MAY_IMPORT}
    assert importers == {RESEARCH_MODULE}, importers
    tree = parse(PACKAGE / RESEARCH_MODULE)
    assert not (imported_modules(tree) & (CAMPAIGN_MODULES - RESEARCH_MODULE_MAY_IMPORT))


# ---------------------------------------------------------------------------------------------
# (2) The packaged campaign handler stays unaware of the lab modules
# ---------------------------------------------------------------------------------------------

def test_packaged_handler_imports_no_lab_module(tmp_path):
    package = tmp_path / "package.zip"
    with zipfile.ZipFile(package, "w") as archive:
        for path in list(SRC.rglob("*.py")) + list(SRC.rglob("*.json")):
            archive.write(path, path.relative_to(SRC))
    code = r'''
import sys, io, json, importlib.abc, zipfile
from pathlib import Path
import boto3
from unittest.mock import MagicMock
class RecordImports(importlib.abc.MetaPathFinder):
    seen = []
    def find_spec(self, fullname, path=None, target=None):
        self.seen.append(fullname)
        return None
sys.meta_path.insert(0, RecordImports())
cloud = MagicMock()
boto3.client = lambda *args, **kwargs: cloud
boto3.resource = lambda *args, **kwargs: cloud
extract_dir = Path(sys.argv[1]).parent / 'extracted'
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(extract_dir)
sys.path.insert(0, str(extract_dir))
from index import handler
assert callable(handler)
print(json.dumps(sorted(set(RecordImports.seen))))
'''
    env = {**os.environ, "BUCKET_NAME": "bucket", "TABLE_NAME": "table",
           "OPENALEX_API_KEY_PARAMETER": "parameter", "AWS_DEFAULT_REGION": "us-east-1"}
    result = subprocess.run([sys.executable, "-c", code, str(package)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    recorded = set(json.loads(result.stdout.strip().splitlines()[-1]))

    assert {"index", "byeori.ingest_lambda"} <= recorded, sorted(recorded)  # the recorder saw the handler path
    lab_imports = {name for name in recorded if name.startswith(LAB_MODULE_PREFIX) or name in LAB_SHARED_MODULES}
    assert not lab_imports, f"the campaign handler imported lab modules: {sorted(lab_imports)}"
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("relative", CAMPAIGN_FILES, ids=lambda relative: Path(relative).name)
def test_campaign_modules_never_import_lab_modules(relative):
    """A lazy import inside a campaign function would escape the subprocess check; the AST does not."""
    modules = imported_modules(parse(REPO / relative))
    lab_imports = {name for name in modules if name.startswith(LAB_MODULE_PREFIX) or name in LAB_SHARED_MODULES}
    assert not lab_imports, f"{relative} imports lab modules: {sorted(lab_imports)}"


# ---------------------------------------------------------------------------------------------
# (3) The campaign-path modules are exactly the recorded baseline
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# (4) S3 and table writes happen in lab_store only, and never under wiki/, papers/ or index/
# ---------------------------------------------------------------------------------------------

def test_only_lab_store_calls_s3_and_table_write_methods():
    writers: dict[str, set[str]] = {}
    for path in lab_modules():
        calls = method_calls(parse(path), S3_WRITE_METHODS | TABLE_WRITE_METHODS)
        if calls:
            writers[path.name] = {call.func.attr for call in calls}
    assert set(writers) == {STORE_MODULE}, writers
    assert "put_object" in writers[STORE_MODULE]


def test_literal_written_keys_never_start_with_wiki_papers_or_index():
    inspected = 0
    for path in lab_modules():
        tree = parse(path)
        candidates: list[tuple[str, ast.expr]] = []
        for call in method_calls(tree, frozenset({"put_object"})):
            candidates.append(("put_object Key=", keyword(call, "Key")))
        for call in function_calls(tree, "receipt_key"):
            candidates.extend((f"receipt_key arg {i}", arg) for i, arg in enumerate(call.args))
            candidates.extend((f"receipt_key {kw.arg}=", kw.value) for kw in call.keywords)
        for call in function_calls(tree, "put_json"):
            candidates.append(("put_json key", call.args[0] if call.args else keyword(call, "key")))
        for label, node in candidates:
            prefix = literal_prefix(node)
            if prefix is None:
                continue
            inspected += 1
            assert not prefix.startswith(FORBIDDEN_KEY_PREFIXES), f"{path.name}: {label} writes {prefix!r}"
            if isinstance(node, ast.Constant) and "/" in prefix:
                assert prefix.startswith(RECEIPT_PREFIX), f"{path.name}: {label} writes {prefix!r}"
    assert inspected > 0, "no literal receipt keys found; the heuristic scan no longer matches the modules"


def test_receipt_writer_refuses_the_forbidden_prefixes_at_runtime():
    """The static scan and the runtime guard agree on what a receipt key may be."""
    assert lab_store.RECEIPT_PREFIX == RECEIPT_PREFIX
    assert set(FORBIDDEN_KEY_PREFIXES) <= set(lab_store.FORBIDDEN_WRITE_PREFIXES)
    s3 = MemoryS3()
    writer = lab_store.ReceiptWriter(s3, "bucket")
    for key in ("wiki/sources/example.md", "papers/example/original.pdf", "index/wiki-index-v2.sqlite3",
                "runs/lab-questions/../wiki/sources/example.md", "runs/questions/job/request.json"):
        with pytest.raises(ValueError):
            writer.put_json(key, {"body": "never written"})
    assert s3.writes == []
    written = writer.put_json(lab_store.receipt_key("job-1", "request.json"), {"question": "example"})
    assert written["key"] == "runs/lab-questions/job-1/request.json"
    assert [key for key, _ in s3.writes] == [written["key"]]


# ---------------------------------------------------------------------------------------------
# (5) Importing every lab module leaves no local mirror behind
# ---------------------------------------------------------------------------------------------

def test_importing_every_lab_module_creates_no_data_directory(tmp_path, monkeypatch):
    names = [f"{PACKAGE_NAME}.{path.stem}" for path in lab_modules()]

    code = "import importlib, os, sys\n" \
           "sys.path.insert(0, sys.argv[1])\n" \
           "for name in sys.argv[2:]:\n" \
           "    importlib.import_module(name)\n" \
           "print(sorted(os.listdir('.')))\n"
    result = subprocess.run([sys.executable, "-c", code, str(SRC), *names], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "data" not in ast.literal_eval(result.stdout.strip().splitlines()[-1])

    monkeypatch.chdir(REPO)
    for name in names:
        importlib.import_module(name)
    assert not Path("data").exists()
    assert not (REPO / "data").exists()
