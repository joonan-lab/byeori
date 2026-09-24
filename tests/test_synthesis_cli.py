from __future__ import annotations

import json
from pathlib import Path

import pytest

from byeori import runs, synthesis
from byeori.cli import build_parser
from byeori.config import Settings


def settings_for(root: Path) -> Settings:
    settings = Settings(root=root, data_dir=root / "data", state_dir=root / "state", openalex_api_key=None,
                        aws_region="ap-northeast-2", aws_bucket="bucket", aws_table="table")
    settings.ensure_directories()
    return settings


def test_start_synthesis_sends_the_full_input(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(runs, "start_run", lambda settings, output_key, payload=None: calls.append((output_key, payload)) or {"execution": "arn"})
    monkeypatch.setattr(synthesis, "start_run", runs.start_run)
    synthesis.start_synthesis(settings_for(tmp_path), scope="autism", plan_only=True)
    assert calls == [("SynthesisStateMachineArn", {"scope": "autism", "plan_only": True, "categories": None,
                                                    "skip_concepts": False, "replan": False})]
    with pytest.raises(ValueError):
        synthesis.start_synthesis(settings_for(tmp_path), scope="nope")


def test_synthesis_page_loops_until_the_page_is_done(monkeypatch, tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    monkeypatch.setattr(synthesis, "stack_outputs", lambda s: {"SynthesisFunctionName": "fn-test"})
    answers = [{"status": "partial", "calls": 3, "input_tokens": 30, "output_tokens": 3, "seconds": 400.0},
               {"status": "ready", "key": "wiki/concepts/scn2a.md", "calls": 1, "input_tokens": 10, "output_tokens": 1, "seconds": 100.0}]
    sent = []
    monkeypatch.setattr(synthesis, "invoke_synthesis",
                        lambda s, payload, function_name=None: sent.append((payload, function_name)) or answers.pop(0))
    result = synthesis.synthesis_page(settings, kind="concept", ident="scn2a")
    assert result["status"] == "ready" and result["rounds"] == 2
    assert [payload for payload, _ in sent] == [{"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"}] * 2
    assert [name for _, name in sent] == ["fn-test", "fn-test"]
    ledger = [json.loads(l) for l in (settings.state_dir / "cost-ledger.jsonl").read_text().splitlines()]
    assert [e["step"] for e in ledger] == ["synthesis_concept", "synthesis_concept"] and ledger[0]["input_tokens"] == 30
    with pytest.raises(ValueError):
        synthesis.synthesis_page(settings, kind="subtopic", ident="no-slash")


def test_synthesis_page_stops_after_max_rounds(monkeypatch, tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    monkeypatch.setattr(synthesis, "stack_outputs", lambda s: {"SynthesisFunctionName": "fn-test"})
    monkeypatch.setattr(synthesis, "invoke_synthesis",
                        lambda s, payload, function_name=None: {"status": "partial", "calls": 1, "seconds": 1.0})
    result = synthesis.synthesis_page(settings, kind="concept", ident="scn2a", max_rounds=3)
    assert result["status"] == "partial" and result["rounds"] == 3


def test_synthesis_page_prices_the_model_the_lambda_used(monkeypatch, tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    monkeypatch.setattr(synthesis, "stack_outputs", lambda s: {"SynthesisFunctionName": "fn-test"})
    monkeypatch.setattr(synthesis, "invoke_synthesis", lambda s, payload, function_name=None: {
        "status": "ready", "key": "wiki/concepts/scn2a.md", "calls": 1, "input_tokens": 10, "output_tokens": 1,
        "seconds": 1.0, "model": "global.anthropic.claude-sonnet-5"})
    synthesis.synthesis_page(settings, kind="concept", ident="scn2a", model_id="global.anthropic.claude-opus-5")
    ledger = [json.loads(l) for l in (settings.state_dir / "cost-ledger.jsonl").read_text().splitlines()]
    assert ledger[0]["model_id"] == "global.anthropic.claude-sonnet-5"


def test_manifest_clients_submit_requests_without_s3_or_local_validation(monkeypatch, tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    sent = []
    monkeypatch.setattr(synthesis.boto3, "Session", lambda *a, **kw: pytest.fail("manifest clients must only call Lambda"))
    monkeypatch.setattr(synthesis, "invoke_synthesis", lambda settings, payload: sent.append(payload) or {"validated": "aws"})
    synthesis.pull_manifests(settings, scope="autism")
    assert sent[-1] == {"action": "manifest_summary", "scope": "autism"}
    synthesis.read_manifest(settings, kind="concepts", section="scn2a", offset=4000, max_chars=8000)
    assert sent[-1] == {"action": "manifest_read", "kind": "concepts", "category": None, "section": "scn2a",
                        "offset": 4000, "max_chars": 8000}
    assert synthesis.push_manifest(settings, content={"subtopics": []}, kind="subtopics", category="asd-ndd") == {"validated": "aws"}
    assert sent[-1] == {"action": "manifest_submit", "kind": "subtopics", "category": "asd-ndd", "content": {"subtopics": []}}


def test_push_manifest_preserves_existing_user_input_and_uses_aws_validation(monkeypatch, tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    sent = []
    monkeypatch.setattr(synthesis, "invoke_synthesis", lambda settings, payload: sent.append(payload) or {"key": "s3-key"})
    path = tmp_path / "asd-ndd" / "subtopics.json"
    path.parent.mkdir()
    content = json.dumps({"category": "asd-ndd", "subtopics": []})
    path.write_text(content)
    assert synthesis.push_manifest(settings, path)["key"] == "s3-key"
    assert sent == [{"action": "manifest_submit", "kind": "subtopics", "category": None, "content": content}]
    assert path.read_text() == content
    with pytest.raises(ValueError):
        synthesis.push_manifest(settings, tmp_path / "something.json")
    with pytest.raises(ValueError):
        synthesis.push_manifest(settings, path, content={})
    with pytest.raises(ValueError):
        synthesis.push_manifest(settings)


def test_cli_parses_the_synthesis_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(["aws-synthesis-run", "--scope", "autism", "--category", "germline-mutation", "--skip-concepts"])
    assert args.scope == "autism" and args.category == ["germline-mutation"] and args.skip_concepts is True
    assert parser.parse_args(["aws-synthesis-plan"]).scope == "autism"
    page_args = parser.parse_args(["aws-synthesis-page", "subtopic", "asd-ndd/de-novo-variants"])
    assert page_args.mode == "generate" and page_args.max_rounds == 30
    assert parser.parse_args(["aws-synthesis-page", "concept", "scn2a", "--max-rounds", "5"]).max_rounds == 5
    assert parser.parse_args(["aws-synthesis-manifest", "--push", "x.json"]).push == "x.json"
    manifest = parser.parse_args(["aws-synthesis-manifest", "--kind", "subtopics", "--category", "asd-ndd",
                                  "--section", "de-novo", "--offset", "4000", "--max-chars", "8000"])
    assert (manifest.kind, manifest.category, manifest.section, manifest.offset, manifest.max_chars) == (
        "subtopics", "asd-ndd", "de-novo", 4000, 8000)
    assert parser.parse_args(["aws-synthesis-manifest", "--kind", "overrides", "--content", "{}"]).content == "{}"
    assert parser.parse_args(["wiki-backlinks", "note", "abdi-2023-x"]).backend == "auto"
    assert parser.parse_args(["aws-pipeline-failures", "--synthesis"]).synthesis is True
