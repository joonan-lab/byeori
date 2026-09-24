"""The CLI names no path of this lab and does not demand Kiro (spec 2026-09-24)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from byeori import cli

ROOT = Path(__file__).parents[1]


def test_no_user_home_path_or_lab_name_in_package_or_tests():
    pattern = re.compile("|".join(["/Us" + "ers/", "Drop" + "box", "joo" + "nan"]))
    files = [*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").glob("test_*.py"), *(ROOT / "templates").glob("*.md")]
    hits = [p.name for p in files if pattern.search(p.read_text(encoding="utf-8")) and p.name != "test_export_public.py"]
    assert hits == []


def test_llm_wiki_arguments_have_no_default():
    parser = cli.build_parser()
    for command in ("upload-papers", "resolve-ids", "benchmark-questions"):
        sub = parser._subparsers._group_actions[0].choices[command]
        action = next(a for a in sub._actions if "--llm-wiki" in a.option_strings)
        assert action.required and action.default is None, command


def test_doctor_does_not_require_kiro(monkeypatch, tmp_path, capsys):
    from byeori.config import Settings
    settings = Settings(tmp_path, tmp_path / "data", tmp_path / "state", None, "us-east-1", None, None, None)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(cli, "KIRO_APP_BINARY", tmp_path / "absent")
    monkeypatch.setattr(cli.AwsStore, "status", lambda self: {"configured": False})

    class Client:
        def __init__(self, key): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get_work(self, work_id): return {"work_id": work_id}
    monkeypatch.setattr(cli, "OpenAlexClient", Client)
    assert cli.command_doctor(settings, argparse.Namespace()) == 0
    assert '"kiro_cli": false' in capsys.readouterr().out
