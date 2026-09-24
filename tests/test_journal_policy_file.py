"""The journal lists are a data file an installer may replace; the shipped file is the lab's list."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from byeori import journal_policy


def test_shipped_file_is_inside_the_package_and_loads():
    assert journal_policy.POLICY_PATH == Path(journal_policy.__file__).with_name("policies") / "journals.json"
    data = json.loads(journal_policy.POLICY_PATH.read_text(encoding="utf-8"))
    assert set(data) == {"journals", "lab_additions", "preprint_servers", "denied_publishers",
                         "denied_journals", "nature_portfolio_below_threshold"}
    assert len(data["journals"]) == 65
    assert data["journals"]["nature"] == {"title": "Nature", "family": "Nature",
                                          "issns": ["0028-0836", "1476-4687"], "source_id": "S137773608"}


def test_module_constants_come_from_the_file():
    data = journal_policy.load_policy()
    assert journal_policy.JOURNALS["nature"] == ("Nature", "Nature", ("0028-0836", "1476-4687"), "S137773608")
    assert set(journal_policy.JOURNALS) == set(data["journals"])
    assert journal_policy.LAB_ADDITIONS["proceedings of machine learning research"][3] == ("S4363608721",)
    assert journal_policy.PREPRINT_SERVERS == tuple(data["preprint_servers"])
    assert journal_policy.DENIED_JOURNALS == tuple(data["denied_journals"])
    assert journal_policy.journal_verdict("Nature")["verdict"] == "include"
    assert journal_policy.journal_verdict("Scientific Reports")["verdict"] == "exclude"


def test_environment_variable_points_at_another_file(tmp_path, monkeypatch):
    data = json.loads(journal_policy.POLICY_PATH.read_text(encoding="utf-8"))
    data["journals"]["journal of testing"] = {"title": "Journal of Testing", "family": "Test family",
                                              "issns": ["1234-5678"], "source_id": "S1"}
    other = tmp_path / "journals.json"
    other.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("BYEORI_JOURNAL_POLICY", str(other))
    try:
        module = importlib.reload(journal_policy)
        assert module.journal_verdict("Journal of Testing")["verdict"] == "include"
        assert module.journal_verdict(None, issn="1234-5678")["verdict"] == "include"
    finally:
        monkeypatch.delenv("BYEORI_JOURNAL_POLICY")
        importlib.reload(journal_policy)


def test_missing_file_names_the_path(tmp_path, monkeypatch):
    monkeypatch.setenv("BYEORI_JOURNAL_POLICY", str(tmp_path / "absent.json"))
    try:
        with pytest.raises(FileNotFoundError, match="absent.json"):
            importlib.reload(journal_policy)
    finally:
        monkeypatch.delenv("BYEORI_JOURNAL_POLICY")
        importlib.reload(journal_policy)
