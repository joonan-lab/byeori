"""The contact address is configuration only, never a built-in default (2026-09-24 fix round 1)."""
from __future__ import annotations

import io
import json

from byeori import contact, openalex_match as matcher


def test_unset_env_returns_empty_values(monkeypatch) -> None:
    monkeypatch.delenv(contact.ENV_NAME, raising=False)
    assert contact.contact_email() == ""
    assert contact.mailto_parameter() == {}
    assert contact.user_agent("byeori/0.1") == "byeori/0.1"


def test_set_env_carries_the_address(monkeypatch) -> None:
    monkeypatch.setenv(contact.ENV_NAME, "lab@example.org")
    assert contact.contact_email() == "lab@example.org"
    assert contact.mailto_parameter() == {"mailto": "lab@example.org"}
    assert contact.user_agent("byeori/0.1") == "byeori/0.1 (mailto:lab@example.org)"


def test_contact_email_strips_whitespace(monkeypatch) -> None:
    monkeypatch.setenv(contact.ENV_NAME, "  lab@example.org  ")
    assert contact.contact_email() == "lab@example.org"


def test_openalex_fetch_one_request_has_no_at_sign_when_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv(contact.ENV_NAME, raising=False)
    captured = {}

    class Response(io.BytesIO):
        headers = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["user_agent"] = request.get_header("User-agent")
        return Response(json.dumps({}).encode())

    monkeypatch.setattr(matcher.urllib.request, "urlopen", fake_urlopen)
    matcher.fetch_one("10.123/example", attempts=1)
    assert "@" not in captured["url"]
    assert "@" not in captured["user_agent"]
    assert captured["user_agent"] == "byeori/0.1"


def test_openalex_fetch_one_request_carries_the_configured_address(monkeypatch) -> None:
    monkeypatch.setenv(contact.ENV_NAME, "lab@example.org")
    captured = {}

    class Response(io.BytesIO):
        headers = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["user_agent"] = request.get_header("User-agent")
        return Response(json.dumps({}).encode())

    monkeypatch.setattr(matcher.urllib.request, "urlopen", fake_urlopen)
    matcher.fetch_one("10.123/example", attempts=1)
    assert "mailto=lab@example.org" in captured["url"]
    assert captured["user_agent"] == "byeori/0.1 (mailto:lab@example.org)"
