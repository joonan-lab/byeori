"""Keep the question MCP tool a transparent client of the AWS answer worker."""
import asyncio
import io
import json
import os
import sys
from dataclasses import replace
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import pytest

from byeori import mcp_server
from byeori.aws_store import AwsStore
from byeori.config import Settings


def settings_for(root):
    return Settings(root, root / "data", root / "state", None, "us-east-1", "bucket", "table", "function")


def worker_response():
    return {
        "answer": "The result follows from the stored evidence.\n\n| Evidence | Interpretation |\n|---|---|\n| A | B |\n",
        "status": "answer_ready",
        "question_key": "wiki/questions/example.md",
        "pages_written": [{"key": "wiki/overviews/example.md", "chars": 987, "replaced": True}],
        "page_errors": [{"key": "wiki/concepts/other.md", "error": "Write conflict"}],
        "connections": {"updated": ["wiki/sources/example.md", "wiki/indexes/overviews.md"]},
        "tool_calls": 6,
        "trace_key": "runs/answers/example/trace.json",
        "usage": {"inputTokens": 100, "outputTokens": 40},
        "future_metadata": {"retained": True},
    }


def test_answer_tool_preserves_worker_body_connections_and_metadata(monkeypatch, tmp_path):
    settings = settings_for(tmp_path)
    response = worker_response()
    calls = []

    class Store:
        def __init__(self, configured):
            assert configured is settings

        def answer_question(self, question, **options):
            calls.append((question, options))
            return response

    monkeypatch.setattr(mcp_server.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(mcp_server, "AwsStore", Store)
    question = "이 근거를 기존 합성과 연결해 설명해줘"
    tags = ["methylation"]
    result = mcp_server.answer_wiki_question(question, tags=tags, reread="never")

    assert result is response
    assert calls == [(question, {"tags": tags, "reread": "never"})]
    assert not list(tmp_path.iterdir())


def test_answer_tool_invokes_only_aws_and_defaults_to_auto(monkeypatch, tmp_path):
    response = worker_response()
    services, requests = [], []

    class Session:
        def client(self, service, **options):
            services.append(service)
            assert service in {"sts", "lambda"}, "The client must not generate or download wiki bodies"
            return self

        def get_caller_identity(self):
            return {"Arn": "arn:aws:iam::123456789012:user/test-user"}

        def invoke(self, **request):
            requests.append(request)
            return {"Payload": io.BytesIO(json.dumps(response).encode())}

    monkeypatch.setattr(mcp_server.Settings, "from_env", lambda: settings_for(tmp_path))
    monkeypatch.setattr("byeori.aws_store.boto3.Session", lambda **options: Session())
    result = mcp_server.answer_wiki_question("Which mechanism explains this?")

    assert result == response
    assert services == ["sts", "lambda"]
    assert requests[0]["FunctionName"] == "function"
    assert requests[0]["InvocationType"] == "RequestResponse"
    assert json.loads(requests[0]["Payload"]) == {
        "action": "answer_question", "title": "Which mechanism explains this?",
        "tags": [], "author": "test-user", "reread": "auto",
    }
    assert not list(tmp_path.iterdir())


def test_missing_aws_function_is_reported_without_local_fallback(monkeypatch, tmp_path):
    settings = replace(settings_for(tmp_path), aws_ingest_function=None)
    monkeypatch.setattr(mcp_server.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(AwsStore, "caller_name", lambda self: "test-user")
    monkeypatch.setattr("byeori.aws_store.boto3.Session", lambda **options: object())

    with pytest.raises(RuntimeError, match="AWS_KIRO_WIKI_INGEST_FUNCTION is not configured"):
        mcp_server.answer_wiki_question("Which mechanism explains this?")
    assert not list(tmp_path.iterdir())


def test_answer_tool_is_exposed_with_auto_reread_default():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    tool = next(tool for tool in tools if tool.name == "answer_wiki_question")

    assert tool.inputSchema["required"] == ["question"]
    assert tool.inputSchema["properties"]["reread"]["default"] == "auto"


def test_stdio_client_discovers_research_and_resume_from_another_workspace(tmp_path):
    async def inspect():
        server = StdioServerParameters(
            command=sys.executable, args=["-m", "byeori.mcp_server"],
            cwd=str(tmp_path), env=dict(os.environ),
        )
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=20)) as session:
                initialized = await session.initialize()
                tools = (await session.list_tools()).tools
                return initialized, {tool.name: tool for tool in tools}

    initialized, tools = asyncio.run(inspect())
    assert "answer_wiki_question" in initialized.instructions
    assert "shared wiki" in initialized.instructions
    assert tools["answer_wiki_question"].inputSchema["required"] == ["question"]
    assert tools["resume_wiki_question"].inputSchema["required"] == ["trace_key"]
    assert not list(tmp_path.iterdir())
