"""The ingest Lambda's answer_question action hands the whole question to the AWS research agent."""
import ast
from pathlib import Path

import botocore.exceptions


def load_lambda():
    source = (Path(__file__).parents[1] / "src/byeori/ingest_lambda.py").read_text()
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_answer_question"]
    ns = dict(botocore=botocore)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "lambda-under-test", "exec"), ns)
    ns.update(BUCKET_NAME="test", INGEST_REASONING="none", bedrock=object())
    return ns


def test_lambda_routes_questions_to_the_aws_agent_with_remaining_time(monkeypatch):
    from types import SimpleNamespace
    from byeori import question_agent
    ns = load_lambda()
    ns.update(boto3=SimpleNamespace(client=lambda *a, **kw: object()), s3=object(), _model_id=lambda event: "global.anthropic.claude-opus-5", _wiki_search=object())
    expected = {"answer": "The actual model answer", "pages_written": [{"key": "wiki/concepts/test.md"}]}
    calls = []
    def run(event, **kwargs):
        calls.append((event, kwargs))
        return expected
    monkeypatch.setattr(question_agent, "run_answer", run)
    context = SimpleNamespace(get_remaining_time_in_millis=lambda: 123456)
    result = ns["_answer_question"]({"title": "Test question?", "reread": "never"}, context)
    assert result is expected
    assert calls[0][1]["remaining_ms"]() == 123456
    assert calls[0][1]["search"] is ns["_wiki_search"]
    assert calls[0][0]["reread"] == "never"
