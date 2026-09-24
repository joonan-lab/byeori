import ast
from pathlib import Path

import botocore.exceptions
import pytest
from test_wiki_connections import S3, source_note
from byeori.wiki_connections import publish_page


def load(cloud):
    source = Path("src/byeori/ingest_lambda.py").read_text()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
             and n.name in {"_read_published", "_inspect_wiki_connections"}]
    namespace = {"s3": cloud, "BUCKET_NAME": "bucket", "botocore": botocore}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "inspection-test", "exec"), namespace)
    return namespace["_inspect_wiki_connections"]


def test_inspects_actual_reciprocal_markdown_and_navigation_without_returning_bodies():
    cloud = S3({"wiki/sources/paper-one.md": source_note()})
    key = "wiki/concepts/new.md"
    publish_page(cloud, "bucket", key, "# New\n[[sources/paper-one]]")
    result = load(cloud)({"key": key})
    assert result["targets"] == [{"key": "wiki/sources/paper-one.md", "exists": True, "reciprocal": True}]
    assert result["catalog_registered"] and result["root_catalog_link"]
    assert result["chars"] > 0 and "text" not in result and "body" not in result


def test_broken_connection_is_reported_from_current_objects():
    cloud = S3({"wiki/concepts/new.md": "# New\n[[sources/missing]]"})
    result = load(cloud)({"key": "wiki/concepts/new.md"})
    assert result["targets"] == [{"key": "wiki/sources/missing.md", "exists": False, "reciprocal": False}]
    assert not result["catalog_registered"] and not result["root_catalog_link"]


def test_inspection_rejects_keys_outside_published_wiki():
    inspect = load(S3())
    with pytest.raises(ValueError):
        inspect({"key": "papers/original.pdf"})
