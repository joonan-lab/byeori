from __future__ import annotations

import ast
import io
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError
import pytest


TEMPLATE = Path(__file__).parents[1] / "infra" / "template.yaml"


def _lambda_source() -> str:
    return TEMPLATE.parents[1].joinpath("src/byeori/ingest_lambda.py").read_text()


def _compact_search_work():
    tree = ast.parse(_lambda_source())
    selected_nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"MAX_SEARCH_AUTHORS", "MAX_SEARCH_TOPICS"}
        )
        or isinstance(node, ast.FunctionDef)
        and node.name == "_compact_search_work"
    ]
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=selected_nodes, type_ignores=[]), "index.py", "exec"), namespace)
    return namespace["_compact_search_work"]


def test_lambda_search_compacts_large_openalex_records_below_sync_limit() -> None:
    compact = _compact_search_work()
    large_record = {
        "id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.1000/test",
        "display_name": "A useful paper",
        "authorships": [
            {"author": {"display_name": f"Author {index}"}, "institutions": ["x" * 1000]}
            for index in range(1000)
        ],
        "topics": [
            {"display_name": f"Topic {index}", "description": "x" * 1000}
            for index in range(100)
        ],
        "referenced_works": [f"W{index}" for index in range(10_000)],
    }

    results = [compact(large_record) for _ in range(100)]
    payload = json.dumps({"results": results}).encode()

    assert len(payload) < 6 * 1024 * 1024
    assert len(results[0]["authorships"]) == 25
    assert len(results[0]["topics"]) == 10
    assert "referenced_works" not in results[0]


def test_lambda_search_requests_only_fields_needed_for_discovery() -> None:
    source = _lambda_source()

    assert '"select": SEARCH_SELECT_FIELDS' in source
    assert "_compact_search_work(work)" in source


def _extract(names: set[str], extra: dict[str, object] | None = None) -> dict:
    """The named functions, compiled with the module's own plain constants and then ``extra``.

    Seeding the constants means a new module-level setting a function reads (``LAB_QUESTION_PREFIX``
    is one) does not turn every test here into a NameError; ``extra`` still overrides any of them.
    """
    tree = ast.parse(_lambda_source())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace: dict[str, object] = {"re": re, "json": json}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (str, int, float, bool))):
            namespace[node.targets[0].id] = node.value.value
    if extra:
        namespace.update(extra)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "index.py", "exec"), namespace)
    return namespace


def test_the_index_skips_exactly_the_prefix_the_student_path_writes() -> None:
    """One value in two places: the campaign Lambda never imports a lab module, so they must agree."""
    from byeori.lab_store import ANSWER_PAGE_PREFIX

    namespace = _extract(set())
    assert namespace["LAB_QUESTION_PREFIX"] == ANSWER_PAGE_PREFIX == "wiki/lab-questions/"
    assert 'startswith(LAB_QUESTION_PREFIX)' in _lambda_source()


def test_index_documents_types_nested_pages_and_collects_links() -> None:
    index_documents = _extract({"_index_documents", "_split_sections"})["_index_documents"]
    docs = list(index_documents([
        ("wiki/sources/a-2020-x.md", '---\ntitle: "A"\ncategory: "asd-ndd"\n---\n## One-line Summary\ns\n'),
        ("wiki/overviews/asd-ndd/de-novo.md", '---\ntitle: "De novo"\nkind: "subtopic"\ncategory: "asd-ndd"\n---\n## Scope\nx [[sources/a-2020-x]]\n'),
        ("wiki/overviews/asd-ndd/index.md", '---\ntitle: "asd-ndd: landscape"\n---\n## Landscape\n[[overviews/asd-ndd/de-novo]]\n'),
        ("wiki/concepts/scn2a.md", '---\ntitle: "SCN2A"\n---\n## Definition\nd [[sources/a-2020-x]] [[concepts/chd8|CHD8]]\n'),
        ("wiki/overviews/de-novo.md", '---\ntopic: "de-novo"\ntitle: "Old"\n---\n## Scope\ny\n'),
    ]))
    rows = {(meta[0], meta[1]): (meta, links) for meta, _sections, links in docs}
    assert rows[("overview", "asd-ndd/de-novo")][0][8] == "asd-ndd"
    assert rows[("overview", "asd-ndd/de-novo")][0][3] == "data/wiki/overviews/asd-ndd/de-novo.md"
    assert rows[("overview", "asd-ndd/index")][1] == [("overview", "asd-ndd/de-novo")]
    assert rows[("concept", "scn2a")][0][8] == "concepts"
    assert rows[("concept", "scn2a")][1] == [("note", "a-2020-x"), ("concept", "chd8")]
    assert rows[("overview", "de-novo")][0][8] == "overviews", "the pilot's flat overviews keep their folder"
    assert rows[("note", "a-2020-x")][1] == []


def test_index_builder_lists_nested_pages_and_serves_backlinks() -> None:
    source = _lambda_source()
    assert 'paginator.paginate(Bucket=BUCKET_NAME, Prefix="wiki/")' in source
    assert '"/failed/" not in o["Key"]' in source
    assert "CREATE TABLE links" in source and 'if action == "wiki_backlinks":' in source
    assert '"runs/synthesis/orphans.json"' in source
    assert 'if doc_type not in (None, "note", "paper", "overview", "question", "concept"):' in source, \
        "wiki_search accepts the concept doc type"


def test_build_wiki_index_dedupes_links_and_reports_orphans(tmp_path: Path) -> None:
    """End-to-end through the real `_build_wiki_index` and `_index_documents`, with a stub S3.

    A note gets an extra, uncited sibling so the orphan report has something to say: the cited
    note is linked twice from the concept page (checking dedup) and once from the subtopic page,
    while the sibling note is linked from nowhere and must be the only orphan.
    """

    class _Body:
        def __init__(self, text: str) -> None:
            self._text = text

        def read(self) -> bytes:
            return self._text.encode("utf-8")

    class _StubS3:
        def __init__(self, bodies: dict[str, str]) -> None:
            self.bodies = bodies
            self.puts: dict[str, bytes] = {}
            self.uploaded_index_path: Path | None = None

        def get_paginator(self, name: str) -> "_StubS3":
            assert name == "list_objects_v2"
            return self

        def head_object(self, **kwargs: object) -> dict:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

        def paginate(self, **kwargs: object) -> list[dict]:
            assert kwargs.get("Prefix") == "wiki/"
            return [{"Contents": [{"Key": key} for key in self.bodies]}]

        def get_object(self, Bucket: str, Key: str) -> dict:
            assert "retained-failed" not in Key and "retained-stale" not in Key
            return {"Body": _Body(self.bodies[Key])}

        def put_object(self, Bucket: str, Key: str, ContentType=None, Body=None, **conditions) -> None:
            if Key == "index/wiki-index-v2.sqlite3":
                assert conditions == {"IfNoneMatch": "*"}
                assert hasattr(Body, "read"), "Publish the index as a stream"
                self.uploaded_index_path = tmp_path / "captured-index.sqlite3"
                self.uploaded_index_path.write_bytes(Body.read())
            else:
                self.puts[Key] = Body

    bodies = {
        "wiki/sources/w1-2020-x.md": '---\ntitle: "W1"\ncategory: "asd-ndd"\n---\n## Summary\ncited note\n',
        "wiki/sources/w2-2021-y.md": '---\ntitle: "W2"\ncategory: "asd-ndd"\n---\n## Summary\nuncited note\n',
        "wiki/sources/retained-failed.md": '## Summary\nPrevious note before a failed retry.\n',
        "wiki/sources/retained-stale.md": '## Summary\nPrevious note whose extraction changed.\n',
        "wiki/concepts/scn2a.md": '---\ntitle: "SCN2A"\n---\n## Definition\nd [[sources/w1-2020-x]] again [[sources/w1-2020-x]]\n',
        "wiki/overviews/asd-ndd/de-novo.md": '---\ntitle: "De novo"\nkind: "subtopic"\ncategory: "asd-ndd"\n---\n## Scope\nx [[sources/w1-2020-x]]\n',
        "wiki/overviews/legacy.md": '---\ntopic: "legacy"\ntitle: "Legacy"\n---\n## Scope\nprose only\n',
        "wiki/questions/q1.md": '---\ntitle: "Q1?"\n---\n## Question\nq\n',
        "wiki/drafts/x.md": '---\ntitle: "Draft"\n---\n## Scope\nx\n',
        "wiki/sources/failed/y.md": '---\ntitle: "Failed"\n---\n## Scope\nx\n',
        "wiki/stray.md": '---\ntitle: "Stray"\n---\n## Scope\nx\n',
    }
    class _Table:
        def __init__(self):
            self.requests = []

        def scan(self, **request):
            self.requests.append(dict(request))
            if "ExclusiveStartKey" not in request:
                return {"Items": [{"work_id": "retained-failed", "source_note_status": "source_failed"},
                                  {"work_id": "w1-2020-x", "source_note_status": "source_ready"}],
                        "LastEvaluatedKey": {"work_id": "retained-failed"}}
            return {"Items": [{"work_id": "retained-stale", "source_note_status": "stale"}]}
    table_stub = _Table()
    s3_stub = _StubS3(bodies)
    ns = _extract(
        {"_index_documents", "_index_excluded_notes", "_build_wiki_index", "_split_sections"},
        extra={
            "s3": s3_stub,
            "table": table_stub,
            "Attr": Attr,
            "BUCKET_NAME": "test-bucket",
            "WIKI_INDEX_KEY": "index/wiki-index-v2.sqlite3",
            "WIKI_INDEX_LOCAL": tmp_path / "local-index.sqlite3",
            "ThreadPoolExecutor": ThreadPoolExecutor,
            "sqlite3": sqlite3,
            "Path": Path,
            "datetime": datetime,
            "timezone": timezone,
        },
    )

    result = ns["_build_wiki_index"]({})

    assert result["documents"] == {"note": 2, "concept": 1, "overview": 2, "question": 1}
    assert result["excluded_nonready_notes"] == 2
    assert len(table_stub.requests) == 2 and table_stub.requests[1]["ExclusiveStartKey"] == {"work_id": "retained-failed"}
    assert "wiki/sources/retained-failed.md" in s3_stub.bodies, "Index exclusion must preserve prior S3 content"
    assert result["links"] == 2, "the concept's two citations of the same note dedupe to one edge"
    assert result["orphans"] == 1

    orphans_payload = json.loads(s3_stub.puts["runs/synthesis/orphans.json"])
    assert orphans_payload["stems"] == ["w2-2021-y"], "only the uncited note is an orphan"

    assert s3_stub.uploaded_index_path is not None
    con = sqlite3.connect(f"file:{s3_stub.uploaded_index_path}?mode=ro", uri=True)
    rows = con.execute("SELECT from_type, from_id, to_type, to_id FROM links ORDER BY from_type, from_id").fetchall()
    con.close()
    assert rows == [
        ("concept", "scn2a", "note", "w1-2020-x"),
        ("overview", "asd-ndd/de-novo", "note", "w1-2020-x"),
    ]

    # The published index must carry no page text: a hit is resolved through section_map and the
    # body is read from S3. Storing it again is what made the file twice the size of the corpus.
    con = sqlite3.connect(f"file:{s3_stub.uploaded_index_path}?mode=ro", uri=True)
    schema = con.execute("SELECT sql FROM sqlite_master WHERE name = 'sections'").fetchone()[0]
    assert "content=''" in schema, "the FTS table must be contentless"
    assert "detail=" not in schema, "detail=column/none make bm25() return 0 for every row"
    hit = con.execute("SELECT m.doc_type, m.doc_id, m.section, d.s3_key FROM sections s "
                      "JOIN section_map m ON m.rowid = s.rowid "
                      "JOIN docs d ON d.doc_type = m.doc_type AND d.doc_id = m.doc_id "
                      "WHERE sections MATCH ? ORDER BY bm25(sections, 5.0, 2.0, 1.0) LIMIT 1",
                      ('"cited"',)).fetchone()
    assert hit == ("note", "w1-2020-x", "Summary", "wiki/sources/w1-2020-x.md")
    assert con.execute("SELECT snippet(sections, 2, '[', ']', '…', 8) FROM sections "
                       "WHERE sections MATCH ? LIMIT 1", ('"cited"',)).fetchone()[0] is None, \
        "snippet() cannot work without stored text; the reader builds one from the fetched page"
    con.close()


def test_index_exclusion_preserves_notes_without_explicit_status():
    class Table:
        def scan(self, **request):
            return {"Items": [{"work_id": "manual"}, {"work_id": "empty", "source_note_status": ""},
                              {"work_id": "null", "source_note_status": None},
                              {"work_id": "ready", "source_note_status": "source_ready"},
                              {"work_id": "failed", "source_note_status": "source_failed"}]}
    ns = _extract({"_index_excluded_notes"}, extra={"table": Table(), "Attr": Attr})
    assert ns["_index_excluded_notes"]() == {"failed"}


def test_section_texts_reads_each_page_once_and_snippets_come_from_the_fetched_body() -> None:
    """A contentless index has no text, so a hit's body is fetched and shared across its hits."""
    page = ('---\ntitle: "T"\n---\n# T\n\n## Summary\nCHD8 targets chromatin at promoters.\n\n'
            '## Results\nCHD8 haploinsufficiency changed 1,200 genes.\n')
    reads: list[str] = []

    class _S3:
        def get_object(self, Bucket, Key):
            reads.append(Key)
            if Key != "wiki/sources/chd8.md":
                raise KeyError(Key)
            return {"Body": io.BytesIO(page.encode("utf-8"))}

    ns = _extract({"_section_texts", "_split_sections", "_snippet"},
                  extra={"s3": _S3(), "BUCKET_NAME": "b", "ThreadPoolExecutor": ThreadPoolExecutor})
    hits = [{"s3_key": "wiki/sources/chd8.md", "section": "Summary"},
            {"s3_key": "wiki/sources/chd8.md", "section": "Results"},
            {"s3_key": "wiki/sources/missing.md", "section": "Summary"}]
    texts = ns["_section_texts"](hits)

    assert reads.count("wiki/sources/chd8.md") == 1, "two hits on one page must share a single read"
    assert texts[("wiki/sources/chd8.md", "Summary")] == "CHD8 targets chromatin at promoters."
    assert texts[("wiki/sources/chd8.md", "Results")] == "CHD8 haploinsufficiency changed 1,200 genes."
    assert ("wiki/sources/missing.md", "Summary") not in texts, "an unreadable page drops out, it does not raise"

    snippet = ns["_snippet"](texts[("wiki/sources/chd8.md", "Summary")], ["chd8", "chromatin"])
    assert "[CHD8]" in snippet and "[chromatin]" in snippet
    assert ns["_snippet"]("", ["chd8"]) == "", "a hit with no recoverable text yields no snippet"


def test_search_ors_content_words_and_returns_one_row_per_document() -> None:
    """A question matched only documents containing all of its words, which is not how BM25 works.

    Measured 2026-09-20 on eight complete questions: two returned nothing and six returned one
    page's five sections, where the local bm25s wrapper returned twenty distinct documents every
    time. BM25 ranks rather than filters, and the answer and synthesis paths already OR'd.
    """
    source = _lambda_source()
    search = source.split("def _wiki_search(event):", 1)[1].split("\ndef ", 1)[0]
    assert "search_index(con, query, limit" in search
    # Actual FTS5 ranking, question exclusion and dedupe regressions live in test_wiki_search.py.
    assert "limit * 8" not in search


def test_index_normalizes_wiki_prefixed_links_and_note_backlinks():
    build = _extract({"_index_documents", "_split_sections"})["_index_documents"]
    note = ("wiki/sources/paper.md", "# Paper\n[[wiki/overviews/topic.md|Topic]]")
    overview = ("wiki/overviews/topic.md", "# Topic\n[[wiki/sources/paper#Results|Evidence]]")
    rows = list(build([note, overview]))
    assert rows[0][2] == [("overview", "topic")]
    assert rows[1][2] == [("note", "paper")]


def _index_guard_fixture(tmp_path, monkeypatch, initial_index, *, conflict=None, denied=None):
    index_key = "index/wiki-index-v2.sqlite3"
    orphan_key = "runs/synthesis/orphans.json"
    building = tmp_path / "wiki-index.building"
    catalogs = []
    monkeypatch.setattr("byeori.wiki_connections.rebuild_catalogs",
                        lambda *args: catalogs.append(args) or {"pages": [], "errors": []})

    class Cloud:
        def __init__(self):
            self.index = initial_index
            self.orphans = b"Previous orphan report"
            self.calls = []
            self.conditions = None
            self.stream = None

        def head_object(self, **request):
            assert request["Key"] == index_key
            self.calls.append("head")
            if denied == "head":
                raise ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")
            if self.index is None:
                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
            return {"ETag": '"initial-index-version"'}

        def get_paginator(self, name):
            assert self.calls == ["head"], "Capture the index version before listing wiki pages"
            return self

        def paginate(self, **request):
            self.calls.append("snapshot")
            if conflict:
                # Another build publishes after our version read, before our own publication.
                self.index = b"Concurrent published index"
                self.orphans = b"Concurrent orphan report"
            return [{"Contents": [{"Key": "wiki/concepts/example.md"}]}]

        def get_object(self, **request):
            assert request["Key"] == "wiki/concepts/example.md"
            return {"Body": io.BytesIO(b"# Example\n\nA useful concept.\n")}

        def put_object(self, *, Key, Body, **request):
            if Key == index_key:
                assert hasattr(Body, "read") and not isinstance(Body, bytes)
                self.stream = Body
                self.conditions = {key: value for key, value in request.items()
                                   if key in {"IfMatch", "IfNoneMatch"}}
                if denied == "put":
                    raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
                if conflict:
                    raise ClientError({"Error": {"Code": conflict}}, "PutObject")
                self.index = Body.read()
                return {"ETag": '"published-index-version"'}
            assert Key == orphan_key
            self.orphans = Body

    cloud = Cloud()
    namespace = _extract({"_build_wiki_index", "_index_documents", "_split_sections"}, {
        "s3": cloud, "BUCKET_NAME": "test-bucket", "WIKI_INDEX_KEY": index_key,
        "WIKI_INDEX_LOCAL": tmp_path / "cached-index.sqlite3", "_index_excluded_notes": lambda: set(),
        "ThreadPoolExecutor": ThreadPoolExecutor, "sqlite3": sqlite3,
        "Path": lambda value: building, "datetime": datetime, "timezone": timezone,
    })
    return namespace["_build_wiki_index"], cloud, building, catalogs


@pytest.mark.parametrize("initial_index", [None, b"Existing index"])
def test_index_publication_uses_snapshot_version_and_stream(tmp_path, monkeypatch, initial_index):
    build, cloud, building, catalogs = _index_guard_fixture(tmp_path, monkeypatch, initial_index)
    result = build({})

    expected = {"IfNoneMatch": "*"} if initial_index is None else {"IfMatch": '"initial-index-version"'}
    assert cloud.conditions == expected
    assert cloud.index.startswith(b"SQLite format 3")
    assert result["documents"] == {"concept": 1}
    assert json.loads(cloud.orphans)["stems"] == []
    assert catalogs and cloud.stream.closed and not building.exists()


@pytest.mark.parametrize("initial_index", [None, b"Existing index"])
@pytest.mark.parametrize("conflict", ["PreconditionFailed", "ConditionalRequestConflict", "412", "409"])
def test_index_conflict_preserves_concurrent_index_and_orphan_report(tmp_path, monkeypatch, initial_index, conflict):
    build, cloud, building, catalogs = _index_guard_fixture(tmp_path, monkeypatch, initial_index, conflict=conflict)
    result = build({})

    assert result["status"] == "index_superseded" and result["published"] is False
    assert cloud.index == b"Concurrent published index"
    assert cloud.orphans == b"Concurrent orphan report"
    assert not catalogs and cloud.stream.closed and not building.exists()


@pytest.mark.parametrize("denied", ["head", "put"])
def test_index_permission_errors_propagate_and_remove_building_file(tmp_path, monkeypatch, denied):
    build, cloud, building, catalogs = _index_guard_fixture(tmp_path, monkeypatch, b"Existing index", denied=denied)
    with pytest.raises(ClientError) as error:
        build({})

    assert error.value.response["Error"]["Code"] == "AccessDenied"
    assert cloud.index == b"Existing index" and cloud.orphans == b"Previous orphan report"
    assert not catalogs and not building.exists()
    if cloud.stream is not None:
        assert cloud.stream.closed
    if denied == "head":
        assert cloud.calls == ["head"]
