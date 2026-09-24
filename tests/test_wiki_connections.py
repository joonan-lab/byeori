import hashlib
import io
import re
import threading

import pytest
from botocore.exceptions import ClientError

from byeori.wiki_connections import (
    BACKLINK_END, BACKLINK_START, PageConflictError, publish_page, rebuild_catalogs,
)
from byeori import wiki_connections as connections


class S3:
    def __init__(self, objects=None):
        self.objects = {key: value.encode() for key, value in (objects or {}).items()}
        self.writes = []
        self.before_put = None
        self.denied = set()

    def etag(self, key):
        return '"' + hashlib.md5(self.objects[key]).hexdigest() + '"'

    def get_object(self, Bucket, Key):
        if Key in self.denied:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key]), "ETag": self.etag(Key)}

    def put_object(self, Bucket, Key, Body, ContentType, **conditions):
        if self.before_put:
            self.before_put(self, Key)
        assert "IfMatch" in conditions or conditions.get("IfNoneMatch") == "*"
        if ((conditions.get("IfNoneMatch") == "*" and Key in self.objects)
                or ("IfMatch" in conditions and
                    (Key not in self.objects or self.etag(Key) != conditions["IfMatch"]))):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if Key in self.denied:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
        self.objects[Key] = Body
        self.writes.append((Key, conditions))
        return {"ETag": self.etag(Key)}


def source_note():
    return ("---\ntitle: \"Original paper\"\ndoi: \"10.1/example\"\n---\n# Original paper\n"
            + "\n".join(heading + "\nOriginal scientific text.\n" for heading in (
                "## 1. Document Information", "## One-line Summary", "## 2. Key Contributions",
                "## 3. Methodology and Architecture", "## 4. Key Results and Benchmarks",
                "## 5. Limitations and Future Work", "## 6. Related Work", "## 7. Glossary")))


def test_new_page_adds_real_note_backlink_and_catalog_without_changing_note_sections():
    original = source_note()
    cloud = S3({"wiki/sources/paper-one.md": original})
    text = "# A concept\n\nScience [[sources/paper-one#Results|Paper]] and [[wiki/sources/paper-one.md]].\n"
    result = publish_page(cloud, "bucket", "wiki/concepts/a-concept.md", text, create_only=True)
    assert result["errors"] == [] and result["replaced"] is False
    assert cloud.objects[result["key"]].decode() == text
    note = cloud.objects["wiki/sources/paper-one.md"].decode()
    assert re.findall(r"^## .*", note, re.M) == re.findall(r"^## .*", original, re.M)
    assert note.count("[[concepts/a-concept|A concept]]") == 1
    assert note.index("## 6. Related Work") < note.index(BACKLINK_START) < note.index("## 7. Glossary")
    assert note.startswith(original.split("## 6. Related Work")[0])
    assert "[[concepts/a-concept|A concept]]" in cloud.objects["wiki/indexes/concepts.md"].decode()
    assert "[[indexes/concepts|Concepts]]" in cloud.objects["wiki/index.md"].decode()
    assert result["sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_rewrite_preserves_frontmatter_and_incoming_links_and_deduplicates():
    managed = f"\n{BACKLINK_START}\n### Linked pages\n- [[questions/old-question|Old question]]\n{BACKLINK_END}\n"
    original = '---\ntitle: "Existing title"\ncreated: 2026-01-01\n---\n# Old body\n' + managed
    cloud = S3({"wiki/concepts/a-concept.md": original, "wiki/sources/paper-one.md": source_note()})
    text = '# Revised body\nNew science [[sources/paper-one]].\n' + managed
    first = publish_page(cloud, "bucket", "wiki/concepts/a-concept.md", text,
                         expected_etag=cloud.etag("wiki/concepts/a-concept.md"))
    saved = cloud.objects[first["key"]].decode()
    assert saved.startswith('---\ntitle: "Existing title"\ncreated: 2026-01-01\n---\n# Revised body')
    assert saved.count("[[questions/old-question|Old question]]") == 1
    second = publish_page(cloud, "bucket", first["key"], text, expected_etag=first["etag"])
    assert cloud.objects[second["key"]].decode() == saved
    assert cloud.objects["wiki/sources/paper-one.md"].decode().count("[[concepts/a-concept|Existing title]]") == 1


@pytest.mark.parametrize("kwargs", [{}, {"create_only": True}, {"expected_etag": '"outdated"'}])
def test_existing_scientific_body_cannot_be_overwritten_without_its_version(kwargs):
    cloud = S3({"wiki/concepts/a-concept.md": "# Original\nScience."})
    with pytest.raises(PageConflictError):
        publish_page(cloud, "bucket", "wiki/concepts/a-concept.md", "# Replacement", **kwargs)
    assert cloud.objects["wiki/concepts/a-concept.md"] == b"# Original\nScience."
    assert cloud.writes == []


def test_scientific_cas_conflict_is_not_retried():
    key = "wiki/concepts/a-concept.md"
    cloud = S3({key: "# Original\nScience."})
    expected = cloud.etag(key)
    calls = []
    def concurrent(store, target):
        calls.append(target)
        store.objects[target] = b"# Another author's science"
    cloud.before_put = concurrent
    with pytest.raises(PageConflictError):
        publish_page(cloud, "bucket", key, "# My revision", expected_etag=expected)
    assert calls == [key]
    assert cloud.objects[key] == b"# Another author's science"


def test_backlink_cas_retry_preserves_concurrent_body_and_other_backlink():
    note = "wiki/sources/paper-one.md"
    cloud = S3({note: source_note()})
    def concurrent(store, key):
        if key != note:
            return
        store.before_put = None
        body = store.objects[key].decode().replace("Original scientific text.", "Concurrent scientific text.")
        body = body.replace("## 7. Glossary", f"{BACKLINK_START}\n### Linked pages\n- [[concepts/other|Other]]\n{BACKLINK_END}\n\n## 7. Glossary")
        store.objects[key] = body.encode()
    cloud.before_put = concurrent
    result = publish_page(cloud, "bucket", "wiki/concepts/a-concept.md", "# A concept\n[[sources/paper-one]]")
    assert result["errors"] == []
    body = cloud.objects[note].decode()
    assert "Concurrent scientific text." in body and "[[concepts/other|Other]]" in body
    assert "[[concepts/a-concept|A concept]]" in body
    assert body.count(BACKLINK_START) == 1


def test_catalog_cas_retry_keeps_other_new_page_and_repeated_publication_is_unique():
    catalog = "wiki/indexes/concepts.md"
    cloud = S3()
    def concurrent(store, key):
        if key != catalog:
            return
        store.before_put = None
        store.objects[key] = ("# Concepts\n\n<!-- byeori:catalog:start -->\n"
                              "- [[concepts/other|Other]]\n<!-- byeori:catalog:end -->\n").encode()
    cloud.before_put = concurrent
    result = publish_page(cloud, "bucket", "wiki/concepts/a-concept.md", "# A concept")
    assert result["errors"] == []
    body = cloud.objects[catalog].decode()
    assert body.count("[[concepts/a-concept|A concept]]") == 1
    assert "[[concepts/other|Other]]" in body


def test_missing_and_denied_targets_are_visible_without_losing_primary_page():
    cloud = S3({"wiki/sources/denied.md": source_note()})
    cloud.denied.update({"wiki/sources/denied.md", "wiki/indexes/concepts.md"})
    result = publish_page(cloud, "bucket", "wiki/concepts/a-concept.md",
                          "# A concept\n[[sources/denied]] [[sources/missing]]")
    assert result["key"] in cloud.objects
    assert result["connections"]["pages"] == [{"key": "wiki/sources/missing.md", "status": "missing", "relationship": "unresolved"}]
    assert len(result["connections"]["errors"]) == 2
    assert len(result["catalogs"]["errors"]) == 1
    assert len(result["errors"]) == 3


def test_removing_a_citation_removes_only_its_managed_reverse_link():
    note = "wiki/sources/paper-one.md"
    manual = "A manual link [[concepts/a-concept]].\n"
    cloud = S3({note: source_note() + manual})
    first = publish_page(cloud, "bucket", "wiki/concepts/a-concept.md", "# A concept\n[[sources/paper-one]]")
    publish_page(cloud, "bucket", first["key"], "# A concept\nRevised science.", expected_etag=first["etag"])
    body = cloud.objects[note].decode()
    assert "[[concepts/a-concept|A concept]]" not in body
    assert manual in body and "## 7. Glossary" in body


def test_rebuild_catalogs_keeps_newer_entries_and_preserves_manual_material():
    key = "wiki/indexes/concepts.md"
    cloud = S3({key: "---\nowner: team\n---\n# Concepts\nManual introduction.\n"
                    "<!-- byeori:catalog:start -->\n- [[concepts/newer|Newer]]\n<!-- byeori:catalog:end -->\n"})
    result = rebuild_catalogs(cloud, "bucket", [("wiki/concepts/current.md", "Current"),
                                                ("wiki/overviews/biology/index.md", "Biology"),
                                                ("wiki/indexes/concepts.md", "Concepts"),
                                                ("wiki/index.md", "Wiki"),
                                                ("wiki/questions/failed/old.md", "Failed")])
    assert result["errors"] == [] and result["documents"] == 2
    body = cloud.objects[key].decode()
    assert "[[concepts/newer|Newer]]" in body and "[[concepts/current|Current]]" in body
    assert "owner: team" in body and "Manual introduction." in body
    assert "[[overviews/biology/index|Biology]]" in cloud.objects["wiki/indexes/overviews.md"].decode()
    assert set(cloud.objects) >= {"wiki/index.md", *(f"wiki/indexes/{name}.md" for name in ("sources", "concepts", "overviews", "questions"))}


def test_rebuild_catalog_cas_retry_preserves_a_publication_after_the_snapshot():
    key = "wiki/indexes/concepts.md"
    cloud = S3({key: "# Concepts\n\n<!-- byeori:catalog:start -->\n<!-- byeori:catalog:end -->\n"})
    def concurrent(store, target):
        if target == key:
            store.before_put = None
            store.objects[key] = ("# Concepts\n\n<!-- byeori:catalog:start -->\n"
                                  "- [[concepts/newer|Newer]]\n<!-- byeori:catalog:end -->\n").encode()
    cloud.before_put = concurrent
    result = rebuild_catalogs(cloud, "bucket", [("wiki/concepts/current.md", "Current")])
    assert result["errors"] == []
    assert "[[concepts/newer|Newer]]" in cloud.objects[key].decode()
    assert "[[concepts/current|Current]]" in cloud.objects[key].decode()


def test_regenerated_source_note_preserves_incoming_links_and_original_metadata():
    note = "wiki/sources/paper-one.md"
    cloud = S3({note: source_note()})
    publish_page(cloud, "bucket", "wiki/concepts/a-concept.md", "# A concept\n[[sources/paper-one]]")
    replacement = source_note().replace('title: "Original paper"', 'title: "Model title"')
    replacement = replacement.replace("Original scientific text.", "Newly read scientific text.")
    result = publish_page(cloud, "bucket", note, replacement, expected_etag=cloud.etag(note))
    assert result["errors"] == []
    body = cloud.objects[note].decode()
    assert 'title: "Original paper"' in body and "Model title" not in body
    assert "Newly read scientific text." in body
    assert body.count("[[concepts/a-concept|A concept]]") == 1
    assert len(re.findall(r"^## ", body, re.M)) == 8


def test_existing_authored_reverse_link_does_not_get_a_second_managed_copy():
    note = "wiki/sources/paper-one.md"
    cloud = S3({note: source_note().replace("## 6. Related Work\n", "## 6. Related Work\n[[concepts/a-concept|Authored context]]\n")})
    result = publish_page(cloud, "bucket", "wiki/concepts/a-concept.md", "# A concept\n[[sources/paper-one]]")
    assert result["errors"] == []
    body = cloud.objects[note].decode()
    assert body.count("[[concepts/a-concept|") == 1
    assert "Authored context" in body and BACKLINK_START not in body


def test_regenerated_question_updates_model_metadata_and_keeps_original_created_date():
    key = "wiki/questions/which-evidence.md"
    cloud = S3({key: '---\ntitle: "Question"\ncreated: 2026-01-01\nupdated: 2026-01-01\ningest_model: "Old"\n---\n# Old answer\n'})
    new = '---\ntitle: "Question refined"\ncreated: 2026-09-20\nupdated: 2026-09-20\ningest_model: "New"\n---\n# New answer\n'
    result = publish_page(cloud, "bucket", key, new, expected_etag=cloud.etag(key))
    body = cloud.objects[key].decode()
    assert result["errors"] == []
    assert 'ingest_model: "New"' in body and 'title: "Question refined"' in body
    assert "created: 2026-01-01" in body and "updated: 2026-09-20" in body
    assert body.endswith("# New answer\n")


def test_late_publisher_reconciles_backlinks_against_the_newest_origin(monkeypatch):
    origin, old_target, new_target = "wiki/concepts/a.md", "wiki/sources/b.md", "wiki/sources/c.md"
    cloud = S3({old_target: "# B\n## Related pages\nEvidence.\n",
                new_target: "# C\n## Related pages\nEvidence.\n"})
    ready, resume = threading.Event(), threading.Event()
    original = connections._reciprocal
    def paused(*args, **kwargs):
        if threading.current_thread().name == "first-publisher":
            ready.set()
            assert resume.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(connections, "_reciprocal", paused)
    results, failures = [], []
    def first():
        try:
            results.append(publish_page(cloud, "bucket", origin, "# A\n[[sources/b]]", create_only=True))
        except Exception as exc:
            failures.append(exc)
    thread = threading.Thread(target=first, name="first-publisher", daemon=True)
    thread.start()
    try:
        assert ready.wait(5)
        second = publish_page(cloud, "bucket", origin, "# A\n[[sources/c]]", expected_etag=cloud.etag(origin))
    finally:
        resume.set()
        thread.join(5)
    assert not thread.is_alive() and failures == []
    assert second["errors"] == [] and results[0]["errors"] == []
    assert results[0]["connections"]["attempts"] == 2
    assert origin not in connections._entries(cloud.objects[old_target].decode(), connections.BACKLINK_BLOCK)
    assert origin in connections._entries(cloud.objects[new_target].decode(), connections.BACKLINK_BLOCK)


def test_continuously_changing_origin_reports_graph_conflict_without_rewriting_science(monkeypatch):
    key = "wiki/concepts/a.md"
    cloud = S3({"wiki/sources/b.md": "# B\nEvidence.\n"})
    original = connections._reciprocal
    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        cloud.objects[key] += b"\nAnother concurrent edit."
        return result
    monkeypatch.setattr(connections, "_reciprocal", changing)
    result = publish_page(cloud, "bucket", key, "# A\n[[sources/b]]")
    assert result["connections"]["converged"] is False
    assert result["connections"]["attempts"] == connections.MAX_ATTEMPTS
    assert any(error["key"] == key and "kept changing" in error["error"] for error in result["errors"])
    assert cloud.objects[key].count(b"Another concurrent edit.") == connections.MAX_ATTEMPTS


def test_deadline_stops_connections_and_catalogs_but_keeps_primary_saved_page():
    cloud = S3({"wiki/sources/b.md": "# B\nEvidence.\n"})
    def expired():
        raise TimeoutError("Publication time budget exhausted")
    result = publish_page(cloud, "bucket", "wiki/concepts/a.md", "# A\n[[sources/b]]", check_remaining=expired)
    assert cloud.objects["wiki/concepts/a.md"] == b"# A\n[[sources/b]]"
    assert cloud.objects["wiki/sources/b.md"] == b"# B\nEvidence.\n"
    assert result["connections"]["errors"] and len(result["catalogs"]["errors"]) == 5
    assert len(cloud.writes) == 1


def test_invalid_publish_target_and_legacy_note_keep_original_scientific_text():
    cloud = S3({"wiki/sources/old.md": "# Legacy note\n## Results\nEvidence."})
    with pytest.raises(ValueError):
        publish_page(cloud, "bucket", "wiki/../escape.md", "# Escape")
    result = publish_page(cloud, "bucket", "wiki/concepts/a-concept.md", "# Concept\n[[sources/old]]")
    assert result["connections"]["errors"] == []
    body = cloud.objects["wiki/sources/old.md"].decode()
    assert body.startswith("# Legacy note\n## Results\nEvidence.")
    assert "### Linked pages" in body and "[[concepts/a-concept|Concept]]" in body
    assert re.findall(r"^## .*", body, re.M) == ["## Results"]


@pytest.mark.parametrize("heading", ["## Related Work", "## Related Papers", "## Related pages"])
def test_legacy_related_headings_receive_backlinks_without_new_level_two_sections(heading):
    note = "wiki/sources/legacy.md"
    original = f"# Legacy\n{heading}\nOriginal related text.\n\n## Glossary\nOriginal glossary.\n"
    cloud = S3({note: original})
    result = publish_page(cloud, "bucket", "wiki/concepts/a-concept.md", "# A concept\n[[sources/legacy]]")
    assert result["errors"] == []
    body = cloud.objects[note].decode()
    assert body.index(heading) < body.index(BACKLINK_START) < body.index("## Glossary")
    assert "Original related text." in body and "Original glossary." in body
    assert re.findall(r"^## .*", body, re.M) == [heading, "## Glossary"]


def test_the_root_catalog_names_the_per_field_catalogs():
    """A browse path nothing links to is not a browse path: the root index is where a reader and
    the research agent both start (2026-09-23)."""
    from byeori.wiki_connections import rebuild_catalogs

    cloud = S3({})
    rebuild_catalogs(cloud, "bucket", [("wiki/sources/paper-one.md", "Paper one")])
    root = cloud.objects["wiki/index.md"].decode()
    assert "[[indexes/categories|Categories (one catalog per field)]]" in root
    assert "[[indexes/sources|Sources]]" in root
