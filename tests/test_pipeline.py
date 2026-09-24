from __future__ import annotations

from pathlib import Path

import pytest

from byeori.catalog import Catalog
from byeori.config import Settings
from byeori.pipeline import select_candidates, tag_slug


pytestmark = pytest.mark.usefixtures("cloud_catalog")

def test_select_candidates_keeps_allowlisted_hosted_articles_only(tmp_path: Path) -> None:
    settings = Settings(root=tmp_path, data_dir=tmp_path / "data", state_dir=tmp_path / "state",
                        openalex_api_key=None, aws_region="us-east-1", aws_bucket="bucket", aws_table="table")
    settings.ensure_directories()
    base = {"is_open_access": True, "oa_license": "cc-by", "openalex_pdf_url": "https://x/p.pdf",
            "grobid_xml_url": "https://x/g.xml", "publication_date": "2020-01-01", "type": "article",
            "authors": ["A B"], "publication_year": 2020,
            "corpus": {"id": "autism-genomics", "fulltext_eligible": True, "journal_verdict": "include", "tags": ["De novo"]}}
    works = [
        {**base, "work_id": "W1", "title": "Article in Nature", "source": "Nature", "cited_by_count": 5},
        {**base, "work_id": "W2", "title": "A review", "source": "Nature", "type": "review", "cited_by_count": 50},
        {**base, "work_id": "W3", "title": "Not hosted", "source": "Nature", "grobid_xml_url": None,
         "corpus": {**base["corpus"], "fulltext_eligible": False}},
        {**base, "work_id": "W4", "title": "Excluded journal", "source": "Scientific Reports",
         "corpus": {**base["corpus"], "journal_verdict": "exclude"}},
    ]
    with Catalog(settings) as catalog:
        for work in works:
            catalog.save_candidate(work)
    selected = select_candidates(settings)
    assert [item["work_id"] for item in selected] == ["W1"]
    assert select_candidates(settings, types=("article", "review"))[0]["work_id"] == "W2"


def test_tag_slug() -> None:
    assert tag_slug("CNV/SV") == "cnv-sv" and tag_slug("GWAS/polygenic") == "gwas-polygenic"
    assert tag_slug("De novo") == "de-novo"


def test_pipeline_rebuilds_the_aws_index_at_the_end() -> None:
    """Nothing outside the pipeline has to be scheduled to keep search current."""
    source = Path(__file__).parents[1].joinpath("src/byeori/pipeline.py").read_text()
    assert "AwsStore(settings).build_wiki_index()" in source
    assert 'not skip_index' in source and '"step": "build_index"' in source


def test_failed_stems_separates_a_failed_attempt_from_one_not_yet_reached(monkeypatch) -> None:
    """A paper still queued has no status at all; retrying it is the ordinary run's job, not a retry's."""
    from byeori import pipeline

    rows = [
        {"work_id": "abbad-2025-advances", "source_note_status": "source_failed",
         "source_note_problems": ["note is too short to be a full evidence note"], "category": "renal-cell-biology"},
        {"work_id": "anderson-2020-single-cell", "source_note_status": "source_ready", "category": "asd-ndd"},
        {"work_id": "zhang-2024-not-yet-started", "category": "asd-ndd"},          # queued, never attempted
    ]
    from byeori.corpus_ops import failure_rows
    for row in rows:
        row["ingest_status"] = "fulltext_ready"
    class Store:
        def pipeline_failures(self, **kw):
            return {"papers": failure_rows(rows), "next_offset": None}
    monkeypatch.setattr(pipeline, "AwsStore", lambda settings: Store())

    notes = pipeline.failed_stems(None)
    assert [r["stem"] for r in notes] == ["abbad-2025-advances"]
    assert notes[0]["problems"] == ["note is too short to be a full evidence note"]


def test_only_failed_selects_from_the_failures_not_the_whole_table() -> None:
    source = Path(__file__).parents[1].joinpath("src/byeori/pipeline.py").read_text()
    assert "elif only_failed:" in source
    assert 'run["previous_failures"] = run_failures' in source


def test_failed_stems_follows_aws_pages_without_catalogue_scan(monkeypatch) -> None:
    from byeori import pipeline
    calls = []
    pages = [{"papers": [{"stem": "first", "stage": "source_note"}], "next_offset": 100},
             {"papers": [{"stem": "last", "stage": "source_note"}], "next_offset": None}]

    class Store:
        def pipeline_failures(self, **request):
            calls.append(request)
            return pages.pop(0)

        def scan(self, **request):
            pytest.fail("Client must not select failures from a downloaded catalogue")

    monkeypatch.setattr(pipeline, "AwsStore", lambda settings: Store())
    assert [row["stem"] for row in pipeline.failed_stems(None)] == ["first", "last"]
    assert calls == [{"verbose": True, "offset": 0}, {"verbose": True, "offset": 100}]


def test_a_declined_note_attempt_gets_its_own_ledger_row(tmp_path) -> None:
    import json
    from types import SimpleNamespace
    from byeori.pipeline import _bedrock_step
    settings = SimpleNamespace(state_dir=tmp_path)
    result = {"status": "source_ready", "model_id": "global.anthropic.claude-opus-5", "seconds": 90.0,
              "usage": {"inputTokens": 30000, "outputTokens": 8000},
              "filtered_attempt": {"model_id": "global.anthropic.claude-opus-5-5", "stop_reason": "content_filtered",
                                   "usage": {"inputTokens": 30000, "outputTokens": 6}, "seconds": 3.0}}
    _bedrock_step(settings, "source_note", "paper-one", result, 0.0)
    rows = [json.loads(line) for line in (tmp_path / "cost-ledger.jsonl").read_text().splitlines()]
    assert [(r["step"], r["model_id"], r["status"]) for r in rows] == [
        ("source_note_filtered", "global.anthropic.claude-opus-5-5", "content_filtered"),
        ("source_note", "global.anthropic.claude-opus-5", "source_ready")]
    assert rows[0]["estimated_usd"] == 0.12012 and rows[0]["input_tokens"] == 30000
