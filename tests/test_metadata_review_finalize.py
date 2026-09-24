from __future__ import annotations

import io
import json
from copy import deepcopy

import pytest
from botocore.exceptions import ClientError

from byeori import metadata_review_finalize as finalizer


RUN = "finalize-test"
PREFIX = f"runs/metadata-review/{RUN}/"
FINAL = PREFIX + "finalization/v1/"
DOI = "10.1234/example"
TITLE = "A coherent study title"


def item(name="paper-a", *, issues=(), **metadata):
    return {"work_id": name, "input_sha256": finalizer._hash(name), "issues": list(issues),
            "metadata": {"doi": DOI, "title": TITLE, "pmid": "123", "pmid_openalex": "456",
                         "pmcid": "PMC111", "year": 2020, "openalex_year": 2023, **metadata}}


def records(**winning):
    return {"123": {"pmid": "123", "title": "An unrelated article", "dois": ["10.1234/other"],
                    "record_type": "article", "publication_types": ["Journal Article"]},
            "456": {"pmid": "456", "title": TITLE, "dois": [DOI], "pmcid": "PMC456",
                    "record_type": "article", "publication_types": ["Journal Article"],
                    "publication_dates": [{"kind": "journal", "year": "2023"}], **winning}}


def source(target, *, pdf=True, **updates):
    return {"work_id": target["work_id"], "input_sha256": target["input_sha256"],
            "decisions": {"pmid_disagreement": "pdf_confirmed"}, "proposals": {"pmid": "456"},
            "original": deepcopy(target["metadata"]),
            "pmc_authority": {"status": "ok", "requested_doi": DOI, "doi": DOI,
                              "pmid": "456", "pmcid": "PMC456"},
            "pdf": pdf if isinstance(pdf, dict) else
                   {"sha256_matches": pdf, "doi_present": pdf, "title_present": pdf}, **updates}


@pytest.mark.parametrize("pdf,classification", [(True, "pdf_confirmed"), (None, "metadata_confirmed")])
def test_existing_pmcid_disagreement_proposes_converter_identity_without_changing_inputs(pdf, classification):
    target = item(issues=["pmid_disagreement"])
    original, pubmed = source(target, pdf=pdf), records()
    before = deepcopy((target, original, pubmed))
    result = finalizer._finalize_row(target, original, pubmed)
    assert result["decisions"]["pmcid_disagreement"] == classification
    assert result["proposals"] == {"pmcid": "PMC456"}
    assert result["original"]["pmcid"] == "PMC111"
    assert result["evidence"]["pubmed_identity"]["dois"] == [DOI]
    assert result["evidence"]["pubmed_identity"]["title"] == TITLE
    assert result["canonical_writes"] is False
    assert (target, original, pubmed) == before


@pytest.mark.parametrize("conversion", [None, {"status": "error", "error": "missing_record"},
    {"status": "mismatch", "doi": DOI, "pmcid": "PMC456"},
    {"status": "ok", "requested_doi": "10.1234/other", "doi": DOI, "pmid": "456", "pmcid": "PMC456"},
    {"status": "ok", "requested_doi": DOI, "doi": "10.1234/other", "pmid": "456", "pmcid": "PMC456"}])
def test_missing_or_mismatched_converter_cannot_establish_an_existing_pmcid_difference(conversion):
    target = item()
    result = finalizer._finalize_row(target, source(target, pmc_authority=conversion), records())
    assert result["proposals"] == {}
    assert "pmcid_disagreement" not in result["decisions"]


@pytest.mark.parametrize("metadata", [{"pmcid": None}, {"pmcid": "PMC456"}])
def test_finalizer_does_not_repropose_missing_or_already_equal_pmcid(metadata):
    target = item(**metadata)
    result = finalizer._finalize_row(target, source(target), records())
    assert result["proposals"] == {}
    assert "pmcid_disagreement" not in result["decisions"]


@pytest.mark.parametrize("stored_pmcid", [
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC456/",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC456",
    "pmcid:PMC456",
])
def test_same_pmcid_in_url_or_prefix_form_is_not_a_disagreement_or_correction(stored_pmcid):
    target = item(pmcid=stored_pmcid)
    result = finalizer._finalize_row(target, source(target), records())
    assert result["pmcid_comparison"] == "same"
    assert result["proposals"] == {}
    assert "pmcid_disagreement" not in result["decisions"]
    assert result["original"]["pmcid"] == stored_pmcid


@pytest.mark.parametrize("change,reason", [
    ({"title": "Another coherent study title"}, "pubmed_title_disagrees"),
    ({"dois": ["10.1234/citation-only"]}, "pubmed_doi_identity_not_unique"),
    ({"record_type": "book"}, "pubmed_publication_or_identity_requires_review"),
    ({"publication_types": ["Published Erratum"]}, "pubmed_publication_or_identity_requires_review"),
    ({"publication_relationships": ["CommentIn"]}, "pubmed_publication_or_identity_requires_review"),
    ({"identity_errors": ["multiple_pmcids"]}, "pubmed_publication_or_identity_requires_review"),
    ({"pmcid": "PMC999"}, "pubmed_pmc_pmcid_conflict"),
    ({"pmid": "789"}, "cached_pubmed_candidate_missing"),
])
def test_pubmed_identity_title_and_publication_holds_block_both_proposals(change, reason):
    target = item(issues=["year_difference_over_one"])
    result = finalizer._finalize_row(target, source(target), records(**change))
    assert result["proposals"] == {}
    assert reason in result["reasons"]["pmcid_disagreement"]
    assert reason in result["reasons"]["year_difference_over_one"]


def test_both_pubmed_candidates_with_own_doi_remain_ambiguous_even_if_one_title_matches():
    target = item()
    pubmed = records()
    pubmed["123"]["dois"] = [DOI]
    result = finalizer._finalize_row(target, source(target), pubmed)
    assert result["proposals"] == {}
    assert result["reasons"]["pmcid_disagreement"] == ["pubmed_doi_identity_not_unique"]


@pytest.mark.parametrize("conversion_pmid", [None, "789", "123"])
def test_converter_pmid_must_be_cached_and_same_coherent_identity(conversion_pmid):
    target = item()
    original = source(target)
    original["pmc_authority"]["pmid"] = conversion_pmid
    result = finalizer._finalize_row(target, original, records())
    assert result["proposals"] == {}
    assert result["decisions"]["pmcid_disagreement"] == "review_required"


@pytest.mark.parametrize("decisions,pdf", [
    ({"pmid_disagreement": "review_required"}, {}),
    ({"missing_pmcid": "authority_conflict"}, {}),
    ({"missing_references": "identity_conflict"}, {}),
    ({"pmid_disagreement": "pdf_identity_conflict"}, {}),
    ({}, {"sha256_matches": False}),
    ({}, {"sha256_matches": True, "doi_present": False, "doi_candidates": ["10.1234/other"]}),
])
def test_prior_conflicts_or_unresolved_pmid_block_new_proposals(decisions, pdf):
    target = item(issues=["year_difference_over_one"])
    result = finalizer._finalize_row(target, source(target, decisions=decisions, pdf=pdf), records())
    assert result["proposals"] == {}
    assert set(result["decisions"].values()) == {"review_required"}


def test_pmid_issue_without_resolved_source_decision_is_a_hold():
    target = item(issues=["pmid_disagreement"])
    result = finalizer._finalize_row(target, source(target, decisions={}), records())
    assert result["proposals"] == {}
    assert "original_pmid_unresolved" in result["reasons"]["pmcid_disagreement"]


@pytest.mark.parametrize("kind", sorted(finalizer.PUBLICATION_DATES))
def test_only_unambiguous_publication_dates_support_metadata_year_proposals(kind):
    target = item(issues=["year_difference_over_one"])
    result = finalizer._finalize_row(target, source(target), records(publication_dates=[{"kind": kind, "year": "2023"}]))
    assert result["proposals"]["year"] == 2023
    assert result["decisions"]["year_difference_over_one"] == "metadata_confirmed"
    assert result["evidence"]["year_difference_over_one"]["pdf_identity_verified"] is True
    assert result["evidence"]["year_difference_over_one"]["source"] == "PubMed publication dates"


@pytest.mark.parametrize("kind", ["history_received", "history_accepted", "history_pubmed", "history_medline",
    "history_entrez", "history_pmc", "history_pmcr", "history_ecollection", "article_unknown", "book"])
def test_indexing_receipt_acceptance_pmc_release_and_unsupported_dates_cannot_propose_year(kind):
    target = item(issues=["year_difference_over_one"], pmcid=None)
    pdf = {"sha256_matches": True, "doi_present": True, "title_present": True, "years": [2023]}
    result = finalizer._finalize_row(target, source(target, pdf=pdf), records(
        publication_year=2023, publication_dates=[{"kind": kind, "year": "2023"}]))
    assert result["proposals"] == {}
    assert result["decisions"]["year_difference_over_one"] == "review_required"


@pytest.mark.parametrize("dates", [[{"kind": "journal", "year": "2023"}, {"kind": "article_electronic", "year": "2022"}],
    [{"kind": "journal", "medline_date": "2022 Dec-2023 Jan"}], [],
    [{"kind": "journal", "year": "2021"}]])
def test_conflicting_missing_or_openalex_disagreeing_publication_years_hold(dates):
    target = item(issues=["year_difference_over_one"], pmcid=None)
    result = finalizer._finalize_row(target, source(target), records(publication_dates=dates))
    assert result["proposals"] == {}


def test_unambiguous_medline_publication_date_ignores_later_indexing_date():
    target = item(issues=["year_difference_over_one"], pmcid=None)
    result = finalizer._finalize_row(target, source(target), records(publication_dates=[
        {"kind": "journal", "medline_date": "2023 Jan-Feb"}, {"kind": "history_pubmed", "year": "2024"}]))
    assert result["proposals"] == {"year": 2023}


@pytest.mark.parametrize("original,oa", [(2022, 2023), (2023, 2023), (None, 2023), (True, 2023), (2020, "unknown")])
def test_large_year_issue_still_requires_valid_measured_difference(original, oa):
    target = item(issues=["year_difference_over_one"], year=original, openalex_year=oa, pmcid=None)
    result = finalizer._finalize_row(target, source(target), records())
    assert result["proposals"] == {}


def test_year_only_case_still_holds_for_converter_identity_mismatch():
    target = item(issues=["year_difference_over_one"], pmcid=None)
    result = finalizer._finalize_row(target, source(target, decisions={}, pmc_authority={"status": "mismatch"}), records())
    assert result["proposals"] == {}
    assert "original_pmc_identity_conflict" in result["reasons"]["year_difference_over_one"]


class MemoryS3:
    """Allow finalization-only conditional writes; all reads remain in memory."""
    def __init__(self):
        self.objects, self.gets, self.puts = {}, [], []
        self.before_put = None

    def seed(self, key, value):
        self.objects[key] = finalizer._encode(value)

    def get_object(self, **kwargs):
        key = kwargs["Key"]
        assert key.startswith(PREFIX)
        self.gets.append(key)
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[key]), "ETag": '"opaque-etag"'}

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        assert key.startswith(FINAL), "Only separate finalization objects may be written"
        assert kwargs.get("IfNoneMatch") == "*", "Finalization objects are immutable"
        if self.before_put:
            callback, self.before_put = self.before_put, None
            callback(self, kwargs)
        if key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.puts.append(key)
        self.objects[key] = kwargs["Body"]
        return {"ETag": '"new-opaque-etag"'}


def seed_run(s3, items=None, *, sources=None, pubmed=None):
    items = [item(issues=["year_difference_over_one"])] if items is None else items
    plan = {"schema_version": 1, "run_id": RUN, "items": items}
    digest = finalizer._hash(plan)
    s3.seed(PREFIX + "plan.json", plan)
    s3.seed(PREFIX + "progress.json", {"run_id": RUN, "done": True, "next_index": len(items), "plan_sha256": digest})
    for start in range(0, len(items), 50):
        selected = items[start:start + 50]
        s3.seed(PREFIX + f"results/{start:06d}.json", {"run_id": RUN, "plan_sha256": digest, "start": start,
                "rows": (sources[start:start + 50] if sources is not None else [source(target) for target in selected])})
        ids = sorted({finalizer._pmid(target["metadata"].get(field)) for target in selected
                      for field in ("pmid", "pmid_openalex")} - {None})
        if ids:
            s3.seed(PREFIX + "authority/pubmed/" + finalizer._hash(ids) + ".json", {
                "provider": "pubmed", "requested_ids": ids, "records": records() if pubmed is None else pubmed})
    return plan


def run(s3):
    return finalizer.finalize_review(s3=s3, bucket="bucket", run_id=RUN)


def test_finalization_writes_only_immutable_separate_results_and_resume_returns_summary():
    s3 = MemoryS3()
    seed_run(s3)
    originals = deepcopy(s3.objects)
    result = run(s3)
    assert result["proposal_counts"] == {"pmcid": 1, "year": 1}
    assert result["existing_pmcid_disagreements"] == 1
    assert result["pmcid_pairs_checked"] == 1
    assert result["decision_counts"] == {"pmcid_disagreement:pdf_confirmed": 1,
                                          "year_difference_over_one:metadata_confirmed": 1}
    assert result["processed_rows"] == 1 and result["canonical_writes"] is False
    assert s3.puts == [FINAL + "results/000000.json", FINAL + "summary.json"]
    assert {key: s3.objects[key] for key in originals} == originals
    s3.puts.clear()
    s3.gets.clear()
    assert run(s3) == result and not s3.puts
    assert s3.gets == [PREFIX + "progress.json", PREFIX + "plan.json", FINAL + "summary.json"]


def test_multiple_batches_use_normalized_sorted_exact_ids_and_bounded_samples():
    items = [item("paper-" + str(i) + "😀" * 1000, issues=["year_difference_over_one"],
                  pmid="PMID: 123", pmid_openalex="https://pubmed.ncbi.nlm.nih.gov/456/") for i in range(101)]
    s3 = MemoryS3()
    seed_run(s3, items)
    result = run(s3)
    assert result["proposal_counts"] == {"pmcid": 101, "year": 101}
    assert result["processed_rows"] == 101
    assert all(len(examples) == 3 for examples in result["samples"].values())
    assert len(json.dumps(result, ensure_ascii=True).encode()) <= 131_072
    assert s3.puts == [FINAL + f"results/{start:06d}.json" for start in (0, 50, 100)] + [FINAL + "summary.json"]
    cachekey = PREFIX + "authority/pubmed/" + finalizer._hash(["123", "456"]) + ".json"
    assert s3.gets.count(cachekey) == 3


@pytest.mark.parametrize("updates", [{"done": False}, {"done": 1}, {"run_id": "other"},
    {"plan_sha256": "different"}, {"next_index": 0}])
def test_incomplete_or_changed_progress_refuses_all_writes(updates):
    s3 = MemoryS3()
    seed_run(s3)
    key = PREFIX + "progress.json"
    state = json.loads(s3.objects[key])
    s3.seed(key, {**state, **updates})
    with pytest.raises(ValueError):
        run(s3)
    assert not s3.puts


@pytest.mark.parametrize("mutation", ["run", "plan", "start", "count", "work_id", "input_sha256", "order"])
def test_mismatched_source_batch_is_rejected_before_finalization_writes(mutation):
    s3 = MemoryS3()
    seed_run(s3, [item("a"), item("b")])
    key = PREFIX + "results/000000.json"
    batch = json.loads(s3.objects[key])
    if mutation in ("run", "plan", "start"):
        batch[{"run": "run_id", "plan": "plan_sha256", "start": "start"}[mutation]] = "wrong"
    elif mutation == "count":
        batch["rows"].pop()
    elif mutation == "order":
        batch["rows"].reverse()
    else:
        batch["rows"][0][mutation] = "wrong"
    s3.seed(key, batch)
    with pytest.raises(ValueError):
        run(s3)
    assert not s3.puts


@pytest.mark.parametrize("mutation", ["missing", "provider", "requested_ids", "records", "unrequested"])
def test_missing_or_incompatible_authority_cache_cannot_be_replaced_by_network(mutation):
    s3 = MemoryS3()
    seed_run(s3)
    key = PREFIX + "authority/pubmed/" + finalizer._hash(["123", "456"]) + ".json"
    cache = json.loads(s3.objects[key])
    if mutation == "missing":
        del s3.objects[key]
    else:
        if mutation == "unrequested":
            cache["records"]["789"] = {"pmid": "789"}
        else:
            cache[mutation] = {"provider": "other", "requested_ids": ["123"], "records": []}[mutation]
        s3.seed(key, cache)
    with pytest.raises(ValueError, match="PubMed cache"):
        run(s3)
    assert not s3.puts


def test_empty_plan_completes_without_results_or_authority_reads():
    s3 = MemoryS3()
    seed_run(s3, [])
    result = run(s3)
    assert result["processed_rows"] == 0
    assert result["proposal_counts"] == {"pmcid": 0, "year": 0}
    assert s3.puts == [FINAL + "summary.json"]


def test_resume_reuses_immutable_batch_without_authority_cache():
    s3 = MemoryS3()
    seed_run(s3)
    expected = run(s3)
    del s3.objects[FINAL + "summary.json"]
    del s3.objects[PREFIX + "authority/pubmed/" + finalizer._hash(["123", "456"]) + ".json"]
    s3.puts.clear()
    result = run(s3)
    assert result["proposal_counts"] == expected["proposal_counts"]
    assert result["decision_counts"] == expected["decision_counts"]
    assert s3.puts == [FINAL + "summary.json"]


@pytest.mark.parametrize("mutation", ["source_sha256", "finalization_version", "input_sha256", "canonical_writes", "decision", "proposal", "year_type", "pmcid_type", "held_proposal"])
def test_cached_finalization_batch_must_match_source_plan_and_supported_proposals(mutation):
    s3 = MemoryS3()
    seed_run(s3)
    run(s3)
    del s3.objects[FINAL + "summary.json"]
    key = FINAL + "results/000000.json"
    batch = json.loads(s3.objects[key])
    if mutation in ("source_sha256", "finalization_version"):
        batch[mutation] = "wrong"
    elif mutation == "decision":
        batch["rows"][0]["decisions"]["year_difference_over_one"] = "pdf_confirmed"
    elif mutation == "proposal":
        batch["rows"][0]["proposals"]["doi"] = DOI
    elif mutation == "year_type":
        batch["rows"][0]["proposals"]["year"] = "2023"
    elif mutation == "pmcid_type":
        batch["rows"][0]["proposals"]["pmcid"] = "invalid"
    elif mutation == "held_proposal":
        batch["rows"][0]["decisions"]["year_difference_over_one"] = "review_required"
    else:
        batch["rows"][0][mutation] = "wrong"
    s3.seed(key, batch)
    s3.puts.clear()
    with pytest.raises(ValueError):
        run(s3)
    assert not s3.puts


@pytest.mark.parametrize("updates", [{"run_id": "wrong"}, {"plan_sha256": "wrong"},
    {"processed_rows": 9}, {"status": "failed"}, {"canonical_writes": True},
    {"samples": {"pmcid_disagreement:pdf_confirmed": [{}, {}, {}, {}]}}])
def test_cached_summary_must_match_completed_plan_and_sample_bound(updates):
    s3 = MemoryS3()
    seed_run(s3)
    summary = run(s3)
    s3.seed(FINAL + "summary.json", {**summary, **updates})
    s3.puts.clear()
    with pytest.raises(ValueError):
        run(s3)
    assert not s3.puts


def test_concurrent_immutable_batch_winner_is_read_and_validated():
    s3 = MemoryS3()
    seed_run(s3)
    s3.before_put = lambda store, kwargs: store.seed(kwargs["Key"], json.loads(kwargs["Body"]))
    result = run(s3)
    assert result["proposal_counts"] == {"pmcid": 1, "year": 1}
    assert s3.puts == [FINAL + "summary.json"]


def test_concurrent_immutable_batch_with_wrong_identity_is_rejected():
    s3 = MemoryS3()
    seed_run(s3)
    def corrupt_winner(store, kwargs):
        value = json.loads(kwargs["Body"])
        value["rows"][0]["input_sha256"] = "wrong"
        store.seed(kwargs["Key"], value)
    s3.before_put = corrupt_winner
    with pytest.raises(ValueError, match="row identities"):
        run(s3)
    assert FINAL + "summary.json" not in s3.objects


def test_cached_summary_cannot_exceed_serialized_response_bound():
    s3 = MemoryS3()
    seed_run(s3)
    summary = run(s3)
    s3.seed(FINAL + "summary.json", {**summary, "extra": "😀" * 131_072})
    s3.puts.clear()
    with pytest.raises(ValueError, match="bounded response size"):
        run(s3)
    assert not s3.puts


def test_failed_summary_write_resumes_from_completed_batch():
    s3 = MemoryS3()
    seed_run(s3)
    original_put = s3.put_object
    def fail_summary(**kwargs):
        if kwargs["Key"] == FINAL + "summary.json":
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
        return original_put(**kwargs)
    s3.put_object = fail_summary
    with pytest.raises(ClientError):
        run(s3)
    assert FINAL + "results/000000.json" in s3.objects
    assert FINAL + "summary.json" not in s3.objects
    s3.put_object = original_put
    s3.puts.clear()
    assert run(s3)["status"] == "complete"
    assert s3.puts == [FINAL + "summary.json"]


def test_placeholder_groups_are_distinct_from_valid_doi_duplicate_groups():
    items = [item(f"{raw}-{i}", issues=["duplicate_stem_doi"], doi=raw, pmid=None, pmid_openalex=None)
             for raw, count in [("null", 5), ("n-a", 5), ("not-in-pdf", 4), ("unknown", 2),
                                (DOI, 3), ("broken-id", 2)] for i in range(count)]
    s3 = MemoryS3()
    seed_run(s3, items)
    result = run(s3)
    counts = result["duplicate_identifier_groups"]
    assert counts["placeholder_groups"] == 4 and counts["placeholder_rows"] == 16
    assert counts["valid_doi_groups"] == 1 and counts["valid_doi_rows"] == 3
    assert counts["other_invalid_groups"] == 1 and counts["other_invalid_rows"] == 2
    assert counts["placeholders"]["not-in-pdf"] == {"groups": 1, "rows": 4}
    assert result["decision_counts"]["duplicate_stem_doi:placeholder_repetition"] == 16
    assert result["decision_counts"]["duplicate_stem_doi:valid_doi_duplicate"] == 3
    assert result["decision_counts"]["duplicate_stem_doi:invalid_identifier_repetition"] == 2
    assert result["proposal_counts"] == {"pmcid": 0, "year": 0}
