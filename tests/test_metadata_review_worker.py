from __future__ import annotations

import hashlib
import io
import json
from copy import deepcopy
from datetime import UTC, datetime
from email.utils import format_datetime

import pytest
from botocore.exceptions import ClientError

from byeori import metadata_review_worker as worker


PREFIX = "runs/metadata-review/review-test/"


def item(name="paper-a", *, issues=(), **metadata):
    return {"work_id": name, "input_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "issues": list(issues), "priority": 1, "metadata": {
                "title": "A coherent study title", "doi": "10.1234/example", "pmid": "123",
                "pmid_openalex": "456", "openalex_id": "W123", **metadata}}


def pubmed_record(pmid, doi="10.1234/example", title="A coherent study title"):
    return {"pmid": pmid, "title": title, "dois": [doi], "record_type": "article",
            "publication_types": ["Journal Article"], "publication_relationships": [], "identity_errors": []}


def pdf_flags(**updates):
    return {"checked": True, "sha256_matches": True, "doi_present": True, "title_present": True, **updates}


@pytest.mark.parametrize("winner,source", [("123", "catalogue"), ("456", "openalex")])
def test_pmid_choice_keeps_originals_and_proposes_only_confirmed_identity(winner, source):
    target = item(issues=["pmid_disagreement"])
    records = {pmid: pubmed_record(pmid, "10.1234/example" if pmid == winner else "10.1234/other")
               for pmid in ("123", "456")}
    before = deepcopy((target, records))
    result = worker.classify_row(target, records, {}, None, pdf_flags())
    assert result["decisions"]["pmid_disagreement"] == "pdf_confirmed"
    assert result["pmid_assessment"]["source"] == source
    assert result["proposals"]["pmid"] == winner
    assert result["original"]["pmid"] == "123"
    assert (target, records) == before


def test_pmc_pmid_disagreement_removes_proposal_and_reports_authority_conflict():
    target = item(issues=["pmid_disagreement", "pmcid_disagreement"], pmcid="PMC123")
    records = {"123": pubmed_record("123", "10.1234/wrong"), "456": pubmed_record("456")}
    pmc = {"10.1234/example": {"status": "ok", "doi": "10.1234/example", "pmid": "789", "pmcid": "PMC789"}}
    result = worker.classify_row(target, records, pmc, None, pdf_flags())
    assert result["decisions"]["pmid_disagreement"] == "authority_conflict"
    assert "pmid" not in result["proposals"]
    assert result["supplements"]["pmid_ncbi"] == "789"
    assert "pmcid" not in result["proposals"]


@pytest.mark.parametrize("conversion", [None, {"status": "error", "error": "missing_record"},
    {"status": "mismatch", "doi": "10.1234/example", "pmcid": "PMC789"},
    {"status": "ok", "doi": "10.1234/unrelated", "pmcid": "PMC789"}])
def test_legacy_missing_pmcid_is_allowed_without_converter_supplements(conversion):
    pmc = {"10.1234/example": conversion} if conversion is not None else {}
    result = worker.classify_row(item(issues=["missing_pmcid"]), {}, pmc, None, {})
    assert result["proposals"] == {} and result["supplements"] == {}
    assert result["decisions"]["missing_pmcid"] == "allowed_empty"
    assert "pmc_authority" not in result
    assert result["original"]["pmid"] == "123"


@pytest.mark.parametrize("issue,doi", [("doi_missing", ""), ("doi_invalid", "arxiv:1234")])
def test_pdf_doi_candidate_requires_hash_and_title_then_stays_a_proposal(issue, doi):
    target = item(issues=[issue], doi=doi)
    pdf = pdf_flags(doi_candidates=["10.1234/recovered"], doi_present=False)
    result = worker.classify_row(target, {}, {}, None, pdf)
    assert result["proposals"]["doi"] == "10.1234/recovered"
    assert result["decisions"][issue] == "pdf_doi_candidate"
    assert result["original"]["doi"] == doi


@pytest.mark.parametrize("updates", [{"sha256_matches": False}, {"title_present": False},
    {"sha256_matches": 1}, {"doi_candidates": []},
    {"doi_candidates": ["10.1234/one", "10.1234/two"]}, {"doi_candidates": ["not-a-doi"]}])
def test_pdf_doi_without_unique_valid_identity_or_literal_hash_title_flags_remains_review(updates):
    pdf = pdf_flags(doi_candidates=["10.1234/recovered"], **{k: v for k, v in updates.items() if k != "doi_candidates"})
    pdf.update(updates)
    result = worker.classify_row(item(issues=["doi_missing"], doi=""), {}, {}, None, pdf)
    assert "doi" not in result["proposals"]
    assert result["decisions"]["doi_missing"] == "review_required"


def oa_work(**updates):
    return {"id": "W123", "doi": "10.1234/example", "referenced_works": ["https://openalex.org/W456"],
            "authorships": [{"author": {"display_name": f"Author {i}"}} for i in range(45)], **updates}


@pytest.mark.parametrize("updates,existing", [({"id": "W999"}, "W123"),
    ({"doi": "10.1234/other"}, "W123"), ({"_ambiguous_openalex_ids": ["W123", "W999"]}, "W123"),
    ({}, ""), ({"id": "malformed"}, "W123")])
def test_no_openalex_supplement_for_id_doi_ambiguity_or_missing_identity_conflicts(updates, existing):
    result = worker.classify_row(item(issues=["missing_references", "authors_at_cap"], openalex_id=existing),
                                 {}, {}, oa_work(**updates), {})
    assert not result["supplements"]
    assert result["decisions"] == {"missing_references": "identity_conflict", "authors_at_cap": "identity_conflict"}


@pytest.mark.parametrize("count,decision,cap", [(30, "no_expansion", False),
    (45, "supplement_available", False), (100, "expanded_upstream_cap_possible", True),
    (101, "expanded_upstream_cap_possible", True)])
def test_author_supplement_retains_upstream_cap_warning(count, decision, cap):
    work = oa_work(authorships=[{"author": {"display_name": f"Author {i}"}} for i in range(count)])
    result = worker.classify_row(item(issues=["authors_at_cap", "missing_references"]), {}, {}, work, {})
    assert len(result["supplements"]["openalex_authors"]) == count
    assert result["supplements"]["openalex_authors_upstream_cap_possible"] is cap
    assert result["decisions"]["authors_at_cap"] == decision
    assert result["supplements"]["openalex_referenced_works"] == ["W456"]


class MemoryS3:
    """S3 fake with opaque ETags and real conditional-write behavior, no other writes."""
    def __init__(self):
        self.objects = {}
        self.etags = {}
        self.generation = 0
        self.gets = []
        self.puts = []
        self.before_put = None

    def seed(self, key, value):
        self.generation += 1
        self.objects[key] = worker._encode(value)
        self.etags[key] = f'"opaque-{self.generation}"'

    def get_object(self, **kwargs):
        self.gets.append(kwargs.copy())
        key = kwargs["Key"]
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[key]), "ETag": self.etags[key]}

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        assert key.startswith(PREFIX), "Runner must write only its S3 run prefix"
        assert len(self.puts) < 200, "A malformed cached batch must not cause an infinite progress loop"
        if self.before_put is not None:
            action, self.before_put = self.before_put, None
            action(self, kwargs)
        if ((kwargs.get("IfNoneMatch") == "*" and key in self.objects) or
                ("IfMatch" in kwargs and self.etags.get(key) != kwargs["IfMatch"])):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.puts.append(kwargs.copy())
        self.seed(key, json.loads(kwargs["Body"]))
        return {"ETag": self.etags[key]}

    def forbidden(self, **kwargs):
        pytest.fail("No DynamoDB, canonical metadata, or destructive S3 write is allowed")

    update_item = put_item = delete_item = delete_object = delete_objects = forbidden


def make_plan(items=None):
    return {"schema_version": 1, "run_id": "review-test", "items": items if items is not None else [item()],
            "created_at": "2026-09-20T00:00:00+00:00", "catalogued_stems": 1, "scan_pages": 1,
            "duplicate_doi_groups": 0}


def progress(**updates):
    return {"run_id": "review-test", "started_at": "2026-09-20T00:00:00+00:00", "next_index": 0,
            "requests": {}, "decisions": {}, "samples": {}, "papers": 0, "pdf_checked": 0,
            "errors": 0, "done": False, "lease_until": 0, **updates}


def result_row(target):
    return {"work_id": target["work_id"], "input_sha256": target["input_sha256"],
            "decisions": {"test": "review_required"}, "proposals": {}, "supplements": {},
            "original": deepcopy(target["metadata"]), "pdf": {"checked": False}}


def seed_run(s3, plan=None, state=None):
    plan = plan or make_plan()
    s3.seed(PREFIX + "plan.json", plan)
    if state is not None:
        s3.seed(PREFIX + "progress.json", state)
    return plan, hashlib.sha256(worker._encode(plan)).hexdigest()


def runner(s3, attempt="attempt-1"):
    return worker.ReviewRunner(s3=s3, bucket="bucket", run_id="review-test", attempt_id=attempt)


def no_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(worker.ReviewRunner, "wait", lambda self, seconds: waits.append(seconds))
    return waits


def test_done_resume_reads_existing_progress_without_acquiring_lease_or_work(monkeypatch):
    s3 = MemoryS3()
    s3.seed(PREFIX + "progress.json", progress(done=True, status="complete", next_index=1, papers=1,
                                              lease_until=10**12, lease_owner="finished-owner"))
    monkeypatch.setattr(worker.ReviewRunner, "batch", lambda *a: pytest.fail("Completed runs must not execute batches"))
    result = runner(s3).run()
    assert result["status"] == "complete" and result["papers"] == 1
    assert s3.puts == []
    assert {entry["Key"] for entry in s3.gets} == {PREFIX + "progress.json"}


def test_resume_rejects_changed_plan_hash_and_releases_failed_lease(monkeypatch):
    no_wait(monkeypatch)
    s3 = MemoryS3()
    seed_run(s3, state=progress(plan_sha256="different-prior-plan"))
    current = runner(s3)
    with pytest.raises(ValueError, match="(?i)plan"):
        current.run()
    state = json.loads(s3.objects[PREFIX + "progress.json"])
    assert state["status"] == "failed" and state["lease_until"] == 0 and not state["done"]
    assert state["plan_sha256"] == "different-prior-plan"


@pytest.mark.parametrize("defect", ["empty", "extra", "work_id", "input_sha256", "order", "run_id", "plan_sha256", "start"])
def test_cached_batch_must_match_exact_plan_slice(monkeypatch, defect):
    no_wait(monkeypatch)
    s3 = MemoryS3()
    plan, plan_hash = seed_run(s3, make_plan([item("paper-a"), item("paper-b")]))
    rows = [result_row(target) for target in plan["items"]]
    cached = {"run_id": "review-test", "start": 0, "plan_sha256": plan_hash, "rows": rows}
    if defect == "empty":
        cached["rows"] = []
    elif defect == "extra":
        rows.append(result_row(item("paper-c")))
    elif defect in ("work_id", "input_sha256"):
        rows[0][defect] = "wrong"
    elif defect == "order":
        rows.reverse()
    elif defect == "start":
        cached["start"] = 1
    else:
        cached[defect] = "wrong"
    s3.seed(PREFIX + "results/000000.json", cached)
    monkeypatch.setattr(worker.ReviewRunner, "batch", lambda *a: pytest.fail("Existing results must be checked before work"))
    current = runner(s3)
    with pytest.raises(ValueError, match="(?i)batch|plan|identity|slice"):
        current.run()
    saved = json.loads(s3.objects[PREFIX + "progress.json"])
    assert saved["papers"] == 0 and saved["next_index"] == 0
    assert saved["status"] == "failed" and saved["lease_until"] == 0


def test_completed_cached_batch_is_folded_once_without_refetch_or_rewrite(monkeypatch):
    no_wait(monkeypatch)
    s3 = MemoryS3()
    plan, plan_hash = seed_run(s3, state=progress())
    s3.seed(PREFIX + "results/000000.json", {"run_id": "review-test", "start": 0,
            "plan_sha256": plan_hash, "rows": [result_row(plan["items"][0])]})
    monkeypatch.setattr(worker.ReviewRunner, "batch", lambda *a: pytest.fail("Cached batch must avoid repeated calls"))
    result = runner(s3).run()
    assert result["done"] and result["papers"] == result["next_index"] == 1
    assert result["decisions"] == {"test:review_required": 1}
    assert all(call["Key"] == PREFIX + "progress.json" for call in s3.puts)
    count = len(s3.puts)
    repeated = runner(s3, "attempt-2").run()
    assert repeated["papers"] == 1 and len(s3.puts) == count


def test_resume_after_folded_batch_starts_at_checkpoint_without_double_counting(monkeypatch):
    no_wait(monkeypatch)
    monkeypatch.setattr(worker, "BATCH_SIZE", 2)
    s3 = MemoryS3()
    plan = make_plan([item("paper-a"), item("paper-b"), item("paper-c")])
    plan_hash = hashlib.sha256(worker._encode(plan)).hexdigest()
    seed_run(s3, plan, progress(next_index=2, papers=2, plan_sha256=plan_hash,
                               decisions={"test:review_required": 2}))
    for start, targets in ((0, plan["items"][:2]), (2, plan["items"][2:])):
        s3.seed(PREFIX + f"results/{start:06d}.json", {"run_id": "review-test", "start": start,
                "plan_sha256": plan_hash, "rows": [result_row(target) for target in targets]})
    monkeypatch.setattr(worker.ReviewRunner, "batch", lambda *a: pytest.fail("Resume should use remaining cached batch"))
    result = runner(s3).run()
    assert result["papers"] == result["next_index"] == 3
    assert result["decisions"] == {"test:review_required": 3}
    assert PREFIX + "results/000000.json" not in {entry["Key"] for entry in s3.gets}


def test_active_lease_is_not_stolen_or_rewritten():
    s3 = MemoryS3()
    s3.seed(PREFIX + "progress.json", progress(lease_until=10**12, lease_owner="another-worker"))
    with pytest.raises(ValueError, match="(?i)owns|lease"):
        runner(s3)
    assert s3.puts == []


@pytest.mark.parametrize("existing_progress", [False, True])
def test_lease_acquisition_uses_conditional_creation_or_read_etag_cas(existing_progress):
    s3 = MemoryS3()
    if existing_progress:
        s3.seed(PREFIX + "progress.json", progress())
    def concurrent_claim(store, request):
        store.seed(request["Key"], progress(lease_owner="concurrent", lease_until=10**12))
    s3.before_put = concurrent_claim
    with pytest.raises(ClientError):
        runner(s3)
    assert json.loads(s3.objects[PREFIX + "progress.json"])["lease_owner"] == "concurrent"
    assert s3.puts == []


def test_stale_worker_cannot_overwrite_another_lease():
    s3 = MemoryS3()
    current = runner(s3)
    s3.seed(PREFIX + "progress.json", progress(lease_owner="new-owner", lease_until=10**12))
    with pytest.raises(ClientError):
        current.save()
    assert json.loads(s3.objects[PREFIX + "progress.json"])["lease_owner"] == "new-owner"


def test_run_failure_releases_lease_and_allows_a_later_resume(monkeypatch):
    no_wait(monkeypatch)
    s3 = MemoryS3()
    seed_run(s3)
    def failing_batch(self, items):
        raise RuntimeError("Simulated batch failure")
    monkeypatch.setattr(worker.ReviewRunner, "batch", failing_batch)
    with pytest.raises(RuntimeError, match="Simulated"):
        runner(s3).run()
    saved = json.loads(s3.objects[PREFIX + "progress.json"])
    assert saved["status"] == "failed" and saved["lease_until"] == 0 and not saved["done"]
    monkeypatch.setattr(worker.ReviewRunner, "batch", lambda self, items: [result_row(target) for target in items])
    result = runner(s3, "attempt-2").run()
    assert result["done"] and result["papers"] == 1
    assert json.loads(s3.objects[PREFIX + "progress.json"])["lease_until"] == 0
    assert all(call["Key"].startswith(PREFIX) for call in s3.puts)


def test_authority_cache_reuses_exact_id_set_without_new_request_intent(monkeypatch):
    waits = no_wait(monkeypatch)
    s3 = MemoryS3()
    current = runner(s3)
    fetched = []
    def fetch(ids):
        fetched.append(ids)
        return {"123": {"pmid": "123"}, "456": {"pmid": "456"}}
    first = current.request("pubmed", ["456", "123", "123"], fetch)
    second = current.request("pubmed", ["123", "456"], lambda ids: pytest.fail("Cache should avoid HTTP"))
    assert first == second and fetched == [["123", "456"]]
    assert current.state["requests"] == {"pubmed": 1} and waits == [0.4]
    assert all(call["Key"].startswith(PREFIX) for call in s3.puts)


@pytest.mark.parametrize("retry_after", ["7", "date"])
def test_authority_retry_after_seconds_and_http_date_are_parsed_before_wait(monkeypatch, retry_after):
    waits = no_wait(monkeypatch)
    instant = 1_789_891_200.0
    monkeypatch.setattr(worker.time, "time", lambda: instant)
    if retry_after == "date":
        retry_after = format_datetime(datetime.fromtimestamp(instant + 7, UTC), usegmt=True)
    current = runner(MemoryS3())
    attempts = []
    def fetch(ids):
        attempts.append(ids)
        if len(attempts) == 1:
            raise worker.authority.AuthorityRequestError("PubMed", "Rate limited", status_code=429,
                                                        retry_after=retry_after, retryable=True)
        return {"123": {"pmid": "123"}}
    assert current.request("pubmed", ["123"], fetch) == {"123": {"pmid": "123"}}
    assert len(attempts) == 2 and waits == [0.4, 7.0]
    assert current.state["requests"]["pubmed"] == 2


def test_terminal_authority_error_has_no_retry_and_preserves_request_intent(monkeypatch):
    waits = no_wait(monkeypatch)
    current = runner(MemoryS3())
    def fetch(ids):
        raise worker.authority.AuthorityRequestError("PubMed", "Not found", status_code=404, retryable=False)
    with pytest.raises(worker.authority.AuthorityRequestError):
        current.request("pubmed", ["123"], fetch)
    assert current.state["requests"]["pubmed"] == 1 and current.state["errors"] == 1
    assert waits == [0.4]


@pytest.mark.parametrize("updates,existing,expected", [
    ({}, "W123", "pdf_and_metadata_coherent"), ({}, "", "pdf_and_metadata_coherent"),
    ({"doi": "10.1234/wrong"}, "W123", "review_required"),
    ({"id": "W999"}, "W123", "review_required"),
    ({"_ambiguous_openalex_ids": ["W123", "W999"]}, "W123", "review_required"),
    ({"display_name": "An unrelated study title"}, "W123", "review_required"),
    ({}, "malformed-existing-id", "review_required")])
def test_recovered_doi_batch_requires_exact_identity_title_and_existing_id_guard(monkeypatch, updates, existing, expected):
    no_wait(monkeypatch)
    s3 = MemoryS3()
    current = runner(s3)
    target = item(issues=["doi_missing"], doi="", openalex_id=existing)
    before = deepcopy(target)
    work = oa_work(doi="10.1234/recovered", display_name=target["metadata"]["title"])
    work.update(updates)
    requests = []
    def request(provider, ids, fetch):
        if ids:
            requests.append((provider, ids))
        return {"10.1234/recovered": work} if provider == "openalex" and ids else {}
    monkeypatch.setattr(current, "request", request)
    monkeypatch.setattr(worker, "inspect_pdf_identity", lambda **kwargs:
                        pdf_flags(doi_candidates=["10.1234/recovered"], doi_present=False))
    result = current.batch([target])[0]
    assert result["doi_recheck"]["status"] == expected
    assert result["doi_recheck"]["canonical_writes"] is False
    assert result["proposals"]["doi"] == "10.1234/recovered"
    assert ("openalex_id" in result["proposals"]) is (expected == "pdf_and_metadata_coherent")
    assert requests.count(("openalex", ["10.1234/recovered"])) == 1
    assert result["external_snapshot"]["openalex_id"] == existing
    assert target == before
    assert all(call["Key"].startswith(PREFIX) for call in s3.puts)


@pytest.mark.parametrize("flags", [{"sha256_matches": False}, {"title_present": False},
                                   {"doi_candidates": ["10.1234/one", "10.1234/two"]}])
def test_inadequate_pdf_identity_does_not_trigger_recovered_doi_lookup(monkeypatch, flags):
    no_wait(monkeypatch)
    current = runner(MemoryS3())
    requested = []
    monkeypatch.setattr(current, "request", lambda provider, ids, fetch: requested.append((provider, ids)) or {})
    evidence = pdf_flags(doi_candidates=["10.1234/recovered"])
    evidence.update(flags)
    monkeypatch.setattr(worker, "inspect_pdf_identity", lambda **kwargs: deepcopy(evidence))
    result = current.batch([item(issues=["doi_missing"], doi="")])[0]
    assert not result["proposals"]
    assert all(not ids for provider, ids in requested if provider == "openalex")


@pytest.mark.parametrize("pmcid", [None, "", " \t"])
@pytest.mark.parametrize("issues", [["missing_pmcid"], ["pmid_disagreement", "missing_pmcid"],
                                     ["year_difference_over_one"], ["authors_at_cap", "missing_references"]])
def test_batch_never_calls_converter_for_empty_original_pmcid(monkeypatch, pmcid, issues):
    no_wait(monkeypatch)
    current = runner(MemoryS3())
    monkeypatch.setattr(worker.authority, "fetch_pmc_ids", lambda *args, **kwargs:
                        pytest.fail("Blank PMCID must not trigger a PMC converter request"))
    monkeypatch.setattr(worker.authority, "fetch_pubmed", lambda *args, **kwargs: {})
    monkeypatch.setattr(worker.matcher, "fetch_batch", lambda *args, **kwargs: {})
    monkeypatch.setattr(worker, "inspect_pdf_identity", lambda **kwargs: pdf_flags())
    target = item(issues=issues, pmcid=pmcid, pmcid_openalex="PMC456")
    result = current.batch([target])[0]
    assert "pmcid" not in result["proposals"]
    assert not {"pmcid_ncbi", "pmid_ncbi"} & result["supplements"].keys()
    assert "pmc_authority" not in result
    assert "pmc" not in current.state["requests"]
    if "missing_pmcid" in issues:
        assert result["decisions"]["missing_pmcid"] == "allowed_empty"


def test_mixed_batch_limits_converter_to_existing_pmcid_and_keeps_exact_pubmed_cache_group(monkeypatch):
    no_wait(monkeypatch)
    s3 = MemoryS3()
    current = runner(s3)
    targets = [item("empty-shared", issues=["missing_pmcid"], pmcid=None),
               item("existing", issues=["pmcid_disagreement"], pmcid="PMC123", pmcid_openalex="PMC456"),
               item("empty-other", issues=["year_difference_over_one"], pmcid="",
                    doi="10.1234/other", pmid="https://pubmed.ncbi.nlm.nih.gov/789/", pmid_openalex="PMID:456")]
    calls = []
    def pubmed(ids, **kwargs):
        calls.append(("pubmed", ids))
        return {"123": pubmed_record("123", doi="10.1234/unrelated"), "456": pubmed_record("456"),
                "789": pubmed_record("789", doi="10.1234/other")}
    def pmc(ids, **kwargs):
        calls.append(("pmc", ids))
        return {"10.1234/example": {"status": "ok", "requested_doi": "10.1234/example", "doi": "10.1234/example",
                                    "pmid": "456", "pmcid": "PMC456"}}
    monkeypatch.setattr(worker.authority, "fetch_pubmed", pubmed)
    monkeypatch.setattr(worker.authority, "fetch_pmc_ids", pmc)
    monkeypatch.setattr(worker.matcher, "fetch_batch", lambda *args, **kwargs:
                        pytest.fail("These issues do not request OpenAlex"))
    monkeypatch.setattr(worker, "inspect_pdf_identity", lambda **kwargs: pdf_flags())
    results = current.batch(targets)
    assert calls == [("pubmed", ["123", "456", "789"]), ("pmc", ["10.1234/example"])]
    pubmed_key = PREFIX + "authority/pubmed/" + hashlib.sha256(worker._encode(["123", "456", "789"])).hexdigest() + ".json"
    assert json.loads(s3.objects[pubmed_key])["requested_ids"] == ["123", "456", "789"]
    assert results[1]["decisions"]["pmcid_disagreement"] == "metadata_confirmed"
    assert results[1]["proposals"]["pmcid"] == "PMC456"
    for row in (results[0], results[2]):
        assert "pmcid" not in row["proposals"]
        assert not {"pmcid_ncbi", "pmid_ncbi"} & row["supplements"].keys()
        assert "pmc_authority" not in row
    assert results[0]["decisions"]["missing_pmcid"] == "allowed_empty"
    assert all(call["Key"].startswith(PREFIX) for call in s3.puts)
