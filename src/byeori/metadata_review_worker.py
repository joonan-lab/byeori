"""Resumable AWS metadata review. Originals and catalogue fields are never written.

Run in the existing ECS worker, where PyMuPDF is available. Complete review rows,
authority snapshots and supplements remain in S3. Lambda callers receive bounded views.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from botocore.exceptions import ClientError

from . import metadata_authorities as authority
from . import openalex_match as matcher
from .metadata_pdf import inspect_pdf_identity
from .metadata_review_plan import _bounded, _encode, _read, _run_id, _text

BATCH_SIZE = 50
PDF_ISSUES = {"pmid_disagreement", "pmcid_disagreement", "year_difference_over_one",
              "doi_missing", "doi_invalid", "duplicate_stem_doi", "title_disagreement",
              "authors_disagreement", "journal_disagreement"}


def _now():
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _valid_doi(value):
    value = matcher.normalize_doi(value)
    return value if value and len(value) <= 512 and re.fullmatch(r"10\.[0-9]{4,9}/\S+", value) else None


def _pmid(value):
    value = str(value or "").strip().rstrip("/").rsplit("/", 1)[-1]
    value = re.sub(r"^pmid:\s*", "", value, flags=re.I)
    return authority._pmid(value)


def _review_existing_pmcid(item):
    # Legacy missing-PMCID work is allowed to stay empty, even in mixed batches.
    return ("missing_pmcid" not in item["issues"]
            and bool(str(item["metadata"].get("pmcid") or "").strip()))


def _delay(value, default):
    try:
        return max(0, float(value))
    except (TypeError, ValueError, OverflowError):
        try:
            return max(0, parsedate_to_datetime(str(value)).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return default


def _rate_delay():
    remaining = matcher.RATE.get("remaining")
    if remaining is None or remaining > matcher.CREDIT_FLOOR:
        return 0
    reset = _delay(matcher.RATE.get("reset"), 60)
    return max(1, reset - time.time() if reset > 1_000_000_000 else reset) + 2


def _get(s3, bucket, key):
    try:
        result = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None, None
        raise
    try:
        return json.loads(result["Body"].read()), result.get("ETag")
    finally:
        result["Body"].close()


def _put(s3, bucket, key, value, **conditions):
    return s3.put_object(Bucket=bucket, Key=key, Body=_encode(value),
                         ContentType="application/json", **conditions)


def _format_title(value):
    text = html.unescape(str(value or "")).lower()
    for word, number in zip(("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"), range(1, 11)):
        text = re.sub(rf"\b{word}\b", str(number), text)
    return matcher._squash(text)


def _format_author(value):
    if isinstance(value, list):
        value = value[0] if value else ""
    first = re.split(r",|;|\bet al\b", str(value or ""))[0]
    first = re.sub(r"[0-9*†‡]+", "", first).replace("ı", "i")
    return matcher._squash(first)


def classify_row(item, pubmed, pmc, oa_work, pdf):
    """Return evidence-linked proposals only; heuristics never alter original values."""
    meta = item["metadata"]
    doi = _valid_doi(meta.get("doi"))
    result = {"work_id": item["work_id"], "issues": item["issues"],
              "input_sha256": item["input_sha256"], "original": {
                  key: meta.get(key) for key in ("doi", "pmid", "pmcid", "title", "year", "authors", "journal")},
              "decisions": {}, "proposals": {}, "supplements": {}, "pdf": pdf}
    result["external_snapshot"] = {key: meta.get(key) for key in (
        "openalex_id", "openalex_title", "openalex_year", "openalex_authors",
        "openalex_venue", "pmid_openalex", "pmcid_openalex")}
    if "pmid_disagreement" in item["issues"]:
        normalized = {**meta, **{field: _pmid(meta.get(field)) for field in ("pmid", "pmid_openalex")}}
        result["pmid_assessment"] = authority.assess_pmid_pair(normalized, pubmed, pdf_evidence=pdf)
        assessment = result["pmid_assessment"]
        result["decisions"]["pmid_disagreement"] = assessment["classification"]
        if assessment.get("proposed_pmid"):
            result["proposals"]["pmid"] = assessment["proposed_pmid"]
    conversion = pmc.get(doi) if doi and _review_existing_pmcid(item) else None
    if conversion and conversion.get("status") == "ok" and conversion.get("doi") == doi:
        result["pmc_authority"] = conversion
        for field in ("pmid", "pmcid"):
            if conversion.get(field):
                result["supplements"][f"{field}_ncbi"] = conversion[field]
        if "pmcid_disagreement" in item["issues"]:
            candidates = {str(meta.get(field) or "").rstrip("/").rsplit("/", 1)[-1].upper()
                          for field in ("pmcid", "pmcid_openalex")}
            if conversion.get("pmcid") in candidates:
                result["decisions"]["pmcid_disagreement"] = "metadata_confirmed"
                result["proposals"]["pmcid"] = conversion["pmcid"]
            else:
                result["decisions"]["pmcid_disagreement"] = "authority_conflict"
        proposed = result["proposals"].get("pmid")
        if proposed and conversion.get("pmid") and conversion["pmid"] != proposed:
            result["decisions"]["pmid_disagreement"] = "authority_conflict"
            result["proposals"].pop("pmid", None)
            result["proposals"].pop("pmcid", None)
            if "pmcid_disagreement" in item["issues"]:
                result["decisions"]["pmcid_disagreement"] = "authority_conflict"
    else:
        if "pmcid_disagreement" in item["issues"]:
            result["decisions"]["pmcid_disagreement"] = "authority_conflict" if conversion and conversion.get("status") == "mismatch" else "not_returned" if conversion is not None else "lookup_unavailable"
        if conversion:
            result["pmc_authority"] = conversion
    candidates = pdf.get("doi_candidates", []) if pdf else []
    if {"doi_missing", "doi_invalid"} & set(item["issues"]):
        if pdf.get("sha256_matches") is True and pdf.get("title_present") is True and len(candidates) == 1 and _valid_doi(candidates[0]):
            result["proposals"]["doi"] = _valid_doi(candidates[0])
        for issue in ("doi_missing", "doi_invalid"):
            if issue in item["issues"]:
                result["decisions"][issue] = "pdf_doi_candidate" if "doi" in result["proposals"] else "review_required"
    if "duplicate_stem_doi" in item["issues"]:
        result["duplicate_stem_doi"] = item.get("duplicate_doi_stems", [])
        result["decisions"]["duplicate_stem_doi"] = "review_required"
    own_pmid, oa_pmid = _pmid(meta.get("pmid")), _pmid(meta.get("pmid_openalex"))
    own_records = [pubmed[p] for p in dict.fromkeys([own_pmid, oa_pmid]) if p in pubmed and doi in pubmed[p].get("dois", [])]
    own_ids = {record["pmid"] for record in own_records}
    unresolved_pair = result.get("pmid_assessment", {}).get("classification") == "review_required"
    converter_conflict = (conversion and conversion.get("status") == "ok" and conversion.get("pmid")
                          and own_ids and (len(own_ids) != 1 or conversion["pmid"] not in own_ids))
    own_pmcids = {record.get("pmcid") for record in own_records if record.get("pmcid")}
    converter_conflict = converter_conflict or (conversion and conversion.get("status") == "ok"
                          and own_pmcids and (len(own_pmcids) != 1 or conversion.get("pmcid") not in own_pmcids))
    if unresolved_pair or converter_conflict:
        for issue in ("pmid_disagreement", "pmcid_disagreement"):
            if issue in item["issues"]:
                result["decisions"][issue] = "authority_conflict" if converter_conflict else "review_required"
        result["proposals"].pop("pmid", None)
        result["proposals"].pop("pmcid", None)
    if "year_difference_over_one" in item["issues"]:
        result["year_authority"] = [{"pmid": r["pmid"], "publication_dates": r.get("publication_dates", [])} for r in own_records]
        result["decisions"]["year_difference_over_one"] = "review_required"
    if "title_disagreement" in item["issues"]:
        result["decisions"]["title_disagreement"] = ("format_variant" if _format_title(meta.get("title")) == _format_title(meta.get("openalex_title")) else "review_required")
    if "authors_disagreement" in item["issues"]:
        result["decisions"]["authors_disagreement"] = ("format_variant" if _format_author(meta.get("authors")) == _format_author(meta.get("openalex_authors")) else "review_required")
    if "journal_disagreement" in item["issues"]:
        def journal(value):
            return matcher._squash(re.sub(r"^the\s+", "", str(value or ""), flags=re.I))
        result["decisions"]["journal_disagreement"] = "format_variant" if journal(meta.get("journal")) == journal(meta.get("openalex_venue")) else "review_required"
    if oa_work:
        result["openalex_identity"] = {"requested_doi": doi, "returned_doi": matcher.normalize_doi(oa_work.get("doi")),
                                      "existing_id": meta.get("openalex_id"), "returned_id": matcher.normalize_openalex_id(oa_work.get("id"))}
        expected = matcher.normalize_openalex_id(meta.get("openalex_id"))
        identity_ok = (not oa_work.get("_ambiguous_openalex_ids") and doi == matcher.normalize_doi(oa_work.get("doi"))
                       and expected is not None and expected == matcher.normalize_openalex_id(oa_work.get("id")))
        if identity_ok:
            if "missing_references" in item["issues"]:
                refs = [matcher.normalize_openalex_id(value) for value in (oa_work.get("referenced_works") or [])]
                if refs and all(refs):
                    result["supplements"]["openalex_referenced_works"] = refs
                    result["decisions"]["missing_references"] = "supplement_available"
                else:
                    result["decisions"]["missing_references"] = "still_missing"
            if "authors_at_cap" in item["issues"]:
                authors = [(a.get("author") or {}).get("display_name") for a in (oa_work.get("authorships") or [])]
                authors = [name for name in authors if name]
                if len(authors) > len(meta.get("openalex_authors") or []):
                    result["supplements"]["openalex_authors"] = authors
                result["openalex_authors_returned_count"] = len(authors)
                # Current public API can cap even singleton responses at 100 names.
                result["supplements"]["openalex_authors_upstream_cap_possible"] = len(authors) >= 100
                result["decisions"]["authors_at_cap"] = "expanded_upstream_cap_possible" if len(authors) >= 100 else "supplement_available" if len(authors) > 30 else "no_expansion"
        else:
            for issue in ("missing_references", "authors_at_cap"):
                if issue in item["issues"]:
                    result["decisions"][issue] = "identity_conflict"
    for issue in item["issues"]:
        result["decisions"].setdefault(issue, "lookup_unavailable")
    # A verified first page explicitly pointing to another DOI is a hold, not a
    # metadata-only confirmation. Failure to extract a title alone is inconclusive.
    if doi and pdf.get("sha256_matches") is True and pdf.get("doi_present") is False and candidates:
        for issue in ("pmid_disagreement", "pmcid_disagreement"):
            if issue in item["issues"]:
                result["decisions"][issue] = "pdf_identity_conflict"
        result["proposals"].pop("pmid", None)
        result["proposals"].pop("pmcid", None)
    if "missing_pmcid" in item["issues"]:
        result["decisions"]["missing_pmcid"] = "allowed_empty"
    return result


class ReviewRunner:
    def __init__(self, *, s3, bucket, run_id, attempt_id, email=None):
        _run_id({"run_id": run_id})
        _run_id({"run_id": attempt_id})
        if not bucket:
            raise ValueError("bucket must be explicitly configured")
        self.s3, self.bucket, self.run_id, self.attempt_id, self.email = s3, bucket, run_id, attempt_id, email
        self.prefix = f"runs/metadata-review/{run_id}/"
        self.state, self.etag = _get(s3, bucket, self.prefix + "progress.json")
        if self.state and self.state.get("done"):
            return
        if self.state and self.state.get("lease_until", 0) > time.time():
            raise ValueError("An AWS metadata review worker still owns this run")
        self.state = self.state or {"run_id": run_id, "started_at": _now(), "next_index": 0,
                                   "requests": {}, "decisions": {}, "samples": {}, "papers": 0,
                                   "pdf_checked": 0, "errors": 0, "done": False}
        self.state["lease_owner"] = attempt_id
        self.state["status"] = "running"
        self.state.pop("last_error", None)
        self.save()

    def save(self):
        self.state["lease_until"] = 0 if self.state.get("done") or self.state.get("status") == "failed" else time.time() + 300
        self.state["updated_at"] = _now()
        result = _put(self.s3, self.bucket, self.prefix + "progress.json", self.state,
                      **({"IfMatch": self.etag} if self.etag else {"IfNoneMatch": "*"}))
        self.etag = result["ETag"]

    def wait(self, seconds):
        end = max(time.time() + seconds, self.state.get("blocked_until", 0))
        self.state["blocked_until"] = end
        self.save()
        while time.time() < end:
            time.sleep(min(30, end - time.time()))
            self.save()

    def request(self, provider, ids, fetch):
        ids = sorted(set(ids))
        if not ids:
            return {}
        key = self.prefix + "authority/" + provider + "/" + hashlib.sha256(_encode(ids)).hexdigest() + ".json"
        cached, _ = _get(self.s3, self.bucket, key)
        if cached is not None:
            if cached.get("provider") != provider or cached.get("requested_ids") != ids:
                raise ValueError("Authority cache does not match requested identifiers")
            return cached["records"]
        self.wait(0.4)
        for attempt in range(3):
            # Persist attempts before network I/O; this is an intent count, never billed usage.
            self.state["requests"][provider] = self.state["requests"].get(provider, 0) + 1
            self.save()
            matcher.RATE.clear()
            try:
                records = fetch(ids)
            except (authority.AuthorityRequestError, matcher.OpenAlexRequestError, TimeoutError, OSError) as exc:
                retryable = getattr(exc, "retryable", isinstance(exc, (TimeoutError, OSError)))
                retry_after = _delay(getattr(exc, "retry_after", None) or matcher.RATE.get("retry_after"), 2 ** (attempt + 1))
                if provider == "openalex":
                    retry_after = max(retry_after, _rate_delay())
                self.state["blocked_until"] = time.time() + retry_after
                if not retryable or attempt == 2:
                    self.state["errors"] += 1
                    self.save()
                    raise
                self.wait(retry_after)
            else:
                record = {"provider": provider, "requested_ids": ids, "fetched_at": _now(), "records": records}
                _put(self.s3, self.bucket, key, record, IfNoneMatch="*")
                if provider == "openalex" and _rate_delay():
                    self.wait(_rate_delay())
                return records
        raise AssertionError("Unreachable request exhaustion")

    def batch(self, items):
        dois = [_valid_doi(item["metadata"].get("doi")) for item in items]
        pmids = [_pmid(item["metadata"].get(field)) for item in items for field in ("pmid", "pmid_openalex")]
        pubmed = self.request("pubmed", [value for value in pmids if value], lambda ids: authority.fetch_pubmed(ids, email=self.email))
        pmc_dois = [doi for item, doi in zip(items, dois) if doi and _review_existing_pmcid(item)]
        pmc = self.request("pmc", pmc_dois, lambda ids: authority.fetch_pmc_ids(ids, email=self.email))
        requested = [doi for item, doi in zip(items, dois) if doi and {"missing_references", "authors_at_cap"} & set(item["issues"])]
        oa = self.request("openalex", requested, lambda ids: matcher.fetch_batch(ids, attempts=1))
        def inspect(item):
            meta = item["metadata"]
            pdf = {"checked": False}
            if PDF_ISSUES & set(item["issues"]):
                pdf = inspect_pdf_identity(s3=self.s3, bucket=self.bucket, pdf_key=meta.get("pdf_key"),
                                           pdf_sha256=meta.get("pdf_sha256"), doi=_valid_doi(meta.get("doi")), title=meta.get("title"))
                pdf["checked"] = True
            return pdf
        # PDF reads/parsing are independent AWS work; external APIs remain serialized.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(inspect, item) for item in items]
            pdfs = []
            for future in futures:
                self.save()
                pdfs.append(future.result())
        results = []
        for item, doi, pdf in zip(items, dois, pdfs):
            result = classify_row(item, pubmed, pmc, oa.get(doi), pdf)
            result["reviewed_at"] = _now()
            results.append(result)
        recovered = [row["proposals"]["doi"] for row in results if row["proposals"].get("doi")]
        recovered_works = self.request("openalex", recovered, lambda ids: matcher.fetch_batch(ids, attempts=1))
        for row in results:
            candidate_doi = row["proposals"].get("doi")
            if not candidate_doi:
                continue
            work = recovered_works.get(candidate_doi)
            if not work:
                row["doi_recheck"] = {"status": "not_returned", "doi": candidate_doi}
                continue
            returned_id = matcher.normalize_openalex_id(work.get("id"))
            raw_existing_id = row["external_snapshot"].get("openalex_id")
            existing_id = matcher.normalize_openalex_id(raw_existing_id)
            coherent = (_valid_doi(work.get("doi")) == candidate_doi and returned_id
                        and not work.get("_ambiguous_openalex_ids")
                        and _format_title(work.get("display_name")) == _format_title(row["original"].get("title"))
                        and (not raw_existing_id or returned_id == existing_id))
            row["doi_recheck"] = {"status": "pdf_and_metadata_coherent" if coherent else "review_required",
                                  "doi": candidate_doi, "openalex_id": returned_id,
                                  "returned_title": work.get("display_name"), "canonical_writes": False}
            if coherent:
                row["proposals"]["openalex_id"] = returned_id
        return results

    def fold(self, rows):
        for row in rows:
            self.state["papers"] += 1
            self.state["pdf_checked"] += int(row.get("pdf", {}).get("checked", False))
            pdf = row.get("pdf", {})
            for counter, valid in (("pdf_hash_verified", pdf.get("sha256_matches") is True),
                                   ("pdf_identity_confirmed", all(pdf.get(key) is True for key in ("sha256_matches", "doi_present", "title_present")))):
                self.state[counter] = self.state.get(counter, 0) + int(valid)
            for error in pdf.get("errors", []):
                counts = self.state.setdefault("pdf_error_counts", {})
                counts[error] = counts.get(error, 0) + 1
            if row.get("doi_recheck"):
                counts = self.state.setdefault("doi_recheck_counts", {})
                status = row["doi_recheck"]["status"]
                counts[status] = counts.get(status, 0) + 1
            for field, value in row["proposals"].items():
                old = row["original"].get(field)
                if field == "pmid":
                    old = _pmid(old)
                elif field == "doi":
                    old = _valid_doi(old)
                elif field == "openalex_id":
                    old = matcher.normalize_openalex_id(row.get("external_snapshot", {}).get(field))
                if old != value:
                    counts = self.state.setdefault("changed_proposal_counts", {})
                    counts[field] = counts.get(field, 0) + 1
                    samples = self.state.setdefault("changed_proposal_samples", {}).setdefault(field, [])
                    if len(samples) < 3:
                        samples.append({"work_id": _text(row["work_id"], 180), "item_index": self.state["papers"] - 1,
                                        "old": _text(old), "proposed": _text(value)})
            for field in row["supplements"]:
                counts = self.state.setdefault("supplement_counts", {})
                counts[field] = counts.get(field, 0) + 1
            for issue, decision in row["decisions"].items():
                key = issue + ":" + decision
                self.state["decisions"][key] = self.state["decisions"].get(key, 0) + 1
                samples = self.state["samples"].setdefault(key, [])
                if len(samples) < 3:
                    samples.append({"work_id": _text(row["work_id"], 180), "item_index": self.state["papers"] - 1,
                                    "proposals": {k: _text(v) for k, v in row["proposals"].items()}})

    def run(self):
        if self.state.get("done"):
            return review_progress({"run_id": self.run_id}, s3=self.s3, bucket=self.bucket)
        try:
            return self._run()
        except Exception as exc:
            # Only compact error classes/codes are durable; request bodies and credentials are not.
            self.state.update(status="failed", last_error={"type": type(exc).__name__,
                              "provider": getattr(exc, "provider", None),
                              "status_code": getattr(exc, "status_code", None)})
            try:
                self.save()
            except ClientError:
                pass  # A lost conditional lease must never be overwritten.
            raise

    def _run(self):
        plan = _read(self.s3, self.bucket, self.run_id)
        items = plan["items"]
        self.state["planned_papers"] = len(items)
        plan_hash = hashlib.sha256(_encode(plan)).hexdigest()
        if self.state.get("plan_sha256") not in (None, plan_hash):
            raise ValueError("Immutable metadata review plan changed")
        self.state["plan_sha256"] = plan_hash
        self.wait(0)
        while self.state["next_index"] < len(items):
            start = self.state["next_index"]
            key = self.prefix + f"results/{start:06d}.json"
            cached, _ = _get(self.s3, self.bucket, key)
            if cached is None:
                rows = self.batch(items[start:start + BATCH_SIZE])
                cached = {"run_id": self.run_id, "start": start, "plan_sha256": self.state["plan_sha256"], "rows": rows}
                _put(self.s3, self.bucket, key, cached, IfNoneMatch="*")
            if cached.get("run_id") != self.run_id or cached.get("plan_sha256") != self.state["plan_sha256"] or cached.get("start") != start:
                raise ValueError("Review batch does not match immutable plan")
            expected = [(item["work_id"], item["input_sha256"]) for item in items[start:start + BATCH_SIZE]]
            actual = [(row.get("work_id"), row.get("input_sha256")) for row in cached.get("rows", [])]
            if actual != expected:
                raise ValueError("Review batch identities do not match immutable plan")
            self.fold(cached["rows"])
            self.state["next_index"] += len(cached["rows"])
            self.save()
        self.state.update(done=True, status="complete", finished_at=_now(), canonical_writes=False)
        self.save()
        return review_progress({"run_id": self.run_id}, s3=self.s3, bucket=self.bucket)


def review_progress(event, *, s3, bucket):
    run_id = _run_id(event)
    state, _ = _get(s3, bucket, f"runs/metadata-review/{run_id}/progress.json")
    if state is None:
        return {"run_id": run_id, "status": "not_started"}
    keys = ("run_id", "started_at", "updated_at", "finished_at", "next_index", "planned_papers", "papers",
            "pdf_checked", "errors", "done", "requests", "decisions", "samples", "plan_sha256", "last_error", "blocked_until",
            "changed_proposal_counts", "changed_proposal_samples", "supplement_counts", "doi_recheck_counts",
            "pdf_hash_verified", "pdf_identity_confirmed", "pdf_error_counts")
    result = {key: state[key] for key in keys if key in state}
    status = "complete" if state.get("done") else "failed" if state.get("status") == "failed" else "stalled" if state.get("lease_until", 0) < time.time() else "running"
    result.update(canonical_writes=False, status=status,
                  requests_semantics="Persisted request intents, not exact HTTP or billing counts")
    return _bounded(result)


def review_results(event, *, s3, bucket):
    run_id = _run_id(event)
    offset, limit = event.get("offset", 0), event.get("limit", 5)
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("offset must be nonnegative and limit must be 1..10")
    state, _ = _get(s3, bucket, f"runs/metadata-review/{run_id}/progress.json")
    completed = int((state or {}).get("next_index", 0))
    rows = []
    cache = {}
    for index in range(offset, min(offset + limit, completed)):
        start = index // BATCH_SIZE * BATCH_SIZE
        if start not in cache:
            cache[start], _ = _get(s3, bucket, f"runs/metadata-review/{run_id}/results/{start:06d}.json")
        row = cache[start]["rows"][index - start]
        def preview(value, depth=0):
            if isinstance(value, dict):
                return {k: preview(v, depth + 1) for k, v in value.items()}
            if isinstance(value, list):
                return {"count": len(value), "first": [preview(v, depth + 1) for v in value[:3]]}
            if isinstance(value, (bool, int)) or value is None:
                return value
            return _text(value, 350)
        rows.append({"item_index": index, **preview(row)})
    end = offset + len(rows)
    return _bounded({"run_id": run_id, "total_completed": completed, "offset": offset,
                     "next_offset": end if end < completed else None, "rows": rows, "preview": True})
