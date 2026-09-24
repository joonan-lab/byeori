"""Finalize completed review evidence in AWS without changing catalogue or source rows.

Only immutable finalization/v1 objects are written. PubMed/PMC snapshots already
collected by the review worker are reused; this module makes no authority requests.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import ClientError

from . import metadata_authorities as authority
from .metadata_review_plan import _bounded, _encode, _read, _run_id, _text
from .metadata_review_worker import BATCH_SIZE, _get, _pmid, _put

VERSION = 1
PLACEHOLDERS = ("null", "n-a", "not-in-pdf", "unknown", "other")
PUBLICATION_DATES = {"journal", "article_electronic", "article_print",
                     "history_epublish", "history_ppublish", "history_aheadofprint"}
DECISIONS = {
    "pmcid_disagreement": {"metadata_confirmed", "pdf_confirmed", "review_required"},
    "year_difference_over_one": {"metadata_confirmed", "review_required"},
    "duplicate_stem_doi": {"placeholder_repetition", "valid_doi_duplicate", "invalid_identifier_repetition"},
}


def _hash(value: Any) -> str:
    return hashlib.sha256(_encode(value)).hexdigest()


def _year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    text = str(value or "")
    return int(text) if re.fullmatch(r"[12][0-9]{3}", text) else None


def _publication_years(record: dict[str, Any]) -> set[int]:
    years = set()
    for date in record.get("publication_dates", []):
        if not isinstance(date, dict) or date.get("kind") not in PUBLICATION_DATES:
            continue
        year = _year(date.get("year"))
        if year is not None:
            years.add(year)
        years.update(int(value) for value in re.findall(r"\b[12][0-9]{3}\b", str(date.get("medline_date", ""))))
    return years


def _placeholder(value: Any) -> str | None:
    token = re.sub(r"[^a-z0-9]", "", str(value or "").strip().casefold())
    aliases = {"null": "null", "none": "null", "na": "n-a", "notavailable": "n-a",
               "notinpdf": "not-in-pdf", "unknown": "unknown", "missing": "other", "notfound": "other"}
    return aliases.get(token)


def _stored_pmcid(value: Any) -> str | None:
    text = str(value or "").strip().rstrip("/").rsplit("/", 1)[-1]
    return authority._pmcid(re.sub(r"^pmcid:\s*", "", text, flags=re.I))


def _duplicates(items: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        raw = item["metadata"].get("doi")
        key = authority.normalize_doi(raw) or str(raw or "").strip().casefold()
        if key:
            groups.setdefault(key, []).append(item)
    result = {"valid_doi_groups": 0, "valid_doi_rows": 0, "placeholder_groups": 0, "placeholder_rows": 0,
              "other_invalid_groups": 0, "other_invalid_rows": 0,
              "placeholders": {name: {"groups": 0, "rows": 0} for name in PLACEHOLDERS}}
    for key, members in groups.items():
        if len(members) < 2:
            continue
        if authority.normalize_doi(key):
            scope = "valid_doi"
        elif (placeholder := _placeholder(key)) is not None:
            scope = "placeholder"
            result["placeholders"][placeholder]["groups"] += 1
            result["placeholders"][placeholder]["rows"] += len(members)
        else:
            scope = "other_invalid"
        result[scope + "_groups"] += 1
        result[scope + "_rows"] += len(members)
    return result


def _holds(item: dict[str, Any], row: dict[str, Any]) -> list[str]:
    decisions = row.get("decisions", {})
    reasons = []
    if any("conflict" in str(value) for value in decisions.values()):
        reasons.append("original_identity_or_authority_conflict")
    if ("pmid_disagreement" in item.get("issues", []) or "pmid_disagreement" in decisions) and (
            decisions.get("pmid_disagreement") not in ("metadata_confirmed", "pdf_confirmed")):
        reasons.append("original_pmid_unresolved")
    if row.get("pdf", {}).get("sha256_matches") is False:
        reasons.append("original_pdf_hash_mismatch")
    pdf = row.get("pdf", {})
    if (pdf.get("sha256_matches") is True and pdf.get("doi_present") is False
            and pdf.get("doi_candidates") and authority.normalize_doi(item["metadata"].get("doi"))):
        reasons.append("original_pdf_doi_conflict")
    conversion = row.get("pmc_authority")
    if isinstance(conversion, dict) and (conversion.get("status") == "mismatch" or (
            conversion.get("status") == "ok" and _conversion(item["metadata"], row) is None)):
        reasons.append("original_pmc_identity_conflict")
    return reasons


def _conversion(meta: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    doi = authority.normalize_doi(meta.get("doi"))
    value = row.get("pmc_authority")
    if (doi and isinstance(value, dict) and value.get("status") == "ok"
            and authority.normalize_doi(value.get("doi")) == doi
            and authority.normalize_doi(value.get("requested_doi")) == doi
            and authority._pmcid(value.get("pmcid"))):
        return value
    return None


def _choose_pubmed(meta: dict[str, Any], conversion: dict[str, Any] | None,
                   records: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, list[str]]:
    doi, title = authority.normalize_doi(meta.get("doi")), authority._title_key(meta.get("title"))
    if not doi or not title:
        return None, ["original_doi_or_title_missing"]
    candidates = {_pmid(meta.get(field)) for field in ("pmid", "pmid_openalex")}
    if conversion and conversion.get("pmid"):
        candidates.add(authority._pmid(conversion["pmid"]))
    candidates.discard(None)
    matching = []
    for pmid in sorted(candidates):
        record = records.get(pmid)
        if not isinstance(record, dict) or record.get("pmid") != pmid:
            return None, ["cached_pubmed_candidate_missing"]
        if authority._publication_exclusions(record):
            return None, ["pubmed_publication_or_identity_requires_review"]
        if doi in {authority.normalize_doi(value) for value in record.get("dois", [])}:
            matching.append(record)
    if len(matching) != 1:
        return None, ["pubmed_doi_identity_not_unique"]
    record = matching[0]
    if authority._title_key(record.get("title")) != title:
        return None, ["pubmed_title_disagrees"]
    if conversion:
        if conversion.get("pmid") and authority._pmid(conversion["pmid"]) != record["pmid"]:
            return None, ["pubmed_pmc_pmid_conflict"]
        pubmed_pmcid = record.get("pmcid")
        if pubmed_pmcid and authority._pmcid(pubmed_pmcid) != authority._pmcid(conversion["pmcid"]):
            return None, ["pubmed_pmc_pmcid_conflict"]
    return record, []


def _finalize_row(item: dict[str, Any], source: dict[str, Any], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    meta = item["metadata"]
    pdf_verified = all(source.get("pdf", {}).get(name) is True for name in
                       ("sha256_matches", "doi_present", "title_present"))
    row = {"work_id": item["work_id"], "input_sha256": item["input_sha256"],
           "original": {field: meta.get(field) for field in ("doi", "pmid", "pmcid", "title", "year")},
           "decisions": {}, "proposals": {}, "reasons": {}, "evidence": {},
           "pdf_identity_verified": pdf_verified, "canonical_writes": False}
    if "duplicate_stem_doi" in item.get("issues", []):
        row["decisions"]["duplicate_stem_doi"] = (
            "valid_doi_duplicate" if authority.normalize_doi(meta.get("doi")) else
            "placeholder_repetition" if _placeholder(meta.get("doi")) else "invalid_identifier_repetition")
    conversion = _conversion(meta, source)
    existing = meta.get("pmcid")
    pmcid_difference = False
    if existing and conversion:
        pmcid_difference = _stored_pmcid(existing) != authority._pmcid(conversion["pmcid"])
        row["pmcid_comparison"] = "different" if pmcid_difference else "same"
    year_case = "year_difference_over_one" in item.get("issues", [])
    if not (pmcid_difference or year_case):
        return row
    holds = _holds(item, source)
    record, identity_reasons = _choose_pubmed(meta, conversion, records)
    reasons = holds + identity_reasons
    if record:
        row["evidence"]["pubmed_identity"] = {field: record.get(field) for field in
                                             ("pmid", "title", "dois", "pmcid")}
    if pmcid_difference:
        key = "pmcid_disagreement"
        row["decisions"][key] = "review_required"
        row["evidence"][key] = {"requested_doi": conversion["requested_doi"], "returned_doi": conversion["doi"],
                                "converter_pmid": conversion.get("pmid"), "converter_pmcid": conversion["pmcid"]}
        pmcid_reasons = reasons.copy()
        if not conversion.get("pmid"):
            pmcid_reasons.append("converter_pmid_missing")
        if not pmcid_reasons:
            row["proposals"]["pmcid"] = authority._pmcid(conversion["pmcid"])
            row["decisions"][key] = "pdf_confirmed" if pdf_verified else "metadata_confirmed"
            row["evidence"][key]["pubmed_pmid"] = record["pmid"]
        row["reasons"][key] = pmcid_reasons or ["exact_doi_pubmed_title_and_pmc_agree"]
    if year_case:
        key = "year_difference_over_one"
        row["decisions"][key] = "review_required"
        year_reasons = reasons.copy()
        years = _publication_years(record) if record else set()
        original_year, openalex_year = _year(meta.get("year")), _year(meta.get("openalex_year"))
        if len(years) != 1:
            year_reasons.append("publication_year_missing_or_ambiguous")
        elif openalex_year not in years:
            year_reasons.append("pubmed_publication_year_disagrees_with_openalex")
        if original_year is None or openalex_year is None or abs(original_year - openalex_year) <= 1:
            year_reasons.append("original_large_year_difference_not_established")
        row["evidence"][key] = {"pubmed_pmid": record.get("pmid") if record else None,
                                "publication_years": sorted(years), "openalex_year": openalex_year,
                                "publication_dates": record.get("publication_dates", []) if record else [],
                                "source": "PubMed publication dates", "pdf_identity_verified": pdf_verified}
        if not year_reasons:
            row["proposals"]["year"] = openalex_year
            row["decisions"][key] = "metadata_confirmed"
        row["reasons"][key] = year_reasons or ["unique_pubmed_publication_year_matches_openalex"]
    return row


def _validate_batch(batch: Any, *, run_id: str, plan_hash: str, start: int,
                    items: list[dict[str, Any]], finalized: bool = False, source_hash: str | None = None) -> None:
    if (not isinstance(batch, dict) or batch.get("run_id") != run_id or batch.get("plan_sha256") != plan_hash
            or batch.get("start") != start or not isinstance(batch.get("rows"), list)):
        raise ValueError("Batch does not match the immutable review plan")
    expected = [(item["work_id"], item["input_sha256"]) for item in items]
    actual = [(row.get("work_id"), row.get("input_sha256")) if isinstance(row, dict) else None for row in batch["rows"]]
    if actual != expected:
        raise ValueError("Batch row identities or count do not match the review plan")
    if finalized:
        if batch.get("finalization_version") != VERSION or batch.get("source_sha256") != source_hash:
            raise ValueError("Cached finalization batch has incompatible source evidence")
        for row in batch["rows"]:
            if (row.get("canonical_writes") is not False or not isinstance(row.get("decisions"), dict)
                    or not isinstance(row.get("proposals"), dict) or set(row["proposals"]) - {"pmcid", "year"}):
                raise ValueError("Invalid cached finalization row")
            if any(field not in DECISIONS or value not in DECISIONS[field] for field, value in row["decisions"].items()):
                raise ValueError("Unsupported cached finalization decision")
            for field, value in row["proposals"].items():
                decision = row["decisions"].get("pmcid_disagreement" if field == "pmcid" else "year_difference_over_one")
                valid = (authority._pmcid(value) == value and decision in ("metadata_confirmed", "pdf_confirmed")
                         if field == "pmcid" else isinstance(value, int) and not isinstance(value, bool)
                         and _year(value) == value and decision == "metadata_confirmed")
                if not valid:
                    raise ValueError("Cached proposal is malformed or lacks a confirming decision")


def _immutable(s3: Any, bucket: str, key: str, value: dict[str, Any]) -> dict[str, Any]:
    try:
        _put(s3, bucket, key, value, IfNoneMatch="*")
        return value
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("PreconditionFailed", "412", "ConditionalRequestConflict", "409"):
            raise
        existing, _ = _get(s3, bucket, key)
        if existing is None:
            raise ValueError("Concurrent finalization object could not be read") from None
        return existing


def _check_summary(summary: Any, run_id: str, plan_hash: str, count: int) -> dict[str, Any]:
    if (not isinstance(summary, dict) or summary.get("run_id") != run_id or summary.get("plan_sha256") != plan_hash
            or summary.get("finalization_version") != VERSION or summary.get("processed_rows") != count
            or summary.get("status") != "complete" or summary.get("canonical_writes") is not False):
        raise ValueError("Cached finalization summary does not match the completed review plan")
    if not isinstance(summary.get("samples"), dict) or any(
            not isinstance(examples, list) or len(examples) > 3 for examples in summary["samples"].values()):
        raise ValueError("Cached finalization summary has unbounded samples")
    return _bounded(summary)


def finalize_review(*, s3: Any, bucket: str, run_id: str) -> dict[str, Any]:
    """Reuse completed AWS review evidence and write resumable, immutable proposals."""
    _run_id({"run_id": run_id})
    if not bucket:
        raise ValueError("bucket must be explicitly configured")
    prefix = f"runs/metadata-review/{run_id}/"
    progress, _ = _get(s3, bucket, prefix + "progress.json")
    if not progress or progress.get("run_id") != run_id or progress.get("done") is not True:
        raise ValueError("Metadata review must be complete before finalization")
    plan = _read(s3, bucket, run_id)
    items, plan_hash = plan["items"], _hash(plan)
    if progress.get("plan_sha256") != plan_hash or progress.get("next_index") != len(items):
        raise ValueError("Completed review progress does not match its immutable plan")
    final_prefix = prefix + "finalization/v1/"
    summary_key = final_prefix + "summary.json"
    cached_summary, _ = _get(s3, bucket, summary_key)
    if cached_summary is not None:
        return _check_summary(cached_summary, run_id, plan_hash, len(items))
    decisions, proposals = Counter(), Counter()
    samples: dict[str, list[dict[str, Any]]] = {}
    pmcid_checked = pmcid_differences = pdf_verified = 0
    for start in range(0, len(items), BATCH_SIZE):
        selected = items[start:start + BATCH_SIZE]
        source, _ = _get(s3, bucket, prefix + f"results/{start:06d}.json")
        _validate_batch(source, run_id=run_id, plan_hash=plan_hash, start=start, items=selected)
        source_hash = _hash(source)
        batch_key = final_prefix + f"results/{start:06d}.json"
        finalized, _ = _get(s3, bucket, batch_key)
        if finalized is None:
            ids = sorted({_pmid(item["metadata"].get(field)) for item in selected for field in ("pmid", "pmid_openalex")} - {None})
            records = {}
            if ids:
                cache_key = prefix + "authority/pubmed/" + _hash(ids) + ".json"
                cache, _ = _get(s3, bucket, cache_key)
                if (not isinstance(cache, dict) or cache.get("provider") != "pubmed" or cache.get("requested_ids") != ids
                        or not isinstance(cache.get("records"), dict) or set(cache["records"]) - set(ids)):
                    raise ValueError("Completed review PubMed cache is missing or mismatched")
                records = cache["records"]
            finalized = {"finalization_version": VERSION, "run_id": run_id, "plan_sha256": plan_hash,
                         "source_sha256": source_hash, "start": start,
                         "rows": [_finalize_row(item, row, records) for item, row in zip(selected, source["rows"])]}
            finalized = _immutable(s3, bucket, batch_key, finalized)
        _validate_batch(finalized, run_id=run_id, plan_hash=plan_hash, start=start, items=selected,
                        finalized=True, source_hash=source_hash)
        for row in finalized["rows"]:
            pmcid_checked += int(row.get("pmcid_comparison") in ("same", "different"))
            pmcid_differences += int(row.get("pmcid_comparison") == "different")
            pdf_verified += int(row.get("pdf_identity_verified") is True)
            proposals.update(row["proposals"].keys())
            for field, decision in row["decisions"].items():
                key = field + ":" + decision
                decisions[key] += 1
                examples = samples.setdefault(key, [])
                if len(examples) < 3:
                    examples.append({"work_id": _text(row["work_id"], 180),
                                     "proposals": {name: _text(value, 80) if isinstance(value, str) else value
                                                   for name, value in row["proposals"].items()},
                                     "reasons": [_text(value, 100) for value in row.get("reasons", {}).get(field, [])[:3]]})
    summary = {"finalization_version": VERSION, "run_id": run_id, "plan_sha256": plan_hash,
               "status": "complete", "canonical_writes": False, "processed_rows": len(items),
               "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(), "summary_key": summary_key,
               "results_prefix": final_prefix + "results/", "pmcid_pairs_checked": pmcid_checked,
               "existing_pmcid_disagreements": pmcid_differences, "pdf_identity_verified_rows": pdf_verified,
               "proposal_counts": {field: proposals[field] for field in ("pmcid", "year")},
               "decision_counts": dict(sorted(decisions.items())), "samples": dict(sorted(samples.items())),
               "duplicate_identifier_groups": _duplicates(items),
               "limitations": [
                   "Proposals are separate metadata review evidence; catalogue and original review results were not changed.",
                   "Publication-year proposals use unambiguous PubMed publication dates, never numeric PDF years or indexing/release dates.",
                   "PDF DOI/title/hash checks establish paper identity only; year proposals remain metadata-confirmed.",
                   "Repeated placeholder identifiers are not evidence that papers are duplicates.",
                   "Metadata proposals do not establish scientific claims or authorize catalogue corrections.",
                   "Full finalization rows remain in S3; this summary includes at most three examples per decision.",
               ]}
    _bounded(summary)
    return _check_summary(_immutable(s3, bucket, summary_key, summary), run_id, plan_hash, len(items))
