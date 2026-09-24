"""Settle an upload's identity where the paper is: in AWS, as soon as its text is stored.

The lab drops a PDF in the shared folder, the GROBID worker stores its text, and this decides which
paper it is and whether the wiki may read it. Until 2026-09-23 that decision ran on a laptop
(`scripts/backfill_to_s3_identity.py`, `scripts/resolve_identity_by_pdf_title.py`) and a paper sat
at `fulltext_ready_unclassified` until somebody remembered to run it. The user asked for the whole
intake to run by itself ("AWS에 올려서 자동으로 하세요"), so `ingest_lambda` calls this on the
object-created event for `papers/{stem}/clean.md`.

Every rule about which paper this is comes from `identity`; what is here is the reading and writing
around it, and the order they happen in:

1. the paper's own title and authors, as GROBID read them into the TEI header;
2. the DOI printed in that header, resolved at OpenAlex, which has to agree with 1;
3. failing that, a search on the PDF's title and then on the file name's words, judged the same way;
4. on agreement, the record, `meta.json` and `identity_status = verified` are written, and a paper
   whose journal is not refused moves to `fulltext_ready`, which is what note generation waits for;
5. when another row already holds the DOI, a noted paper makes this upload `duplicate_of_noted_paper`,
   a bare OpenAlex discovery of the same work is retired to the upload, and anything else stays parked.

Nothing is deleted and nothing the pipeline owns is overwritten. A paper that does not agree keeps
its parked status and says why, for a person to look at. The clients are passed in rather than
taken from the module, so this runs the same in a test as in the Lambda.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable

from botocore.exceptions import ClientError

from . import identity
from .papers import merge_bibliography

# What the extraction and upload own. This writes none of them; `promote` moves `ingest_status`
# alone, and only out of the parked state.
UNTOUCHABLE = ("ingest_status", "pdf_key", "pdf_sha256", "pdf_bytes", "tei_key", "source_key",
               "source_note_key", "source_note_status", "source_note_sha256")
SKIP_STATUSES = (identity.NOT_A_PAPER, "duplicate_of_noted_paper")


def _read(s3: Any, bucket: str, key: str, size: int | None = None) -> str | None:
    request: dict[str, Any] = {"Bucket": bucket, "Key": key}
    if size:
        request["Range"] = f"bytes=0-{size}"
    try:
        return s3.get_object(**request)["Body"].read().decode("utf-8", "ignore")
    except ClientError:
        return None


def stem_of(key: str) -> str | None:
    """The paper an extraction object belongs to, or None when the object is not one."""
    if not (key.startswith("papers/") and key.endswith("/clean.md")):
        return None
    stem = key[len("papers/"):-len("/clean.md")]
    return stem if stem and "/" not in stem else None


def header_of(s3: Any, bucket: str, stem: str) -> dict[str, Any]:
    """The paper's own title, authors and year, or the file name when the TEI holds no title."""
    tei = _read(s3, bucket, f"papers/{stem}/grobid.tei.xml", identity.TEI_HEAD_BYTES)
    header = identity.parse_header(tei) if tei else None
    if header and header.get("title"):
        return header
    author, _year, title = identity.split_stem(stem)
    # The TEI's authors stay when it has them; the file name holds only the first author.
    surnames = (header or {}).get("surnames") or ([author] if author else [])
    return {"title": title.replace("-", " "), "surnames": surnames, "authors": (header or {}).get("authors") or [],
            "year": None, "from_file_name": True}


def doi_in_header(s3: Any, bucket: str, stem: str) -> str | None:
    """The DOI GROBID put in the extraction header, repaired, or None when it is unusable.

    Only the header is read: a DOI deeper in the file belongs to the reference list.
    """
    import re
    text = _read(s3, bucket, f"papers/{stem}/grobid.tei.xml", identity.TEI_HEAD_BYTES) or ""
    match = re.search(r'<idno type="DOI">([^<]+)</idno>', text.split("</teiHeader>")[0])
    if not match:
        return None
    doi = identity.repair_doi(match.group(1).strip())
    return doi if doi and not identity.looks_truncated(doi) else None


def find(header: dict[str, Any], stem: str, text_head: str | None, *, doi: str | None,
         fetch_work: Callable[[str], dict[str, Any] | None],
         search: Callable[[str, int | None], list[dict[str, Any]]]) -> dict[str, Any]:
    """The paper this PDF is, with the judgement that accepted it.

    The DOI printed in the PDF is tried first and still has to agree: a PDF's DOI can name the
    supplement, the abstract record, or another paper of the same author (2026-09-23).
    """
    _author, stem_year, stem_title = identity.split_stem(stem)
    years = [int(stem_year) if stem_year.isdigit() else None, header.get("year")]

    def judged(work: dict[str, Any], *, from_pdf_doi: bool = False) -> tuple[dict, dict]:
        record = identity.build_record(work)
        return identity.judge(header, record, text_head=text_head, stem=stem,
                              doi_from_pdf=from_pdf_doi), record

    if doi:
        work = fetch_work(doi)
        if work and work.get("type") in identity.PAPER_TYPES:
            check, record = judged(work, from_pdf_doi=True)
            if check["agrees"]:
                return {"state": "found", "how": "pdf_doi", "check": check, "record": record, "work": work}
    # A year can be wrong in either place: GROBID reads a download date as the publication date,
    # and OpenAlex files Jennings and Cribbie (2016) under 2021. The file name's year is tried
    # first, then the header's, then no window at all.
    for query in identity.queries(header, stem_title):
        for year in [*dict.fromkeys(y for y in years if y), None]:
            for compact in search(identity._query(query), year):
                if compact.get("type") not in identity.PAPER_TYPES:
                    continue
                check, record = judged(compact)
                if not check["agrees"]:
                    continue
                # The search returns a projection; the work itself carries the identifiers a note needs.
                full = fetch_work(compact["doi"]) if compact.get("doi") else None
                return {"state": "found", "how": "search", "query": query, "check": check,
                        "record": identity.build_record(full) if full else record, "work": full or compact}
    return {"state": "not_found"}


def _dynamodb_safe(value: Any) -> Any:
    """Floats as Decimals, as DynamoDB takes them. A twin of `aws_store._dynamodb_safe`, which the
    Lambda cannot import: that module is the client's store."""
    from decimal import Decimal
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {key: _dynamodb_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dynamodb_safe(item) for item in value]
    return value


def _put_item(table: Any, stem: str, updates: dict[str, Any], preserve: dict[str, Any]) -> None:
    """`updates` is set, `preserve` only fills a blank. Nothing else on the item moves."""
    assert not (set(updates) | set(preserve)) & set(UNTOUCHABLE), "refusing to write a pipeline-owned field"
    fields = {**updates, **preserve}
    names = {f"#f{index}": key for index, key in enumerate(fields)}
    values, parts = {}, []
    for name, key in names.items():
        slot = f":v{name[2:]}"
        values[slot] = _dynamodb_safe(fields[key])
        parts.append(f"{name} = if_not_exists({name}, {slot})" if key in preserve else f"{name} = {slot}")
    table.update_item(Key={"work_id": stem}, UpdateExpression="SET " + ", ".join(parts),
                      ExpressionAttributeNames=names, ExpressionAttributeValues=values)


def _put_meta(s3: Any, bucket: str, stem: str, fields: dict[str, Any], record: dict[str, Any]) -> None:
    """Merge the identity into `papers/{stem}/meta.json`, never replacing what is there.

    The Lambda writes a note's title, authors and year from this file, so a verified identity has
    to reach it or the note carries its file stem as the title.
    """
    import json
    key = f"papers/{stem}/meta.json"
    body = _read(s3, bucket, key)
    meta = json.loads(body) if body else {"stem": stem}
    meta.update(fields)
    meta = merge_bibliography(meta, record, verified=True)
    s3.put_object(Bucket=bucket, Key=key, ContentType="application/json",
                  Body=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))


def doi_holder(table: Any, doi: str, stem: str) -> str | None:
    """Another catalogue row already holding this DOI, if there is one.

    Writing a DOI onto a second row is how the catalogue gets two entries for one paper, and the
    older one is then refused for the rest of its life.
    """
    from boto3.dynamodb.conditions import Key
    try:
        rows = table.query(IndexName="doi-index", KeyConditionExpression=Key("doi").eq(doi)).get("Items", [])
    except ClientError:
        return None
    others = [row["work_id"] for row in rows if row.get("work_id") not in (stem, None)]
    return others[0] if others else None


def _set_status(table: Any, stem: str, status: str, extra: dict[str, Any]) -> bool:
    """Move a parked paper to `status`, recording why; like `promote`, only out of the parked state."""
    from boto3.dynamodb.conditions import Attr
    names = {"#s": "ingest_status", **{f"#e{i}": key for i, key in enumerate(extra)}}
    values = {":s": status, **{f":e{i}": value for i, value in enumerate(extra.values())}}
    sets = ["#s = :s", *(f"#e{i} = :e{i}" for i in range(len(extra)))]
    try:
        table.update_item(Key={"work_id": stem}, UpdateExpression="SET " + ", ".join(sets),
                          ConditionExpression=Attr("ingest_status").eq(identity.UNCLASSIFIED),
                          ExpressionAttributeNames=names, ExpressionAttributeValues=values)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise


def retire_discovery(table: Any, holder: dict[str, Any], stem: str, now: str) -> bool:
    """Retire the OpenAlex discovery row of this paper to the upload, as `repair_catalog_dois` does.

    The row keeps everything but the DOI, which becomes the `openalex:` sentinel so the DOI index
    names one row, and `superseded_by` takes it out of corpus search. The write holds only while
    the row still has the DOI read and has not been retired.
    """
    from boto3.dynamodb.conditions import Attr
    work_id = holder["work_id"]
    try:
        table.update_item(
            Key={"work_id": work_id},
            UpdateExpression=("SET #doi = :released, superseded_by = :kept, superseded_at = :at, "
                              "superseded_reason = :why"),
            ConditionExpression=Attr("doi").eq(holder["doi"]) & Attr("superseded_by").not_exists(),
            ExpressionAttributeNames={"#doi": "doi"},
            ExpressionAttributeValues={
                ":released": f"openalex:{work_id}", ":kept": stem, ":at": now,
                ":why": (f"the OpenAlex discovery of {stem}, the verified upload holding the original PDF "
                         f"(identity settled in AWS)")})
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise


def holder_kind(holder: dict[str, Any], record: dict[str, Any]) -> str:
    """What the other row holding this DOI is: a noted paper, a bare discovery of this work, or other."""
    if holder.get("source_note_key"):
        return "noted"
    work = record.get("work_id")
    if not holder.get("pdf_key") and work and work in (holder.get("work_id"), holder.get("openalex_id")):
        return "discovery"
    return "other"


def promote(table: Any, stem: str) -> bool:
    """Release a paper that was waiting on its journal verdict, and nothing else.

    The condition carries the safety: this can only ever move a paper out of
    `fulltext_ready_unclassified`. One already drafted, noted, or failed in extraction stays.
    """
    from boto3.dynamodb.conditions import Attr
    try:
        table.update_item(Key={"work_id": stem}, UpdateExpression="SET ingest_status = :ready",
                          ConditionExpression=Attr("ingest_status").eq(identity.UNCLASSIFIED),
                          ExpressionAttributeValues={":ready": "fulltext_ready"})
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise


def resolve(stem: str, *, s3: Any, table: Any, bucket: str,
            fetch_work: Callable[[str], dict[str, Any] | None],
            search: Callable[[str, int | None], list[dict[str, Any]]],
            apply: bool = True) -> dict[str, Any]:
    """Settle one paper's identity, and release it when its journal is not refused."""
    item = (table.get_item(Key={"work_id": stem}).get("Item") or {})
    result: dict[str, Any] = {"stem": stem, "ingest_status": item.get("ingest_status")}
    if not item:
        return result | {"state": "not_in_catalog"}
    if item.get("ingest_status") in SKIP_STATUSES:
        return result | {"state": "skipped"}
    if item.get("identity_status") == "verified":
        return result | {"state": "already_verified"}
    header = header_of(s3, bucket, stem)
    text_head = _read(s3, bucket, f"papers/{stem}/clean.md", identity.TEXT_HEAD_BYTES)
    result["query_from"] = "file_name" if header.get("from_file_name") else "pdf_header"
    found = find(header, stem, text_head, doi=doi_in_header(s3, bucket, stem),
                 fetch_work=fetch_work, search=search)
    if found["state"] != "found":
        return result | {"state": "not_found"}
    record, check, work = found["record"], found["check"], found["work"]
    verdict = record["corpus"]["journal_verdict"]
    result |= {"state": "would_verify", "how": found["how"], "check": check, "doi": record.get("doi"),
               "journal": record.get("source"), "journal_verdict": verdict}
    if not apply:
        return result
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    ident = {**(item.get("identity") or {}), **check,
             "method": "aws_" + ("pdf_doi" if found["how"] == "pdf_doi" else "pdf_title_search"),
             "checked_at": now}
    record["corpus"]["screening"] = "identity_verified_in_aws"
    updates = {"record": record, "identity": ident, "identity_status": "verified",
               "title": record.get("title") or stem, "search_text": record["catalog_search_text"],
               "backfilled_at": now}
    if record.get("publication_year") is not None:
        updates["publication_year"] = record["publication_year"]
    from .papers import _ids_from_work
    ids = _ids_from_work(work) if work.get("ids") else {}
    doi = record.get("doi")
    conflict = doi_holder(table, doi, stem) if doi else None
    # Another row with this DOI is either the paper already noted, which this upload duplicates
    # (Ahanger, 2026-09-23), or its bare OpenAlex discovery, which the upload replaces as the
    # 25 held uploads were that day. Anything else is left for a person.
    kind = None
    if conflict:
        holder = table.get_item(Key={"work_id": conflict}).get("Item") or {"work_id": conflict}
        kind = holder_kind(holder, record)
        if kind == "discovery" and not retire_discovery(table, holder, stem, now):
            kind = "other"
    keep_doi = not conflict or kind == "discovery"
    preserve = {key: value for key, value in (({"doi": doi} if keep_doi else {}) | ids).items() if value}
    _put_item(table, stem, updates, preserve)
    _put_meta(s3, bucket, stem, {**ids, "doi": doi, "journal": record.get("source"),
                                 "journal_verdict": verdict, "identity_status": "verified",
                                 **({"openalex_title": record.get("title")} if record.get("work_id") else {})},
              record)
    result |= {"state": "verified", "doi_conflict": conflict, "doi_holder_kind": kind}
    if kind == "noted":
        result["duplicate_of"] = conflict
        result["marked_duplicate"] = _set_status(table, stem, "duplicate_of_noted_paper",
                                                 {"duplicate_of": conflict, "duplicate_marked_at": now})
        return result
    if kind == "other":
        result["released_for_notes"] = False
        result["held"] = "another row holding this DOI has a PDF or is another work"
        return result
    if verdict == "include":
        result["released_for_notes"] = promote(table, stem)
    return result
