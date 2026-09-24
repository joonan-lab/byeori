from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from .config import Settings
from .openalex import normalize_work, normalize_work_id


def _dynamodb_safe(value: Any) -> Any:
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {key: _dynamodb_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dynamodb_safe(item) for item in value]
    return value


class AwsStore:
    def __init__(self, settings: Settings, session: boto3.Session | None = None) -> None:
        self.settings = settings
        self.session = session or boto3.Session(region_name=settings.aws_region)

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "region": self.settings.aws_region,
            "credentials": False,
            "bucket_configured": bool(self.settings.aws_bucket),
            "table_configured": bool(self.settings.aws_table),
            "ingest_function_configured": bool(self.settings.aws_ingest_function),
        }
        try:
            self.session.client("sts").get_caller_identity()
            result["credentials"] = True
        except (BotoCoreError, ClientError, NoCredentialsError) as exc:
            result["error"] = str(exc)
        return result

    def push_candidate(self, candidate: dict[str, Any]) -> dict[str, str]:
        if not self.settings.aws_table:
            raise RuntimeError("AWS_KIRO_WIKI_TABLE is not configured")
        item = {
            "work_id": candidate["work_id"],
            "doi": candidate.get("doi") or f"openalex:{candidate['work_id']}",
            "title": candidate["title"],
            "publication_year": candidate.get("publication_year"),
            "status": candidate["status"],
            "stem": candidate["stem"],
            "record": {key: value for key, value in candidate["record"].items() if key != "raw"},
            "updated_at": candidate["updated_at"],
            "search_text": candidate["record"].get("catalog_search_text", candidate["title"].lower()),
        }
        table = self.session.resource("dynamodb").Table(self.settings.aws_table)
        # Updating metadata must not erase the Lambda's ingest status, hashes, or S3 keys.
        names = {f"#f{index}": key for index, key in enumerate(item) if key != "work_id"}
        values = {f":v{index}": _dynamodb_safe(item[key]) for index, key in enumerate(item) if key != "work_id"}
        updates = [
            f"{name} = if_not_exists({name}, :v{name[2:]})" if key == "status"
            else f"{name} = :v{name[2:]}"
            for name, key in names.items()
        ]
        table.update_item(
            Key={"work_id": candidate["work_id"]},
            UpdateExpression="SET " + ", ".join(updates),
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )
        return {"table": self.settings.aws_table, "work_id": candidate["work_id"]}

    def search_corpus(
        self, *, query: str = "", from_year: int | None = None,
        to_year: int | None = None, tag: str | None = None,
        oa_only: bool = False, fulltext_only: bool = False,
        journal_verdicts: tuple[str, ...] | None = None,
        corpus_id: str | None = None,
    ) -> dict[str, Any]:
        """Filter the catalog. Scan filters do not reduce read capacity costs.

        ``corpus_id`` names one intake; ``None`` searches every one of them. Either way the filter
        requires a record with a corpus, which keeps rows written by a bulk PDF upload and never
        classified out of the results rather than letting them through unlabelled.
        """
        if not self.settings.aws_table:
            raise RuntimeError("AWS_KIRO_WIKI_TABLE is not configured")
        condition = (Attr("record.corpus.id").eq(corpus_id) if corpus_id
                     else Attr("record.corpus.id").exists())
        # A row retired as a duplicate is not a second paper; the row named in `superseded_by` is
        # the one that holds the PDF and the evidence note.
        condition &= Attr("superseded_by").not_exists()
        if journal_verdicts:
            condition &= Attr("record.corpus.journal_verdict").is_in(list(journal_verdicts))
        for term in query.lower().split():
            condition &= Attr("search_text").contains(term)
        if from_year is not None:
            condition &= Attr("publication_year").gte(from_year)
        if to_year is not None:
            condition &= Attr("publication_year").lte(to_year)
        if tag:
            condition &= Attr("record.corpus.tags").contains(tag)
        if oa_only:
            condition &= Attr("record.is_open_access").eq(True)
        if fulltext_only:
            condition &= Attr("record.corpus.fulltext_eligible").eq(True)
        table = self.session.resource("dynamodb").Table(self.settings.aws_table)
        request: dict[str, Any] = {"FilterExpression": condition, "ConsistentRead": True,
                                   "ReturnConsumedCapacity": "TOTAL"}
        items = []
        scanned = 0
        capacity = 0.0
        while True:
            response = table.scan(**request)
            items.extend(response.get("Items", []))
            scanned += response.get("ScannedCount", 0)
            capacity += float(response.get("ConsumedCapacity", {}).get("CapacityUnits", 0))
            if not response.get("LastEvaluatedKey"):
                break
            request["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        # Boto3 decodes numbers as Decimal; return ordinary JSON-compatible numbers.
        items = json.loads(json.dumps(items, default=lambda value: int(value) if value % 1 == 0 else float(value)))
        return {"items": items, "scanned_count": scanned, "read_capacity_units": capacity}

    # Bedrock page writing can take several minutes; never let botocore time out and re-invoke
    # (a retry would run the model, and bill it, twice).
    LAMBDA_CONFIG = Config(read_timeout=920, connect_timeout=10, retries={"max_attempts": 0})

    def _lambda_client(self):
        client = getattr(self, "_lambda", None)
        if client is None:
            client = self.session.client("lambda", config=self.LAMBDA_CONFIG)
            self._lambda = client
        return client

    def _invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.aws_ingest_function:
            raise RuntimeError("AWS_KIRO_WIKI_INGEST_FUNCTION is not configured")
        response = self._lambda_client().invoke(
            FunctionName=self.settings.aws_ingest_function,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload).encode("utf-8"),
        )
        body = json.loads(response["Payload"].read())
        if response.get("FunctionError"):
            message = body.get("errorMessage") if isinstance(body, dict) else str(body)
            raise RuntimeError(f"AWS ingest Lambda failed: {message}")
        if not isinstance(body, dict):
            raise RuntimeError("AWS workshop Lambda returned a non-object response")
        return body

    def ingest_work(self, work_id: str) -> dict[str, Any]:
        return self._invoke({"action": "ingest", "work_id": work_id})

    def pipeline_failures(self, *, verbose=False, offset=0, limit=100):
        return self._invoke({"action": "pipeline_failures", "verbose": verbose, "offset": offset, "limit": limit})

    def corpus_status(self, *, check_index=False):
        return self._invoke({"action": "corpus_status", "check_index": check_index})

    def draft_work(self, work_id: str, model_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": "draft", "work_id": work_id}
        if model_id:
            payload["model_id"] = model_id
        return self._invoke(payload)

    def get_item(self, work_id: str) -> dict[str, Any]:
        if not self.settings.aws_table:
            raise RuntimeError("AWS_KIRO_WIKI_TABLE is not configured")
        table = self.session.resource("dynamodb").Table(self.settings.aws_table)
        item = table.get_item(Key={"work_id": work_id}).get("Item") or {}
        return json.loads(json.dumps(item, default=lambda value: int(value) if value % 1 == 0 else float(value)))

    def get_openalex_work(self, identifier: str) -> dict[str, Any]:
        work_id = normalize_work_id(identifier)
        if not work_id:
            raise ValueError("AWS workshop lookup requires an OpenAlex work ID")
        result = self._invoke({"action": "get", "work_id": work_id})
        return normalize_work(result["work"])

    def search_openalex(
        self,
        query: str,
        *,
        limit: int = 10,
        from_year: int | None = None,
        to_year: int | None = None,
        oa_only: bool = False,
        fulltext_only: bool = False,
        journal_scope: str = "list",
    ) -> list[dict[str, Any]]:
        return self.search_openalex_report(
            query, limit=limit, from_year=from_year, to_year=to_year, oa_only=oa_only,
            fulltext_only=fulltext_only, journal_scope=journal_scope)["results"]

    def search_openalex_report(
        self,
        query: str,
        *,
        limit: int = 10,
        from_year: int | None = None,
        to_year: int | None = None,
        oa_only: bool = False,
        fulltext_only: bool = False,
        journal_scope: str = "list",
    ) -> dict[str, Any]:
        """The works plus what the journal policy did to the search.

        ``journal_scope`` is ``list`` (the lab's 65 journals only) or ``wide`` (anywhere except the
        refused houses and titles). A ``wide`` result carries ``journal_warning`` on every work
        outside the list, and the report says how many were refused and which journals they were.
        """
        result = self._invoke(
            {
                "action": "search",
                "query": query,
                "limit": limit,
                "from_year": from_year,
                "to_year": to_year,
                "oa_only": oa_only,
                "fulltext_only": fulltext_only,
                "journal_scope": journal_scope,
            }
        )
        works = []
        for work in result.get("results") or []:
            normalized = normalize_work(work)
            normalized["journal_scope_verdict"] = work.get("journal_scope_verdict")
            normalized["journal_warning"] = work.get("journal_warning")
            works.append(normalized)
        return {
            "results": works,
            "journal_scope": result.get("journal_scope", journal_scope),
            "outside_list_count": result.get("outside_list_count", 0),
            "refused_count": result.get("refused_count", 0),
            "refused_journals": result.get("refused_journals") or [],
        }

    def read_text(self, key: str, *, start: int = 0, max_chars: int = 4000) -> dict[str, Any]:
        """Ask AWS for a bounded source excerpt; the client never retrieves the full object."""
        return self._invoke({"action": "read_text", "key": key, "start": start, "max_chars": max_chars})

    def upload_file(self, local_path: Path, key: str) -> dict[str, str]:
        if not self.settings.aws_bucket:
            raise RuntimeError("AWS_KIRO_WIKI_BUCKET is not configured")
        self.session.client("s3").upload_file(str(local_path), self.settings.aws_bucket, key)
        return {"bucket": self.settings.aws_bucket, "key": key}

    def upload_json(self, value: dict[str, Any], key: str) -> dict[str, str]:
        if not self.settings.aws_bucket:
            raise RuntimeError("AWS_KIRO_WIKI_BUCKET is not configured")
        body = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        self.session.client("s3").put_object(
            Bucket=self.settings.aws_bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return {"bucket": self.settings.aws_bucket, "key": key}

    def get_text(self, key: str) -> str:
        """Read JSON control metadata only; wiki/source/index bodies must stay in AWS."""
        if not key.endswith(".json") or not key.startswith(("candidates/", "runs/")):
            raise ValueError("Client body/index downloads are disabled; use AWS wiki_read/read_text/metrics")
        if not self.settings.aws_bucket:
            raise RuntimeError("AWS_KIRO_WIKI_BUCKET is not configured")
        response = self.session.client("s3").get_object(Bucket=self.settings.aws_bucket, Key=key)
        return response["Body"].read().decode("utf-8")

    def put_text(self, key: str, text: str, *, create_only: bool = False,
                 content_type: str = "text/markdown; charset=utf-8") -> dict[str, str]:
        """Publish text directly from memory, without a local wiki mirror."""
        if not self.settings.aws_bucket:
            raise RuntimeError("AWS_KIRO_WIKI_BUCKET is not configured")
        options = {"IfNoneMatch": "*"} if create_only else {}
        try:
            self.session.client("s3").put_object(
                Bucket=self.settings.aws_bucket, Key=key, Body=text.encode("utf-8"),
                ContentType=content_type, **options,
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"PreconditionFailed", "412"}:
                raise FileExistsError(f"s3://{self.settings.aws_bucket}/{key} already exists") from exc
            raise
        return {"bucket": self.settings.aws_bucket, "key": key}

    def wiki_objects(self):
        """List published wiki objects in S3; never download a local copy."""
        if not self.settings.aws_bucket:
            raise RuntimeError("AWS_KIRO_WIKI_BUCKET is not configured")
        paginator = self.session.client("s3").get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.settings.aws_bucket, Prefix="wiki/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".md") and not key.startswith("wiki/drafts/") and "/failed/" not in key:
                    yield obj

    def wiki_categories(self) -> dict[str, Any]:
        return self._invoke({"action": "wiki_categories"})

    def synthesis_coverage(self) -> dict[str, Any]:
        """Per category, the notes a concept or overview cites and the ones nothing cites."""
        return self._invoke({"action": "synthesis_coverage"})

    def notes_in_category(self, category: str, *, limit: int = 0) -> list[str]:
        """Every note stem filed under one category, following the index's own pages."""
        stems: list[str] = []
        offset: int | None = 0
        while offset is not None and (not limit or len(stems) < limit):
            page = self._invoke({"action": "notes_in_category", "category": category,
                                 "offset": offset, **({"limit": limit - len(stems)} if limit else {})})
            stems.extend(page["stems"])
            offset = page["next_offset"]
        return stems[:limit] if limit else stems

    def read_extraction(self, stem: str, *, start: int = 0, max_chars: int = 40000) -> dict[str, Any]:
        """A window of one paper's stored extraction, for writing its note outside AWS."""
        return self._invoke({"action": "read_extraction", "stem": stem, "start": start,
                             "max_chars": max_chars})

    def publish_source_note(self, stem: str, markdown: str, *, model_id: str) -> dict[str, Any]:
        """Publish a note written outside AWS; the frontmatter and connections are built there."""
        return self._invoke({"action": "publish_source_note", "stem": stem, "markdown": markdown,
                             "model_id": model_id})

    def build_category_catalogs(self, *, min_notes: int = 1) -> dict[str, Any]:
        """One browse catalog per field, and the root table of fields, written in AWS."""
        return self._invoke({"action": "build_category_catalogs", "min_notes": min_notes})

    def classify_notes(self, stems: list[str], *, apply: bool = False,
                       model_id: str | None = None,
                       new_folders: list[dict[str, str]] | None = None,
                       only_new_folders: bool = False,
                       only_into: list[str] | None = None) -> dict[str, Any]:
        """File notes into the folders the wiki already uses, plus any field a person is opening."""
        return self._invoke({"action": "classify_notes", "stems": stems, "apply": apply,
                             **({"new_folders": new_folders} if new_folders else {}),
                             **({"only_new_folders": True} if only_new_folders else {}),
                             **({"only_into": only_into} if only_into else {}),
                             **({"model_id": model_id} if model_id else {})})

    def file_notes(self, stems: list[str], category: str, *, apply: bool = False) -> dict[str, Any]:
        """Put named notes in a named field because a person said so; no model is called."""
        return self._invoke({"action": "file_notes", "stems": stems, "category": category,
                             "apply": apply})

    def fields(self, *, open_fields: list[dict[str, str]] | None = None,
               close: list[str] | None = None) -> dict[str, Any]:
        """Read the fields a person opened, and open or close one."""
        return self._invoke({"action": "fields", **({"open": open_fields} if open_fields else {}),
                             **({"close": close} if close else {})})

    def sync_note_categories(self, *, apply: bool = False) -> dict[str, Any]:
        """Make the catalogue agree with the notes about which field each one is in."""
        return self._invoke({"action": "sync_note_categories", "apply": apply})

    def wiki_read(self, doc_type: str, doc_id: str, *, section: str | None = None,
                  start: int = 0, max_chars: int = 4000) -> dict[str, Any]:
        return self._invoke({"action": "wiki_read", "doc_type": doc_type, "doc_id": doc_id,
                             "section": section, "start": start, "max_chars": max_chars})

    def validate_wiki(self, *, cursor: str | None = None) -> dict[str, Any]:
        return self._invoke({"action": "wiki_validate", "cursor": cursor})

    def wiki_metrics(self, key: str, *, compare: bool = False) -> dict[str, Any]:
        return self._invoke({"action": "wiki_metrics", "key": key, "compare": compare})

    def promote_draft(self, work_id: str, **review: Any) -> dict[str, Any]:
        return self._invoke({"action": "promote_draft", "work_id": work_id, **review})

    def mark_reviewed(self, work_id: str, fields: dict[str, Any], *,
                      expected_draft_sha256: str) -> dict[str, Any]:
        """Set review fields on the catalog item only while it still holds the reviewed draft's hash."""
        if not self.settings.aws_table:
            raise RuntimeError("AWS_KIRO_WIKI_TABLE is not configured")
        if not fields or "review_status" not in fields:
            raise ValueError("review fields must include review_status")
        table = self.session.resource("dynamodb").Table(self.settings.aws_table)
        names = {f"#r{index}": key for index, key in enumerate(fields)}
        values = {f":r{index}": _dynamodb_safe(value) for index, value in enumerate(fields.values())}
        values[":expected_draft"] = expected_draft_sha256
        values[":model_draft"] = "model_draft"
        try:
            table.update_item(
                Key={"work_id": work_id},
                UpdateExpression="SET " + ", ".join(f"{name} = :r{name[2:]}" for name in names),
                ConditionExpression="draft_sha256 = :expected_draft AND ingest_status = :model_draft",
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=values,
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                raise RuntimeError(
                    f"{work_id} in DynamoDB no longer holds the reviewed draft (re-drafted or not model_draft); "
                    "re-read the draft before promoting"
                ) from exc
            raise
        return {"table": self.settings.aws_table, "work_id": work_id, "fields": sorted(fields)}

    def synthesize_topic(self, slug: str, title: str, work_ids: list[str],
                         model_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": "synthesize", "topic": slug, "title": title, "work_ids": work_ids}
        if model_id:
            payload["model_id"] = model_id
        return self._invoke(payload)

    def wiki_search(self, query: str, *, limit: int = 10, doc_type: str | None = None,
                    category: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": "wiki_search", "query": query, "limit": limit}
        if doc_type:
            payload["doc_type"] = doc_type
        if category:
            payload["category"] = category
        return self._invoke(payload)

    def wiki_backlinks(self, doc_type: str, doc_id: str) -> dict[str, Any]:
        return self._invoke({"action": "wiki_backlinks", "doc_type": doc_type, "doc_id": doc_id})

    def get_openalex_work_by_doi(self, doi: str) -> dict[str, Any]:
        """Raw OpenAlex work for a DOI (includes ids.pmid / ids.pmcid); raises on a missing work."""
        return self._invoke({"action": "get", "doi": doi})["work"]

    def source_note(self, stem: str, model_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": "source_note", "stem": stem}
        if model_id:
            payload["model_id"] = model_id
        return self._invoke(payload)

    def caller_name(self) -> str:
        """The IAM identity behind this session's credentials, e.g. `alice`.

        Read from STS rather than taken on trust, so a page records the account that wrote it and
        not a name the caller typed.
        """
        name = getattr(self, "_caller", None)
        if name is None:
            arn = self.session.client("sts").get_caller_identity()["Arn"]
            name = arn.rsplit("/", 1)[-1]
            self._caller = name
        return name

    def answer_question(self, title: str, *, tags: list[str] | None = None,
                        model_id: str | None = None, author: str | None = None,
                        reread: str = "auto") -> dict[str, Any]:
        """Answer from the wiki; `reread` decides whether an original's full text may be re-read.

        auto follows the llm-wiki rule: the research agent reads an original only when the wiki
        pages cannot settle the question. never keeps it to the wiki, always lets it read regardless.
        The reading happens in the Lambda: no original reaches this client.
        """
        payload: dict[str, Any] = {"action": "answer_question", "title": title,
                                   "tags": tags or [], "author": author or self.caller_name(),
                                   "reread": reread}
        if model_id:
            payload["model_id"] = model_id
        return self._invoke(payload)

    def resume_question(self, trace_key: str) -> dict[str, Any]:
        """Continue an AWS checkpoint without downloading its evidence to this client."""
        payload = {"action": "answer_question", "resume_trace": trace_key, "author": self.caller_name()}
        return self._invoke(payload)

    def filtered_questions(self, *, since: str | None = None, limit: int = 200) -> dict[str, Any]:
        """List the questions Bedrock's content filter stopped, newest first.

        Reads the small JSON records under runs/filtered/; no wiki page or index is downloaded.
        Only the single-shot answer path retired on 2026-09-20 wrote them. The research agent keeps
        a filtered stop as an answer_skipped result with its trace instead, which this does not
        read. `since` is a YYYY-MM-DD date, and it filters by the date in the key.
        """
        client, rows = self.session.client("s3"), []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.settings.aws_bucket, Prefix="runs/filtered/"):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(".json"):
                    continue
                day = obj["Key"].split("/")[2] if obj["Key"].count("/") >= 3 else ""
                if since and day < since:
                    continue
                rows.append((day, obj["Key"]))
        rows.sort(reverse=True)
        out = []
        for _day, key in rows[:limit]:
            body = client.get_object(Bucket=self.settings.aws_bucket, Key=key)["Body"].read()
            out.append(json.loads(body))
        spend = sum((r["usage"].get("inputTokens", 0) * 5 + r["usage"].get("outputTokens", 0) * 25) / 1e6
                    for r in out)
        return {"count": len(rows), "returned": len(out),
                "estimated_usd_for_returned": round(spend, 4),
                "basis": "Anthropic list prices applied to the reported token counts of stopped calls",
                "records": out}

    def build_wiki_index(self) -> dict[str, Any]:
        """Have the Lambda rebuild the BM25 index from the pages in S3 and publish it."""
        return self._invoke({"action": "build_index"})
