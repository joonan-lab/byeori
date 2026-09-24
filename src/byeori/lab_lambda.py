"""Composition root of the ``byeori-lab`` stack (docs/LAB-QUESTION-WORKFLOW.md, section 3).

Every function of the stack runs ``handler`` and takes its role from ``LAB_HANDLER``: the
authenticated ``gateway`` behind the Function URL, the SQS-driven ``answer`` and ``triage``
workers, the ``outbox`` relay an administrator invokes by hand while its schedule is disabled,
and the ``research`` consumer, which runs approved research through ``lab_research`` only while
``LAB_RESEARCH_CONSUMER_ENABLED`` is ``true`` and otherwise reports itself disabled. The module
binds the environment, the Lambda context and the AWS clients to the modules that own the
records (``lab_gateway``, ``lab_answer``, ``lab_triage``, ``lab_research``, ``lab_jobs``); it holds
no rule of the workflow itself.

Configuration is read when the handler runs, never at import, so packaging tests import the
module without the stack's environment. The account id and the region come from the invoked
function's ARN, not from the environment. Clients are created on first use within one
invocation and can be replaced with fakes; the Bedrock client makes exactly one attempt per
request, so the answer worker's budget bookkeeping sees every call the service received. The
shared index is opened once per invocation and closed at its end; the answer worker opens it
only after a claim succeeded, so a duplicate delivery never pays for the download. The outbox
relay also runs the expired-lease sweep of design section 8 once per invocation. All S3 writes
go through ``lab_store.ReceiptWriter`` inside the owning modules. Log records carry
identifiers, actions, statuses and error codes only: never a question, a request body, a
principal ARN, a token or the Jev secret.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config as ClientConfig
from botocore.exceptions import ClientError

from byeori import evidence_packet, jev_client, lab_answer, lab_gateway, lab_jobs, lab_research, lab_triage
from byeori.lab_jobs import InvalidTransition, NotFound
from byeori.lab_policy import LEASE_SECONDS, PACKET_LIMITS, POLICY_REVISION
from byeori.lab_store import (
    ConditionFailed,
    DynamoTable,
    PageWriter,
    ReceiptWriter,
    StoreError,
    TablePort,
)

__all__ = [
    "Config", "DEFAULT_RESEARCH_REASONING", "Deps", "HANDLERS", "LEASE_SWEEP_LIMIT", "RESEARCH_DISABLED",
    "bedrock_config", "bedrock_model", "build_deps", "handler", "invocation_identity", "send_message", "sqs_sender",
]

LOGGER = logging.getLogger(__name__)

GATEWAY, ANSWER, TRIAGE, OUTBOX, RESEARCH = "gateway", "answer", "triage", "outbox", "research"
HANDLERS = (GATEWAY, ANSWER, TRIAGE, OUTBOX, RESEARCH)
DEFAULT_INDEX_KEY = "index/wiki-index-v2.sqlite3"
INDEX_CACHE_DIR = Path("/tmp/lab-index")     # the only writable path of a Lambda; reused by warm invocations
BEDROCK_CONNECT_TIMEOUT = 10
# One Converse call with a 4,096-token answer finishes well inside this; a hung call must surface
# as ReadTimeoutError (an unknown outcome for lab_answer) before the 900 s function timeout ends
# the process without a record.
BEDROCK_READ_TIMEOUT = 300
OUTBOX_BATCH = 25                          # rows relayed per kind per invocation (lab_jobs.pending_outbox limit)
LEASE_SWEEP_LIMIT = 100                    # expired running jobs closed per outbox invocation (lab_jobs.sweep_expired_leases)
RELAY_KINDS = (lab_jobs.KIND_ANSWER, lab_jobs.KIND_TRIAGE, lab_jobs.KIND_RESEARCH)
RESEARCH_DISABLED = {"status": "disabled", "reason": "research consumer not activated"}
RESEARCH_SKIPPED = {"status": "skipped", "reason": "research consumer not activated"}
DEFAULT_RESEARCH_REASONING = "high"        # the user unified everything on Opus 5 at high on 2026-09-23
QUEUE_UNCONFIGURED = {"status": "skipped", "reason": "queue url not configured"}
LEASE_HELD = "lease_held"
INVALID_MESSAGE = "invalid_message"


# ---------------------------------------------------------------------------------------------
# Configuration and dependencies
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """The stack's environment for one function, read at invocation time by ``from_env``."""

    table_name: str
    bucket: str
    handler: str
    index_key: str = DEFAULT_INDEX_KEY
    answer_model_id: str = ""
    answer_reasoning: str | None = None
    jev_parameter: str = ""
    answer_queue_url: str = ""
    triage_queue_url: str = ""
    research_queue_url: str = ""
    policy_revision: str = POLICY_REVISION
    research_consumer_enabled: bool = False
    research_model_id: str = ""
    research_reasoning: str | None = DEFAULT_RESEARCH_REASONING

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Config:
        """Build the configuration from ``environ`` (the process environment by default).

        A missing ``LAB_TABLE`` or ``LAB_BUCKET`` and an unknown ``LAB_HANDLER`` raise
        ``RuntimeError`` naming the variable, so a misconfigured function fails on its first
        invocation instead of serving requests against nothing. ``LAB_RESEARCH_MODEL_ID`` falls
        back to ``LAB_ANSWER_MODEL_ID`` and ``LAB_RESEARCH_REASONING`` to ``xhigh`` when empty.
        """
        env = os.environ if environ is None else environ

        def text(name: str, default: str = "") -> str:
            value = env.get(name)
            return default if value is None else str(value).strip()

        table_name, bucket = text("LAB_TABLE"), text("LAB_BUCKET")
        missing = [name for name, value in (("LAB_TABLE", table_name), ("LAB_BUCKET", bucket)) if not value]
        if missing:
            raise RuntimeError(f"lab_lambda needs the environment variable(s) {', '.join(missing)}")
        kind = text("LAB_HANDLER")
        if kind not in HANDLERS:
            raise RuntimeError(f"LAB_HANDLER must be one of {', '.join(HANDLERS)}; got {kind!r}")
        deployed_policy = text("LAB_POLICY_REVISION")
        if deployed_policy and deployed_policy != POLICY_REVISION:
            # Verdicts, offers and approvals are stamped with the code constant; a stack deployed
            # with another revision would record two policies for one run. Fail loudly instead.
            raise RuntimeError(f"LAB_POLICY_REVISION {deployed_policy!r} does not match the packaged policy {POLICY_REVISION!r}")
        return cls(
            table_name=table_name, bucket=bucket, handler=kind,
            index_key=text("LAB_INDEX_KEY") or DEFAULT_INDEX_KEY,
            answer_model_id=text("LAB_ANSWER_MODEL_ID"),
            answer_reasoning=text("LAB_ANSWER_REASONING") or None,
            jev_parameter=text("LAB_JEV_PARAMETER"),
            answer_queue_url=text("LAB_ANSWER_QUEUE_URL"),
            triage_queue_url=text("LAB_TRIAGE_QUEUE_URL"),
            research_queue_url=text("LAB_RESEARCH_QUEUE_URL"),
            policy_revision=text("LAB_POLICY_REVISION") or POLICY_REVISION,
            research_consumer_enabled=text("LAB_RESEARCH_CONSUMER_ENABLED").lower() == "true",
            research_model_id=text("LAB_RESEARCH_MODEL_ID") or text("LAB_ANSWER_MODEL_ID"),
            research_reasoning=text("LAB_RESEARCH_REASONING") or DEFAULT_RESEARCH_REASONING,
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Deps:
    """Ports and clients of one invocation.

    Anything not supplied is created on first use through ``client_factory`` (``boto3.client``
    with the invocation's region); tests pass the in-memory fakes instead. ``open_index``
    downloads the shared index once per invocation and ``close`` releases the connection.
    ``jev_post`` defaults to ``jev_client.post`` with its fixed endpoint; nothing here reads a
    provider address from the environment.
    """

    def __init__(self, config: Config, *, account_id: str, region: str, table: TablePort | None = None,
                 receipts: ReceiptWriter | None = None, pages: PageWriter | None = None,
                 s3: Any = None, sqs: Any = None, ssm: Any = None,
                 bedrock: Any = None, index: tuple[sqlite3.Connection, str | None] | None = None,
                 now: Callable[[], datetime] | None = None,
                 jev_post: Callable[[bytes, str], bytes] = jev_client.post,
                 client_factory: Callable[..., Any] | None = None,
                 index_cache_dir: Path | str = INDEX_CACHE_DIR):
        self.config = config
        self.account_id = account_id
        self.region = region
        self.now = now or _utc_now
        self.jev_post = jev_post
        self.index = index
        self._index_cache_dir = Path(index_cache_dir)
        self._factory = client_factory or boto3.client
        self._table, self._receipts, self._pages = table, receipts, pages
        self._s3, self._sqs, self._ssm, self._bedrock = s3, sqs, ssm, bedrock

    def _client(self, service: str, **kwargs: Any) -> Any:
        return self._factory(service, region_name=self.region, **kwargs)

    @property
    def s3(self) -> Any:
        if self._s3 is None:
            self._s3 = self._client("s3")
        return self._s3

    @property
    def table(self) -> TablePort:
        if self._table is None:
            self._table = DynamoTable(self._client("dynamodb"), self.config.table_name)
        return self._table

    @property
    def receipts(self) -> ReceiptWriter:
        if self._receipts is None:
            self._receipts = ReceiptWriter(self.s3, self.config.bucket)
        return self._receipts

    @property
    def pages(self) -> PageWriter:
        """Writes answered questions under ``wiki/lab-questions/`` and refuses every other key."""
        if self._pages is None:
            self._pages = PageWriter(self.s3, self.config.bucket)
        return self._pages

    @property
    def sqs(self) -> Any:
        if self._sqs is None:
            self._sqs = self._client("sqs")
        return self._sqs

    @property
    def ssm(self) -> Any:
        if self._ssm is None:
            self._ssm = self._client("ssm")
        return self._ssm

    @property
    def bedrock(self) -> Any:
        if self._bedrock is None:
            self._bedrock = self._client("bedrock-runtime", config=bedrock_config())
        return self._bedrock

    def open_index(self) -> tuple[sqlite3.Connection, str | None]:
        """The ``(connection, etag)`` pair of the shared index, opened once per invocation."""
        if self.index is None:
            self.index = evidence_packet.open_index(self.s3, self.config.bucket, self.config.index_key,
                                                    self._index_cache_dir)
        return self.index

    def close(self) -> None:
        if self.index is not None:
            connection, _etag = self.index
            with contextlib.suppress(sqlite3.Error):
                connection.close()
            self.index = None


def invocation_identity(context: Any) -> tuple[str, str]:
    """``(account_id, region)`` from ``context.invoked_function_arn``; the environment plays no part."""
    arn = getattr(context, "invoked_function_arn", None)
    parts = arn.split(":") if isinstance(arn, str) else []
    if len(parts) < 7 or parts[0] != "arn" or parts[2] != "lambda" or not parts[3] or not parts[4].isdigit():
        raise RuntimeError("the Lambda context must carry invoked_function_arn")
    return parts[4], parts[3]


def build_deps(config: Config, context: Any, **overrides: Any) -> Deps:
    """Dependencies for one invocation; ``overrides`` inject fakes or a pre-opened index."""
    account_id, region = invocation_identity(context)
    return Deps(config, account_id=account_id, region=region, **overrides)


# ---------------------------------------------------------------------------------------------
# Client adapters
# ---------------------------------------------------------------------------------------------

def bedrock_config() -> ClientConfig:
    """One attempt per Converse call: the worker, not the SDK, decides what a failure means."""
    return ClientConfig(retries={"total_max_attempts": 1}, read_timeout=BEDROCK_READ_TIMEOUT,
                        connect_timeout=BEDROCK_CONNECT_TIMEOUT)


def bedrock_model(client: Any) -> Callable[[dict[str, Any]], Mapping[str, Any]]:
    """``request -> client.converse(**request)``; exceptions pass through unchanged, nothing is retried."""

    def converse(request: dict[str, Any]) -> Mapping[str, Any]:
        return client.converse(**request)

    return converse


def send_message(client: Any, url: str, body: Mapping[str, Any]) -> Any:
    """Send one compact JSON message to the queue at ``url``."""
    if not isinstance(url, str) or not url:
        raise ValueError("queue url is not configured")
    return client.send_message(QueueUrl=url, MessageBody=json.dumps(body, sort_keys=True, separators=(",", ":")))


def sqs_sender(client: Any) -> Callable[[str, Mapping[str, Any]], Any]:
    """The ``(url, body)`` sender shape ``lab_gateway.GatewayDeps.queue_sender`` expects."""

    def send(url: str, body: Mapping[str, Any]) -> Any:
        return send_message(client, url, body)

    return send


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------

def handler(event: Any, context: Any, deps: Deps | None = None) -> dict[str, Any]:
    """Lambda entry for every function of the stack; ``deps`` is injected by tests only."""
    if deps is None:
        deps = build_deps(Config.from_env(), context)
    route = ROUTES[deps.config.handler]
    try:
        return route(event, context, deps)
    finally:
        deps.close()


def gateway_handler(event: Any, context: Any, deps: Deps) -> dict[str, Any]:
    """Serve one Function URL event through ``lab_gateway.handle``."""
    config = deps.config
    gateway_deps = lab_gateway.GatewayDeps(
        table=deps.table, receipts=deps.receipts, s3=deps.s3, bucket=config.bucket, account_id=deps.account_id,
        answer_queue_url=config.answer_queue_url, queue_sender=lambda url, body: send_message(deps.sqs, url, body),
        index_opener=deps.open_index, now=deps.now, policy_revision=config.policy_revision,
    )
    response = lab_gateway.handle(event, deps=gateway_deps)
    LOGGER.info("gateway status=%s error=%s", response.get("statusCode"), _response_error(response))
    return response


def answer_handler(event: Any, context: Any, deps: Deps) -> dict[str, Any]:
    """Claim and answer each delivered job; report retriable records as partial batch failures.

    The claim comes first and the shared index is opened only after it succeeded: a live lease
    (``lease_held``) is retriable and reported, any other refused claim is a duplicate delivery
    of a job that already moved on and is dropped, and neither pays for the index download. An
    index that cannot be opened, or an exception the worker did not classify, leaves the job to
    its lease and reports the record; ``lab_answer`` owns every close of a claimed job.
    """
    config = deps.config
    if not config.answer_model_id:
        raise RuntimeError("LAB_ANSWER_MODEL_ID is required by the answer handler")
    failed: list[str] = []
    for message_id, record in _records(event):
        try:
            delivery = _delivery(record, lab_jobs.KIND_ANSWER)
        except ValueError:
            LOGGER.warning("answer message_id=%s action=parse status=failed error=%s", message_id, INVALID_MESSAGE)
            failed.append(message_id)
            continue
        job_id = delivery["job_id"]
        try:
            job = lab_jobs.claim(deps.table, job_id, delivery["outbox_id"], LEASE_SECONDS, deps.now())
        except InvalidTransition as exc:
            if exc.code == LEASE_HELD:
                LOGGER.info("answer job_id=%s action=claim status=retry error=%s", job_id, exc.code)
                failed.append(message_id)
            else:
                LOGGER.info("answer job_id=%s action=claim status=dropped error=%s", job_id, exc.code)
            continue
        except NotFound as exc:
            LOGGER.info("answer job_id=%s action=claim status=dropped error=%s", job_id, exc.code)
            continue
        except Exception as exc:  # noqa: BLE001 - the record is redelivered
            LOGGER.warning("answer job_id=%s action=claim status=failed error=%s", job_id, _code(exc))
            failed.append(message_id)
            continue
        try:
            index = deps.open_index()
        except Exception as exc:  # noqa: BLE001 - the job keeps its lease; SQS redelivers after it expires
            LOGGER.warning("answer job_id=%s action=open_index status=failed error=%s", job_id, _code(exc))
            failed.append(message_id)
            continue
        try:
            result = lab_answer.answer_job(
                job, table=deps.table, receipts=deps.receipts, s3=deps.s3, bucket=config.bucket,
                index=index, model=bedrock_model(deps.bedrock), model_id=config.answer_model_id,
                reasoning=config.answer_reasoning, now=deps.now(), clock=deps.now, limits=PACKET_LIMITS,
                remaining_ms=_remaining(context), pages=deps.pages,
            )
        except Exception as exc:  # noqa: BLE001 - the job keeps its lease; SQS redelivers after it expires
            LOGGER.warning("answer job_id=%s action=answer status=failed error=%s", job_id, _code(exc))
            failed.append(message_id)
            continue
        LOGGER.info("answer job_id=%s member_id=%s status=%s calls=%s", job_id, job.get("member_id"),
                    result.get("status"), result.get("calls"))
    return _batch(failed)


def triage_handler(event: Any, context: Any, deps: Deps) -> dict[str, Any]:
    """Triage each delivered answer job; drop duplicates, report unexpected failures.

    Triage jobs are finished answer jobs, so nothing is claimed here. ``InvalidTransition`` (the
    job is not ``completed``/``partial``) and ``NotFound`` are duplicate deliveries and are
    dropped; ``lab_triage`` records Jev failures itself, so only an unclassified exception
    reports the record.
    """
    config = deps.config
    if not config.jev_parameter:
        raise RuntimeError("LAB_JEV_PARAMETER is required by the triage handler")

    def secret_reader() -> str:
        return jev_client.read_secret(deps.ssm, config.jev_parameter)

    failed: list[str] = []
    for message_id, record in _records(event):
        try:
            delivery = _delivery(record, lab_jobs.KIND_TRIAGE)
        except ValueError:
            LOGGER.warning("triage message_id=%s action=parse status=failed error=%s", message_id, INVALID_MESSAGE)
            failed.append(message_id)
            continue
        job_id = delivery["job_id"]
        try:
            verdict = lab_triage.triage_job(
                {"job_id": job_id}, table=deps.table, receipts=deps.receipts, s3=deps.s3, bucket=config.bucket,
                index=deps.open_index(), jev_post=deps.jev_post, secret_reader=secret_reader, now=deps.now(),
            )
        except (InvalidTransition, NotFound) as exc:
            LOGGER.info("triage job_id=%s action=triage status=dropped error=%s", job_id, exc.code)
            continue
        except Exception as exc:  # noqa: BLE001 - nothing was written; the record is redelivered
            LOGGER.warning("triage job_id=%s action=triage status=failed error=%s", job_id, _code(exc))
            failed.append(message_id)
            continue
        LOGGER.info("triage job_id=%s status=%s candidate_status=%s error=%s", job_id, verdict.get("status"),
                    verdict.get("candidate_status"), verdict.get("error_code"))
    return _batch(failed)


def outbox_handler(event: Any, context: Any, deps: Deps) -> dict[str, Any]:
    """Relay pending outbox rows to their queues, then sweep expired leases once; the event is ignored.

    A row is marked ``sent`` only after ``send_message`` returned. Research rows are relayed only
    when ``LAB_RESEARCH_QUEUE_URL`` is set and ``LAB_RESEARCH_CONSUMER_ENABLED`` is ``true``.
    The sweep (``lab_jobs.sweep_expired_leases``, design section 8) closes at most
    ``LEASE_SWEEP_LIMIT`` ``running`` jobs whose lease has ended as ``outcome_unknown`` with their
    money still held, and its counts are reported under ``lease_sweep``; a sweep that fails is
    reported there with its error code and never hides the relay counts.
    """
    config = deps.config
    urls = {lab_jobs.KIND_ANSWER: config.answer_queue_url, lab_jobs.KIND_TRIAGE: config.triage_queue_url,
            lab_jobs.KIND_RESEARCH: config.research_queue_url}
    report: dict[str, dict[str, Any]] = {}
    for kind in RELAY_KINDS:
        if kind == lab_jobs.KIND_RESEARCH and not config.research_consumer_enabled:
            report[kind] = dict(RESEARCH_SKIPPED)
            continue
        if not urls[kind]:
            report[kind] = dict(QUEUE_UNCONFIGURED)
            continue
        report[kind] = _relay(deps, kind, urls[kind])
    return {"status": "ok", "handler": OUTBOX, "kinds": report, "lease_sweep": _sweep_leases(deps)}


def _sweep_leases(deps: Deps) -> dict[str, Any]:
    """Close expired leases once and report ``lab_jobs.sweep_expired_leases``'s counts, or the failure's code."""
    try:
        counts = lab_jobs.sweep_expired_leases(deps.table, deps.now(), limit=LEASE_SWEEP_LIMIT)
    except Exception as exc:  # noqa: BLE001 - reconciliation must not fail the relay that ran before it
        LOGGER.warning("outbox action=lease_sweep status=failed error=%s", _code(exc))
        return {"status": "failed", "error": _code(exc)}
    LOGGER.info("outbox action=lease_sweep expired=%s closed=%s skipped=%s job_ids=%s", counts["expired"],
                counts["closed"], counts["skipped"], ",".join(counts["job_ids"]))
    return {"status": "swept", **counts}


def _relay(deps: Deps, kind: str, url: str) -> dict[str, Any]:
    rows = lab_jobs.pending_outbox(deps.table, kind, OUTBOX_BATCH)
    counts: dict[str, Any] = {"status": "relayed", "pending": len(rows), "sent": 0, "failed": 0, "unmarked": 0}
    for row in rows:
        outbox_id, job_id = row["outbox_id"], row["job_id"]
        try:
            send_message(deps.sqs, url, {"outbox_id": outbox_id, "job_id": job_id, "kind": kind})
        except Exception as exc:  # noqa: BLE001 - the row stays pending for the next relay
            counts["failed"] += 1
            LOGGER.warning("outbox kind=%s job_id=%s action=send status=failed error=%s", kind, job_id, _code(exc))
            continue
        counts["sent"] += 1
        try:
            lab_jobs.mark_sent(deps.table, outbox_id, row["revision"], now=deps.now())
        except (ConditionFailed, StoreError) as exc:
            counts["unmarked"] += 1
            LOGGER.warning("outbox kind=%s job_id=%s action=mark_sent status=unmarked error=%s", kind, job_id, _code(exc))
    LOGGER.info("outbox kind=%s pending=%s sent=%s failed=%s unmarked=%s", kind, counts["pending"], counts["sent"],
                counts["failed"], counts["unmarked"])
    return counts


def research_handler(event: Any, context: Any, deps: Deps) -> dict[str, Any]:
    """Claim and run each delivered approved research job; inert until the consumer is enabled.

    While ``LAB_RESEARCH_CONSUMER_ENABLED`` is not ``true`` the handler reads and writes nothing
    and returns ``RESEARCH_DISABLED``, so the code path can be deployed before the event source
    is switched on. Enabled, it follows ``answer_handler``: the claim comes first, the shared
    index is opened only after it succeeded, a live lease is retriable, other refused claims are
    duplicate deliveries and are dropped, and ``lab_research`` owns every close of a claimed job
    (approval check, one attempt reservation, the engine, the receipt, the terminal state). An
    exception it did not classify leaves the job to its lease and reports the record.
    """
    config = deps.config
    if not config.research_consumer_enabled:
        LOGGER.info("research status=disabled")
        return dict(RESEARCH_DISABLED)
    model_id = config.research_model_id or config.answer_model_id
    if not model_id:
        raise RuntimeError("LAB_RESEARCH_MODEL_ID (or LAB_ANSWER_MODEL_ID) is required by the research handler")
    failed: list[str] = []
    for message_id, record in _records(event):
        try:
            delivery = _delivery(record, lab_jobs.KIND_RESEARCH)
        except ValueError:
            LOGGER.warning("research message_id=%s action=parse status=failed error=%s", message_id, INVALID_MESSAGE)
            failed.append(message_id)
            continue
        job_id = delivery["job_id"]
        try:
            job = lab_jobs.claim(deps.table, job_id, delivery["outbox_id"], LEASE_SECONDS, deps.now())
        except InvalidTransition as exc:
            if exc.code == LEASE_HELD:
                LOGGER.info("research job_id=%s action=claim status=retry error=%s", job_id, exc.code)
                failed.append(message_id)
            else:
                LOGGER.info("research job_id=%s action=claim status=dropped error=%s", job_id, exc.code)
            continue
        except NotFound as exc:
            LOGGER.info("research job_id=%s action=claim status=dropped error=%s", job_id, exc.code)
            continue
        except Exception as exc:  # noqa: BLE001 - the record is redelivered
            LOGGER.warning("research job_id=%s action=claim status=failed error=%s", job_id, _code(exc))
            failed.append(message_id)
            continue
        try:
            index = deps.open_index()
        except Exception as exc:  # noqa: BLE001 - the job keeps its lease; SQS redelivers after it expires
            LOGGER.warning("research job_id=%s action=open_index status=failed error=%s", job_id, _code(exc))
            failed.append(message_id)
            continue
        try:
            result = lab_research.run_research(
                job, table=deps.table, receipts=deps.receipts, s3=deps.s3, bucket=config.bucket, index=index,
                model_client=deps.bedrock, converse=lab_research.converse_once, model_id=model_id,
                reasoning=config.research_reasoning, now=deps.now(), clock=deps.now, remaining_ms=_remaining(context),
            )
        except Exception as exc:  # noqa: BLE001 - the job keeps its lease; SQS redelivers after it expires
            LOGGER.warning("research job_id=%s action=research status=failed error=%s", job_id, _code(exc))
            failed.append(message_id)
            continue
        LOGGER.info("research job_id=%s member_id=%s status=%s research_status=%s calls=%s pages=%s error=%s",
                    job_id, job.get("member_id"), result.get("status"), result.get("research_status"),
                    result.get("model_calls"), len(result.get("published_keys") or []), result.get("error_code"))
    return _batch(failed)


ROUTES: dict[str, Callable[[Any, Any, Deps], dict[str, Any]]] = {
    GATEWAY: gateway_handler, ANSWER: answer_handler, TRIAGE: triage_handler, OUTBOX: outbox_handler,
    RESEARCH: research_handler,
}


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

def _records(event: Any) -> list[tuple[str, Mapping[str, Any]]]:
    """``(messageId, record)`` pairs of an SQS event; records without an id cannot be reported and are skipped."""
    records = event.get("Records") if isinstance(event, Mapping) else None
    if not isinstance(records, list):
        return []
    usable: list[tuple[str, Mapping[str, Any]]] = []
    for record in records:
        message_id = record.get("messageId") if isinstance(record, Mapping) else None
        if isinstance(message_id, str) and message_id:
            usable.append((message_id, record))
        else:
            LOGGER.warning("sqs action=parse status=skipped error=missing_message_id")
    return usable


def _delivery(record: Mapping[str, Any], kind: str) -> dict[str, str]:
    """The ``{outbox_id, job_id, kind}`` body of one outbox message, or ``ValueError``."""
    try:
        body = json.loads(record.get("body") or "")
    except (TypeError, ValueError):
        raise ValueError("message body is not JSON") from None
    if not isinstance(body, Mapping):
        raise ValueError("message body must be a JSON object")
    job_id, outbox_id = body.get("job_id"), body.get("outbox_id")
    if not isinstance(job_id, str) or not job_id or not isinstance(outbox_id, str) or not outbox_id:
        raise ValueError("message needs job_id and outbox_id")
    if body.get("kind") != kind:
        raise ValueError(f"message kind must be {kind}")
    return {"job_id": job_id, "outbox_id": outbox_id, "kind": kind}


def _batch(failed: list[str]) -> dict[str, Any]:
    return {"batchItemFailures": [{"itemIdentifier": message_id} for message_id in failed]}


def _remaining(context: Any) -> Callable[[], int] | None:
    remaining = getattr(context, "get_remaining_time_in_millis", None)
    return remaining if callable(remaining) else None


def _code(exc: BaseException) -> str:
    """A stable error code for a log record: never the exception text."""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    if isinstance(exc, ClientError):
        service_code = exc.response.get("Error", {}).get("Code")
        if isinstance(service_code, str) and service_code:
            return service_code
    return type(exc).__name__


def _response_error(response: Mapping[str, Any]) -> str | None:
    try:
        payload = json.loads(response.get("body") or "")
    except (TypeError, ValueError):
        return None
    if isinstance(payload, Mapping) and payload.get("ok") is False:
        error = payload.get("error")
        return error if isinstance(error, str) else None
    return None
