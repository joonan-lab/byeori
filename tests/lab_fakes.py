"""In-memory AWS surfaces shared by the lab (student workflow) tests.

Every fake mirrors the exact conditional semantics the production code relies on: S3 puts with
IfNoneMatch/IfMatch, versioned objects with ETags, and a DynamoDB port whose transactions are
all-or-nothing. Nothing here touches the network or the file system.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import sqlite3
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import ClientError

from byeori.lab_store import Check, ConditionFailed, Delete, Put, Update


# ---------------------------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------------------------

class MemoryS3:
    """Versioned bucket with conditional writes. ``writes`` records every put key in order.

    ``conflict_keys`` scripts S3's 409 ``ConditionalRequestConflict``: the first put of each
    listed key raises it (a concurrent conditional write was in progress) and stores nothing;
    the key is then removed from the set, so the retry sees ordinary semantics. The default
    constructor never conflicts.
    """

    def __init__(self, objects: dict[str, bytes | str] | None = None, *, conflict_keys: set[str] | None = None):
        self.objects: dict[str, bytes] = {}
        self.versions: dict[str, list[bytes]] = {}
        self.writes: list[tuple[str, dict[str, Any]]] = []
        self.reads: list[str] = []
        self.conflict_keys: set[str] = set(conflict_keys or ())
        for key, body in (objects or {}).items():
            self._store(key, body.encode("utf-8") if isinstance(body, str) else body)

    def _store(self, key: str, body: bytes) -> None:
        self.objects[key] = body
        self.versions.setdefault(key, []).append(body)

    def etag(self, key: str) -> str:
        return '"' + hashlib.md5(self.objects[key]).hexdigest() + '"'

    def version_id(self, key: str) -> str:
        return f"v{len(self.versions[key])}"

    def put_object(self, *, Bucket, Key, Body, ContentType=None, **conditions):
        body = Body.read() if hasattr(Body, "read") else (Body.encode("utf-8") if isinstance(Body, str) else bytes(Body))
        if Key in self.conflict_keys:
            self.conflict_keys.discard(Key)
            raise ClientError({"Error": {"Code": "ConditionalRequestConflict",
                                         "Message": "Conditional request cannot be completed"},
                               "ResponseMetadata": {"HTTPStatusCode": 409}}, "PutObject")
        if conditions.get("IfNoneMatch") == "*" and Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}, "ResponseMetadata": {"HTTPStatusCode": 412}}, "PutObject")
        if "IfMatch" in conditions and (Key not in self.objects or self.etag(Key) != conditions["IfMatch"]):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}, "ResponseMetadata": {"HTTPStatusCode": 412}}, "PutObject")
        self._store(Key, body)
        self.writes.append((Key, dict(conditions)))
        return {"ETag": self.etag(Key), "VersionId": self.version_id(Key)}

    def get_object(self, *, Bucket, Key, VersionId=None, Range=None, **_):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        self.reads.append(Key)
        body = self.objects[Key]
        if VersionId is not None:
            index = int(VersionId[1:]) - 1
            body = self.versions[Key][index]
        if Range is not None:
            # ``bytes=start-end``, inclusive, as S3 serves it.
            start, end = (int(x) for x in Range.removeprefix("bytes=").split("-"))
            body = body[start:end + 1]
        return {"Body": io.BytesIO(body), "ETag": self.etag(Key), "VersionId": self.version_id(Key),
                "ContentLength": len(body)}

    def head_object(self, *, Bucket, Key, **_):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ETag": self.etag(Key), "VersionId": self.version_id(Key), "ContentLength": len(self.objects[Key])}

    def download_file(self, Bucket, Key, Filename):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "GetObject")
        self.reads.append(Key)
        with open(Filename, "wb") as handle:
            handle.write(self.objects[Key])

    def list_objects_v2(self, *, Bucket, Prefix="", MaxKeys=1000, ContinuationToken=None, **_):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + MaxKeys]
        result = {"Contents": [{"Key": k, "Size": len(self.objects[k])} for k in page], "KeyCount": len(page)}
        if start + MaxKeys < len(keys):
            result["NextContinuationToken"] = str(start + MaxKeys)
            result["IsTruncated"] = True
        return result

    def text(self, key: str) -> str:
        return self.objects[key].decode("utf-8")

    def json(self, key: str) -> Any:
        return json.loads(self.objects[key])


# ---------------------------------------------------------------------------------------------
# DynamoDB port
# ---------------------------------------------------------------------------------------------

class MemoryTable:
    """Implements ``byeori.lab_store.TablePort`` with all-or-nothing transactions.

    ``query`` follows DynamoDB's ``LastEvaluatedKey`` contract: a cursor comes back whenever the
    page holds exactly ``limit`` rows, even when the next page turns out to be empty, so cursor
    loops must tolerate an empty final page.
    """

    def __init__(self, now=None):
        self.items: dict[tuple[str, str], dict[str, Any]] = {}
        self.now = now or (lambda: datetime.now(UTC).isoformat(timespec="microseconds"))
        self.transactions: list[list[Any]] = []

    def get(self, pk, sk):
        item = self.items.get((pk, sk))
        return copy.deepcopy(item) if item is not None else None

    def put(self, item):
        self.transact([Put(item)])
        return copy.deepcopy(item)

    def update(self, pk, sk, expected_revision, changes):
        self.transact([Update(pk, sk, expected_revision, changes)])
        return self.get(pk, sk)

    def delete(self, pk, sk, expected_revision):
        self.transact([Delete(pk, sk, expected_revision)])

    def query(self, pk, *, sk_prefix="", limit=100, start_after=None, ascending=True):
        if not 1 <= int(limit) <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        rows = sorted((k[1], v) for k, v in self.items.items() if k[0] == pk and k[1].startswith(sk_prefix))
        if not ascending:
            rows.reverse()
        if start_after is not None:
            rows = [(sk, v) for sk, v in rows if (sk > start_after if ascending else sk < start_after)]
        page = rows[:limit]
        next_key = page[-1][0] if len(page) == limit else None
        return [copy.deepcopy(v) for _, v in page], next_key

    def transact(self, operations):
        if not operations:
            raise ValueError("a transaction needs at least one operation")
        if len(operations) > 100:
            raise ValueError("DynamoDB transactions accept at most 100 items")
        touched: set[tuple[str, str]] = set()
        # Staged outcome per key: the new item, or None for a delete. Applied only after every
        # operation passed its condition, so a failing item leaves the table untouched.
        staged: dict[tuple[str, str], dict[str, Any] | None] = {}

        def touch(key):
            if key in touched:
                raise ValueError("a transaction may touch each item once")
            touched.add(key)

        def current_at(key, expected_revision, what):
            current = self.items.get(key)
            if current is None or current.get("revision") != expected_revision:
                raise ConditionFailed(f"{what}: {key}")
            return current

        for op in operations:
            if isinstance(op, Put):
                key = (op.item["pk"], op.item["sk"])
                if op.item.get("revision") != 1:
                    raise ValueError("new items start at revision 1")
                touch(key)
                if key in self.items:
                    raise ConditionFailed(f"exists: {key}")
                staged[key] = copy.deepcopy(op.item)
            elif isinstance(op, Update):
                key = (op.pk, op.sk)
                touch(key)
                current = current_at(key, op.expected_revision, "revision mismatch")
                if {"pk", "sk", "revision", "updated_at"} & set(op.changes):
                    raise ValueError("pk, sk, revision and updated_at are managed by the store")
                if not op.changes:
                    raise ValueError("an update needs at least one change")
                staged[key] = {**copy.deepcopy(current), **copy.deepcopy(op.changes),
                               "revision": current["revision"] + 1, "updated_at": self.now()}
            elif isinstance(op, Check):
                key = (op.pk, op.sk)
                touch(key)
                current_at(key, op.expected_revision, "check failed")
            elif isinstance(op, Delete):
                key = (op.pk, op.sk)
                touch(key)
                current_at(key, op.expected_revision, "revision mismatch")
                staged[key] = None
            else:
                raise TypeError(f"Unsupported transaction operation: {op!r}")
        for key, value in staged.items():
            if value is None:
                del self.items[key]
            else:
                self.items[key] = value
        self.transactions.append(list(operations))

    # test helpers -----------------------------------------------------------------------
    def rows(self, pk_prefix: str = "") -> list[dict[str, Any]]:
        return [copy.deepcopy(v) for k, v in sorted(self.items.items()) if k[0].startswith(pk_prefix)]


def member(table: MemoryTable, member_id: str, *, role: str = "student", user_id: str | None = None,
           user_arn: str | None = None, active: bool = True, policy_revision: str = "2026-09-21-v1",
           created_at: str = "2026-09-21T00:00:00+00:00") -> dict[str, Any]:
    """Register a member and its principal pointer the way the registry would."""
    user_id = user_id or f"AIDA{member_id.upper()}"
    user_arn = user_arn or f"arn:aws:iam::123456789012:user/{member_id}"
    profile = {"pk": f"MEMBER#{member_id}", "sk": "PROFILE", "revision": 1, "created_at": created_at,
               "updated_at": created_at, "member_id": member_id, "principal_id": user_id, "principal_arn": user_arn,
               "role": role, "active": active, "policy_revision": policy_revision}
    pointer = {"pk": f"PRINCIPAL#{user_id}", "sk": "MEMBER", "revision": 1, "created_at": created_at,
               "updated_at": created_at, "member_id": member_id}
    table.put(profile)
    table.put(pointer)
    # lab_members.register also writes the MEMBERS index row that list_members and the usage report
    # read. Seed it directly so this setup helper keeps its two-transaction footprint, which several
    # tests assert on.
    table.items[("MEMBERS", member_id)] = {"pk": "MEMBERS", "sk": member_id, "revision": 1,
                                           "created_at": created_at, "updated_at": created_at,
                                           "member_id": member_id}
    return profile


def iam_event(action: str, body: dict[str, Any] | None = None, *, user_id: str, user_arn: str,
              account_id: str = "123456789012", method: str = "POST", raw_body: str | None = None) -> dict[str, Any]:
    """A Lambda Function URL (payload format 2.0) event authenticated with AWS_IAM."""
    payload = raw_body if raw_body is not None else json.dumps({"action": action, **(body or {})}, ensure_ascii=False)
    return {"version": "2.0", "rawPath": "/", "rawQueryString": "",
            "headers": {"content-type": "application/json"},
            "requestContext": {"accountId": account_id, "http": {"method": method, "path": "/"},
                               "authorizer": {"iam": {"accessKey": "AKIAEXAMPLE", "accountId": account_id,
                                                      "callerId": user_id, "cognitoIdentity": None,
                                                      "principalOrgId": None, "userArn": user_arn, "userId": user_id}}},
            "body": payload, "isBase64Encoded": False}


# ---------------------------------------------------------------------------------------------
# Search index (same schema as the AWS index built by the ingest Lambda)
# ---------------------------------------------------------------------------------------------

_LINK_TYPES = {"sources": "note", "concepts": "concept", "overviews": "overview", "questions": "question", "papers": "paper"}


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    fields: dict[str, str] = {}
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end > 0:
            for line in text[4:end].splitlines():
                key, sep, value = line.partition(":")
                if sep and not line.startswith(" "):
                    fields[key.strip()] = value.strip().strip('"')
            body = text[end + 5:]
    return fields, body


def split_sections(body: str) -> list[tuple[str, str]]:
    parts = re.split(r"^## (.+)$", body, flags=re.M)
    preamble = re.sub(r"^# .+\n?", "", parts[0].strip()).strip()
    rows = ([("", preamble)] if preamble else []) + list(zip(parts[1::2], parts[2::2]))
    return [(name.strip(), content.strip()) for name, content in rows if content.strip()]


def build_index(pages: dict[str, str]) -> bytes:
    """Serialise a contentless FTS5 index over ``{s3_key: markdown}`` exactly like the AWS builder."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE docs (doc_type TEXT, doc_id TEXT, title TEXT, path TEXT, year TEXT, journal TEXT, "
                "doi TEXT, work_ids TEXT, category TEXT, s3_key TEXT, summary TEXT, PRIMARY KEY (doc_type, doc_id))")
    con.execute("CREATE VIRTUAL TABLE sections USING fts5(title, section, content, tokenize='porter unicode61', content='')")
    con.execute("CREATE TABLE section_map (rowid INTEGER PRIMARY KEY, doc_type TEXT, doc_id TEXT, section TEXT)")
    con.execute("CREATE INDEX section_map_doc ON section_map (doc_type, doc_id)")
    con.execute("CREATE TABLE links (from_type TEXT, from_id TEXT, to_type TEXT, to_id TEXT)")
    con.execute("CREATE INDEX links_to ON links (to_type, to_id)")
    for key, text in pages.items():
        if key == "wiki/index.md" or key.startswith("wiki/indexes/"):
            continue
        parts = key.split("/")
        folder = parts[1] if len(parts) > 2 else "papers"
        doc_id = ("/".join(parts[2:]) if len(parts) > 2 else parts[-1])[:-3]
        doc_type = {"sources": "note", "overviews": "overview", "questions": "question", "concepts": "concept"}.get(folder, "paper")
        fields, body = _frontmatter(text)
        if folder == "sources":
            category = fields.get("category") or "note"
        elif folder == "overviews" and len(parts) > 3:
            category = parts[2]
        else:
            category = folder
        title = fields.get("title") or ""
        if not title:
            head = re.search(r"^# (.+)$", body, re.M)
            title = head.group(1).strip() if head else doc_id
        path = f"data/{'sources' if folder == 'sources' else 'wiki/' + folder}/{doc_id}.md"
        summary = next((" ".join(content.split())[:260] for name, content in split_sections(body)
                         if name.startswith("One-line Summary")), "")
        con.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (doc_type, doc_id, title, path, str(fields.get("publication_year") or fields.get("year") or ""),
                     fields.get("journal", ""), fields.get("doi", ""), fields.get("work_ids", ""), category, key,
                     summary))
        for name, content in split_sections(body):
            cur = con.execute("INSERT INTO sections(title, section, content) VALUES (?,?,?)", (title, name, content))
            con.execute("INSERT INTO section_map VALUES (?,?,?,?)", (cur.lastrowid, doc_type, doc_id, name))
        links = []
        for kind, ident in re.findall(r"\[\[(?:wiki/)?([a-z]+)/([^\]|#]+)(?:[|#][^\]]*)?\]\]", body):
            if kind in _LINK_TYPES:
                links.append((_LINK_TYPES[kind], ident.strip().removesuffix(".md")))
        con.executemany("INSERT INTO links VALUES (?,?,?,?)", [(doc_type, doc_id, t, i) for t, i in dict.fromkeys(links)])
    con.commit()
    data = con.serialize()
    con.close()
    return bytes(data)


def index_connection(raw: bytes) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.deserialize(raw)
    return con


def wiki_with_index(pages: dict[str, str], index_key: str = "index/wiki-index-v2.sqlite3") -> MemoryS3:
    """A bucket holding the pages and an index built over them."""
    s3 = MemoryS3(pages)
    s3._store(index_key, build_index(pages))
    return s3


def source_note(title: str = "Paper one", *, stem: str = "paper-one", results: str = "Regional inheritance was stable (n = 120, p = 0.01).",
                limitations: str = "The cohort was small and single-site.", pdf_path: str | None = None) -> str:
    """One evidence note. ``pdf_path`` gives it a stored extraction the reader can open."""
    front = f"pdf_path: {pdf_path}\n" if pdf_path else ""
    return (f"---\ntitle: {title}\ncategory: asd-ndd\nyear: 2020\njournal: Nature\ndoi: 10.1000/{stem}\n{front}---\n\n"
            f"# {title}\n\nOne-line summary of the paper.\n\n## Methods\n\nA cohort design with 120 families.\n\n"
            f"## Results\n\n{results}\n\n## Limitations\n\n{limitations}\n\n"
            "## Related pages\n\n- [[overviews/existing]]\n\n"
            "<!-- byeori:backlinks:start -->\n### Linked pages\n\n- [[concepts/new-insight]]\n<!-- byeori:backlinks:end -->\n")


# ---------------------------------------------------------------------------------------------
# Model, Jev, SSM, SQS, Lambda context
# ---------------------------------------------------------------------------------------------

class FakeConverse:
    """Callable ``request -> response`` returning scripted Bedrock Converse responses.

    ``responses`` items are dicts (returned as-is), exceptions (raised) or ``("after_send", exc)``
    tuples that raise after recording the request, so callers can distinguish an error before
    the request left from an unknown outcome afterwards.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.sent = 0

    def __call__(self, request):
        if not self.responses:
            raise AssertionError("FakeConverse received more calls than scripted")
        response = self.responses.pop(0)
        if isinstance(response, tuple) and response[0] == "after_send":
            self.requests.append(copy.deepcopy(request))
            self.sent += 1
            raise response[1]
        if isinstance(response, Exception):
            raise response
        self.requests.append(copy.deepcopy(request))
        self.sent += 1
        return response


def tool_use(name: str, arguments: dict[str, Any], *, tool_use_id: str = "call-1", usage=None, text: str | None = None,
             stop_reason: str = "tool_use"):
    content = ([{"text": text}] if text else []) + [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": arguments}}]
    return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": stop_reason,
            "usage": usage or {"inputTokens": 1200, "outputTokens": 300, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}}


def cut_off(name: str = "submit_answer", arguments: dict[str, Any] | None = None, *, tool_use_id: str = "cut-1",
            usage=None):
    """A tool call the output limit cut short: Bedrock drops the field it could not finish."""
    return tool_use(name, arguments if arguments is not None else {}, tool_use_id=tool_use_id, usage=usage,
                    stop_reason="max_tokens")


def text_only(text: str, usage=None):
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}, "stopReason": "end_turn",
            "usage": usage or {"inputTokens": 500, "outputTokens": 100, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}}


def jev_response(choice: str = "answer_only", *, probabilities: dict[str, float] | None = None,
                 confidence: float = 0.8, model: str = "jev-1.13.0", input_tokens: int = 3000, output_tokens: int = 40) -> bytes:
    probabilities = probabilities or {"answer_only": 0.9, "needs_lookup": 0.05, "review_candidate": 0.05}
    return json.dumps({"model": model, "answers": {"route": {"type": "choice", "choice": choice, "confidence": confidence,
                                                              "probabilities": probabilities}},
                       "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}).encode("utf-8")


class FakeJev:
    """Callable ``(payload_bytes, secret) -> raw_bytes`` with scripted responses or exceptions."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[bytes, str]] = []

    def __call__(self, payload: bytes, secret: str, **_):
        if not self.responses:
            raise AssertionError("FakeJev received more calls than scripted")
        self.calls.append((payload, secret))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeSsm:
    def __init__(self, value: str = "secret-key-value", name: str = "/byeori/jev/api-key"):
        self.value, self.name = value, name
        self.calls: list[dict[str, Any]] = []

    def get_parameter(self, **request):
        self.calls.append(request)
        if request.get("Name") != self.name or not request.get("WithDecryption"):
            raise ClientError({"Error": {"Code": "ParameterNotFound"}}, "GetParameter")
        return {"Parameter": {"Name": self.name, "Value": self.value, "Version": 2}}


class FakeSqs:
    def __init__(self, fail_urls: set[str] | None = None):
        self.messages: list[tuple[str, dict[str, Any]]] = []
        self.fail_urls = fail_urls or set()

    def send_message(self, *, QueueUrl, MessageBody, **_):
        if QueueUrl in self.fail_urls:
            raise ClientError({"Error": {"Code": "ServiceUnavailable"}}, "SendMessage")
        self.messages.append((QueueUrl, json.loads(MessageBody)))
        return {"MessageId": f"m{len(self.messages)}"}


class FakeContext:
    def __init__(self, remaining_ms: int = 850_000, function_arn: str = "arn:aws:lambda:ap-northeast-2:123456789012:function:byeori-lab"):
        self._remaining, self.invoked_function_arn = remaining_ms, function_arn
        self.aws_request_id = "req-1"

    def get_remaining_time_in_millis(self):
        return self._remaining


def sqs_event(*bodies: dict[str, Any]) -> dict[str, Any]:
    return {"Records": [{"messageId": f"m{i}", "receiptHandle": f"h{i}", "body": json.dumps(body),
                         "eventSource": "aws:sqs"} for i, body in enumerate(bodies, 1)]}


def fixed_clock(start: str = "2026-09-21T09:00:00+00:00"):
    """A ``now()`` callable that advances one second per call, for deterministic ordering."""
    moment = datetime.fromisoformat(start)
    state = {"tick": 0}

    def now():
        from datetime import timedelta
        current = moment + timedelta(seconds=state["tick"])
        state["tick"] += 1
        return current
    return now


def paper_text(stem: str = "paper-one", *, results: str = "Across 312 probands the odds ratio was 2.4 (95% CI 1.8-3.2).") -> str:
    """A stored extraction: the paper itself, carrying numbers the note only summarises."""
    return (f"# The full paper for {stem}\n\n## Abstract\n\nWhat the paper reports.\n\n"
            f"## Methods\n\nRecruitment, sequencing and the analysis plan in full.\n\n"
            f"## Results\n\n{results}\n\n## Discussion\n\nWhat the authors take it to mean.\n")


def paper_assets(stem: str = "paper-one") -> str:
    """The figure and table text uploaded beside an extraction on 2026-09-22."""
    return (f"# {stem}\n\n2 items - Figures 2\n\n## Figure 1\n\n![Figure 1](figure1.png)\n\n"
            "### Caption\n\nFigure 1. Odds ratio by ancestry group; error bars are 95% confidence intervals.\n\n"
            "### Panels\n\na, b\n\n### Mentioned in the text\n\n"
            "- The effect held in every ancestry group ( Fig. 1a ).\n"
            "- Group sizes are given in Fig. 1b (n = 118, 97 and 97).\n\n### Source\n\npage 4\n")
