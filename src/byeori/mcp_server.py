from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .aws_store import AwsStore
from .catalog import AwsCatalog as Catalog
from .config import Settings
from .synthesis import synthesis_page
from .corpus import search_corpus
from .journal_policy import INCLUDE, journal_verdict
from .openalex import OpenAlexClient
from .pipeline import synthesize_step
from .promote import promote_draft
from .search import backlinks, read_page, save_question, search


mcp = FastMCP(
    "OpenAlex Wiki",
    instructions=(
        "Use answer_wiki_question for research questions that need an evidence-grounded answer "
        "from Byeori. The AWS researcher searches, reads stored originals when needed, and "
        "decides whether to revise existing knowledge or create a useful synthesis. This tool "
        "writes to the shared wiki: final answers are archived as question pages, while concept "
        "and overview changes are chosen during research. Use search_wiki and read_wiki_page "
        "for lookup-only requests. Use save_wiki_question to save supplied text; it does not research "
        "or answer a question. Return the substantive answer and distinguish incomplete work "
        "using status, page_errors and connection_errors. To continue an interrupted answer "
        "with a returned trace_key, use resume_wiki_question."
    ),
)


@mcp.tool()
def search_openalex(
    query: str,
    limit: int = 10,
    from_year: int | None = None,
    to_year: int | None = None,
    open_access_only: bool = False,
    openalex_fulltext_only: bool = False,
) -> list[dict[str, Any]]:
    """Search OpenAlex for candidate papers. Results are metadata, not scientific evidence."""
    settings = Settings.from_env()
    search_options = {
        "limit": limit,
        "from_year": from_year,
        "to_year": to_year,
        "oa_only": open_access_only,
        "fulltext_only": openalex_fulltext_only,
    }
    if settings.aws_ingest_function:
        results = AwsStore(settings).search_openalex(query, **search_options)
    else:
        with OpenAlexClient(settings.openalex_api_key) as client:
            results = client.search(
                query,
                **search_options,
            )
    return [
        {
            "work_id": work["work_id"],
            "doi": work["doi"],
            "title": work["title"],
            "publication_year": work["publication_year"],
            "authors": work["authors"],
            "source": work["source"],
            "cited_by_count": work["cited_by_count"],
            "is_open_access": work["is_open_access"],
            "oa_license": work["oa_license"],
            "landing_page_url": work["landing_page_url"],
            "pdf_url": work["pdf_url"],
            "openalex_pdf_url": work["openalex_pdf_url"],
            "grobid_xml_url": work["grobid_xml_url"],
            "topics": work["topics"],
        }
        for work in results
    ]


@mcp.tool()
def get_openalex_work(identifier: str) -> dict[str, Any]:
    """Get metadata by OpenAlex work ID. The direct API path also accepts DOI; AWS lookup requires a work ID."""
    settings = Settings.from_env()
    if settings.aws_ingest_function:
        return AwsStore(settings).get_openalex_work(identifier)
    with OpenAlexClient(settings.openalex_api_key) as client:
        return client.get_work(identifier)


@mcp.tool()
def save_candidate(identifier: str) -> dict[str, Any]:
    """Save an OpenAlex work to this project's candidate catalog without creating a wiki page."""
    settings = Settings.from_env()
    if settings.aws_ingest_function:
        work = AwsStore(settings).get_openalex_work(identifier)
    else:
        with OpenAlexClient(settings.openalex_api_key) as client:
            work = client.get_work(identifier)
    with Catalog(settings) as catalog:
        return catalog.save_candidate(work)


@mcp.tool()
def list_candidates(status: str | None = None) -> list[dict[str, Any]]:
    """List candidates stored in this independent project."""
    settings = Settings.from_env()
    with Catalog(settings) as catalog:
        candidates = catalog.list_candidates(status)
    return [
        {
            "work_id": item["work_id"],
            "doi": item["doi"],
            "title": item["title"],
            "publication_year": item["publication_year"],
            "status": item["status"],
            "stem": item["stem"],
        }
        for item in candidates
    ]


@mcp.tool()
def search_saved_autism_papers(
    query: str = "", backend: str = "aws", from_year: int | None = None,
    to_year: int | None = None, tag: str | None = None,
    open_access_only: bool = False, fulltext_eligible_only: bool = False, limit: int = 50,
    journal_policy: str = "include", corpus: str = "autism-genomics",
) -> dict[str, Any]:
    """Search saved autism-genomics metadata without calling OpenAlex.

    Use backend='aws' for the configured DynamoDB table. Tags are title-keyword labels,
    not verified methods. Full-text eligibility is metadata only. This does not search
    paper bodies or establish scientific evidence. AWS uses a paginated Scan for this pilot.
    journal_policy='include' (default) returns allowlisted journals only; 'review' adds
    same-family titles awaiting the user's decision; 'all' disables the journal filter.
    corpus='autism-genomics' (default) keeps this tool to the corpus it is named for; the
    catalog also holds 'lab-shared', the PDFs lab members put in the shared folder, and
    corpus='all' searches every intake. Each result names the corpus it came from.
    """
    return search_corpus(Settings.from_env(), backend=backend, query=query,
                         from_year=from_year, to_year=to_year, tag=tag,
                         oa_only=open_access_only, fulltext_only=fulltext_eligible_only, limit=limit,
                         journal_policy=journal_policy, corpus=corpus)


@mcp.tool()
def attach_original_pdf(identifier: str, absolute_path: str) -> dict[str, Any]:
    """Upload a user-provided PDF directly to S3 and record its SHA-256; preserve the source file."""
    settings = Settings.from_env()
    with Catalog(settings) as catalog:
        return catalog.attach_pdf(identifier, Path(absolute_path).expanduser().resolve())


@mcp.tool()
def aws_status() -> dict[str, Any]:
    """Check AWS credentials and whether optional S3/DynamoDB destinations are configured."""
    return AwsStore(Settings.from_env()).status()


@mcp.tool()
def ingest_candidate_to_aws(identifier: str) -> dict[str, Any]:
    """Store one candidate in AWS and run the serverless OpenAlex full-text ingest function.

    Refuses any journal outside the test allowlist; check journal_verdict in search results first.
    """
    settings = Settings.from_env()
    with Catalog(settings) as catalog:
        candidate = catalog.get_candidate(identifier)
    verdict = journal_verdict(candidate["record"].get("source"))
    if verdict["verdict"] != INCLUDE:
        raise ValueError(
            f"{candidate['work_id']} is in '{candidate['record'].get('source') or 'unknown journal'}' "
            f"({verdict['verdict']}: {verdict['reason']}); the test corpus ingests allowlisted journals only"
        )
    store = AwsStore(settings)
    catalog_result = store.push_candidate(candidate)
    if settings.aws_bucket:
        store.upload_json(candidate["record"], f"candidates/{candidate['work_id']}.json")
    ingest_result = store.ingest_work(candidate["work_id"])
    return {"catalog": catalog_result, "ingest": ingest_result}


@mcp.tool()
def read_aws_source(
    identifier: str,
    start: int = 0,
    max_chars: int = 4000,
    kind: str = "source",
) -> dict[str, Any]:
    """Ask AWS for a bounded excerpt of extracted text (kind='source') or a draft (kind='draft')."""
    settings = Settings.from_env()
    with Catalog(settings) as catalog:
        work_id = catalog.get_candidate(identifier)["work_id"]
    store = AwsStore(settings)
    if kind == "draft":
        key = store.get_item(work_id).get("draft_key") or f"wiki/drafts/{work_id}.md"
    elif kind == "source":
        key = f"sources/{work_id}.md"
    else:
        raise ValueError("kind must be source or draft")
    return store.read_text(key, start=start, max_chars=max_chars)


@mcp.tool()
def draft_page_in_aws(identifier: str, model_id: str | None = None) -> dict[str, Any]:
    """Have the AWS Lambda draft an evidence note with Bedrock from an ingested paper's extraction.

    The result is a model_draft, not reviewed wiki content. Refuses journals outside the allowlist.
    Records tokens, seconds, and an estimated cost in the local ledger.
    """
    settings = Settings.from_env()
    with Catalog(settings) as catalog:
        candidate = catalog.get_candidate(identifier)
    verdict = journal_verdict(candidate["record"].get("source"))
    if verdict["verdict"] != INCLUDE:
        raise ValueError(f"{candidate['work_id']} is not in an allowlisted journal ({verdict['reason']})")
    import time
    from . import costs
    started = time.monotonic()
    result = AwsStore(settings).draft_work(candidate["work_id"], model_id)
    usage = result.get("usage") or {}
    result["estimated_usd"] = costs.estimate_draft_usd(result.get("model_id", ""), usage)
    costs.record(settings.state_dir, {
        "step": "draft", "work_id": candidate["work_id"], "status": result.get("status"),
        "model_id": result.get("model_id"), "seconds": round(time.monotonic() - started, 1),
        "lambda_seconds": result.get("seconds"), "input_tokens": usage.get("inputTokens"),
        "output_tokens": usage.get("outputTokens"), "estimated_usd": result["estimated_usd"],
        "basis": "Anthropic list prices applied to reported token counts; not reconciled against the AWS bill",
    })
    return result


def wiki_upload_key(relative_path: str) -> str:
    """Restrict direct text publication to the shared wiki namespace."""
    from pathlib import PurePosixPath
    path = PurePosixPath(relative_path)
    if (path.is_absolute() or ".." in path.parts or "\\" in relative_path
            or path.suffix != ".md" or len(path.parts) < 2
            or path.parts[0] not in {"sources", "wiki"}):
        raise ValueError("relative_path must be a Markdown key under sources/ or wiki/")
    key = path.as_posix()
    return f"wiki/{key}" if key.startswith("sources/") else key


@mcp.tool()
def upload_wiki_markdown(relative_path: str, markdown: str) -> dict[str, str]:
    """Publish supplied Markdown directly to S3 without a local file.

    sources/<id>.md maps to wiki/sources/<id>.md, preserving extraction objects.
    """
    settings = Settings.from_env()
    return AwsStore(settings).put_text(wiki_upload_key(relative_path), markdown)


@mcp.tool()
def promote_draft_in_aws(
    identifier: str,
    reviewer: str,
    method: str = "agent-session",
    note: str | None = None,
    edited: bool = False,
    reviewed_text: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Read a draft from S3, check its provenance, and publish the reviewed note to S3.

    Explicit reviewed_text with edited=True supports corrections without a local mirror.
    dry_run performs the checks without publishing.
    """
    settings = Settings.from_env()
    with Catalog(settings) as catalog:
        work_id = catalog.get_candidate(identifier)["work_id"]
    store = AwsStore(settings) if (settings.aws_bucket or settings.aws_table) else None
    import time
    from . import costs
    started = time.monotonic()
    result = promote_draft(settings, work_id, reviewer=reviewer, method=method, note=note,
                           edited=edited, dry_run=dry_run, store=store, reviewed_text=reviewed_text)
    if not dry_run:
        costs.record(settings.state_dir, {
            "step": "promote", "work_id": work_id, "reviewer": reviewer, "review_method": method,
            "seconds": round(time.monotonic() - started, 1), "estimated_usd": 0.0,
            "basis": "review effort is not metered here; a bedrock-review pass would record its own draft step",
        })
    return result


@mcp.tool()
def synthesize_topic_in_aws(topic: str, title: str, identifiers: list[str], model_id: str | None = None) -> dict[str, Any]:
    """Have the Lambda write an overview page (Scope, Synthesis, Open questions, Related papers)
    with Bedrock from up to 20 reviewed evidence notes. topic is the page's lowercase slug. The
    page lands in S3 wiki/overviews/<slug>.md."""
    settings = Settings.from_env()
    with Catalog(settings) as catalog:
        work_ids = [catalog.get_candidate(identifier)["work_id"] for identifier in identifiers]
    return synthesize_step(settings, AwsStore(settings), topic, title, work_ids, model_id)


@mcp.tool()
def answer_wiki_question(question: str, tags: list[str] | None = None,
                         reread: str = "auto") -> dict[str, Any]:
    """Ask Byeori to answer a research question and maintain its wiki in AWS.

    Returns the answer together with saved pages, connection updates, and trace metadata.
    Final answers are archived as question pages. Byeori decides which existing scientific
    passages to revise and whether an insight deserves a new concept or overview page.
    Search, reading, generation, and page updates run in AWS. With reread='auto', Byeori
    decides when to return to stored originals; 'never' and 'always' select other modes.
    """
    return AwsStore(Settings.from_env()).answer_question(question, tags=tags, reread=reread)


@mcp.tool()
def resume_wiki_question(trace_key: str) -> dict[str, Any]:
    """Continue an interrupted Byeori answer from its returned AWS trace_key.

    The saved evidence and prior wiki changes stay in AWS and are reused there.
    Returns the actual answer and the completed or remaining publication results.
    """
    return AwsStore(Settings.from_env()).resume_question(trace_key)


@mcp.tool()
def search_wiki(query: str, limit: int = 10, doc_type: str | None = None, category: str | None = None,
                backend: str = "auto") -> dict[str, Any]:
    """BM25 search over this wiki's reviewed content: evidence notes (per paper), paper pages, and
    overview pages, and question pages. Runs in the AWS Lambda against the S3 index when configured (backend auto/aws),
    No local fallback is used. Each hit names the document, the section (Methods, Results,
    Synthesis, ...), a snippet, and the path. Use read_wiki_page to read the hit. doc_type filters
    to note, paper, overview, concept, or question. category limits the search to one of the lab's sub-wikis
    (asd-ndd, asd-models, long-read, drug-resistance, ...); leave it out to search across all of
    them, which is how a cross-topic connection surfaces. Cite hits as work IDs or page slugs."""
    return search(Settings.from_env(), query, limit=limit, doc_type=doc_type, category=category, backend=backend)


@mcp.tool()
def read_wiki_page(doc_type: str, doc_id: str, section: str | None = None,
                   start: int = 0, max_chars: int = 4000) -> dict[str, Any]:
    """Ask AWS for a page outline, or at most 8000 characters of a named section.

    With no section, returns metadata and section names without the body. Use the exact heading
    to read an excerpt and next_start/has_more to continue. AWS resolves the S3 key and selects
    the text; no whole page or index is downloaded into the client. Fetch backlinks separately.
    """
    return read_page(Settings.from_env(), doc_type, doc_id, section=section,
                     start=start, max_chars=max_chars)


@mcp.tool()
def wiki_backlinks(doc_type: str, doc_id: str, backend: str = "auto") -> dict[str, Any]:
    """Which pages cite a page. For a note (doc_type note, doc_id its stem): the concept and subtopic
    pages built on it. For a concept or overview: the pages that link to it. Computed from the
    search index's links table; a note that no concept or subtopic cites is a synthesis orphan."""
    return backlinks(Settings.from_env(), doc_type, doc_id, backend=backend)


@mcp.tool()
def save_wiki_question(title: str, question: str, sharper_followup: str, holdings: str, tentative_answer: str,
                       related: list[str], tags: list[str], author: str | None = None) -> dict[str, Any]:
    """Save supplied text; use answer_wiki_question to have Byeori research and answer instead.

    Write a research-question page under wiki/questions/ (S3 only) in llm-wiki's
    question schema: title is the question; sections Question, Sharper follow-up, What the
    knowledge base holds, Tentative answer from the knowledge base, Related Pages (first link is
    the source paper). related lists vault-root-relative wikilinks such as papers/W1992748173 or
    overviews/de-novo. author defaults to the IAM user behind the credentials in use; leave it
    out unless a page is being written on someone else's behalf."""
    return save_question(Settings.from_env(), title=title, question=question, sharper_followup=sharper_followup,
                         holdings=holdings, tentative_answer=tentative_answer, related=related, tags=tags,
                         author=author)


@mcp.tool()
def update_wiki_page(doc_type: str, doc_id: str, established: str = "", corrections: str = "",
                     evidence_notes: list[str] | None = None) -> dict[str, Any]:
    """Revise an existing synthesis page with what you have just established.

    This is how a page grows. It is not planned and written once: when an answer you have just
    given bears on a subtopic or concept page, call this and the page carries it from then on.
    You decide whether it is worth carrying; nothing here does that for you.

    doc_type is 'subtopic' or 'concept'. doc_id is 'category/slug' for a subtopic (for example
    asd-ndd/rare-coding-de-novo-discovery) and the slug for a concept. `established` is what the
    evidence now shows, in your own words. `corrections` names what the page currently gets wrong
    and what it should say instead; the wrong statement is replaced, not annotated. `evidence_notes`
    are the stems behind both, and they join the page's members, which is how a paper the original
    partition never assigned gets into the page.

    Everything the update does not touch comes back unchanged. The previous version stays in S3, so
    a revision is recoverable and the page's history is readable.
    """
    if doc_type not in ("subtopic", "concept"):
        raise ValueError("doc_type must be subtopic or concept; a category page is assembled from its subtopics")
    return synthesis_page(Settings.from_env(), kind=doc_type, ident=doc_id, mode="update",
                          established=established, corrections=corrections,
                          evidence_notes=evidence_notes or [])


@mcp.tool()
def rebuild_wiki_index() -> dict[str, Any]:
    """Rebuild and publish the index in AWS from S3; no client-side download or build."""
    return AwsStore(Settings.from_env()).build_wiki_index()



def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
