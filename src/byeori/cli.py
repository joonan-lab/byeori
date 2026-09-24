from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from . import costs
from . import installer
from .aws_store import AwsStore
from .catalog import AwsCatalog as Catalog
from .catalog_export import export_papers_csv
from .config import Settings
from .corpus import apply_policy_to_catalog, collect, date_range, export_report, search_corpus
from .journal_policy import INCLUDE, journal_verdict
from .openalex import OpenAlexClient, OpenAlexError
from .openalex_match import match_openalex
from .benchmark import run_question_benchmark
from .extract import extraction_status, run_extraction
from . import paper_upload
from .papers import resolve_ids, resolve_ids_ncbi, upload_papers
from .pipeline import (draft_step, failed_stems, ingest_step, require_allowlisted_journal,
                       run_pipeline, run_stem_pipeline, source_note_step, synthesize_step)
from .promote import REVIEW_METHODS, promote_draft
from .runs import notes_status, start_notes
from .search import backlinks, categories, read_page, save_question, search
from .synthesis import (failed_synthesis, pull_manifests, push_manifest, read_manifest, start_synthesis, synthesis_page,
                        synthesis_status, upload_hgnc)
from .validation import validate


KIRO_APP_BINARY = Path("/Applications/Kiro CLI.app/Contents/MacOS/kiro-cli")


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def command_init(settings: Settings, args: argparse.Namespace) -> int:
    """Write the env file another lab's install reads; asks on the terminal unless --non-interactive."""
    settings.ensure_directories()
    path = Path(args.env_file)
    existing = installer.read_env(path)
    if args.non_interactive:
        given = dict(item.split("=", 1) for item in args.set or [])
        values = installer.ask_settings(lambda key, default: given.get(key, default), existing)
    else:
        values = installer.ask_settings(installer.prompt_on_terminal, existing)
    installer.write_env(path, values)
    print(f"wrote {path}; next: source it and run `deploy`")
    return 0


def _env_for_scripts(args: argparse.Namespace) -> dict[str, str]:
    env = installer.read_env(Path(args.env_file))
    if not env:
        raise FileNotFoundError(f"{args.env_file} not found; run `init` first")
    return env


def command_deploy(settings: Settings, args: argparse.Namespace) -> int:
    env = _env_for_scripts(args)
    run_env = env
    if not env.get("AWS_KIRO_WIKI_BUCKET"):
        # A first install has no bucket yet: the stack's own data bucket is one of its outputs, so
        # `deploy.sh` (which packages templates into a bucket via `aws cloudformation package`)
        # cannot use it before the stack exists. A small bootstrap bucket holds the package this once.
        session = boto3.Session(profile_name=env.get("AWS_PROFILE"), region_name=env.get("AWS_REGION"))
        account = session.client("sts").get_caller_identity()["Account"]
        region = session.region_name or env.get("AWS_REGION", "us-east-1")
        bucket = installer.ensure_deploy_bucket(session, account, region)
        run_env = {**env, "AWS_KIRO_WIKI_BUCKET": bucket}
        print(f"packaging the deployment templates into {bucket} (created for this first install)")
    code = installer.run_script("deploy.sh", run_env, *(args.parameter or []))
    if code != 0:
        return code
    session = boto3.Session(profile_name=env.get("AWS_PROFILE"), region_name=env.get("AWS_REGION"))
    outputs = installer.outputs_of(session.client("cloudformation"), env["KIRO_WIKI_STACK"])
    installer.write_env(Path(args.env_file), installer.env_from_outputs(outputs))
    print_json(installer.env_from_outputs(outputs))
    return 0


def command_build_workers(settings: Settings, args: argparse.Namespace) -> int:
    return installer.run_script("build_asset_image.sh", _env_for_scripts(args))


def command_deploy_lab(settings: Settings, args: argparse.Namespace) -> int:
    """Deploy the optional student stack, defaulting LAB_STACK to <main stack>-lab so a second
    installation in the same account does not collide with (and silently redeploy) another lab's
    stack of the same default name."""
    env = _env_for_scripts(args)
    if not env.get("LAB_STACK"):
        env = {**env, "LAB_STACK": f"{env['KIRO_WIKI_STACK']}-lab"}
        installer.write_env(Path(args.env_file), {"LAB_STACK": env["LAB_STACK"]})
    print(f"deploying the lab stack {env['LAB_STACK']}")
    return installer.run_script("deploy_lab.sh", env, "--execute")


def command_deploy_jev_eval(settings: Settings, args: argparse.Namespace) -> int:
    """Deploy the optional Jev evaluation stack, defaulting JEV_EVAL_STACK to <main stack>-jev for
    the same reason command_deploy_lab defaults LAB_STACK."""
    env = _env_for_scripts(args)
    if not env.get("JEV_EVAL_STACK"):
        env = {**env, "JEV_EVAL_STACK": f"{env['KIRO_WIKI_STACK']}-jev"}
        installer.write_env(Path(args.env_file), {"JEV_EVAL_STACK": env["JEV_EVAL_STACK"]})
    if not env.get("LAB_JEV_KMS_KEY_ARN"):
        raise RuntimeError("set LAB_JEV_KMS_KEY_ARN in the env file; see docs/JEV.md")
    print(f"deploying the Jev evaluation stack {env['JEV_EVAL_STACK']}")
    return installer.run_script("deploy_jev_eval.sh", env)


def command_grant_client(settings: Settings, args: argparse.Namespace) -> int:
    env = _env_for_scripts(args)
    session = boto3.Session(profile_name=env.get("AWS_PROFILE"), region_name=env.get("AWS_REGION"))
    outputs = installer.outputs_of(session.client("cloudformation"), env["KIRO_WIKI_STACK"])
    account = session.client("sts").get_caller_identity()["Account"]
    policy = installer.build_client_policy(outputs, session.region_name, account, env["KIRO_WIKI_STACK"])
    if args.attach_to_user:
        session.client("iam").put_user_policy(UserName=args.attach_to_user, PolicyName="ByeoriClient",
                                              PolicyDocument=json.dumps(policy))
        print(f"attached inline policy ByeoriClient to {args.attach_to_user}")
    else:
        print_json(policy)
    return 0


def command_doctor(settings: Settings, _: argparse.Namespace) -> int:
    settings.ensure_directories()
    kiro_binary = shutil.which("kiro-cli")
    if kiro_binary is None and KIRO_APP_BINARY.is_file():
        kiro_binary = str(KIRO_APP_BINARY)
    checks = {
        "project_root": str(settings.root),
        "python": sys.version.split()[0],
        "uv": shutil.which("uv") is not None,
        "kiro_cli": bool(kiro_binary),
        "kiro_cli_path": kiro_binary,
        "openalex_api_key": bool(settings.openalex_api_key),
        "aws": AwsStore(settings).status(),
    }
    try:
        if settings.aws_ingest_function:
            work = AwsStore(settings).get_openalex_work("W2741809807")
        else:
            with OpenAlexClient(settings.openalex_api_key) as client:
                work = client.get_work("W2741809807")
        checks["openalex_live"] = bool(work.get("work_id"))
    except (OpenAlexError, ValueError) as exc:
        checks["openalex_live"] = False
        checks["openalex_error"] = str(exc)
    stack = os.environ.get("KIRO_WIKI_STACK")
    if stack and settings.aws_bucket:
        try:
            session = boto3.Session(region_name=settings.aws_region)
            models = installer.model_ids_of(session.client("cloudformation"), stack)
            checks["bedrock"] = installer.check_bedrock_models(session.client("bedrock-runtime"), models)
        except (ClientError, BotoCoreError) as exc:
            checks["bedrock"] = {"error": str(exc)}
    print_json(checks)
    return 0 if checks["uv"] and checks["openalex_live"] else 1


def command_search(settings: Settings, args: argparse.Namespace) -> int:
    """Search OpenAlex under the lab's journal policy; nothing is saved or ingested here."""
    search_options = {
        "limit": args.limit,
        "from_year": args.from_year,
        "to_year": args.to_year,
        "oa_only": args.oa_only,
        "fulltext_only": args.fulltext_only,
    }
    report: dict[str, Any] = {}
    if settings.aws_ingest_function:
        report = AwsStore(settings).search_openalex_report(
            args.query, journal_scope=args.journal_scope, **search_options)
        results = report["results"]
    else:
        with OpenAlexClient(settings.openalex_api_key) as client:
            results = client.search(args.query, **search_options)
        results = [dict(work, **{"journal_scope_verdict": None, "journal_warning": None})
                   for work in results]
    compact = [
        {
            "work_id": work["work_id"],
            "doi": work["doi"],
            "title": work["title"],
            "publication_year": work["publication_year"],
            "authors": work["authors"],
            "source": work["source"],
            "journal_scope_verdict": work.get("journal_scope_verdict"),
            "journal_warning": work.get("journal_warning"),
            "cited_by_count": work["cited_by_count"],
            "is_open_access": work["is_open_access"],
            "oa_license": work["oa_license"],
            "pdf_url": work["pdf_url"],
            "openalex_pdf_url": work["openalex_pdf_url"],
            "grobid_xml_url": work["grobid_xml_url"],
        }
        for work in results
    ]
    print_json({
        "query": args.query,
        "journal_scope": report.get("journal_scope", args.journal_scope),
        "returned": len(compact),
        "outside_list_count": report.get("outside_list_count", 0),
        "refused_count": report.get("refused_count", 0),
        "refused_journals": report.get("refused_journals", []),
        "note": ("Results outside the lab's journal list carry journal_warning. Nothing here is "
                 "saved or ingested; the journal list still decides what may be collected."),
        "results": compact,
    })
    return 0


def command_candidate_add(settings: Settings, args: argparse.Namespace) -> int:
    if settings.aws_ingest_function:
        work = AwsStore(settings).get_openalex_work(args.identifier)
    else:
        with OpenAlexClient(settings.openalex_api_key) as client:
            work = client.get_work(args.identifier)
    with Catalog(settings) as catalog:
        candidate = catalog.save_candidate(work)
    print_json(candidate)
    return 0


def command_candidate_list(settings: Settings, args: argparse.Namespace) -> int:
    with Catalog(settings) as catalog:
        candidates = catalog.list_candidates(args.status)
    compact = [
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
    print_json(compact)
    return 0


def command_attach_pdf(settings: Settings, args: argparse.Namespace) -> int:
    with Catalog(settings) as catalog:
        result = catalog.attach_pdf(args.identifier, Path(args.pdf).expanduser().resolve())
    print_json(result)
    return 0


def command_upload_pdf(settings: Settings, args: argparse.Namespace) -> int:
    if not settings.aws_bucket:
        raise RuntimeError("AWS_KIRO_WIKI_BUCKET is not configured")
    result = paper_upload.upload_one(settings, Path(args.pdf).expanduser().resolve(),
                                     stem=args.stem, source=args.source or "upload-pdf")
    print_json(result)
    return 0 if result["state"] in ("uploaded", "already_present") else 1


def command_validate(settings: Settings, _: argparse.Namespace) -> int:
    errors = validate(settings)
    if errors:
        print("\n".join(errors))
        return 1
    print("AWS validation passed for the published S3 wiki")
    return 0



def command_aws_status(settings: Settings, _: argparse.Namespace) -> int:
    result = AwsStore(settings).status()
    print_json(result)
    return 0 if result["credentials"] else 1


def command_aws_push_candidate(settings: Settings, args: argparse.Namespace) -> int:
    with Catalog(settings) as catalog:
        candidate = catalog.get_candidate(args.identifier)
    store = AwsStore(settings)
    table_result = store.push_candidate(candidate)
    object_result = None
    if settings.aws_bucket:
        object_result = store.upload_json(
            candidate["record"], f"candidates/{candidate['stem']}.json"
        )
    print_json({"dynamodb": table_result, "s3": object_result})
    return 0


def command_aws_ingest_candidate(settings: Settings, args: argparse.Namespace) -> int:
    with Catalog(settings) as catalog:
        candidate = catalog.get_candidate(args.identifier)
    print_json(ingest_step(settings, AwsStore(settings), candidate))
    return 0


def command_aws_draft_page(settings: Settings, args: argparse.Namespace) -> int:
    with Catalog(settings) as catalog:
        candidate = catalog.get_candidate(args.identifier)
    result = draft_step(settings, AwsStore(settings), candidate, args.model)
    print_json(result)
    return 0 if result.get("status") == "model_draft" else 1


def command_aws_synthesize(settings: Settings, args: argparse.Namespace) -> int:
    with Catalog(settings) as catalog:
        work_ids = [catalog.get_candidate(identifier)["work_id"] for identifier in args.identifiers]
    result = synthesize_step(settings, AwsStore(settings), args.topic, args.title, work_ids, args.model)
    print_json(result)
    return 0 if result.get("status") == "model_topic" else 1


def command_aws_pipeline(settings: Settings, args: argparse.Namespace) -> int:
    result = run_pipeline(settings, limit=args.limit, dry_run=args.dry_run,
                          skip_topics=args.skip_topics, model_id=args.model, concurrency=args.concurrency)
    print_json(result)
    return 1 if result["errors"] else 0


def command_filtered_list(settings: Settings, args: argparse.Namespace) -> int:
    print_json(AwsStore(settings).filtered_questions(since=args.since, limit=args.limit))
    return 0


def command_aws_build_index(settings: Settings, _: argparse.Namespace) -> int:
    print_json(AwsStore(settings).build_wiki_index())
    return 0


def command_collect_for_gap(settings: Settings, args: argparse.Namespace) -> int:
    """Search the lab's journals for an approved gap's query and save what may be collected.

    Nothing here fetches a PDF or writes a note: it stops at a candidate, and ingest keeps its own
    checks. Without ``--apply`` the search runs and the report names what would be saved.
    """
    from byeori.gap_collection import collect_for_query

    store = AwsStore(settings)
    with Catalog(settings) as catalog:
        report = collect_for_query(
            args.query, job_id=args.job_id, limit=args.limit, dry_run=not args.apply,
            search=store.search_openalex_report, save=catalog.save_candidate)
    print_json(report)
    return 0


def _ingest_once(settings: Settings, store: AwsStore, candidate: dict[str, Any]) -> dict[str, Any]:
    """Fetch the PDF unless it is already stored; a retry after a failed note must not refetch."""
    from byeori.pipeline import ingest_step

    stored = (store.get_item(candidate["work_id"]) or {}).get("ingest_status")
    if stored in ("fulltext_ready", "model_draft", "draft_failed"):
        return {"status": stored, "skipped_fetch": True}
    return ingest_step(settings, store, candidate)


def command_ingest_for_gap(settings: Settings, args: argparse.Namespace) -> int:
    """Bring an approved gap's candidates into the wiki and rebuild the index once at the end.

    Without ``--apply`` nothing is fetched or written: the report names what would be ingested and
    why each of the others cannot be.
    """
    from byeori.gap_ingest import blocked_list, candidates_for_gap, ingest_for_gap
    from byeori.pipeline import run_paper

    store = AwsStore(settings)
    with Catalog(settings) as catalog:
        rows = catalog.list_candidates()
    chosen = candidates_for_gap(rows, args.job_id)
    report = ingest_for_gap(
        args.job_id, chosen, limit=args.limit, dry_run=not args.apply,
        ingest=lambda candidate: _ingest_once(settings, store, candidate),
        # An OpenAlex paper's note is written by draft + promote (run_paper), which publishes
        # wiki/sources/{work_id}.md. run_stem is the uploaded-PDF route and refuses a work id,
        # which a live run on 2026-09-22 proved by failing on exactly that.
        write_note=lambda catalog_id: run_paper(settings, store, catalog_id),
        rebuild_index=store.build_wiki_index,
        already_here=lambda catalog_id: (store.get_item(catalog_id) or {}).get("source_note_status") == "source_ready",
    )
    # The papers this gap needs that serverless ingest cannot fetch. They are handed to the route
    # that can -- the Optimus scout's legal retrieval, or the user -- rather than left in a log.
    blocked = blocked_list(args.job_id, args.query, report["skipped"], chosen)
    report["blocked"] = blocked
    if blocked["count"] and args.apply and settings.aws_bucket:
        key = f"runs/collection/{args.job_id}/blocked.json"
        store.upload_json(blocked, key)
        report["blocked_key"] = key
    print_json(report)
    return 0


def command_link_lab_questions(settings: Settings, args: argparse.Namespace) -> int:
    """Give each indexed page one line down to its lab questions; reads only unless --apply."""
    import boto3

    from byeori import wiki_question_links

    bucket = settings.aws_bucket
    if not bucket:
        raise SystemExit("set the AWS bucket before linking lab questions")
    report = wiki_question_links.link_pages(boto3.client("s3"), bucket, dry_run=not args.apply,
                                            limit=args.limit)
    print_json(report)
    return 0


def command_wiki_search(settings: Settings, args: argparse.Namespace) -> int:
    print_json(search(settings, args.query, limit=args.limit, doc_type=args.type, category=args.category,
                      backend=args.backend))
    return 0


def command_wiki_categories(settings: Settings, _: argparse.Namespace) -> int:
    print_json(categories(settings))
    return 0


def command_synthesis_coverage(settings: Settings, _: argparse.Namespace) -> int:
    """Where synthesis is missing: notes no concept or overview cites, by category."""
    print_json(AwsStore(settings).synthesis_coverage())
    return 0


def command_read_extraction(settings: Settings, args: argparse.Namespace) -> int:
    """Print a window of a paper's extracted text, for writing its note in this session."""
    store = AwsStore(settings)
    start, printed = args.start, 0
    while True:
        window = store.read_extraction(args.stem, start=start, max_chars=args.max_chars)
        sys.stdout.write(window["text"])
        printed += len(window["text"])
        start = window["next_start"]
        if start is None or not args.all:
            break
    print(f"\n\n[{printed:,} of {window['total_chars']:,} characters"
          + (f"; continue with --start {start}" if start else "; end of text") + "]", file=sys.stderr)
    return 0


def command_publish_source_note(settings: Settings, args: argparse.Namespace) -> int:
    """Publish a note written in this session, for a paper that has none."""
    body = args.markdown.read_text(encoding="utf-8")
    result = AwsStore(settings).publish_source_note(args.stem, body, model_id=args.model_id)
    print_json({k: v for k, v in result.items() if k != "publication"})
    return 0 if result.get("published") else 1


def command_build_category_catalogs(settings: Settings, args: argparse.Namespace) -> int:
    """One catalog per field, the way llm-wiki's indexes/ folder works."""
    result = AwsStore(settings).build_category_catalogs(min_notes=args.min_notes)
    print_json({k: v for k, v in result.items() if k != "written"})
    return 1 if result["errors"] else 0


def command_classify_notes(settings: Settings, args: argparse.Namespace) -> int:
    """File the notes the intake left in `other`, in AWS, a batch of stems at a time."""
    import json as _json

    store = AwsStore(settings)
    stems = list(args.stems or [])
    if not stems:
        stems = store.notes_in_category("other", limit=args.limit or 0)
    new_folders = []
    for row in (args.new_folder or []):
        name, _, scope = row.partition(":")
        new_folders.append({"name": name.strip(), "scope": scope.strip()})
    run = {"at": datetime.now(UTC).replace(microsecond=0).isoformat(),
           "applied": bool(args.apply), "selected": len(stems),
           "new_folders": new_folders, "only_new_folders": bool(args.only_new_folders),
           "states": {}, "usage": {}, "papers": [], "errors": []}
    for start in range(0, len(stems), args.batch):
        window = stems[start:start + args.batch]
        try:
            result = store.classify_notes(window, apply=args.apply, model_id=args.model,
                                          new_folders=new_folders,
                                          only_new_folders=args.only_new_folders,
                                          only_into=args.only_into)
        except Exception as exc:  # noqa: BLE001 - one batch must not lose the ones already filed
            run["errors"].append({"stems": window, "error": f"{type(exc).__name__}: {exc}"})
            continue
        run["papers"].extend(result["papers"])
        for state, count in result["states"].items():
            run["states"][state] = run["states"].get(state, 0) + count
        for key, value in (result.get("usage") or {}).items():
            run["usage"][key] = run["usage"].get(key, 0) + value
        print(f"[{min(start + args.batch, len(stems))}/{len(stems)}] {result['states']}",
              file=sys.stderr, flush=True)
    run["estimated_usd"] = costs.estimate_draft_usd(args.model or "global.anthropic.claude-opus-5",
                                                    run["usage"])
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(_json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print_json({k: v for k, v in run.items() if k != "papers"})
    return 1 if run["errors"] else 0


def command_file_notes(settings: Settings, args: argparse.Namespace) -> int:
    """Put named notes in a named field, in AWS, a batch of stems at a time. No model is called."""
    store = AwsStore(settings)
    stems = list(args.stems or [])
    states: dict[str, int] = {}
    for start in range(0, len(stems), args.batch):
        result = store.file_notes(stems[start:start + args.batch], args.category, apply=args.apply)
        for state, count in result["states"].items():
            states[state] = states.get(state, 0) + count
    print_json({"category": args.category, "stems": len(stems), "applied": bool(args.apply),
                "states": states})
    return 0


def command_fields(settings: Settings, args: argparse.Namespace) -> int:
    """Read the fields a person opened; open or close one."""
    opening = []
    for row in (args.open or []):
        name, _, scope = row.partition(":")
        opening.append({"name": name.strip(), "scope": scope.strip()})
    print_json(AwsStore(settings).fields(open_fields=opening, close=args.close or []))
    return 0


def command_sync_note_categories(settings: Settings, args: argparse.Namespace) -> int:
    """Make the catalogue agree with the notes about which field each one is in."""
    print_json(AwsStore(settings).sync_note_categories(apply=args.apply))
    return 0


def command_wiki_read(settings: Settings, args: argparse.Namespace) -> int:
    result = read_page(settings, args.type, args.doc_id, section=args.section,
                       start=args.start, max_chars=args.max_chars)
    print_json(result)
    return 0


def command_wiki_save_question(settings: Settings, args: argparse.Namespace) -> int:
    spec = json.loads(Path(args.spec_file).read_text(encoding="utf-8"))
    print_json(save_question(settings, title=spec["title"], question=spec["question"], sharper_followup=spec["sharper_followup"],
                             holdings=spec["holdings"], tentative_answer=spec["tentative_answer"],
                             related=spec["related"], tags=spec.get("tags", [])))
    return 0


def command_upload_papers(settings: Settings, args: argparse.Namespace) -> int:
    result = upload_papers(settings, Path(args.llm_wiki).expanduser(), select=args.select, stems=args.stems or None,
                           limit=args.limit, concurrency=args.concurrency, dry_run=args.dry_run)
    print_json(result)
    return 1 if result["errors"] else 0


def command_resolve_ids(settings: Settings, args: argparse.Namespace) -> int:
    if args.source == "ncbi":
        print_json(resolve_ids_ncbi(settings, Path(args.llm_wiki).expanduser(), select=args.select, limit=args.limit))
        return 0
    result = resolve_ids(settings, Path(args.llm_wiki).expanduser(), select=args.select, stems=args.stems or None,
                         limit=args.limit, concurrency=args.concurrency, direct=args.direct)
    print_json({k: (v if k not in ("not_found", "no_doi") else len(v)) for k, v in result.items()})
    return 1 if result["errors"] else 0


def command_aws_extract(settings: Settings, args: argparse.Namespace) -> int:
    print_json(run_extraction(settings, stems=args.stems or None, tasks=args.tasks, limit=args.limit,
                              force=args.force, dry_run=args.dry_run, preprocess=args.preprocess))
    return 0


def command_aws_extract_status(settings: Settings, _: argparse.Namespace) -> int:
    print_json(extraction_status(settings))
    return 0


def command_aws_source_note(settings: Settings, args: argparse.Namespace) -> int:
    result = source_note_step(settings, AwsStore(settings), args.stem, args.model)
    print_json(result)
    return 0 if result.get("status") == "source_ready" else 1


def command_aws_pipeline_stems(settings: Settings, args: argparse.Namespace) -> int:
    result = run_stem_pipeline(settings, stems=args.stems or None, limit=args.limit, dry_run=args.dry_run,
                               model_id=args.model, only_failed=args.only_failed,
                               concurrency=args.concurrency, skip_index=args.skip_index)
    print_json(result)
    return 1 if result["errors"] else 0


def command_aws_notes_run(settings: Settings, _: argparse.Namespace) -> int:
    print_json(start_notes(settings))
    return 0


def command_aws_notes_status(settings: Settings, args: argparse.Namespace) -> int:
    print_json(notes_status(settings, args.execution))
    return 0


def command_aws_corpus_status(settings: Settings, args: argparse.Namespace) -> int:
    print_json(AwsStore(settings).corpus_status(check_index=args.check_index))
    return 0


def command_aws_openalex_match(settings: Settings, args: argparse.Namespace) -> int:
    result = match_openalex(settings, limit=args.limit, refresh=args.refresh, dry_run=args.dry_run,
                            batch=args.batch)
    print_json(result)
    return 0


def command_aws_pipeline_failures(settings: Settings, args: argparse.Namespace) -> int:
    if args.synthesis:
        print_json(failed_synthesis(settings, verbose=args.verbose, offset=args.offset))
        return 0
    print_json(AwsStore(settings).pipeline_failures(verbose=args.verbose, offset=args.offset))
    return 0


def command_aws_synthesis_plan(settings: Settings, args: argparse.Namespace) -> int:
    print_json(start_synthesis(settings, scope=args.scope, plan_only=True, categories=args.category or None,
                               skip_concepts=args.skip_concepts, replan=args.replan))
    return 0


def command_aws_synthesis_run(settings: Settings, args: argparse.Namespace) -> int:
    print_json(start_synthesis(settings, scope=args.scope, plan_only=False, categories=args.category or None,
                               skip_concepts=args.skip_concepts, replan=args.replan))
    return 0


def command_aws_synthesis_status(settings: Settings, args: argparse.Namespace) -> int:
    print_json(synthesis_status(settings, args.execution))
    return 0


def command_aws_synthesis_manifest(settings: Settings, args: argparse.Namespace) -> int:
    if args.push:
        print_json(push_manifest(settings, Path(args.push), kind=args.kind, category=args.category))
    elif args.content is not None:
        print_json(push_manifest(settings, content=args.content, kind=args.kind, category=args.category))
    elif args.kind:
        print_json(read_manifest(settings, kind=args.kind, category=args.category, section=args.section,
                                 offset=args.offset, max_chars=args.max_chars))
    else:
        print_json(pull_manifests(settings, scope=args.scope))
    return 0


def command_aws_synthesis_reference(settings: Settings, _: argparse.Namespace) -> int:
    print_json(upload_hgnc(settings))
    return 0


def command_aws_synthesis_page(settings: Settings, args: argparse.Namespace) -> int:
    result = synthesis_page(settings, kind=args.kind, ident=args.id, mode=args.mode, force=args.force,
                            established=args.established or "", corrections=args.corrections or "",
                            evidence_notes=args.evidence_notes or [],
                            max_rounds=args.max_rounds)
    print_json(result)
    return 0 if result.get("status") in ("ready", "skipped") else 1


def command_wiki_backlinks(settings: Settings, args: argparse.Namespace) -> int:
    print_json(backlinks(settings, args.doc_type, args.doc_id, backend=args.backend))
    return 0


def command_aws_answer(settings: Settings, args: argparse.Namespace) -> int:
    store = AwsStore(settings)
    started = time.monotonic()
    result = store.answer_question(args.question, tags=args.tags or [], model_id=args.model,
                                   author=args.author, reread=args.reread)
    usage = result.get("usage") or {}
    result["estimated_usd"] = costs.estimate_draft_usd(result.get("model_id", ""), usage)
    costs.record(settings.state_dir, {"step": "answer_question", "work_id": result.get("slug"), "status": result.get("status"),
                                      "model_id": result.get("model_id"), "seconds": round(time.monotonic() - started, 1),
                                      "lambda_seconds": result.get("seconds"), "input_tokens": usage.get("inputTokens"),
                                      "output_tokens": usage.get("outputTokens"), "estimated_usd": result["estimated_usd"],
                                      "answer_passes": result.get("answer_passes"),
                                      "reread_sources": [r.get("stem") for r in (result.get("reread") or [])],
                                      "basis": "Anthropic list prices applied to reported token counts; not reconciled against the AWS bill"})
    print_json(result)
    return 0 if result.get("status") == "answer_ready" else 1


def command_benchmark_questions(settings: Settings, args: argparse.Namespace) -> int:
    result = run_question_benchmark(settings, Path(args.llm_wiki).expanduser(), select=args.select, limit=args.limit,
                                    concurrency=args.concurrency, model_id=args.model, dry_run=args.dry_run)
    print_json({k: v for k, v in result.items() if k != "rows"})
    return 1 if result["errors"] else 0


def command_cost_ledger(settings: Settings, _: argparse.Namespace) -> int:
    print_json(costs.summarize(settings.state_dir))
    return 0


def command_aws_read_source(settings: Settings, args: argparse.Namespace) -> int:
    with Catalog(settings) as catalog:
        work_id = catalog.get_candidate(args.identifier)["work_id"]
    if args.kind == "draft":
        key = AwsStore(settings).get_item(work_id).get("draft_key") or f"wiki/drafts/{work_id}.md"
    else:
        key = f"sources/{work_id}.md"
    print_json(AwsStore(settings).read_text(key, start=args.start, max_chars=args.max_chars))
    return 0


def command_promote_draft(settings: Settings, args: argparse.Namespace) -> int:
    with Catalog(settings) as catalog:
        work_id = catalog.get_candidate(args.identifier)["work_id"]
    store = AwsStore(settings) if (settings.aws_bucket or settings.aws_table) else None
    started = time.monotonic()
    result = promote_draft(settings, work_id, reviewer=args.reviewer, method=args.method, note=args.note,
                           edited=args.edited, dry_run=args.dry_run, store=store,
                           reviewed_text=Path(args.reviewed_draft).read_text() if args.reviewed_draft else None)
    if not args.dry_run:
        result["ledger"] = costs.record(settings.state_dir, {
            "step": "promote", "work_id": work_id, "reviewer": args.reviewer, "review_method": args.method,
            "seconds": round(time.monotonic() - started, 1), "estimated_usd": 0.0,
            "basis": "review effort is not metered here; a bedrock-review pass would record its own draft step",
        })
    print_json(result)
    return 0


def command_autism_collect(settings: Settings, args: argparse.Namespace) -> int:
    result = collect(settings, start=args.from_date, end=args.to_date,
                     per_query_limit=args.per_query_limit, publish_aws=args.aws,
                     resume=Path(args.resume) if args.resume else None)
    print_json(result)
    return 1 if result["errors"] else 0


def command_corpus_search(settings: Settings, args: argparse.Namespace) -> int:
    print_json(search_corpus(settings, backend=args.backend, query=args.query,
                             from_year=args.from_year, to_year=args.to_year, tag=args.tag,
                             oa_only=args.oa_only, fulltext_only=args.fulltext_only, limit=args.limit,
                             journal_policy=args.journals, corpus=args.corpus))
    return 0


def command_corpus_journal_policy(settings: Settings, args: argparse.Namespace) -> int:
    result = apply_policy_to_catalog(settings, publish_aws=args.aws)
    print_json(result)
    return 1 if result["errors"] else 0


def command_papers_export(settings: Settings, args: argparse.Namespace) -> int:
    print_json(export_papers_csv(settings, Path(args.output), force=args.force))
    return 0


def command_corpus_report(settings: Settings, args: argparse.Namespace) -> int:
    print(export_report(settings, backend=args.backend))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="byeori")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="write the env file for this installation (asks on the terminal)")
    init_parser.add_argument("--env-file", default=installer.ENV_FILE)
    init_parser.add_argument("--non-interactive", action="store_true")
    init_parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="answer a question without a prompt")
    init_parser.set_defaults(handler=command_init)

    doctor_parser = subparsers.add_parser("doctor", help="check local, OpenAlex, and AWS access")
    doctor_parser.set_defaults(handler=command_doctor)

    for name, handler, help_text in (
        ("deploy", command_deploy, "deploy the main stack from infra/template.yaml and record its outputs in the env file"),
        ("build-workers", command_build_workers, "build and push the asset worker image; publish the worker scripts"),
        ("deploy-lab", command_deploy_lab, "deploy the optional student service stack"),
        ("deploy-jev-eval", command_deploy_jev_eval, "deploy the optional Jev evaluation stack"),
        ("grant-client", command_grant_client, "print (or attach) the IAM policy an administrator's user needs"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--env-file", default=installer.ENV_FILE)
        if name == "deploy":
            sub.add_argument("parameter", nargs="*", help="extra Key=Value parameter overrides")
        if name == "grant-client":
            sub.add_argument("--attach-to-user", help="IAM user name to attach the policy to")
        sub.set_defaults(handler=handler)

    search_parser = subparsers.add_parser("search", help="search OpenAlex")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--from-year", type=int)
    search_parser.add_argument("--to-year", type=int)
    search_parser.add_argument("--oa-only", action="store_true")
    search_parser.add_argument(
        "--fulltext-only",
        action="store_true",
        help="require OpenAlex-hosted GROBID full text",
    )
    search_parser.add_argument(
        "--journal-scope", choices=("list", "wide"), default="list",
        help="list: the lab's 65 journals only (default). wide: anywhere except the refused "
             "houses and titles, with a warning on every result outside the list",
    )
    search_parser.set_defaults(handler=command_search)

    add_parser = subparsers.add_parser("candidate-add", help="save an OpenAlex work as a candidate")
    add_parser.add_argument("identifier")
    add_parser.set_defaults(handler=command_candidate_add)

    list_parser = subparsers.add_parser("candidate-list", help="list saved candidates")
    list_parser.add_argument("--status")
    list_parser.set_defaults(handler=command_candidate_list)

    attach_parser = subparsers.add_parser("attach-pdf", help="upload an original PDF directly to S3")
    attach_parser.add_argument("identifier")
    attach_parser.add_argument("pdf")
    attach_parser.set_defaults(handler=command_attach_pdf)

    upload_pdf_parser = subparsers.add_parser(
        "upload-pdf",
        help="store one PDF as papers/{stem}/original.pdf with meta.json and a catalog item, ready for aws-extract")
    upload_pdf_parser.add_argument("pdf")
    upload_pdf_parser.add_argument("--stem", help="lowercase author-year-words stem; defaults to the file's own name")
    upload_pdf_parser.add_argument("--source", help="where the PDF came from (default: upload-pdf)")
    upload_pdf_parser.set_defaults(handler=command_upload_pdf)

    validate_parser = subparsers.add_parser("validate", help="validate source and paper page sections")
    validate_parser.set_defaults(handler=command_validate)

    index_parser = subparsers.add_parser("build-index", help="rebuild the search index in AWS")
    index_parser.set_defaults(handler=command_aws_build_index)
    filtered_parser = subparsers.add_parser(
        "filtered-list", help="questions Bedrock's content filter stopped on the answer path retired on 2026-09-20 (runs/filtered/), with what they cost")
    filtered_parser.add_argument("--since", help="only records from this date onward (YYYY-MM-DD)")
    filtered_parser.add_argument("--limit", type=int, default=200)
    filtered_parser.set_defaults(handler=command_filtered_list)

    aws_status_parser = subparsers.add_parser("aws-status", help="verify AWS credentials and configuration")
    aws_status_parser.set_defaults(handler=command_aws_status)

    aws_push_parser = subparsers.add_parser("aws-push-candidate", help="copy one candidate to configured AWS resources")
    aws_push_parser.add_argument("identifier")
    aws_push_parser.set_defaults(handler=command_aws_push_candidate)

    aws_ingest_parser = subparsers.add_parser(
        "aws-ingest-candidate",
        help="run the AWS Lambda full-text ingest for one saved candidate",
    )
    aws_ingest_parser.add_argument("identifier")
    aws_ingest_parser.set_defaults(handler=command_aws_ingest_candidate)

    aws_read_parser = subparsers.add_parser(
        "aws-read-source",
        help="read a slice of an ingested source document from S3",
    )
    aws_read_parser.add_argument("identifier")
    aws_read_parser.add_argument("--start", type=int, default=0)
    aws_read_parser.add_argument("--max-chars", type=int, default=4000)
    aws_read_parser.add_argument("--kind", choices=("source", "draft"), default="source",
                                 help="source: GROBID extraction; draft: the Bedrock-written evidence note")
    aws_read_parser.set_defaults(handler=command_aws_read_source)

    aws_draft_parser = subparsers.add_parser(
        "aws-draft-page",
        help="ask the AWS Lambda to draft an evidence note with Bedrock from an ingested paper's extraction",
    )
    aws_draft_parser.add_argument("identifier")
    aws_draft_parser.add_argument("--model", help="Bedrock model or inference profile id; default is the stack's DraftModelId")
    aws_draft_parser.set_defaults(handler=command_aws_draft_page)

    promote_parser = subparsers.add_parser(
        "promote-draft",
        help="publish a reviewed Bedrock draft directly to S3 and record review in DynamoDB",
    )
    promote_parser.add_argument("identifier")
    promote_parser.add_argument("--reviewer", required=True, help="who or which session reviewed the draft")
    promote_parser.add_argument("--method", choices=REVIEW_METHODS, default="agent-session",
                                help="agent-session (default): the Claude Code or Kiro session read the draft against the PDF; validator: structural check only (what aws-pipeline records); person; bedrock-review")
    promote_parser.add_argument("--note", help="short review note kept in the page frontmatter")
    promote_parser.add_argument("--edited", action="store_true",
                                help="reviewed text differs from the S3 draft; requires --reviewed-draft")
    promote_parser.add_argument("--reviewed-draft", help="explicit reviewed Markdown input; no copy is retained")
    promote_parser.add_argument("--dry-run", action="store_true", help="run every check and print the result without writing anything")
    promote_parser.set_defaults(handler=command_promote_draft)


    synth_parser = subparsers.add_parser("aws-synthesize", help="ask the Lambda to write an overview page (wiki/overviews/) with Bedrock from reviewed evidence notes")
    synth_parser.add_argument("topic", help="lowercase slug of the overview page, e.g. de-novo-variants")
    synth_parser.add_argument("--title", required=True)
    synth_parser.add_argument("identifiers", nargs="+", help="work IDs or DOIs of reviewed papers (max 20)")
    synth_parser.add_argument("--model")
    synth_parser.set_defaults(handler=command_aws_synthesize)

    pipeline_parser = subparsers.add_parser("aws-pipeline", help="ingest, draft and promote every allowlisted OpenAlex-hosted article to a reviewed evidence note, then synthesize overview pages per tag; resumable")
    pipeline_parser.add_argument("--limit", type=int, default=0, help="stop after this many papers (0 = all)")
    pipeline_parser.add_argument("--dry-run", action="store_true", help="list the selected papers only")
    pipeline_parser.add_argument("--skip-topics", action="store_true")
    pipeline_parser.add_argument("--model")
    pipeline_parser.add_argument("--concurrency", type=int, default=6, help="papers processed at once (1-16); Bedrock and OpenAlex rate limits are the real ceiling")
    pipeline_parser.set_defaults(handler=command_aws_pipeline)

    aws_index_parser = subparsers.add_parser("aws-build-index", help="rebuild the search index in AWS from the pages in S3; no local copy of the wiki is needed")
    aws_index_parser.set_defaults(handler=command_aws_build_index)

    index_parser = subparsers.add_parser("build-search-index", help="rebuild the search index in AWS")
    index_parser.set_defaults(handler=command_aws_build_index)

    gap_parser = subparsers.add_parser(
        "aws-collect-for-gap",
        help="search the lab's journals for a question the wiki could not answer and save candidates")
    gap_parser.add_argument("query", help="the English query; the professor may rewrite the gap's own")
    gap_parser.add_argument("--job-id", default=None, help="the answer job whose gap this collects for")
    gap_parser.add_argument("--limit", type=int, default=25)
    gap_parser.add_argument("--apply", action="store_true",
                            help="save the candidates; without it the search runs and nothing is stored")
    gap_parser.set_defaults(handler=command_collect_for_gap)

    gap_ingest_parser = subparsers.add_parser(
        "aws-ingest-for-gap",
        help="ingest an approved gap's saved candidates and write their evidence notes")
    gap_ingest_parser.add_argument("job_id", help="the answer job whose gap was approved")
    gap_ingest_parser.add_argument("--query", default=None, help="recorded on the blocked-paper list")
    gap_ingest_parser.add_argument("--limit", type=int, default=25)
    gap_ingest_parser.add_argument("--apply", action="store_true",
                                   help="fetch and write; without it nothing is stored")
    gap_ingest_parser.set_defaults(handler=command_ingest_for_gap)

    link_parser = subparsers.add_parser(
        "aws-link-lab-questions",
        help="add one standing line to each indexed page pointing at its hub of answered lab questions")
    link_parser.add_argument("--apply", action="store_true", help="write the pages; without it nothing is changed")
    link_parser.add_argument("--limit", type=int, default=None, help="stop after this many pages (for a trial run)")
    link_parser.set_defaults(handler=command_link_lab_questions)

    coverage_parser = subparsers.add_parser(
        "aws-synthesis-coverage",
        help="per category, how many notes a concept or overview cites and how many nothing cites")
    coverage_parser.set_defaults(handler=command_synthesis_coverage)

    extraction_parser = subparsers.add_parser(
        "aws-read-extraction",
        help="print a window of a paper's extracted text, for writing its note in this session")
    extraction_parser.add_argument("stem")
    extraction_parser.add_argument("--start", type=int, default=0)
    extraction_parser.add_argument("--max-chars", type=int, default=40000)
    extraction_parser.add_argument("--all", action="store_true", help="follow the windows to the end")
    extraction_parser.set_defaults(handler=command_read_extraction)

    publish_parser = subparsers.add_parser(
        "aws-publish-source-note",
        help="publish an evidence note written in this session, for a paper that has none")
    publish_parser.add_argument("stem")
    publish_parser.add_argument("markdown", type=Path, help="the seven sections, no frontmatter")
    publish_parser.add_argument("--model-id", default="claude-opus-5",
                                help="the model that actually wrote the note")
    publish_parser.set_defaults(handler=command_publish_source_note)

    catalogs_parser = subparsers.add_parser(
        "aws-build-category-catalogs",
        help="write one browse catalog per field, and the root table of fields, from the index")
    catalogs_parser.add_argument("--min-notes", type=int, default=1,
                                 help="skip a field with fewer notes than this")
    catalogs_parser.set_defaults(handler=command_build_category_catalogs)

    classify_parser = subparsers.add_parser(
        "aws-classify-notes",
        help="file the notes the intake left in `other` into the folders the wiki already uses")
    classify_parser.add_argument("--stems", nargs="*", help="explicit stems; default: every note in `other`")
    classify_parser.add_argument("--limit", type=int, default=0)
    classify_parser.add_argument("--batch", type=int, default=200, help="stems per Lambda call (1-400)")
    classify_parser.add_argument("--model")
    classify_parser.add_argument(
        "--new-folder", action="append", metavar="SLUG:SCOPE",
        help="open a field no note carries yet, as `slug:what belongs in it` (repeatable); without "
             "this the model may only choose a folder 20 or more notes already use")
    classify_parser.add_argument(
        "--only-into", action="append", metavar="SLUG",
        help="keep every other note where it is: file only the ones that land in this existing field "
             "(repeatable), which is how a subtopic is gathered into a field the wiki already has")
    classify_parser.add_argument(
        "--only-new-folders", action="store_true",
        help="keep every other note where it is: file only the ones that land in a field being opened")
    classify_parser.add_argument("--apply", action="store_true", help="write; without it the run only reports")
    classify_parser.add_argument("--receipt", type=Path, default=None)
    classify_parser.set_defaults(handler=command_classify_notes)

    sync_parser = subparsers.add_parser(
        "aws-sync-note-categories",
        help="make the catalogue the synthesis planner reads agree with the notes about their field")
    sync_parser.add_argument("--apply", action="store_true", help="write; without it the run only reports")
    sync_parser.set_defaults(handler=command_sync_note_categories)

    fields_parser = subparsers.add_parser(
        "aws-fields", help="the fields a person opened, which the classifier is offered whatever their note count")
    fields_parser.add_argument("--open", action="append", metavar="SLUG:SCOPE",
                               help="open a field, as `slug:what belongs in it` (repeatable)")
    fields_parser.add_argument("--close", action="append", metavar="SLUG",
                               help="stop offering this field; the notes carrying it are not moved")
    fields_parser.set_defaults(handler=command_fields)

    file_parser = subparsers.add_parser(
        "aws-file-notes",
        help="put named notes in a named field because you say so; no model is called")
    file_parser.add_argument("category", help="the field slug; it must already exist")
    file_parser.add_argument("--stems", nargs="+", required=True)
    file_parser.add_argument("--batch", type=int, default=200, help="stems per Lambda call (1-400)")
    file_parser.add_argument("--apply", action="store_true", help="write; without it the run only reports")
    file_parser.set_defaults(handler=command_file_notes)

    wsearch_parser = subparsers.add_parser("wiki-search", help="BM25 search over the reviewed wiki; results point at sections")
    wsearch_parser.add_argument("query")
    wsearch_parser.add_argument("--limit", type=int, default=10)
    wsearch_parser.add_argument("--type", choices=("note", "paper", "overview", "question", "concept"))
    wsearch_parser.add_argument("--category", help="one sub-wiki, e.g. asd-ndd, long-read, drug-resistance")
    wsearch_parser.add_argument("--backend", choices=("auto", "aws"), default="aws", help="the Lambda searches the S3 index")
    wsearch_parser.set_defaults(handler=command_wiki_search)

    wcat_parser = subparsers.add_parser("wiki-categories", help="pages per category: the shape of the lab's sub-wikis")
    wcat_parser.set_defaults(handler=command_wiki_categories)

    wread_parser = subparsers.add_parser("wiki-read", help="request an AWS page outline or a bounded section excerpt")
    wread_parser.add_argument("type", choices=("note", "paper", "overview", "question", "concept"))
    wread_parser.add_argument("doc_id", help="work ID (note, paper) or page slug (overview, question)")
    wread_parser.add_argument("--section", help="e.g. Results, Synthesis")
    wread_parser.add_argument("--start", type=int, default=0)
    wread_parser.add_argument("--max-chars", type=int, default=4000, help="section excerpt size, maximum 8000")
    wread_parser.set_defaults(handler=command_wiki_read)

    wsave_parser = subparsers.add_parser("wiki-save-question", help="write a research-question page (llm-wiki questions/ schema) directly to S3 from a JSON spec")
    wsave_parser.add_argument("spec_file", help="JSON with title, question, sharper_followup, holdings, tentative_answer, related, tags")
    wsave_parser.set_defaults(handler=command_wiki_save_question)

    upload_parser = subparsers.add_parser("upload-papers", help="copy llm-wiki PDFs into S3 papers/{stem}/original.pdf with meta.json and a DynamoDB item; nothing local changes")
    upload_parser.add_argument("--llm-wiki", required=True, help="path of an llm-wiki checkout to compare against")
    upload_parser.add_argument("--select", choices=("autism", "all"), default="autism", help="autism: asd-ndd, asd-models, or 'autis' in the note head")
    upload_parser.add_argument("--stems", nargs="*", help="explicit stems instead of a selection")
    upload_parser.add_argument("--limit", type=int, default=0)
    upload_parser.add_argument("--concurrency", type=int, default=8)
    upload_parser.add_argument("--dry-run", action="store_true")
    upload_parser.set_defaults(handler=command_upload_papers)

    resolve_parser = subparsers.add_parser("resolve-ids", help="look uploaded papers up in OpenAlex by DOI and record openalex_id, pmid, pmcid (free singleton lookups)")
    resolve_parser.add_argument("--llm-wiki", required=True, help="path of an llm-wiki checkout to compare against")
    resolve_parser.add_argument("--select", choices=("autism", "all"), default="autism")
    resolve_parser.add_argument("--stems", nargs="*")
    resolve_parser.add_argument("--limit", type=int, default=0)
    resolve_parser.add_argument("--concurrency", type=int, default=4)
    resolve_parser.add_argument("--source", choices=("ncbi", "openalex"), default="ncbi",
                                help="ncbi (default): free batched PMC ID converter, DOI -> pmid/pmcid; openalex: also records openalex_id but costs $0.001 per lookup against the key's $1 daily budget")
    resolve_parser.add_argument("--direct", action="store_true", help="query api.openalex.org from this machine without the key instead of through the Lambda")
    resolve_parser.set_defaults(handler=command_resolve_ids)

    extract_parser = subparsers.add_parser("aws-extract", help="run GROBID extraction on Fargate for uploaded PDFs that are not yet extracted, in N parallel tasks")
    extract_parser.add_argument("--stems", nargs="*", help="explicit stems; default: every paper waiting (pdf_uploaded or extract_failed)")
    extract_parser.add_argument("--tasks", type=int, default=4, help="parallel Fargate tasks (1-20), each GROBID + worker")
    extract_parser.add_argument("--limit", type=int, default=0)
    extract_parser.add_argument("--force", action="store_true", help="re-extract even if clean.md already matches the PDF")
    extract_parser.add_argument("--preprocess", choices=("ocr", "shrink"),
                                help="read each PDF through a derived copy (papers/{stem}/derived.pdf): ocr for scans "
                                     "and image-only pages, shrink for a PDF too large for GROBID")
    extract_parser.add_argument("--dry-run", action="store_true")
    extract_parser.set_defaults(handler=command_aws_extract)

    extract_status_parser = subparsers.add_parser("aws-extract-status", help="list extraction tasks and count uploaded papers by status")
    extract_status_parser.set_defaults(handler=command_aws_extract_status)

    sn_parser = subparsers.add_parser("aws-source-note", help="Bedrock writes the llm-wiki source page (7 sections) for one uploaded, extracted paper")
    sn_parser.add_argument("stem"); sn_parser.add_argument("--model")
    sn_parser.set_defaults(handler=command_aws_source_note)


    ps_parser = subparsers.add_parser("aws-pipeline-stems", help="the evidence note for every extracted upload that has none, concurrently; resumable")
    ps_parser.add_argument("--stems", nargs="*")
    ps_parser.add_argument("--limit", type=int, default=0)
    ps_parser.add_argument("--dry-run", action="store_true")
    ps_parser.add_argument("--model")
    ps_parser.add_argument("--concurrency", type=int, default=6)
    ps_parser.add_argument("--only-failed", action="store_true",
                           help="retry only papers a previous run tried and could not finish; "
                                "papers not yet reached are left alone. Combine with --dry-run to list them first.")
    ps_parser.add_argument("--skip-index", action="store_true", help="do not rebuild the AWS search index at the end")
    ps_parser.set_defaults(handler=command_aws_pipeline_stems)

    run_parser = subparsers.add_parser("aws-notes-run",
                                       help="start the notes run in AWS; takes no arguments and does not need this machine to stay open")
    run_parser.set_defaults(handler=command_aws_notes_run)

    rst_parser = subparsers.add_parser("aws-notes-status", help="how the AWS-side notes run is going")
    rst_parser.add_argument("--execution", help="a specific execution ARN; defaults to the most recent")
    rst_parser.set_defaults(handler=command_aws_notes_status)

    corpus_status_parser = subparsers.add_parser("aws-corpus-status", help="count note states in AWS and optionally check index coverage")
    corpus_status_parser.add_argument("--check-index", action="store_true")
    corpus_status_parser.set_defaults(handler=command_aws_corpus_status)

    oam_parser = subparsers.add_parser("aws-openalex-match",
                                       help="start resumable DOI matching in AWS, up to 50 DOIs per request")
    oam_parser.add_argument("--limit", type=int, default=0)
    oam_parser.add_argument("--batch", type=int, default=50, help="DOIs per request (OpenAlex allows 50)")
    oam_parser.add_argument("--refresh", action="store_true", help="re-match papers already matched")
    oam_parser.add_argument("--dry-run", action="store_true", help="report what would be selected, fetch nothing")
    oam_parser.add_argument("--verbose", action="store_true", help="retained for compatibility; diagnostics are stored in the AWS run report")
    oam_parser.set_defaults(handler=command_aws_openalex_match)

    fail_parser = subparsers.add_parser("aws-pipeline-failures",
                                        help="list the papers a previous run tried and could not finish, with the validator's reasons")
    fail_parser.add_argument("--verbose", action="store_true", help="show each paper's status and problems, not just its stem")
    fail_parser.add_argument("--synthesis", action="store_true", help="list failed synthesis pages (concept, subtopic, category) instead of notes")
    fail_parser.add_argument("--offset", type=int, default=0, help="continue the bounded AWS failure report")
    fail_parser.set_defaults(handler=command_aws_pipeline_failures)

    for name, handler, help_text in (
        ("aws-synthesis-plan", command_aws_synthesis_plan, "plan the synthesis layer for a scope in AWS (manifests only) and stop, for review"),
        ("aws-synthesis-run", command_aws_synthesis_run, "plan what is missing and write the synthesis pages for a scope in AWS"),
    ):
        p = subparsers.add_parser(name, help=help_text)
        p.add_argument("--scope", default="autism", help="autism (five categories) or all")
        p.add_argument("--category", action="append", help="limit to these categories (repeatable); concepts still count over the corpus")
        p.add_argument("--skip-concepts", action="store_true", help="subtopic and category pages only")
        p.add_argument("--replan", action="store_true", help="propose the subtopic partitions again, superseding the manifests")
        p.set_defaults(handler=handler)

    sst_parser = subparsers.add_parser("aws-synthesis-status", help="how the AWS-side synthesis run is going")
    sst_parser.add_argument("--execution", help="a specific execution ARN; defaults to the most recent")
    sst_parser.set_defaults(handler=command_aws_synthesis_status)

    sman_parser = subparsers.add_parser("aws-synthesis-manifest", help="read AWS manifests without local copies, or push an explicitly supplied edit")
    sman_parser.add_argument("--scope", default="autism")
    sman_input = sman_parser.add_mutually_exclusive_group()
    sman_input.add_argument("--push", help="read an existing user input overrides.json or subtopics.json and submit for AWS validation")
    sman_input.add_argument("--content", help="JSON content to submit for AWS validation; requires --kind")
    sman_parser.add_argument("--kind", choices=["concepts", "overrides", "subtopics"], help="read or submit one manifest; omitted returns AWS summary")
    sman_parser.add_argument("--category", help="category slug for a subtopic manifest")
    sman_parser.add_argument("--section", help="read a concept/subtopic slug, or an overrides field")
    sman_parser.add_argument("--offset", type=int, default=0, help="character offset within the selected JSON")
    sman_parser.add_argument("--max-chars", type=int, default=4000, help="excerpt length, at most 8000 characters")
    sman_parser.set_defaults(handler=command_aws_synthesis_manifest)

    sref_parser = subparsers.add_parser("aws-synthesis-reference", help="upload the HGNC gene table the concept planner uses (reference/hgnc.tsv)")
    sref_parser.set_defaults(handler=command_aws_synthesis_reference)

    spage_parser = subparsers.add_parser("aws-synthesis-page", help="write one synthesis page now and wait for it (the pilot)")
    spage_parser.add_argument("kind", choices=["concept", "subtopic", "category"])
    spage_parser.add_argument("id", help="concept slug, category/slug for a subtopic, or a category")
    spage_parser.add_argument("--mode", choices=["generate", "refresh", "update"], default="generate")
    spage_parser.add_argument("--force", action="store_true", help="rewrite a category page whose subtopic pages did not change")
    spage_parser.add_argument("--max-rounds", type=int, default=30, help="stop waiting after this many partial rounds and report status partial")
    spage_parser.add_argument("--established", help="what the answering agent has just shown (mode=update)")
    spage_parser.add_argument("--corrections", help="what the page currently gets wrong, and how (mode=update)")
    spage_parser.add_argument("--evidence-notes", nargs="*", dest="evidence_notes",
                              help="stems of the evidence notes behind the update; they join the page's members")
    spage_parser.set_defaults(handler=command_aws_synthesis_page)

    wback_parser = subparsers.add_parser("wiki-backlinks", help="which concept and overview pages cite a page (a note nothing cites is a synthesis orphan)")
    wback_parser.add_argument("doc_type", choices=["note", "paper", "overview", "concept", "question"])
    wback_parser.add_argument("doc_id")
    wback_parser.add_argument("--backend", choices=["auto", "aws"], default="auto")
    wback_parser.set_defaults(handler=command_wiki_backlinks)

    answer_parser = subparsers.add_parser("aws-answer", help="ask the wiki a research question: the research agent in the Lambda searches and reads the wiki, revises pages, and saves a wiki/questions/ page")
    answer_parser.add_argument("question", help="the question, ending with ?")
    answer_parser.add_argument("--tags", nargs="*")
    answer_parser.add_argument("--model")
    answer_parser.add_argument("--author", help="defaults to the IAM user behind the credentials in use")
    answer_parser.add_argument("--reread", choices=["auto", "never", "always"], default="auto",
                               help="read an original's stored full text when the wiki cannot settle the question "
                                    "(auto), never, or for every answer (always); reading happens in the Lambda")
    answer_parser.set_defaults(handler=command_aws_answer)

    bench_parser = subparsers.add_parser("benchmark-questions", help="ask Byeori the autism-related questions llm-wiki already answered and write a comparison report")
    bench_parser.add_argument("--llm-wiki", required=True, help="path of an llm-wiki checkout to compare against")
    bench_parser.add_argument("--select", choices=("autism", "all"), default="autism")
    bench_parser.add_argument("--limit", type=int, default=0)
    bench_parser.add_argument("--concurrency", type=int, default=4)
    bench_parser.add_argument("--model")
    bench_parser.add_argument("--dry-run", action="store_true")
    bench_parser.set_defaults(handler=command_benchmark_questions)

    ledger_parser = subparsers.add_parser("cost-ledger", help="summarize measured time and estimated cost per step")
    ledger_parser.set_defaults(handler=command_cost_ledger)

    collect_parser = subparsers.add_parser("autism-collect", help="collect a bounded 15-year autism-genomics candidate corpus")
    start, end = date_range()
    collect_parser.add_argument("--from-date", default=start)
    collect_parser.add_argument("--to-date", default=end)
    collect_parser.add_argument("--per-query-limit", type=int, default=100)
    collect_parser.add_argument("--aws", action="store_true", help="also store a per-run export bundle in S3; candidate storage is always AWS")
    collect_parser.add_argument("--resume", help="reuse completed requests from a saved run directory")
    collect_parser.set_defaults(handler=command_autism_collect)

    corpus_search_parser = subparsers.add_parser("corpus-search", help="search saved metadata in DynamoDB; no OpenAlex requests")
    corpus_search_parser.add_argument("query", nargs="?", default="")
    corpus_search_parser.add_argument("--backend", choices=("aws",), default="aws")
    corpus_search_parser.add_argument("--from-year", type=int)
    corpus_search_parser.add_argument("--to-year", type=int)
    corpus_search_parser.add_argument("--tag")
    corpus_search_parser.add_argument("--oa-only", action="store_true")
    corpus_search_parser.add_argument("--fulltext-only", action="store_true", help="require metadata indicating eligibility for the existing full-text ingestion path")
    corpus_search_parser.add_argument("--limit", type=int, default=50)
    corpus_search_parser.add_argument("--journals", choices=("include", "review", "all"), default="include",
                                      help="include: allowlisted journals only (default); review: also same-family titles awaiting a decision; all: no journal filter")
    corpus_search_parser.add_argument("--corpus", default="all",
                                      help="which intake to search: an id such as autism-genomics or lab-shared, or all (default)")
    corpus_search_parser.set_defaults(handler=command_corpus_search)

    policy_parser = subparsers.add_parser("corpus-journal-policy", help="classify every saved candidate's journal against the test allowlist; nothing is deleted")
    policy_parser.add_argument("--aws", action="store_true", help="compatibility flag; verdicts are always written to AWS")
    policy_parser.set_defaults(handler=command_corpus_journal_policy)

    report_parser = subparsers.add_parser("corpus-report", help="publish an HTML catalog report to S3")
    report_parser.add_argument("--backend", choices=("aws",), default="aws")
    report_parser.set_defaults(handler=command_corpus_report)

    export_parser = subparsers.add_parser("papers-export",
                                          help="write one CSV row per PDF the catalogue records in S3, for deciding which newly collected papers are missing")
    export_parser.add_argument("--output", required=True, help="path of the CSV file to write")
    export_parser.add_argument("--force", action="store_true", help="replace the file if it already exists")
    export_parser.set_defaults(handler=command_papers_export)

    return parser


def main() -> None:
    settings = Settings.from_env()
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(args.handler(settings, args))
    except (FileNotFoundError, FileExistsError, KeyError, OpenAlexError, RuntimeError, ValueError,
            ClientError, BotoCoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
