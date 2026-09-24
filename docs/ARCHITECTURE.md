# Architecture

Byeori is two CloudFormation stacks around one S3 bucket. The main stack (`infra/template.yaml`)
takes papers in, writes the wiki and keeps the search index. The optional student stack
(`infra/lab-template.json`) lets lab members ask and read without administrator rights. All of the
work runs in AWS; a client (the `byeori` CLI or an MCP server) signs in, sends a request and reads
a bounded result. Nothing is mirrored on anyone's computer.

## One paper, one note

Each paper gets exactly one page of its own: its evidence note, `wiki/sources/{stem}.md`, written
from the paper's full text as extracted from the stored original. The note records the paper's
identifiers and the SHA-256 of the PDF it was read from, the model that wrote it, the paper's
methods and results as the paper states them, its limitations, and a line between what the paper
reports and what is interpretation. Everything else in the wiki (overviews, concepts, answered
questions) is synthesis across notes, and links back to the notes it cites.

## Storage layout

Everything lives in the data bucket. A paper's folder name, its `stem`, is built from its first
author, year and title.

| Key | What it holds |
|---|---|
| `papers/{stem}/original.pdf` | the original PDF, as uploaded, never overwritten |
| `papers/{stem}/meta.json` | the paper's identifiers, the PDF's hash and where it came from |
| `papers/{stem}/grobid.tei.xml`, `papers/{stem}/clean.md` | GROBID's structured extraction, and the Markdown text the model reads |
| `papers/{stem}/assets/` | figures and tables cut from the PDF, with their captions |
| `papers/{stem}/supplementary/` | the paper's supplementary files, when supplied |
| `wiki/sources/{stem}.md` | the evidence note: one per paper |
| `wiki/overviews/`, `wiki/concepts/` | synthesis pages across notes |
| `wiki/questions/` | answered research questions |
| `wiki/lab-questions/` | answers given through the student service; outside the search index |
| `wiki/indexes/`, `wiki/index.md` | browse catalogs, one per folder and field; outside the search index |
| `index/` | the BM25 search index, rebuilt in AWS from the pages (nightly and by `byeori build-index`) |
| `runs/` | receipts and ledgers: what each run did, read and cost |

The catalog table (DynamoDB) holds one record per paper, keyed by its stem or OpenAlex id, with
its status through the pipeline and each step's token counts.

## The main stack

| Piece | Service | What it does |
|---|---|---|
| Ingest function | Lambda | OpenAlex search and lookups, identity checks, evidence notes, answers, validation, the index rebuild |
| Synthesis function | Lambda | plans and writes pages across notes |
| Asset trigger | Lambda | starts the figure worker when a paper's text is stored |
| Extraction task | Fargate | GROBID: PDF to structured text, written back beside the PDF |
| Asset task | Fargate | cuts figures and tables out of the PDF (image in ECR) |
| Notes workflow | Step Functions | writes notes for many papers at once, resumable |
| OpenAlex match workflow | Step Functions | matches stored papers to OpenAlex records |
| Synthesis workflow | Step Functions | writes synthesis pages in parallel under a page cap |
| Question workflow | Step Functions | runs research questions in bulk under a per-question budget |
| Rules | EventBridge | the nightly index rebuild; the figure and identity steps after an extraction |
| Trail | CloudTrail | records every write to the synthesis pages, kept in a separate audit bucket |

The path of one paper: the PDF is stored under `papers/{stem}/`; an extraction task writes
`clean.md`; that write starts the identity check and the figure worker; the ingest function has
Bedrock write the note from `clean.md`; the index rebuild makes the note searchable.

## The student stack

| Piece | Service | What it does |
|---|---|---|
| Gateway | Lambda with a Function URL | checks the caller's IAM identity against the member list, records questions, serves bounded reads |
| Answer worker | Lambda, SQS queue | answers from a bounded evidence packet in at most two model calls |
| Triage worker | Lambda, SQS queue | asks Jev, after the answer, whether the question deserves synthesis; issues an offer at 0.99 |
| Outbox relay | Lambda, EventBridge timer | passes recorded work to the queues every minute |
| Research worker | Lambda, SQS queue | runs an approved synthesis inside the approved scope; off until `ResearchConsumerEnabled` is true |
| Control table | DynamoDB | members, questions, answers, offers, approvals, budgets |

## Write boundaries

Who may write what is enforced by IAM, not by the client's good behaviour:

- The index is built only in AWS; no client writes `index/`. Originals arrive in `papers/` only
  through the administrator's upload command, which refuses to replace an existing original.
  Nothing in the student service may write `papers/` or `index/`.
- The student gateway, the triage worker and the outbox relay write no wiki page at all; each role
  explicitly denies writes to `wiki/`, `papers/` and `index/`.
- The answer worker may write only its receipts (`runs/lab-questions/`) and answer pages under
  `wiki/lab-questions/`; its role denies writes to `papers/` and `index/`.
- The research worker writes synthesis pages only after a student accepted an offer or the
  professor approved a candidate, and only inside that approval's scope; it too is denied
  `papers/` and `index/`.
- A student's IAM policy allows one thing: calling the gateway's Function URL.

## How it works, in diagrams

The README carries short versions of these diagrams; the ones here are the detailed versions.
Every box and arrow is taken from the code in `src/byeori/` and the templates in `infra/`. Solid
arrows are calls and writes, dotted arrows are reads, and an arrow ending in `x` is a write that
IAM denies.

### Overall architecture

```mermaid
flowchart TB
    subgraph clients["Clients: sign in, send a request, read a bounded result"]
        admin["Administrator: byeori CLI or byeori-mcp"]
        member["Lab member: byeori-lab-mcp"]
    end
    subgraph mainstack["Main stack"]
        ingest["Ingest function"]
        synth["Synthesis function"]
        sfn["Step Functions: Notes, OpenAlexMatch, Synthesis, Question"]
        extract["Extraction task on Fargate: GROBID + worker"]
        assettrigger["Asset trigger function"]
        assets["Asset task on Fargate: figures and tables"]
        rules["EventBridge: nightly rebuild, clean.md stored"]
        catalog[("Catalog table, DynamoDB")]
        ssm["Parameter Store: OpenAlex key, Jev key"]
        trail["CloudTrail: writes to overviews, concepts, questions"]
    end
    subgraph bucket["Data bucket, S3"]
        papers["papers/{stem}/: original, extraction, assets, supplementary"]
        sources["wiki/sources/"]
        synthesis["wiki/overviews/ and wiki/concepts/"]
        questions["wiki/questions/"]
        labq["wiki/lab-questions/"]
        catalogs["wiki/indexes/"]
        index["index/: BM25 search index"]
        runs["runs/: receipts"]
    end
    subgraph labstack["Student stack, optional"]
        gateway["Gateway: Function URL, IAM auth, member list"]
        control[("Control table, DynamoDB")]
        queues["SQS queues and outbox relay"]
        answer["Answer worker"]
        triage["Triage worker"]
        research["Research worker"]
    end
    bedrock["Bedrock: Claude"]
    jev["Jev, external and optional"]

    admin -->|"upload-pdf"| papers
    admin -->|"aws-extract starts tasks"| extract
    admin --> ingest
    admin -->|"start a run"| sfn
    sfn --> ingest
    sfn --> synth
    extract -->|"clean.md, grobid.tei.xml"| papers
    rules -->|"clean.md created"| assettrigger --> assets -->|"assets/"| papers
    rules -->|"resolve_identity, build_index"| ingest
    ingest --> catalog
    ingest -->|"notes"| sources
    ingest -->|"answers"| questions
    ingest -->|"build_index"| index
    ingest --> catalogs
    ingest --> runs
    synth -->|"pages"| synthesis
    synth --> catalog
    ingest --> bedrock
    synth --> bedrock
    ssm -.-> ingest
    trail -.-> synthesis
    member -->|"signed request"| gateway
    gateway --> control
    gateway --> queues
    queues --> answer
    queues --> triage
    queues --> research
    answer -.->|"search and read"| index
    answer -->|"answer pages"| labq
    answer --> bedrock
    triage --> jev
    ssm -.-> triage
    research -->|"approved scope only"| synthesis
    research --> bedrock
```

Write boundaries, as the IAM roles enforce them (`infra/template.yaml`, `infra/lab-template.json`):

```mermaid
flowchart LR
    admin["Administrator client"] -->|"upload-pdf, never replaces an original"| papers["papers/"]
    admin --x index["index/"]
    buildindex["build_index in the ingest function"] --> index
    answer["Answer worker"] --> labq["wiki/lab-questions/ and runs/lab-questions/"]
    answer --x papers
    answer --x index
    research["Research worker"] -->|"inside an approval's scope"| wiki["wiki/ synthesis pages"]
    research --x papers
    research --x index
    gateway["Gateway, triage worker, outbox relay"] --x wiki
    gateway --x papers
    gateway --x index
    student["A student's IAM policy"] -->|"only call"| url["Gateway Function URL"]
```

### Ingesting a paper

Two routes bring a paper in. A person's PDF is the usual one; OpenAlex discovery takes only
OpenAlex-hosted papers under a Creative Commons or public-domain licence. For a PDF the note is
written by Bedrock by default, or, on request, by a Claude Code session; everything else stays in
AWS either way.

```mermaid
flowchart TB
    pdf["A person's PDF"] -->|"byeori upload-pdf"| orig["papers/{stem}/original.pdf and meta.json, catalog status pdf_uploaded"]
    orig -->|"byeori aws-extract"| grobid["Extraction task on Fargate: GROBID + worker"]
    grobid -->|"on error"| exfail["extract_failed, picked up again by the next aws-extract"]
    grobid --> clean["papers/{stem}/clean.md and grobid.tei.xml"]
    clean -->|"journal included"| ready["fulltext_ready"]
    clean -->|"otherwise"| parked["fulltext_ready_unclassified"]
    clean -->|"EventBridge: clean.md created"| identity["resolve_identity: GROBID header, DOI at OpenAlex"]
    identity -->|"verified and journal not refused"| ready
    parked -.->|"released by resolve_identity"| ready
    clean -->|"EventBridge: object created"| assets["Asset task on Fargate: papers/{stem}/assets/"]
    ready --> who{"Who writes the note"}
    who -->|"default"| aws["source_note in the ingest function: Bedrock reads clean.md"]
    runners["aws-source-note, aws-pipeline-stems, or aws-notes-run and the Notes workflow"] --> aws
    who -->|"on request"| session["Claude Code session: aws-read-extraction, writes the seven sections"]
    session -->|"aws-publish-source-note with --model-id"| local["publish_source_note: first note only, ingest_harness claude-code, no token counts"]
    aws -->|"sections check fails"| notefail["wiki/sources/failed/{stem}.md"]
    aws -->|"sections check passes"| publish["publish_page"]
    local -->|"sections check passes"| publish
    publish --> note["wiki/sources/{stem}.md"]
    publish --> links["reciprocal links, wiki/indexes/sources.md"]
    note --> rebuild["build_index: nightly, byeori build-index, or the end of the Notes workflow"]
    rebuild --> index["index/"]

    search["byeori search, candidate-add"] --> cand["candidate in the catalog"]
    cand -->|"aws-ingest-candidate"| oaingest["ingest: OpenAlex-hosted PDF and GROBID XML, CC or public domain only"]
    oaingest --> oastore["papers/{work_id}.pdf, sources/{work_id}.md, status fulltext_ready"]
    oastore -->|"EventBridge: object created"| assets
    oastore -->|"aws-draft-page"| draft["wiki/drafts/{work_id}.md, status model_draft"]
    draft -->|"promote-draft: hashes and review recorded"| note
```

`byeori aws-pipeline` runs the OpenAlex route (ingest, draft, promote) for every allowlisted
candidate. `byeori validate` checks the sections of every published page in AWS afterwards; it
reports and writes nothing.

### Synthesis

The Synthesis workflow plans and writes the pages that sit across notes: concept pages, subtopic
pages inside a category, and one category page written from its subtopic pages. It never rewrites
a note's text.

```mermaid
flowchart TB
    start["byeori aws-synthesis-run: scope, categories"] --> scope["resolve_scope"]
    notes[("Ready notes: catalog items and wiki/sources/")] -.-> plansub
    notes -.-> planconcepts
    scope --> plansub["plan_subtopics per category: the model proposes subtopics"]
    plansub --> submanifest["runs/synthesis/{category}/subtopics.json"]
    plansub --> planconcepts["plan_concepts: Glossary terms carried by 5 or more notes"]
    hgnc["reference/hgnc.tsv: gene identity"] -.-> planconcepts
    planconcepts --> candidates["runs/synthesis/concepts/candidates.json"]
    candidates --> writeconcepts["Write concept pages in parallel"]
    writeconcepts --> plansubpages["plan_subtopic_pages"]
    submanifest -.-> plansubpages
    plansubpages --> writesub["Write subtopic pages in parallel"]
    cap["SynthesisMaxPages caps each work list"] -.-> writeconcepts
    cap -.-> writesub
    index["index/: BM25 retrieval of related pages"] -.-> writeconcepts
    index -.-> writesub
    writeconcepts --> concept["wiki/concepts/{slug}.md"]
    writesub --> subtopic["wiki/overviews/{category}/{subtopic}.md"]
    subtopic --> writecat["Category page from its subtopic pages"]
    writecat --> category["wiki/overviews/{category}/index.md"]
    writeconcepts --> bedrock["Bedrock: synthesis model, fallback model when a page is declined"]
    writesub --> bedrock
    writecat --> bedrock
    concept --> publish["publish_page: body once, backlink blocks and wiki/indexes/ catalogs merged"]
    subtopic --> publish
    category --> publish
    writesub -->|"no usable text"| failed["failed/ beside the page, recorded in the catalog"]
    category --> rebuild["build_index, then retry failures and build_index again"]
```

A page with more member notes than one call takes is written in partial pages of 15 notes, merged
15 at a time (`runs/synthesis/{folder}/partials/`). A note's own text is never regenerated by a
synthesis run: `publish_page` only merges a managed backlink block into the pages a new page cites,
and `wiki_backlinks` answers which pages cite a note from the index's links table.

### A member's question

```mermaid
sequenceDiagram
    autonumber
    participant M as Member agent, byeori-lab-mcp
    participant G as Gateway, Function URL
    participant C as Control table
    participant Q as SQS and outbox relay
    participant A as Answer worker
    participant S as Data bucket
    participant B as Bedrock
    participant T as Triage worker
    participant J as Jev
    participant R as Research worker
    M->>G: ask_byeori, signed with the member's IAM key
    G->>C: check the member list, record the job and its outbox row
    G->>Q: send the job to the answer queue
    G-->>M: job_id
    Q->>A: answer job
    A->>S: BM25 search of index/, read pages once into an evidence packet
    A->>S: read originals or supplementary tables when the packet asks
    A->>B: at most two calls, a third only to shorten a cut answer, 64k output tokens
    A->>S: receipts in runs/lab-questions/{job}/
    A->>C: job completed, spend recorded without a cap by default, triage queued
    A->>S: wiki/lab-questions/{period}/{job}.md and the by-page hubs
    M->>G: get_byeori_answer with the job_id
    G-->>M: answer, citations, limitations
    Q->>T: triage job
    T->>T: skip private material, held answers, insufficient evidence
    T->>J: question, answer and cited excerpts, once
    J-->>T: probabilities
    T->>C: verdict, and an offer when review_candidate >= 0.99 with a concrete gap
    M->>G: get_byeori_answer returns the offer, the student decides
    M->>G: respond_to_synthesis_offer with offer_id, revision, hash
    G->>C: approval bound to scope, budget and model, research job queued
    Q->>R: research job, only while the research worker is switched on
    R->>B: research run under the approval's budget
    R->>S: scoped edits and new concept or overview pages, with backlinks
    R->>C: pages written, index_pending until the next index rebuild
    Note over G,R: A professor (admin role) reaches the same approval through list_question_records, propose_research_from_records and decide_research_candidate
```

### The nightly index rebuild

```mermaid
flowchart LR
    schedule["EventBridge: IndexRebuildSchedule"] --> build["Ingest function: build_index"]
    cli["byeori build-index or aws-build-index"] --> build
    workflows["End of the Notes, Synthesis and Question workflows"] --> build
    build --> list["Every Markdown page under wiki/"]
    list --> skip["Skipped: wiki/drafts/, failed/, wiki/indexes/, wiki/lab-questions/, notes not source_ready"]
    skip --> db["SQLite: FTS5 sections, docs, links"]
    db -->|"only if index/ is unchanged since the build began"| index["index/wiki-index-v2.sqlite3"]
    db --> orphans["runs/synthesis/orphans.json"]
    db --> catalogs["wiki/indexes/{folder}.md and wiki/index.md"]
```

### OpenAlex matching

Matches the stored papers in the catalog to OpenAlex records by DOI. OpenAlex is metadata here,
not evidence: its values are stored under an `openalex_` prefix, and only a missing PMID or PMCID
is filled in.

```mermaid
flowchart LR
    cli["byeori aws-openalex-match"] -->|"--dry-run"| plan["openalex_match_plan: what would be selected"]
    cli --> sm["OpenAlexMatch workflow"]
    sm --> batch["openalex_match_batch: up to 50 DOIs per request"]
    batch --> openalex["OpenAlex API"]
    batch --> catalog[("Catalog: openalex_ fields")]
    batch --> progress["runs/openalex-match/{run}.json: continuation"]
    batch --> done{"done"}
    done -->|"no"| wait["Wait out the rate-limit window"]
    wait --> batch
    done -->|"yes"| finished["Finished"]
```

### The administrator's research question

The administrator's questions run the research loop, which may read originals, revise pages and
write new ones, under a per-question budget (`QuestionBudgetUsd`). A student's question never
takes this path unless an offer was accepted or a professor approved it.

```mermaid
flowchart TB
    ask["byeori aws-answer, or answer_wiki_question in byeori-mcp"] --> fn["Ingest function: answer_question"]
    campaign["A list of questions in runs/questions/{run}/manifest.json"] --> qsm["Question workflow"]
    qsm -->|"question_campaign_answer, per question"| fn
    qsm -->|"between batches"| rebuild["build_index"]
    fn --> loop["Research loop, stopped by QuestionBudgetUsd"]
    loop --> bedrock["Bedrock"]
    loop -.-> reads["search_wiki, read_page, read_original, read_supplementary"]
    loop --> writes["write_page, edit_page, refresh_links"]
    writes --> publish["publish_page: notes, concepts and overviews, with backlinks and catalogs"]
    loop --> qpage["wiki/questions/{slug}.md"]
    loop --> trace["runs/agents/{date}/{run}.json, resumed by resume_wiki_question"]
```

### Other flows

- **Category catalogs.** `byeori aws-build-category-catalogs` writes one browse page per field,
  `wiki/indexes/categories/{field}.md`, and their table `wiki/indexes/categories.md`, from the
  index. Like every page under `wiki/indexes/`, they stay out of the search index.
- **Supplementary files.** A paper's supplementary tables and documents sit in
  `papers/{stem}/supplementary/` with a `README.md` guide and a `manifest.json`; the research loop
  and the answer worker read them with `read_supplementary`. This beta has no command that adds
  them.
- **Receipts.** Each run leaves its record under `runs/`: `runs/notes/` and `runs/retry/` (Notes
  workflow), `runs/synthesis/`, `runs/openalex-match/`, `runs/agents/` (research traces),
  `runs/questions/` and `runs/lab-questions/{job}/` (student answers and triage).
- **Cost ledger.** `byeori cost-ledger` sums a local ledger in the clone's `state/` folder, which
  the CLI appends to after the steps it runs itself; it is an estimate from list prices, not a
  bill.
- **Hubs of answered questions.** `byeori aws-link-lab-questions --apply` adds one standing line
  to each indexed page pointing at its hub under `wiki/lab-questions/by-page/`.
- **Audit.** The CloudTrail trail records every write under `wiki/overviews/`, `wiki/concepts/`
  and `wiki/questions/` in a separate audit bucket.
