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
