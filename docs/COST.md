# What Byeori costs

Three numbers matter: what one paper costs, what one question costs, and what the system costs
in a month when nobody uses it.

These are **one lab's numbers**, measured in that lab's account in September 2026 at Bedrock's
list prices in US dollars, before tax. Your papers, questions, region and model choice will give
different ones. To read your own, open **Billing → Cost Explorer** in the AWS console, set the
date range, and group or filter by "Service": Bedrock appears as "Amazon Bedrock" (or under the
Claude model's own name), next to "Amazon Simple Storage Service", "AWS Lambda" and the others.
Cost Explorer lags by about a day. The stack also records each note's token counts in the catalog
table, and `uv run byeori cost-ledger` summarises the time and estimated cost of the steps run
from your computer.

## Per paper

The evidence note is almost the whole cost of a paper. The lab's estimate assumes a paper of
12,000 input tokens (the extracted full text plus the writing rules) and 3,000 output tokens (the
note):

| Model | One paper | 1,000 papers |
|---|---:|---:|
| Claude Opus 5 (the default) | about $0.135 | about $135 |
| Claude Sonnet 5 | about $0.055 | about $55 |

Add about $0.02 per paper for the rest: GROBID extraction and the figure worker on Fargate (about
a cent together) and the OpenAlex lookups.

Two things move the note's cost. A long paper reads more tokens; the lab's three measured papers
read between 9,000 and 26,000. And "thinking" (`IngestReasoning`, which the installer deploys at
`high`) is billed as output on top of the 3,000: a single note the lab measured on Opus 5 with
thinking on came to about $0.30. Set `IngestReasoning=default` in `byeori deploy` for no thinking.

## Per question

- **A research question** (an administrator's question that may read originals and write
  synthesis pages): with a budget of $12 per question (`QuestionBudgetUsd`), six questions the lab
  ran on 2026-09-23 cost $13.25 together, a median of **$2.21** per question. The budget is a
  ceiling at which the question stops and writes its answer, not a typical cost.
- **A student answer** (the optional student service; answer-only, at most two model calls):
  about **$0.25**. Twenty answers asked at once cost $5.06.
- **Jev triage** (optional): a fraction of a cent per answer; see `docs/JEV.md`.

## Idle month

When nobody adds papers or asks questions:

- **S3**: billed by stored size, about $0.023 per GB-month in us-east-1. A PDF with its extraction
  and note takes a few megabytes, so a thousand papers is a few gigabytes: well under a dollar.
- **KMS**: $1 per key-month for the key the student service and Jev use (`docs/INSTALL.md` step 7);
  nothing if you did not create it.
- **Everything else** (Lambda, Fargate, Step Functions, DynamoDB, Bedrock, EventBridge,
  CloudFormation, IAM, Parameter Store): near zero. They bill only while they work, and the
  nightly index rebuild is one short function run. The student service's one-minute relay is a
  small Lambda that finds nothing to do.
