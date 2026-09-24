# Installing Byeori

Ten steps. Steps 1 to 6 give you a working wiki with one paper in it; 7 to 9 are optional; 10 is
how to remove it again. Each step says what it creates, and `docs/AWS-SERVICES.md` explains the
services by name. Run every command from the root of your clone.

If you use Claude Code, open the clone and type `/byeori-install`: the agent walks through these
same steps with you and asks before each one that creates something or costs money.

## 1. Prerequisites

You need:

- **An AWS account** and a user in it with administrator access, set up as a profile of the AWS
  CLI (`aws configure --profile byeori` asks for the user's access key). Check it with
  `aws sts get-caller-identity --profile byeori`; it prints the account and the user.
- **A region** where Bedrock offers the Claude models, for example `us-east-1`.
- **Bedrock model access.** In the AWS console, switch to your region, open Bedrock → Model access,
  and request access to the Claude models (Byeori's default is Claude Opus 5). Approval is usually
  immediate. Without it every note fails; step 5 checks it.
- **Docker**, running. `docker --version` should answer.
- **`uv`** (`uv --version`) and **the AWS CLI** (`aws --version`).
- **An OpenAlex API key.** Free; create one at [openalex.org](https://openalex.org). Byeori uses it
  to look papers up and to check which paper a PDF is.

Then clone and install the Python package:

```bash
git clone https://github.com/joonan-lab/byeori
cd byeori
uv sync
```

## 2. `byeori init`

```bash
uv run byeori init
```

It asks six questions and writes the answers to `.byeori.env` (readable only by you):

1. The AWS CLI profile to use (`byeori`).
2. The region (`us-east-1`).
3. The name of the CloudFormation stack to create (`byeori`).
4. The Parameter Store name that will hold your OpenAlex key (`/byeori/openalex-api-key`).
5. A contact e-mail. It is sent to OpenAlex, Crossref and NCBI as the "polite pool" address,
   which gets more reliable service. You may leave it empty.
6. Whether the stack should create its own network (`true`). Say `true` unless your account
   already has a VPC you want to use. If you say `false`, it asks for that VPC's id and a
   comma-separated list of its public subnet ids.

Every later command reads its settings from the environment, so load the file in each new
terminal:

```bash
source .byeori.env
```

For a scripted install (no terminal prompts), pass `--non-interactive` and answer each question
with `--set KEY=VALUE`. The keys are `AWS_PROFILE`, `AWS_REGION`, `KIRO_WIKI_STACK`,
`KIRO_WIKI_OPENALEX_PARAMETER`, `KIRO_WIKI_CONTACT_EMAIL` and `KIRO_WIKI_CREATE_VPC`; when
`KIRO_WIKI_CREATE_VPC` is not `true`, also set `KIRO_WIKI_VPC_ID` and `KIRO_WIKI_SUBNET_IDS`. Any
key you leave out keeps its default (or, on a rerun, the value already in `.byeori.env`):

```bash
uv run byeori init --non-interactive \
  --set AWS_PROFILE=byeori --set AWS_REGION=us-east-1 --set KIRO_WIKI_STACK=byeori \
  --set KIRO_WIKI_OPENALEX_PARAMETER=/byeori/openalex-api-key \
  --set KIRO_WIKI_CONTACT_EMAIL=lab@example.org --set KIRO_WIKI_CREATE_VPC=true
```

## 3. `byeori deploy`

First put your OpenAlex key into Parameter Store, under the name you gave in step 2. Run this
yourself and paste the key where it says; do not paste it into a chat with an agent:

```bash
source .byeori.env
aws ssm put-parameter --name /byeori/openalex-api-key --type SecureString --value <your key>
```

(Sourcing the file first makes the AWS CLI use your profile and region.)

`SecureString` means the value is stored encrypted. Use the name you chose if it differs.

Then deploy:

```bash
source .byeori.env
uv run byeori deploy
```

This hands `infra/template.yaml` to CloudFormation, which creates the stack: the data bucket (S3),
the catalog table (DynamoDB), the ingest and synthesis functions (Lambda), the extraction
cluster (Fargate), the image repository (ECR), the workflows (Step Functions), the audit trail
(CloudTrail) and the timers (EventBridge). It takes ten to fifteen minutes. When it finishes, the
command writes the bucket, table and ingest function names back into `.byeori.env`; run
`source .byeori.env` again.

On a first install, before the stack exists, there is no bucket yet to package the CloudFormation
templates into — the data bucket above is itself one of the stack's outputs. `byeori deploy`
creates a small deployment bucket of its own for this, `byeori-deploy-<account>-<region>`, and
prints its name. It stays afterward (it holds a few kilobytes of packaged templates) and is
reused by every later deploy once the env file has a data bucket configured.

What `CreateVpc` means: the Fargate tasks need a network with internet access, to pull their
container images and to reach S3 and DynamoDB. With `true` (your answer in step 2)
the stack creates a small network of its own (a VPC with two public subnets), which costs nothing
while idle. With `false` it uses the VPC and subnets you named.

Other settings can be changed by naming them after the command, as `Key=Value`. The ones you are
most likely to want are the models (`DraftModelId`, `NoteModelId`, `NoteFallbackModelId`,
`SynthesisModelId`, `SynthesisFallbackModelId`; all default to `global.anthropic.claude-opus-5`),
the thinking effort (`IngestReasoning`, `SynthesisReasoning`), the spending cap per research
question (`QuestionBudgetUsd`, 12 dollars as deployed) and the nightly index rebuild
(`IndexRebuildSchedule`, a cron expression in UTC; empty turns it off). For example:

```bash
uv run byeori deploy NoteModelId=global.anthropic.claude-sonnet-5 IndexRebuildSchedule="cron(0 3 * * ? *)"
```

If the deploy fails, CloudFormation rolls back and leaves the previous state in place. Open
CloudFormation → Stacks → your stack → Events in the console: the first `CREATE_FAILED` line says
why. Fix the input and run the command again.

## 4. `byeori build-workers`

```bash
uv run byeori build-workers
```

Builds the asset worker image with Docker (the program that cuts figures and tables out of a
PDF), pushes it to the stack's ECR repository, and publishes the worker script to the bucket.
The first build downloads several gigabytes and takes several minutes. Docker must be running.

No Docker on this machine? You can skip this step for now and come back to it later, from any
machine that has Docker, by running `uv run byeori build-workers` again against the same
`.byeori.env`. Without the image, step 6 still writes the evidence note: extraction and the note
itself use Fargate and Bedrock, not the asset worker. Only the figure- and table-cutting task is
affected, and it fails visibly rather than silently — `aws-extract-status` shows it as STOPPED
with `CannotPullContainerError` (the ECR repository has no image yet) while GROBID extraction and
the note both succeed. The note is complete text with figures and tables simply missing until you
run `build-workers`.

## 5. `byeori doctor`

```bash
source .byeori.env
uv run byeori doctor
```

It prints one JSON object. Read it field by field:

| Field | What it should say | If it does not |
|---|---|---|
| `python`, `uv` | a version, `true` | install `uv` |
| `aws.credentials` | `true` | the profile in `.byeori.env` cannot sign in; rerun `aws configure --profile <profile>` |
| `aws.bucket_configured`, `aws.table_configured`, `aws.ingest_function_configured` | `true` | `.byeori.env` was not sourced, or step 3 did not finish |
| `openalex_live` | `true` | the key in Parameter Store is missing or wrong; `openalex_error` says which |
| `bedrock` | `"ok"` for every model the stack uses | an AWS error code, usually `AccessDeniedException`: model access was not granted in this region (step 1) |
| `openalex_api_key`, `kiro_cli` | may be `false` | not needed; the key is read in AWS, not on your computer |

The command exits with an error only when `uv` or `openalex_live` fails; still read `bedrock`
yourself, because nothing else will work without it.

## 6. First paper

One paper you have as a PDF. Every step runs in AWS; your computer only sends the PDF and reads
the results. It costs roughly what `docs/COST.md` gives for one paper.

Upload the PDF. The file on your computer is only read, never moved or changed. `--stem` is the
paper's folder name in the bucket: lowercase letters, digits and hyphens, usually
`<first author>-<year>-<a few title words>`, for example `smith-2024-cortical-organoids`.

```bash
source .byeori.env
uv run byeori upload-pdf <path to the PDF> --stem <author-year-words>
```

(If you already know the paper's OpenAlex work id and would rather start from there: `uv run
byeori search "<title of the paper>" --limit 5`, then `uv run byeori candidate-add <OpenAlex work
id, for example W1234567890>`, which prints the record including its `stem`, then `uv run byeori
attach-pdf <OpenAlex work id> <path to the PDF>` in place of `upload-pdf` above.)

Extract its text with GROBID on Fargate, and check on the task (a few minutes):

```bash
uv run byeori aws-extract --stems <stem> --tasks 1
uv run byeori aws-extract-status
```

Watch for two things here, both harmless on their own:

- Without step 4's asset worker image, the extraction's asset task fails with
  `CannotPullContainerError` in the status output. That is expected if you skipped `build-workers`;
  extraction itself still finishes, and the note below is unaffected.
- `aws-extract-status` may settle at `fulltext_ready_unclassified` instead of `fulltext_ready`.
  That means GROBID read the PDF's text, but the automatic identity check — matching the title and
  DOI it found against OpenAlex — could not settle which paper this is, so the pipeline parked it
  rather than guess. This is common for commentaries, letters and papers with an unusual first
  page; a straightforward research article with a clear title and a resolvable DOI almost always
  reaches `fulltext_ready` on its own. If it stays at `fulltext_ready_unclassified`, the next
  `aws-source-note` step fails with `has no extracted text yet (ingest_status
  fulltext_ready_unclassified)` — a misleading message, since the text *is* there; only the
  identity is unsettled. `uv run byeori resolve-ids --help` looks like the tool for this, but
  reading it shows it requires a required path argument pointing at a separate local paper-notes
  checkout to compare against, for a bulk migration this repository does not ship and a new
  installer will not have — it is not a fix for a single unresolved upload. For your first paper,
  the practical fix is simply to pick a different PDF — an ordinary journal article rather than a
  commentary or a paper with a non-standard layout — and upload that instead.

Have Bedrock write the evidence note from the extracted full text:

```bash
uv run byeori aws-source-note <stem>
```

It prints `"status": "source_ready"` when the note is stored under `wiki/sources/`. Then check
every page's structure, rebuild the search index, and search:

```bash
uv run byeori validate
uv run byeori build-index
uv run byeori wiki-search "<a phrase from the paper>"
uv run byeori wiki-read note <stem>
```

`validate` only checks notes and syntheses — pages under `wiki/sources/`, `wiki/overviews/`,
`wiki/concepts/` and `wiki/questions/` — never the catalog pages the index builder generates under
`wiki/indexes/` or the lab's own question pages under `wiki/lab-questions/`. A clean first paper
prints nothing and exits 0. A real failure looks like one line per missing section, naming the S3
key and the heading, for example:

```
s3://<your bucket>/wiki/sources/<stem>.md: missing ## Evidence boundary
```

and the command exits 1. That means the note itself is missing a required section; open it
(`uv run byeori wiki-read note <stem>`) and see what is short.

For many papers at once, `uv run byeori aws-pipeline-stems` writes the note for every extracted
paper that has none, in parallel, and rebuilds the index at the end.

## 7. Student service (optional)

A separate stack that lets lab members ask questions and read the wiki without being
administrators. What it does and how members connect is in `docs/LAB-SERVICE.md`.

It needs one KMS key, even if you never use Jev (step 8): the template gives its triage worker
permission to decrypt the Jev key with it. Create the key once and add its ARN to `.byeori.env`:

```bash
aws kms create-key --description "byeori parameters"
aws kms create-alias --alias-name alias/byeori-parameters --target-key-id <KeyId from the output>
echo "export LAB_JEV_KMS_KEY_ARN=<Arn from the create-key output>" >> .byeori.env
source .byeori.env
```

The key costs $1 a month. `byeori deploy-lab` names the stack `<your KIRO_WIKI_STACK>-lab` by
default (recorded as `LAB_STACK` in `.byeori.env` the first time you run it) — if you administer
more than one Byeori installation in this account, each needs its own stack name, so do not reuse
one; setting `LAB_STACK` yourself in `.byeori.env` before the first run picks a different one.
Then deploy the stack and register each member (the member's IAM user must exist first;
`docs/LAB-SERVICE.md` explains how):

```bash
uv run byeori deploy-lab
uv run python scripts/lab_members.py register --member-id <id> --iam-user <IAM user name> --role student --attach-policy
uv run python scripts/lab_members.py outputs
```

`outputs` prints `GatewayUrl`, the address members connect to.

## 8. Jev (optional)

Jev is an external model that, after a student's answer, estimates whether the question deserves
a new synthesis page. It needs a key from TypeSafe, stored in Parameter Store, and the student
service from step 7. Everything is in `docs/JEV.md`.

## 9. Connect a writing agent

Byeori comes with two MCP servers, which let an agent such as Claude Code or Codex use the wiki
as a set of tools.

**Administrator (`byeori-mcp`)**: full access through your administrator profile, from this
clone. Copy the values from `.byeori.env`:

```bash
claude mcp add -s user byeori \
  -e AWS_PROFILE=<profile> -e AWS_REGION=<region> \
  -e AWS_KIRO_WIKI_BUCKET=<bucket> -e AWS_KIRO_WIKI_TABLE=<table> \
  -e AWS_KIRO_WIKI_INGEST_FUNCTION=<ingest function> \
  -- uv run --project <absolute path to this clone> byeori-mcp
```

If a second person administers the stack, give their IAM user the policy it needs:
`uv run byeori grant-client --attach-to-user <IAM user name>` (without `--attach-to-user` it only
prints the policy).

**Members (`byeori-lab-mcp`)**: answer-only, through the student service of step 7. No clone
is needed; `uvx` fetches the package:

```bash
claude mcp add -s user byeori-lab -e LAB_FUNCTION_URL=<gateway url> -e AWS_REGION=<region> -e AWS_PROFILE=<profile> -- uvx --from git+https://github.com/joonan-lab/byeori byeori-lab-mcp
```

For Codex, add to `~/.codex/config.toml`:

```toml
[mcp_servers.byeori-lab]
command = "uvx"
args = ["--from", "git+https://github.com/joonan-lab/byeori", "byeori-lab-mcp"]
startup_timeout_sec = 60

[mcp_servers.byeori-lab.env]
LAB_FUNCTION_URL = "<gateway url>"
AWS_REGION = "<region>"
AWS_PROFILE = "<profile>"
```

The administrator server goes into Codex the same way, with `command = "uv"` and
`args = ["run", "--project", "<absolute path to this clone>", "byeori-mcp"]` and the five
variables above under `[mcp_servers.byeori.env]`.

## 10. Cost and cleanup

What things cost is in `docs/COST.md`. To see what you have actually spent, open Billing → Cost
Explorer in the console and group by service.

To remove Byeori, delete the student stack first (if you made one; its name is `LAB_STACK` in
`.byeori.env`, `<your stack>-lab` unless you chose otherwise), then the main stack:

```bash
aws cloudformation delete-stack --stack-name <your stack name>-lab
aws cloudformation delete-stack --stack-name <your stack name>
```

**If you registered any lab members with `--attach-policy`,** the student stack's delete fails
first, with `Cannot delete a policy attached to entities`: CloudFormation will not delete an IAM
policy that is still attached to a user. Detach it from every member first. List who has it
attached (the policy ARN is the student stack's `StudentAccessPolicyArn` or `AdminAccessPolicyArn`
output, `aws cloudformation describe-stacks --stack-name <your stack name>-lab --query
"Stacks[0].Outputs"`):

```bash
aws iam list-entities-for-policy --policy-arn <StudentAccessPolicyArn or AdminAccessPolicyArn> --entity-filter User
aws iam detach-user-policy --user-name <IAM user name> --policy-arn <that policy arn>
```

Repeat the detach for each user the list shows, then delete the student stack again.

On purpose, deleting a stack does **not** delete data the design wants kept past the stack's own
life. These stay, and keep costing their storage, until you remove them by hand:

- **The data bucket** (`AWS_KIRO_WIKI_BUCKET` in `.byeori.env`), with every PDF and page. To remove
  it, open S3 → the bucket → Empty in the console, confirm, then Delete. This cannot be undone;
  download anything you want to keep first.
- **The audit bucket** (the stack output `AuditBucketName`), emptied and deleted the same way.
- **The catalog table** (`AWS_KIRO_WIKI_TABLE`): `aws dynamodb delete-table --table-name <table>`.
- **The student stack's control table** (`<lab stack name>-control`), retained with deletion
  protection on, so the delete above leaves it behind. Turn protection off, then delete it:

  ```bash
  aws dynamodb update-table --table-name <lab stack name>-control --no-deletion-protection-enabled
  aws dynamodb delete-table --table-name <lab stack name>-control
  ```
- **The main stack's asset-trigger log group** is retained too. Find its name (CloudFormation
  gives Lambda functions a generated name, so it is not simply the stack name) and delete it:

  ```bash
  aws cloudformation describe-stack-resources --stack-name <your stack name> \
    --logical-resource-id AssetTriggerFunction --query "StackResources[0].PhysicalResourceId" --output text
  aws logs delete-log-group --log-group-name /aws/lambda/<that function name>
  ```
- **The ECR repository** with the worker image: in the console, ECR → Repositories → the
  repository → Delete, or `aws ecr delete-repository --repository-name <name> --force`.
- **The deployment bucket** `byeori-deploy-<account>-<region>` that step 3 creates on a first
  install is not part of any stack and is never deleted by CloudFormation. If you are removing
  Byeori entirely from this account, empty and delete it the same way as the data bucket (S3 → the
  bucket → Empty, then Delete).
- **The parameters and the KMS key**: `aws ssm delete-parameter --name <name>` for each key you
  stored, and `aws kms schedule-key-deletion --key-id <key arn> --pending-window-in-days 7`. Never
  delete a Parameter Store parameter or the KMS key if another Byeori installation in this account
  still uses it — the OpenAlex parameter and the Jev key are each named once per installation, but
  a shared KMS key (`LAB_JEV_KMS_KEY_ARN`) can outlive any one of them.
