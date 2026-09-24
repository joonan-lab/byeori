# The AWS services Byeori uses

Each section says what the service is, what Byeori does with it, where to look in the AWS console
when something fails, and what its line on the bill looks like. Open the console at
https://console.aws.amazon.com, check the region shown at the top right is the one in
`.byeori.env`, and type the service's name in the search bar.

Most failures show up in one of four places:

- **CloudFormation → Stacks → your stack → Events**: why a deploy failed (the first `CREATE_FAILED`
  or `UPDATE_FAILED` line).
- **Lambda → Functions → the function → Monitor**: whether a function ran, how long it took, and
  whether it errored; "View CloudWatch logs" opens its log.
- **ECS → Clusters → the extraction cluster → Tasks**: the Fargate tasks, running and stopped; a
  stopped task shows its stop reason.
- **CloudWatch → Log groups**: every function's and task's log, one group each, named after it.

The bill is read in **Billing → Cost Explorer**: group by "Service" to see one line per service.

## IAM

Identity and Access Management decides who may do what. Every user, program and service acts
under an identity, and a policy attached to that identity lists what it may do.

Byeori's stacks create one role per function and per task; each role allows only that piece's
own work (the ingest function may read and write the bucket and call Bedrock; the student
gateway may read the wiki but never write it). Your administrator user gets one policy from
`byeori grant-client`; each student of the optional service gets the stack's student policy,
attached by `scripts/lab_members.py register --attach-policy`, which allows asking and reading only.

When something fails: an error containing `AccessDenied` or `not authorized to perform` names the
action and the identity. Look the identity up in IAM → Users (a person) or IAM → Roles (a function).

On the bill: IAM is free and has no line.

## S3

Simple Storage Service stores files, called objects, in a bucket. An object's name (its "key")
looks like a path, such as `papers/<stem>/original.pdf`.

Byeori's data bucket holds everything the wiki is: the original PDFs, the extracted text, the
evidence notes and synthesis pages, the search index and the run receipts (`docs/ARCHITECTURE.md`
has the layout). The stack keeps the bucket if the stack is deleted, so an uninstall never
removes papers. A second bucket holds the audit trail (see CloudTrail).

When something fails: S3 → Buckets → the bucket, then browse to the key. A missing object is
usually a step that did not run.

On the bill: "Amazon Simple Storage Service", mostly storage (about $0.023 per GB-month in
us-east-1) plus a small amount per request.

## DynamoDB

A database of small records that needs no server and bills per request.

Byeori's catalog table holds one record per paper (identifiers, where its files are, which steps
are done and what they cost) and the state of running jobs. The student service has its own
control table (members, questions, offers, approvals).

When something fails: DynamoDB → Tables → the table → Explore items, and search by `work_id`.

On the bill: "Amazon DynamoDB", per read and write; near zero when nobody uses it.

## Lambda

Lambda runs code when it is asked to and stops when the code is done. You pay for the seconds it
runs, and nothing between.

Byeori's main stack has three functions: the ingest function (OpenAlex lookups, evidence notes,
answers, validation, the index rebuild), the synthesis function (pages across notes) and the
asset trigger (starts the figure worker). The student stack adds a gateway, an answer worker, a
triage worker, an outbox relay and a research worker (`docs/ARCHITECTURE.md`).

When something fails: Lambda → Functions → the function → Monitor, then "View CloudWatch logs".
A function stopped at 15 minutes timed out.

On the bill: "AWS Lambda"; usually cents. Its logs appear as "AmazonCloudWatch".

## Fargate

Fargate runs a container (a program packaged with everything it needs) without a server of your
own. It is part of ECS, the Elastic Container Service.

Byeori starts Fargate tasks for two jobs: GROBID, which turns a PDF into structured text, and the
asset worker, which cuts the figures and tables out of the PDF. Each task stops when its papers
are done.

When something fails: ECS → Clusters → the extraction cluster → Tasks. Tick "Stopped" to see
finished tasks and their stop reason, and open the task's Logs tab. `byeori aws-extract-status`
prints the same list.

On the bill: under "Amazon Elastic Container Service", per vCPU and GB per second while a task
runs; about a cent per paper.

## ECR

Elastic Container Registry stores container images, the packaged programs Fargate runs.

`byeori build-workers` builds the asset worker image on your computer and pushes it to the
stack's ECR repository. GROBID's image is public and is pulled from Docker Hub instead.

When something fails: ECR → Private registry → Repositories → the repository shows whether an
image with the tag `latest` exists. A task that cannot start with `CannotPullContainerError`
means step 4 of `docs/INSTALL.md` did not finish.

On the bill: "Amazon EC2 Container Registry (ECR)", a few cents per GB-month.

## Step Functions

Step Functions runs workflows ("state machines"): a list of steps run in order or in parallel,
with retries, that keeps running whether or not your computer stays on.

Byeori's main stack has four: evidence notes for many papers, OpenAlex matching, synthesis, and
question campaigns. A workflow that fails part-way can be started again and resumes where it
stopped.

When something fails: Step Functions → State machines → the machine → Executions → the failed
execution. The graph shows which step failed and its error.

On the bill: "AWS Step Functions", per state transition; cents for a large run.

## Bedrock

Bedrock is where AWS runs large language models, including Anthropic's Claude, called from
your own account and billed to it.

Every evidence note, synthesis page and answer is written by a Claude model in Bedrock. The model
is a stack setting (`NoteModelId` and the others in `docs/INSTALL.md` step 3). Bedrock is the
only large cost, billed per token (roughly, per word read and written).

When something fails: `byeori doctor` calls each configured model once. `AccessDeniedException`
means model access was not granted in this region (Bedrock → Model access);
`ThrottlingException` means too many calls at once, which the workflows retry.

On the bill: "Amazon Bedrock", or a line named after the Claude model; most of the bill.

## Parameter Store

Part of AWS Systems Manager: a store for small settings and secrets. A `SecureString` parameter
is kept encrypted.

Byeori keeps the OpenAlex key here, and the Jev key if you use Jev (`/byeori/jev/api-key`). The
functions read them under their own roles when they need them; no key is written to a file or a
log.

When something fails: Systems Manager → Parameter Store shows whether the parameter exists and
its version. `openalex_live: false` from `byeori doctor` usually means the name in `.byeori.env`
does not match the parameter.

On the bill: standard parameters are free; no line.

## KMS

Key Management Service holds encryption keys and uses them on behalf of services.

Byeori needs one key you create (`docs/INSTALL.md` step 7): it encrypts the Jev parameter, and
only the Jev triage worker may decrypt it. The OpenAlex parameter uses the AWS-managed default
key, which costs nothing.

When something fails: KMS → Customer managed keys shows whether the key is enabled. An error
mentioning `kms:Decrypt` means the parameter was encrypted with a different key from the one
given as `LAB_JEV_KMS_KEY_ARN`.

On the bill: "AWS Key Management Service", $1 per key-month.

## CloudFormation

CloudFormation builds a set of resources from one template file, called a stack, and can update
or delete them together.

`byeori deploy` hands `infra/template.yaml` to CloudFormation and `byeori deploy-lab` hands it
`infra/lab-template.json`. Everything Byeori has in your account was made this way; nothing is
created by hand.

When something fails: CloudFormation → Stacks → the stack → Events. Read from the bottom up to
the first failure; the lines after it are the rollback. The Outputs tab lists the names the
stack created.

On the bill: free; no line.

## CloudTrail

CloudTrail records calls made to AWS: who did what, and when.

Byeori's trail records every write to the synthesis pages (`wiki/overviews/`, `wiki/questions/`,
`wiki/concepts/`), so a changed page can be traced to the person or function that changed it.
The records go into a second bucket and expire after 400 days.

When something fails: CloudTrail → Trails shows whether the trail is logging. The records
themselves are files in the audit bucket.

On the bill: "AWS CloudTrail" for the recorded events, and the audit bucket's storage under S3;
both small, because only synthesis writes are recorded.

## EventBridge

EventBridge runs timers and reacts to events in the account.

Byeori has three rules: the nightly index rebuild (`IndexRebuildSchedule`), the figure worker
started when a new paper's text is stored, and the identity check started at the same moment.
The student stack adds a one-minute timer that passes work to its queues.

When something fails: EventBridge → Rules → the rule → Monitoring shows whether it fired and
whether the call to its target failed; the target function's own log says why.

On the bill: scheduled rules and events from your own account are free; usually no line.
