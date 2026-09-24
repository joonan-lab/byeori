---
name: byeori-install
description: Install Byeori into the person's own AWS account step by step, running the release's commands and explaining each AWS service as it appears. Use when someone asks to install, set up or deploy Byeori, or says they have an AWS account and want the system running.
---

# Installing Byeori for someone who may not know AWS

You run the release's commands and explain what they do. You never create AWS resources by any other means: no `aws cloudformation create-stack`, no `iam create-role`, no console instructions for resources the templates make. If a step fails, read the error, explain it in the terms of `docs/AWS-SERVICES.md`, and fix the input, not the template.

Follow `docs/INSTALL.md`; its ten steps are the checklist. At each step:

1. Say in one or two sentences what the step creates and which AWS service it lives in
   (`docs/AWS-SERVICES.md` has the paragraph to draw on).
2. If the step creates account resources or spends money (deploy, build-workers, deploy-lab, the
   first paper, the Jev key), say what it will cost or create and **wait for the person's yes**
   before running it. Ask before every such step, not once for all.
3. Run the command exactly as the document gives it. Read the output. Report the result in plain
   words and the next step.

## The commands, in order

- Prerequisites check: `aws --version`, `uv --version`, `docker --version`,
  `aws sts get-caller-identity --profile <profile>`; Bedrock model access is checked later by `doctor`.
- `byeori init` : asks for the profile, region, stack name, the Parameter Store name for the
  OpenAlex key, and whether the stack should create its own network. Writes `.byeori.env`.
- Put the OpenAlex key in Parameter Store (the document gives the `aws ssm put-parameter` line;
  the person pastes their own key; never ask them to paste it into the chat).
- `byeori deploy` : creates the main stack. Ten to fifteen minutes. Explain the stack's parts as
  the events scroll: bucket, table, functions, cluster, workflows, audit trail.
- `byeori build-workers` : builds the asset worker image with Docker and pushes it. Needs Docker
  running. Several minutes the first time.
- `byeori doctor` : checks the profile, the outputs, Bedrock access for every configured model,
  and the OpenAlex key. A `bedrock` entry that is not `ok` means model access was not granted in
  that region; point at the Bedrock console's model access page and stop until it is.
- First paper: `docs/INSTALL.md` step 6. Costs roughly what `docs/COST.md` says for one paper.
- Optional: `byeori deploy-lab` (student service), `docs/JEV.md` (Jev), `docs/LAB-SERVICE.md`
  (members and the `byeori-lab-mcp` server).
- `byeori grant-client --attach-to-user <name>` when a second administrator user needs access.

## What you do not do

- Do not type, store or echo a key or a password. The person runs the `put-parameter` line.
- Do not delete anything. Cleanup is `docs/INSTALL.md` step 10, run by the person.
- Do not change `infra/` to make a deploy pass. Report the error instead.
