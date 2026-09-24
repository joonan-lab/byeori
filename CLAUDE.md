# Byeori

Byeori is a serverless scientific knowledge system a lab installs in its own AWS account:
original papers in S3, one evidence note per paper written from its full text, syntheses across
notes, a search index, and a question service, all with provenance. This file tells a coding
agent how to work here. It is short on purpose; the documents it points at carry the detail.

## Installing

Use the `byeori-install` skill (`.claude/skills/byeori-install/SKILL.md`): it walks the person
through `docs/INSTALL.md` step by step, runs the release's own commands (`byeori init`,
`byeori deploy`, `byeori build-workers`, `byeori doctor`, and the optional `byeori deploy-lab`),
and explains each AWS service as it is created (`docs/AWS-SERVICES.md`). Never write
infrastructure by hand or call `aws cloudformation create-stack` yourself: the templates in
`infra/` are the installation, reviewed and tested, and every account gets the same ones.

## Boundaries that hold in every session

- Application functions (search, extraction, validation, indexing, ingestion, synthesis) run in AWS. The client authenticates, submits and reads bounded results. Do not move a function to the laptop or download the index to answer a small request.
- The wiki lives in S3 and only there. Keep no local copy of PDFs, extractions, pages or the index; there is no local mirror to sync.
- A page needs the paper's full text read from the stored original. Metadata, an abstract or a landing-page snippet is never enough. Record limitations and the line between reported results and interpretation.
- Preserve every original: upload, never move or overwrite, never delete objects in bulk, never run a destructive sync, never disable the audit trail, and never weaken the bucket policy if `infra/storage-protection.yaml` was applied.
- Secrets live in Parameter Store, read by the Lambdas under their roles. Do not put a key in a
  file, a log or a shell history, and do not call a model from the laptop.
- Deploy with the release's commands. A deploy that fails leaves the previous settings in place;
  do not retry with guessed parameters.

## Where things are

`docs/INSTALL.md` (the walkthrough), `docs/AWS-SERVICES.md` (what each service is and why),
`docs/COST.md` (what a paper, a question and an idle month cost), `docs/ARCHITECTURE.md`,
`docs/LAB-SERVICE.md` (the optional student service), `docs/JEV.md` (the optional triage),
`src/byeori/policies/journals.json` (the journal allowlist; replace it for your lab),
`infra/template.yaml` and `infra/lab-template.json` (the stacks), `tests/` (`uv run pytest -q`).
