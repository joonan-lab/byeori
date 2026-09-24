# Changelog

## v0.1.0-beta.1 (2026-09-24)

Copyright 2026 Joon An and the An Lab, Apache-2.0. First public beta. Exported from the An Lab
workspace (see `RELEASE-SOURCE`).

Contains: the `byeori` package and CLI; the main stack (S3, DynamoDB, Lambda ingest and
synthesis, Fargate extraction, Step Functions, audit trail); the optional student service stack
and its MCP server; the optional Jev triage; installer commands `init`, `deploy`,
`build-workers`, `deploy-lab`, `deploy-jev-eval`, `grant-client`, `doctor`; `upload-pdf` for a
paper's original PDF; the journal policy as a data file.

Does not contain: a language setting for prompts (reserved as `BYEORI_LANGUAGE`); the lab's own
intake and repair tools; any shared or hosted instance.

Known limits of this beta:

- `build-workers` is unverified: the machine that verified this release had no Docker.
- A paper whose identity cannot be resolved against OpenAlex stays parked at
  `fulltext_ready_unclassified`; there is no per-paper fix yet.
- The student-facing offer messages, the lab-question page headings and the `byeori-lab` setup
  text are in Korean.
