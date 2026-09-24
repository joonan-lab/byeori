# Changelog

## v0.1.0-beta.1 (unreleased)

Copyright 2026 Joon An and the An Lab, Apache-2.0. First public beta. Exported from the An Lab
workspace (see `RELEASE-SOURCE`).

Contains: the `byeori` package and CLI; the main stack (S3, DynamoDB, Lambda ingest and
synthesis, Fargate extraction, Step Functions, audit trail); the optional student service stack
and its MCP server; the optional Jev triage; installer commands `init`, `deploy`,
`build-workers`, `deploy-lab`, `grant-client`, `doctor`; the journal policy as a data file.

Does not contain: a language setting for prompts (reserved as `BYEORI_LANGUAGE`); the lab's own
intake and repair tools; any shared or hosted instance.
