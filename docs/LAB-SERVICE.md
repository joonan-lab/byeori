# The student question service

An optional second stack (`uv run byeori deploy-lab`, `docs/INSTALL.md` step 7) that lets every
member of a lab ask the wiki questions and read it, from their own writing agent, without
administrator rights. It has been run by one lab so far; treat it as beta.

## What a student gets

- **Answers with citations.** A question gets an answer written from a bounded packet of evidence
  from the wiki, in at most two model calls, with its citations, its limitations and a statement
  of how well the evidence supports it. Answering never edits the wiki; the answer is kept as a
  record under `wiki/lab-questions/` and in the service's receipts, outside the search index.
- **Reading.** Search, bounded reads of any page (8,000 characters at most at a time), the pages
  that cite a page, and the stored full text behind a note, to check a claim against the paper.
- **A request for a missing paper**, recorded for the administrator to add.
- **A synthesis offer, only with consent.** After an answer, the optional Jev triage (`docs/JEV.md`)
  estimates whether the question deserves a new or improved synthesis page. When that probability
  is at least 0.99 and a concrete gap in the wiki was identified, the answer carries an offer. The
  agent shows it to the student and waits: nothing runs unless the student explicitly accepts.
  Accepting starts one research run, within the scope and budget the offer states, and its pages
  join the wiki at the next index rebuild. The research worker is off until the administrator
  turns it on (`LAB_RESEARCH_CONSUMER_ENABLED=true` in `.byeori.env`, then `byeori deploy-lab`);
  until then an accepted offer is recorded and waits.

Questions that contain an unpublished manuscript or lab data are marked private
(`private_material`) and are never sent to Jev.

## Registering a member

Each member signs in with their own IAM user in the lab's AWS account.

1. **Create the user** (skip if the member already has one in this account). In the console,
   IAM → Users → Create user, then Security credentials → Create access key; or:

   ```bash
   aws iam create-user --user-name <name>
   aws iam create-access-key --user-name <name>
   ```

   Give the key pair to the member privately. Never paste it into a chat or a file in a repository.
2. **Register the member and attach the student policy**, from your clone with `.byeori.env`
   sourced. The script finds the student stack by `LAB_STACK` (which `byeori deploy-lab` writes
   into `.byeori.env`), else by `<your KIRO_WIKI_STACK>-lab`:

   ```bash
   uv run python scripts/lab_members.py register --member-id <id> --iam-user <name> --role student --attach-policy
   ```

   The gateway refuses anyone who is not registered, who has been deactivated, or whose IAM user
   was deleted and re-created under the same name. Registration does not check the user's other
   policies: an existing user that can already write the data bucket or call the functions
   directly could go around the gateway, so check such users by hand first.
3. **Give the member the gateway address**: `uv run python scripts/lab_members.py outputs` prints
   `GatewayUrl`.

`uv run python scripts/lab_members.py list` shows every member; `deactivate --member-id <id>`
shuts one out without deleting anything, and `activate --member-id <id>` lets them back in. To
make someone a reviewer who may approve research, register them with `--role admin`.

## The student's own setup

The student needs `uv` and an AWS CLI profile holding their key (`aws configure --profile byeori`).
Nothing else is installed or configured; no bucket, table or function name is needed.

Claude Code:

```bash
claude mcp add -s user byeori-lab -e LAB_FUNCTION_URL=<gateway url> -e AWS_REGION=<region> -e AWS_PROFILE=<profile> -- uvx --from git+https://github.com/joonan-lab/byeori@v0.1.0-beta.1 byeori-lab-mcp
```

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.byeori-lab]
command = "uvx"
args = ["--from", "git+https://github.com/joonan-lab/byeori@v0.1.0-beta.1", "byeori-lab-mcp"]
startup_timeout_sec = 60

[mcp_servers.byeori-lab.env]
LAB_FUNCTION_URL = "<gateway url>"
AWS_REGION = "<region>"
AWS_PROFILE = "<profile>"
```

The gateway address is an HTTPS address that only accepts requests signed with a registered
member's AWS credentials; the MCP server signs them with the profile. Opened in a browser it
returns 403, and knowing it grants nothing. The server refuses to start if administrator
settings (`AWS_KIRO_WIKI_TABLE`, `AWS_KIRO_WIKI_INGEST_FUNCTION`) are in its environment, so a
student configuration cannot reach the main stack by mistake.

The tools the agent then has:

| Need | Tool |
|---|---|
| Find material | `search_wiki` (the wiki is in English; English queries match best) |
| Read an outline or one section | `read_wiki_page` |
| Pages that cite a page | `wiki_backlinks` |
| The paper's stored full text behind a note | `read_source` |
| Ask for a paper the wiki lacks | `request_paper` |
| Ask a question, then collect the answer | `ask_byeori`, then `get_byeori_answer` |
| Accept or decline a synthesis offer | `respond_to_synthesis_offer` |
| Accept or decline an offer to collect papers for an unanswered question | `respond_to_collection_offer` |

## What the professor's approval does

Every question and its triage score is kept, whether or not an offer was made or accepted. A
member registered with `--role admin` can review them and choose questions to research, at any
score, without waiting for a student to accept anything. The gateway actions are
`list_question_records` (questions in a date window, with filters), `propose_research_from_records`
(turn chosen questions into a research candidate), `list_research_candidates` and
`decide_research_candidate` (approve or reject exactly the candidate revision you reviewed).
They have no MCP tool yet; call them from a clone, in a terminal where `.byeori.env` is **not**
sourced, with your own member profile:

```bash
LAB_FUNCTION_URL=<gateway url> AWS_PROFILE=<profile> AWS_REGION=<region> uv run python -c '
from byeori.lab_mcp_server import client
print(client().call("list_question_records", {"from": "2026-10-01"}))'
```

An approval is bound to that candidate's revision, its research question, the pages it may
change or create, and its budget and model. It starts one research run, which writes only inside
that scope; widening the scope needs a new candidate and a new approval. Approving does not
relax the evidence rules: the run works under the same rules as any other research. A student's acceptance of an offer and a professor's approval are the only two ways
research starts.
