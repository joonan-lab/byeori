"""``byeori-lab setup``: review a member's central agent rules and add the Byeori routing block.

Installing the MCP server only registers six tools. Whether an agent reaches for Byeori when it
needs paper knowledge is decided by the member's own rule files, ``~/.claude/CLAUDE.md`` for
Claude Code and ``~/.codex/AGENTS.md`` for Codex. This command reads those files, reports every
line that routes literature lookups elsewhere (a local llm-wiki, bm25s or QMD scripts, web search,
external databases) and any skill that hard-codes such a route, then appends one marked routing
block that names Byeori as the default. Nothing is written without ``--apply``; a backup copy is
kept beside each edited file. The block is idempotent: a file that already carries the marker is
left alone. Copied verbatim into the ``byeori-lab`` package as ``byeori_lab/lab_setup.py``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

MARKER = "<!-- byeori-lab routing v1 -->"

ROUTING_BLOCK = f"""
{MARKER}
## 논문 지식은 벼리(Byeori) MCP에서 찾는다

- 논문 근거, 연구 지식, 선행 연구 확인이 필요하면 웹 검색이나 다른 문헌 도구보다 먼저 벼리 MCP `byeori-lab`을
  쓴다. `search_wiki`(완전한 영어 질문으로 검색)로 후보를 찾고 `read_wiki_page`로 본문을 확인하며, 연결된
  페이지는 `wiki_backlinks`로 본다.
- 여러 논문을 가로지르는 연구 질문은 `ask_byeori`로 제출하고 `get_byeori_answer`로 받는다. 답의 인용과 한계를
  그대로 전달하고, 답에 없는 근거를 덧붙이지 않는다.
- 벼리에서 못 찾으면 못 찾았다고 말한다. 웹이나 외부 데이터베이스는 사용자가 이 턴에서 분명히 요청했을 때만
  찾는다.
- 미발표 원고나 실험 데이터가 질문이나 맥락에 들어가면 `ask_byeori`를 `private_material=true`로 부른다.
- `synthesis_offer`가 오면 제안문을 그대로 보여 주고, 사용자가 분명히 동의했을 때만
  `respond_to_synthesis_offer`를 accept로 부른다. 에이전트가 스스로 동의하지 않는다.
- 노트의 요약이 아니라 논문 원문으로 확인해야 할 주장은 `read_source`로 저장된 원문 발췌를 읽는다. 위키에 없는 논문은
  `request_paper`로 추가를 요청하고, 노트가 생기기 전에는 근거로 쓰지 않는다.
- 이 규칙은 이 파일의 다른 문헌 조회 규칙(로컬 위키, 검색 스크립트, 웹 검색)보다 우선한다.
"""

# Lines that send literature lookups somewhere other than Byeori. Each pattern carries the reading
# the report gives for it; the member decides what to do with the line. A retrieval script is a
# conflict on its own; every other pattern also needs a literature word on the same line, so a
# registry, e-mail or meeting-note pointer into llm-wiki is not reported.
SCRIPT_READING = "bm25s 또는 QMD 검색 스크립트"
CONFLICT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (SCRIPT_READING, re.compile(r"bm25|qmd|search_llm_wiki", re.I)),
    ("로컬 llm-wiki 경로", re.compile(r"llm-wiki", re.I)),
    ("논문 조회에 웹 검색 허용", re.compile(r"(web\s*search|websearch|웹\s*검색|google\s+scholar|pubmed|openalex|semantic\s+scholar)", re.I)),
    ("다른 지식 베이스나 위키를 기본으로 지정", re.compile(r"(knowledge\s*base|지식\s*베이스|기본\s*위키|default\s+wiki|obsidian)", re.I)),
)
# "reference" is left out on purpose: the user's llm-wiki lives under a folder named References.
LITERATURE_WORDS = re.compile(r"(paper|논문|literature|문헌|scientific|과학|근거|evidence|citation|인용|retriev|scholar)", re.I)
# A skill counts when it retrieves from llm-wiki or a knowledge base, or searches for papers; a
# skill that merely writes, reviews or reads papers does not route lookups.
SKILL_RETRIEVAL = re.compile(r"(llm-wiki|bm25|qmd|search_llm_wiki|knowledge\s*base|지식\s*베이스|지식베이스)", re.I)
SKILL_PAPER_SEARCH = re.compile(r"((paper|논문|literature|문헌)s?\W{0,3}\w{0,6}\W{0,3}(\bsearch|검색|찾아|\blookup|조회)|(\bsearch\w*|검색|찾아|\blookup\w*|조회)\W{0,3}\w{0,6}\W{0,3}(paper|논문|literature|문헌))", re.I)
SKILL_BODY_MARKERS = ("search_llm_wiki", "bm25s", "qmd_vector")

DEFAULT_TARGETS = (("Claude Code", Path("~/.claude/CLAUDE.md"), Path("~/.claude/skills")),
                   ("Codex", Path("~/.codex/AGENTS.md"), Path("~/.codex/skills")))


class Finding:
    """One line of a rule file that routes literature lookups elsewhere."""

    def __init__(self, path: Path, line_number: int, text: str, reading: str):
        self.path, self.line_number, self.text, self.reading = path, line_number, text, reading

    def render(self) -> str:
        return f"  {self.path}:{self.line_number}  [{self.reading}]\n      {self.text.strip()[:160]}"


def scan_rules(text: str, path: Path) -> list[Finding]:
    """Every line of ``text`` that matches a conflict pattern, skipping the Byeori block itself."""
    findings: list[Finding] = []
    inside_block = False
    for number, line in enumerate(text.splitlines(), start=1):
        if MARKER in line:
            inside_block = True
            continue
        if inside_block:
            if line.startswith("## ") and "벼리" not in line:
                inside_block = False
            else:
                continue
        for reading, pattern in CONFLICT_PATTERNS:
            if not pattern.search(line):
                continue
            if reading != SCRIPT_READING and not LITERATURE_WORDS.search(line):
                continue
            findings.append(Finding(path, number, line, reading))
            break
    return findings


def scan_skills(directory: Path) -> list[tuple[Path, str, str]]:
    """Skills that route literature lookups: ``(skill_dir, description, reason)``.

    A skill counts when its description names llm-wiki, a knowledge base or a paper search, or when
    its body calls one of the llm-wiki retrieval scripts (``SKILL_BODY_MARKERS``).
    """
    hits: list[tuple[Path, str, str]] = []
    if not directory.is_dir():
        return hits
    for entry in sorted(directory.iterdir()):
        skill = entry / "SKILL.md"
        if not skill.is_file():
            continue
        try:
            head = skill.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            continue
        match = re.search(r"^description:\s*(.+)$", head, re.M)
        description = match.group(1).strip() if match else ""
        markers = [marker for marker in SKILL_BODY_MARKERS if marker in head]
        match = SKILL_RETRIEVAL.search(description) or SKILL_PAPER_SEARCH.search(description)
        if match:
            hits.append((entry, description[:160], f"description의 '{match.group(0)[:50]}'가 문헌 검색을 트리거"))
        elif markers:
            hits.append((entry, description[:160], "본문이 " + ", ".join(markers) + " 호출"))
    return hits


def has_block(text: str) -> bool:
    return MARKER in text


def with_block(text: str) -> str:
    """``text`` followed by the routing block, separated by one blank line."""
    body = text.rstrip("\n")
    return (body + "\n\n" if body else "") + ROUTING_BLOCK.lstrip("\n")


def backup_path(path: Path, now: _dt.datetime | None = None) -> Path:
    stamp = (now or _dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.name}.bak-{stamp}")


def review(targets: Iterable[tuple[str, Path, Path]], out=sys.stdout) -> dict[str, dict[str, object]]:
    """Print the diagnosis for each target and return what was found, keyed by agent name."""
    report: dict[str, dict[str, object]] = {}
    for agent, rules, skills in targets:
        rules = rules.expanduser()
        skills = skills.expanduser()
        entry: dict[str, object] = {"rules": rules, "exists": rules.is_file(), "has_block": False,
                                    "findings": [], "skills": []}
        print(f"\n== {agent}: {rules}", file=out)
        if not rules.is_file():
            if rules.parent.is_dir():
                print("  규칙 파일이 없습니다. --apply 때 벼리 블록만 담은 새 파일을 만듭니다.", file=out)
                entry["creatable"] = True
            else:
                print(f"  {rules.parent} 폴더가 없어 이 에이전트는 설치되지 않은 것으로 봅니다. 건너뜁니다.", file=out)
                entry["creatable"] = False
        else:
            text = rules.read_text(encoding="utf-8", errors="replace")
            entry["has_block"] = has_block(text)
            findings = scan_rules(text, rules)
            entry["findings"] = findings
            if entry["has_block"]:
                print("  벼리 라우팅 블록이 이미 있습니다. 다시 추가하지 않습니다.", file=out)
            if findings:
                print(f"  다른 곳으로 문헌 조회를 보내는 문장 {len(findings)}건:", file=out)
                for finding in findings:
                    print(finding.render(), file=out)
                print("  진단: 벼리 블록은 이 문장들보다 우선한다고 명시하지만, 같은 파일에 두 경로가 남으면 에이전트가"
                      " 오래된 경로를 먼저 고를 수 있습니다. 로컬 위키·검색 스크립트 문장은 지우거나 '원본 PDF 전용'으로"
                      " 좁히고, 웹 검색 문장은 '사용자가 명시 요청할 때만'으로 고치는 것을 권합니다.", file=out)
            else:
                print("  충돌하는 문헌 조회 문장이 없습니다.", file=out)
        skill_hits = scan_skills(skills)
        entry["skills"] = skill_hits
        if skill_hits:
            print(f"  문헌 조회를 자동 트리거하는 스킬 {len(skill_hits)}개 ({skills}):", file=out)
            for skill_dir, description, reason in skill_hits:
                print(f"  {skill_dir.name} [{reason}]: {description or '(description 없음)'}", file=out)
            print("  진단: 이런 스킬은 '논문 찾아줘' 같은 문구에서 규칙 파일보다 먼저 발동합니다. 벼리를 쓰게 하려면"
                  " 스킬을 지우거나 description의 트리거를 원본 PDF·registry 조회로 좁혀야 합니다.", file=out)
        report[agent] = entry
    return report


def apply(report: dict[str, dict[str, object]], out=sys.stdout, now: _dt.datetime | None = None) -> list[Path]:
    """Append the block to every reviewed file that lacks it; returns the files written."""
    written: list[Path] = []
    for agent, entry in report.items():
        rules = entry["rules"]
        assert isinstance(rules, Path)
        if entry.get("has_block"):
            continue
        if entry["exists"]:
            text = rules.read_text(encoding="utf-8", errors="replace")
            backup = backup_path(rules, now)
            backup.write_text(text, encoding="utf-8")
            rules.write_text(with_block(text), encoding="utf-8")
            print(f"{agent}: {rules}에 벼리 블록을 추가했습니다 (백업 {backup.name}).", file=out)
        elif entry.get("creatable"):
            rules.write_text(with_block(""), encoding="utf-8")
            print(f"{agent}: {rules}를 벼리 블록만으로 새로 만들었습니다.", file=out)
        else:
            continue
        written.append(rules)
    return written


def main(argv: Sequence[str] | None = None, out=sys.stdout, targets=DEFAULT_TARGETS) -> int:
    parser = argparse.ArgumentParser(prog="byeori-lab setup",
                                     description="Review the agent rule files and add the Byeori routing block.")
    parser.add_argument("--apply", action="store_true", help="append the block (default: report only)")
    parser.add_argument("--claude-rules", default=None, help="path of the Claude Code rules file")
    parser.add_argument("--codex-rules", default=None, help="path of the Codex rules file")
    args = parser.parse_args(list(argv) if argv is not None else None)
    chosen = []
    for agent, rules, skills in targets:
        override = args.claude_rules if agent == "Claude Code" else args.codex_rules if agent == "Codex" else None
        chosen.append((agent, Path(override) if override else rules, skills))
    print("byeori-lab setup: 중앙 규칙 검토", file=out)
    report = review(chosen, out=out)
    if not args.apply:
        print("\n추가할 블록:", file=out)
        print(ROUTING_BLOCK.rstrip("\n"), file=out)
        print("\n보고만 했습니다. 위 블록을 추가하려면 `byeori-lab setup --apply`를 실행하세요.", file=out)
        return 0
    written = apply(report, out=out)
    if not written:
        print("\n바꿀 파일이 없습니다.", file=out)
    else:
        print("\n완료. 에이전트를 다시 시작하면 새 규칙이 적용됩니다. 위 진단의 충돌 문장은 직접 정리해 주세요.", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
