"""``byeori-lab setup``: the rule-file review and the idempotent routing block."""
from __future__ import annotations

import io
from pathlib import Path

from byeori import lab_setup
from byeori.lab_setup import MARKER, ROUTING_BLOCK, apply, has_block, main, review, scan_rules, scan_skills, with_block

RULES_WITH_CONFLICTS = """# My rules

- Search papers with `bash ~/llm-wiki/scripts/search_llm_wiki.sh "question"` first.
- 논문 근거가 필요하면 web search를 쓴다.
- Use WebSearch for weather forecasts.
- Keep commits small.
"""


def targets(tmp_path: Path, *, claude: str | None, codex: str | None):
    claude_dir, codex_dir = tmp_path / ".claude", tmp_path / ".codex"
    for directory, text in ((claude_dir, claude), (codex_dir, codex)):
        if text is not None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "RULES.md").write_text(text, encoding="utf-8")
    return (("Claude Code", claude_dir / "RULES.md", claude_dir / "skills"),
            ("Codex", codex_dir / "RULES.md", codex_dir / "skills"))


def test_scan_reports_literature_routes_but_not_unrelated_web_search():
    findings = scan_rules(RULES_WITH_CONFLICTS, Path("RULES.md"))
    lines = [f.line_number for f in findings]
    assert lines == [3, 4]
    assert findings[0].reading == "bm25s 또는 QMD 검색 스크립트"      # the script is the more precise reading
    assert findings[1].reading == "논문 조회에 웹 검색 허용"


def test_scan_ignores_the_byeori_block_itself():
    text = with_block(RULES_WITH_CONFLICTS)
    findings = scan_rules(text, Path("RULES.md"))
    assert [f.line_number for f in findings] == [3, 4], "the block's own wording must not be reported"


def test_block_is_appended_once_and_recognised():
    once = with_block("# Rules\n")
    assert has_block(once) and once.count(MARKER) == 1
    assert once.startswith("# Rules\n\n" + MARKER)
    assert ROUTING_BLOCK.strip() in once
    assert with_block("") == ROUTING_BLOCK.lstrip("\n")


def test_review_reports_missing_file_skill_and_findings(tmp_path):
    chosen = targets(tmp_path, claude=RULES_WITH_CONFLICTS, codex=None)
    skills = tmp_path / ".claude" / "skills" / "kb-retriever"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: kb-retriever\ndescription: llm-wiki에서 논문 찾기\n---\nbm25s first\n", encoding="utf-8")
    (tmp_path / ".claude" / "skills" / "unrelated").mkdir()
    (tmp_path / ".claude" / "skills" / "unrelated" / "SKILL.md").write_text("---\ndescription: format tables\n---\n", encoding="utf-8")
    out = io.StringIO()
    report = review(chosen, out=out)
    assert len(report["Claude Code"]["findings"]) == 2
    assert [d.name for d, _, _ in report["Claude Code"]["skills"]] == ["kb-retriever"]
    assert report["Codex"]["exists"] is False and report["Codex"]["creatable"] is False
    text = out.getvalue()
    assert "kb-retriever" in text and "RULES.md:3" in text and "건너뜁니다" in text


def test_dry_run_writes_nothing(tmp_path):
    chosen = targets(tmp_path, claude=RULES_WITH_CONFLICTS, codex="# codex\n")
    before = {p: p.read_text() for _, p, _ in chosen}
    out = io.StringIO()
    assert main([], out=out, targets=chosen) == 0
    assert {p: p.read_text() for _, p, _ in chosen} == before
    assert "--apply" in out.getvalue()
    assert not list((tmp_path / ".claude").glob("*.bak-*"))


def test_apply_appends_with_backup_and_is_idempotent(tmp_path):
    chosen = targets(tmp_path, claude=RULES_WITH_CONFLICTS, codex="# codex\n")
    out = io.StringIO()
    assert main(["--apply"], out=out, targets=chosen) == 0
    for _, path, _ in chosen:
        text = path.read_text()
        assert has_block(text) and text.count(MARKER) == 1
        backups = list(path.parent.glob(path.name + ".bak-*"))
        assert len(backups) == 1
    assert RULES_WITH_CONFLICTS.strip() in chosen[0][1].read_text()

    again = io.StringIO()
    assert main(["--apply"], out=again, targets=chosen) == 0
    for _, path, _ in chosen:
        assert path.read_text().count(MARKER) == 1
        assert len(list(path.parent.glob(path.name + ".bak-*"))) == 1, "no second backup when nothing changes"
    assert "바꿀 파일이 없습니다" in again.getvalue()


def test_apply_creates_the_file_when_the_agent_directory_exists(tmp_path):
    (tmp_path / ".codex").mkdir()
    chosen = targets(tmp_path, claude="# claude\n", codex=None)
    report = review(chosen, out=io.StringIO())
    assert report["Codex"]["creatable"] is True
    written = apply(report, out=io.StringIO())
    codex_rules = tmp_path / ".codex" / "RULES.md"
    assert codex_rules in written and codex_rules.read_text() == ROUTING_BLOCK.lstrip("\n")


def test_setup_module_imports_only_the_standard_library():
    import ast
    source = Path(lab_setup.__file__).read_text(encoding="utf-8")
    modules = {node.names[0].name.split(".")[0] if isinstance(node, ast.Import) else (node.module or "").split(".")[0]
               for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom))}
    assert modules <= {"__future__", "argparse", "datetime", "os", "re", "sys", "collections", "pathlib"}


def test_scan_skills_tolerates_a_missing_directory(tmp_path):
    assert scan_skills(tmp_path / "nope") == []


REGISTRY_AND_WRITING_RULES = """# Rules
- Route people and contact lookup through `~/llm-wiki/agenda/registries/`.
- The meeting wrapper writes a note under llm-wiki `live/notes/`.
- Use WebSearch for weather forecasts.
| `~/llm-wiki` | Literature evidence, reusable scientific knowledge |
- Use `qmd_vector_serialized.sh --query` as the semantic fallback.
"""


def test_registry_pointers_into_llm_wiki_are_not_reported_but_literature_routes_are():
    findings = scan_rules(REGISTRY_AND_WRITING_RULES, Path("RULES.md"))
    assert [(f.line_number, f.reading) for f in findings] == [
        (5, "로컬 llm-wiki 경로"), (6, "bm25s 또는 QMD 검색 스크립트")]


def test_writing_and_review_skills_are_not_reported_as_retrieval_routes(tmp_path):
    skills = tmp_path / "skills"
    for name, description in (
        ("academic-paper", "12-agent academic paper writing pipeline with lit-review and citation-check modes"),
        ("nature-reader", "Build side-by-side readers for journal or conference papers from PDF, DOI, arXiv"),
        ("review-add", "논문을 리뷰 프로젝트에 추가합니다. DOI / 키워드 / PDF 경로를 지원합니다."),
        ("kb-retriever", "로컬 지식베이스 검색·QA 어시스턴트. 기본 지식베이스 = ~/llm-wiki/"),
        ("llm-wiki-retrieval", "Search and answer from llm-wiki with production BM25s as the default"),
        ("paper-finder", "관련 논문을 찾아 주는 도우미"),
        ("review-gap", "질문 트리와 논문 전체를 분석해 빈틈, 모순, 미답 영역을 찾고 L4 질문을 생성합니다."),
        ("gwas-database", "Query the GWAS Catalog. Search variants by rs ID, retrieve p-values and summary statistics"),
        ("academic-pipeline", "Orchestrator for the full academic pipeline from research to paper: write, review, revise"),
        ("literature-search", "Literature search for papers on a topic"),
    ):
        (skills / name).mkdir(parents=True)
        (skills / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\nbody\n", encoding="utf-8")
    assert [d.name for d, _, _ in scan_skills(skills)] == ["kb-retriever", "literature-search", "llm-wiki-retrieval", "paper-finder"]


def test_a_skill_whose_body_calls_the_llm_wiki_scripts_is_reported_with_that_reason(tmp_path):
    skill = tmp_path / "skills" / "cns-writer"
    skill.mkdir(parents=True)
    skill.joinpath("SKILL.md").write_text("---\ndescription: Revise manuscript sections\n---\nRun search_llm_wiki.sh first.\n", encoding="utf-8")
    hits = scan_skills(tmp_path / "skills")
    assert [(d.name, reason) for d, _, reason in hits] == [("cns-writer", "본문이 search_llm_wiki 호출")]


def test_the_reported_reason_quotes_the_description_fragment(tmp_path):
    skill = tmp_path / "skills" / "kb-retriever"
    skill.mkdir(parents=True)
    skill.joinpath("SKILL.md").write_text("---\ndescription: 로컬 지식베이스 검색 도우미\n---\n", encoding="utf-8")
    (_, _, reason), = scan_skills(tmp_path / "skills")
    assert reason == "description의 '지식베이스'가 문헌 검색을 트리거"
