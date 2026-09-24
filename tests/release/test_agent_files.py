"""The release carries CLAUDE.md as the agent file and an install skill that runs the CLI, not the cloud."""
from pathlib import Path
import re

ROOT = Path(__file__).parents[2]


def test_claude_md_states_the_boundaries_and_names_the_skill():
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    for phrase in ("byeori-install", "run in AWS", "no local", "full text", "never delete"):
        assert phrase in text, phrase
    assert (ROOT / "AGENTS.md").read_text(encoding="utf-8").count("CLAUDE.md") >= 1


def test_install_skill_runs_the_release_commands_and_stops_before_spending():
    text = (ROOT / ".claude/skills/byeori-install/SKILL.md").read_text(encoding="utf-8")
    assert re.search(r"^name: byeori-install$", text, re.M)
    for command in ("byeori init", "byeori deploy", "byeori build-workers", "byeori doctor"):
        assert command in text, command
    assert "docs/AWS-SERVICES.md" in text
    assert "wait for the person's yes" in text
    assert "never create AWS resources by any other means" in text
