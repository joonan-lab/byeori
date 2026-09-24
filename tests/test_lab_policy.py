"""Policy constants of the student question workflow (docs/LAB-QUESTION-WORKFLOW.md)."""
import math

from byeori import lab_policy
from byeori.lab_policy import PacketLimits, passes_cutoff


def test_policy_revision_is_pinned():
    assert lab_policy.POLICY_REVISION == "2026-09-21-v1"


def test_cutoff_passes_at_exactly_0_99():
    assert passes_cutoff({"review_candidate": 0.99}) is True


def test_cutoff_fails_just_below_0_99():
    assert passes_cutoff({"review_candidate": 0.9899}) is False


def test_cutoff_passes_above_0_99():
    assert passes_cutoff({"review_candidate": 0.995}) is True
    assert passes_cutoff({"review_candidate": 1}) is True


def test_cutoff_rejects_non_numeric_values():
    assert passes_cutoff({"review_candidate": True}) is False
    assert passes_cutoff({"review_candidate": None}) is False
    assert passes_cutoff({"review_candidate": "0.99"}) is False
    assert passes_cutoff({"review_candidate": math.nan}) is False
    assert passes_cutoff({"review_candidate": math.inf}) is False


def test_cutoff_rejects_missing_probabilities():
    assert passes_cutoff({}) is False
    assert passes_cutoff({"answer_only": 0.99}) is False
    assert passes_cutoff(None) is False
    assert passes_cutoff("0.99") is False


def test_cutoff_constant_matches_the_design():
    assert lab_policy.REVIEW_CANDIDATE_CUTOFF == 0.99
    assert lab_policy.ANSWER_MAX_MODEL_CALLS == 3   # two content calls plus one retry after an output cut


def test_packet_limits_match_the_design_numbers():
    limits = lab_policy.PACKET_LIMITS
    assert isinstance(limits, PacketLimits)
    assert limits.candidates == 8
    assert limits.max_documents == 8
    assert limits.max_sections == 24
    assert limits.section_chars == 6000     # results and limitations
    assert limits.context_chars == 2500     # methods, interpretation, everything else
    assert limits.total_bytes == 96000
    assert limits.searches == 2
    assert PacketLimits() == limits


def test_packet_limits_reject_non_positive_values():
    import pytest

    with pytest.raises(ValueError):
        PacketLimits(candidates=0)
    with pytest.raises(ValueError):
        PacketLimits(total_bytes=-1)


def test_read_and_budget_constants():
    assert lab_policy.READ_MAX_CHARS == 8000
    # No ceiling refuses a student's answer; the ledger still records every micro-USD.
    assert lab_policy.ANSWER_JOB_CAP_MICROS is None
    assert lab_policy.RESEARCH_PROFILE == {"budget_usd_micros": 5_000_000, "reread": "auto", "max_calls": 40}
    assert lab_policy.OFFER_TTL_SECONDS == 604_800
    assert lab_policy.LEASE_SECONDS == 960
    assert lab_policy.SCOPE_MATCH_SCORE == 20.0
    assert lab_policy.JEV_MAX_INPUT_BYTES == 30_000
    assert lab_policy.JEV_MODEL == "jev-1.13.0"


def test_offer_templates_carry_the_verbatim_korean_sentences():
    templates = lab_policy.OFFER_TEMPLATES
    assert templates["new_synthesis"] == (
        "현재 검색한 위키에서는 이 질문을 종합한 문서를 찾지 못했습니다. 관련 논문을 바탕으로 합성 위키를 만들까요?"
    )
    assert templates["supplement_existing"] == "관련 합성 위키가 있지만 이 부분을 보완할 수 있습니다. 보완할까요?"
    assert "비동기" in templates["consent_note"]


def test_offer_templates_are_read_only():
    import pytest

    with pytest.raises(TypeError):
        lab_policy.OFFER_TEMPLATES["new_synthesis"] = "changed"
