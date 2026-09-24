from __future__ import annotations

from pathlib import Path

import pytest

from byeori.catalog import Catalog
from byeori.cli import require_allowlisted_journal
from byeori.config import Settings
from byeori.corpus import annotate, apply_policy_to_catalog, search_corpus
from byeori import journal_policy
from byeori.journal_policy import (
    ALLOWLIST, NATURE_PORTFOLIO_BELOW_THRESHOLD, SAME_FAMILY_EXCLUDED, allowed_verdicts,
    journal_verdict, normalize_journal,
)


pytestmark = pytest.mark.usefixtures("cloud_catalog")

def settings_for(root: Path) -> Settings:
    return Settings(root, root / "data", root / "state", None, "us-east-1", "bucket", "table", "function")


def work(work_id: str, source: str | None, doi: str | None = None):
    return {"work_id": work_id, "doi": doi or f"10.1000/{work_id.lower()}", "title": "Autism de novo variants",
            "publication_year": 2021, "publication_date": "2021-01-01", "authors": ["A"], "type": "article",
            "topics": [], "source": source, "is_open_access": True, "oa_license": "cc-by",
            "openalex_pdf_url": "https://content.openalex.org/x.pdf",
            "grobid_xml_url": "https://content.openalex.org/x.xml", "cited_by_count": 1}


@pytest.mark.parametrize("name, verdict, family", [
    ("Nature", "include", "Nature"),
    ("Nature Communications", "include", "Nature portfolio"),
    ("Nature reviews. Neuroscience", "include", "Nature portfolio"),
    ("Nature reviews. Cancer", "include", "Nature portfolio"),   # added 2026-09-22 on the user's decision
    ("Molecular Psychiatry", "include", "Nature portfolio"),
    ("The American Journal of Human Genetics", "include", "Cell Press"),
    ("Cell", "include", "Cell Press"),
    ("Neuron", "include", "Cell Press"),
    ("Science", "include", "Science"),
    ("The Lancet", "include", "Lancet"),
    ("New England Journal of Medicine", "include", "NEJM"),
    ("JAMA Psychiatry", "include", "JAMA"),
    ("JAMA Network Open", "include", "JAMA"),
    ("Genome biology", "include", "Genome Medicine / Genome Biology"),
    ("Genome Medicine", "include", "Genome Medicine / Genome Biology"),
    # Admitted on 2026-09-22 when the list was unified with the Optimus scout's 65.
    ("The Lancet Psychiatry", "include", "Lancet"),
    ("The Lancet Neurology", "include", "Lancet"),
    ("Cell Genomics", "include", "Cell Press"),
    ("Immunity", "include", "Cell Press"),
    ("Science Translational Medicine", "include", "Science"),
    ("World Psychiatry", "include", "Psychiatry"),
    ("Nature Methods", "include", "Nature portfolio"),
    ("bioRxiv", "include", "Preprint server"),
    # A sibling nobody named is still not collectable.
    ("Cell Reports", "exclude", None),
    ("Science Advances", "exclude", None),
    ("Scientific Reports", "exclude", None),
    ("Translational Psychiatry", "exclude", "Nature portfolio"),
    ("npj Genomic Medicine", "exclude", "Nature portfolio"),
    ("Neuropsychopharmacology", "exclude", "Nature portfolio"),
    ("Frontiers in Psychiatry", "exclude", None),
    ("International Journal of Molecular Sciences", "exclude", None),
    ("Cell Death and Disease", "exclude", "Nature portfolio"),
    (None, "exclude", None),
])
def test_journal_verdicts(name, verdict, family):
    result = journal_verdict(name)
    assert (result["verdict"], result["family"]) == (verdict, family)


def test_titles_are_stored_normalized_and_disjoint():
    include = {t for titles in ALLOWLIST.values() for t in titles}
    review = {t for titles in SAME_FAMILY_EXCLUDED.values() for t in titles}
    below = set(NATURE_PORTFOLIO_BELOW_THRESHOLD)
    for title in include | review | below:
        assert normalize_journal(title) == title
    assert not include & review and not include & below and not review & below


def test_substring_matches_do_not_leak():
    # Exact normalized titles only: "Cell" must not admit every journal containing the word.
    assert journal_verdict("Cell Research")["verdict"] == "exclude"
    assert journal_verdict("Nature Communications Biology")["verdict"] == "exclude"
    assert journal_verdict("Science China Life Sciences")["verdict"] == "exclude"
    assert journal_verdict("Journal of Neuroscience")["verdict"] == "exclude"


def test_allowed_verdicts():
    assert allowed_verdicts("include") == ("include",)
    assert allowed_verdicts("review") == ("include", "review")
    assert allowed_verdicts("all") is None
    with pytest.raises(ValueError):
        allowed_verdicts("everything")


def test_search_defaults_to_allowlisted_journals(tmp_path):
    s = settings_for(tmp_path)
    with Catalog(s) as catalog:
        catalog.save_candidate(annotate(work("W1", "Nature Genetics"), "autism_title", []))
        catalog.save_candidate(annotate(work("W2", "Cell Reports"), "autism_title", []))
        catalog.save_candidate(annotate(work("W3", "Frontiers in Psychiatry"), "autism_title", []))
    assert [r["work_id"] for r in search_corpus(s)["results"]] == ["W1"]
    assert search_corpus(s, journal_policy="review")["total"] == 1  # no title carries review now
    assert search_corpus(s, journal_policy="all")["total"] == 3
    assert search_corpus(s, journal_policy="all")["results"][1]["journal_verdict"] == "exclude"
    assert search_corpus(s)["results"][0]["journal_family"] == "Nature portfolio"


def test_apply_policy_reclassifies_stored_records_without_deleting(tmp_path):
    s = settings_for(tmp_path)
    with Catalog(s) as catalog:
        stale = annotate(work("W1", "Nature Genetics"), "autism_title", [])
        stale["corpus"]["journal_verdict"] = "exclude"  # simulate a record saved before the policy existed
        catalog.save_candidate(stale)
        catalog.save_candidate(annotate(work("W2", "Science Advances"), "autism_title", []))
        catalog.save_candidate(annotate(work("W3", None), "autism_title", []))
    result = apply_policy_to_catalog(s)
    assert result["classified"] == 3
    assert result["verdicts"] == {"include": 1, "exclude": 2}
    assert result["included_by_family"] == {"Nature portfolio": 1}
    assert result["review_journals"] == {}
    assert result["aws_published"] == 3 and result["errors"] == []
    with Catalog(s) as catalog:
        assert len(catalog.list_candidates()) == 3
        assert catalog.get_candidate("W1")["record"]["corpus"]["journal_verdict"] == "include"


def test_ingest_gate_refuses_non_allowlisted_journal(tmp_path):
    s = settings_for(tmp_path)
    with Catalog(s) as catalog:
        allowed = catalog.save_candidate(work("W1", "Neuron"))
        refused = catalog.save_candidate(work("W2", "Translational Psychiatry"))
    require_allowlisted_journal(allowed)
    with pytest.raises(ValueError, match="Translational Psychiatry"):
        require_allowlisted_journal(refused)


# ---------------------------------------------------------------------------------------------
# The search policy: wider than collection, with a hard refusal and a warning (user, 2026-09-22)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("journal, publisher", [
    ("Scientific Reports", "Springer Nature"),
    ("iScience", "Elsevier BV"),
    ("Heliyon", "Elsevier BV"),
    ("Oncotarget", "Impact Journals, LLC"),
    ("International Journal of Molecular Sciences", "MDPI AG"),
    ("Genes", "MDPI AG"),
    ("Frontiers in Psychiatry", "Frontiers Media SA"),
    ("BioMed Research International", "Hindawi Limited"),
    ("Theranostics", "Ivyspring International Publisher"),
    ("Drug Design Development and Therapy", "Dove Medical Press"),
    ("Annals of Indian Academy of Neurology", "Wolters Kluwer Medknow Publications"),
])
def test_these_are_refused_however_relevant_the_topic_is(journal, publisher):
    result = journal_policy.search_verdict(journal, None, publisher)
    assert result["verdict"] == journal_policy.FORBIDDEN, (journal, result)
    assert result["warning"] is None and "refused by the user" in result["reason"]


def test_a_denied_house_is_refused_even_when_the_journal_name_is_unfamiliar():
    """Naming MDPI's journals one by one never finishes, so the refusal is on the house."""
    result = journal_policy.search_verdict("Some Journal Nobody Has Heard Of", None, "MDPI")
    assert result["verdict"] == journal_policy.FORBIDDEN


@pytest.mark.parametrize("journal", ["Nature Genetics", "The Lancet Neurology", "Cell Genomics", "bioRxiv"])
def test_the_collection_list_searches_without_a_warning(journal):
    result = journal_policy.search_verdict(journal)
    assert result["verdict"] == journal_policy.ALLOWED and result["warning"] is None


@pytest.mark.parametrize("journal", ["Cell Reports", "Brain", "eLife", "Nucleic Acids Research",
                                     "Proceedings of the National Academy of Sciences"])
def test_outside_the_list_may_be_searched_but_must_carry_the_warning(journal):
    result = journal_policy.search_verdict(journal, None, "Elsevier BV")
    assert result["verdict"] == journal_policy.OUTSIDE_LIST
    assert journal in result["warning"] and "not on the lab's journal list" in result["warning"]


def test_the_search_policy_is_wider_than_the_collection_policy():
    """The two answer different questions and must not be read as one."""
    assert journal_verdict("Brain")["verdict"] == "exclude"
    assert journal_policy.search_verdict("Brain")["verdict"] == journal_policy.OUTSIDE_LIST


def test_apply_search_policy_records_the_verdict_and_the_warning_on_the_work():
    outside = journal_policy.apply_search_policy(
        {"work_id": "W1", "source": "Brain", "source_publisher": "Oxford University Press"})
    assert outside["corpus"]["search_verdict"] == journal_policy.OUTSIDE_LIST
    assert "Brain" in outside["corpus"]["search_warning"]
    listed = journal_policy.apply_search_policy(
        {"work_id": "W2", "source": "Nature Genetics", "source_issn": ["1061-4036"]})
    assert listed["corpus"]["search_verdict"] == journal_policy.ALLOWED
    assert listed["corpus"]["search_warning"] is None


# ---------------------------------------------------------------------------------------------
# ISSN matching, which is why the 65 carry their identifiers
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("journal, issn, expected", [
    ("Science (New York, N.Y.)", ["0036-8075"], "include"),          # the title OpenAlex actually holds
    ("Archives of Pediatrics and Adolescent Medicine", ["2168-6203"], "include"),  # JAMA Pediatrics' old name
    ("Nature reviews. Genetics", None, "include"),                    # title still matches after normalizing
    ("", ["1546-1718"], "include"),                                   # no name at all, ISSN decides
    ("", ["9999-9999"], "exclude"),
])
def test_an_issn_matches_where_a_title_drifts(journal, issn, expected):
    assert journal_verdict(journal, issn)["verdict"] == expected


def test_every_listed_journal_carries_its_identifiers_and_none_is_shared():
    issns = [issn for _t, _f, issns, _s in journal_policy.JOURNALS.values() for issn in issns]
    assert len(journal_policy.JOURNALS) == 65
    assert all(issns and source_id.startswith("S") for _t, _f, issns, source_id
               in journal_policy.JOURNALS.values())
    issns += [issn for _t, _f, extra, _s in journal_policy.LAB_ADDITIONS.values() for issn in extra]
    assert len(issns) == len(set(issns)), "two titles would answer to the same ISSN"
    additions = [s for _t, _f, _i, ids in journal_policy.LAB_ADDITIONS.values() for s in ids]
    assert len(journal_policy.SOURCE_IDS) == len(set(journal_policy.SOURCE_IDS)) == 65 + len(additions)
    # OpenAlex refuses more than 100 values in one filter, and the ISSNs are over it.
    assert len(journal_policy.SOURCE_IDS) <= 100 < len(issns) + 1


def test_a_source_id_decides_before_an_issn_or_a_title():
    """OpenAlex's own id is the one identifier that never drifts."""
    assert journal_verdict("Nonsense Title", None, "https://openalex.org/S137905309")["verdict"] == "include"
    assert journal_verdict("Nonsense Title", None, "S137905309")["verdict"] == "include"
    assert journal_verdict("Nonsense Title", None, "S999999999")["verdict"] == "exclude"
    assert journal_policy.search_verdict("Nonsense Title", None, None, "S137905309")["verdict"] \
        == journal_policy.ALLOWED


def test_every_journal_the_optimus_scout_discovers_with_is_collectable_here():
    """The scout fails closed on a mismatch, so a drift in a shared title must be visible here."""
    scout = Path("~/Codex/automation-ops/runtime/journal-paper-scout/run_scout.py").expanduser()
    if not scout.exists():                      # the scout lives outside this repository
        pytest.skip("the Optimus workspace is not on this machine")
    import ast
    target = {}
    for node in ast.parse(scout.read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "TARGET_JOURNALS":
            target = ast.literal_eval(node.value)
    # The two lists may differ (user, 2026-09-23): the scout's are the journals the user scans daily,
    # Byeori's are what the lab wants to read. Every scout journal stays collectable here.
    assert {journal_policy.normalize_journal(t) for t in target} <= set(journal_policy.JOURNALS)
    for title, issns in target.items():
        key = journal_policy.normalize_journal(title)
        assert tuple(issns) == journal_policy.JOURNALS[key][2], title


@pytest.mark.parametrize("journal, publisher", [
    # The names OpenAlex actually writes, read from api.openalex.org on 2026-09-22. A live search
    # let MDPI's Nutrients through as merely outside the list because the entry said only "mdpi".
    ("Nutrients", "Multidisciplinary Digital Publishing Institute"),
    ("International Journal of Molecular Sciences", "Multidisciplinary Digital Publishing Institute"),
    ("Genes", "MDPI AG"),
    ("Frontiers in Microbiology", "Frontiers Media"),
    ("Frontiers in Neuroscience", "Frontiers Media SA"),
    ("BioMed Research International", "Hindawi Publishing Corporation"),
    ("Theranostics", "Ivyspring International Publisher"),
    ("Neuropsychiatric Disease and Treatment", "Dove Medical Press"),
    ("Neurology India", "Medknow"),
    ("Oncotarget", "Impact Journals LLC"),
])
def test_a_denied_house_is_matched_by_the_name_openalex_writes(journal, publisher):
    assert journal_policy.search_verdict(journal, None, publisher)["verdict"] == journal_policy.FORBIDDEN


@pytest.mark.parametrize("journal, publisher", [
    ("PLoS ONE", "Public Library of Science"),
    ("Microbiome", "BioMed Central"),
    ("eLife", "eLife Sciences Publications Ltd"),
    ("Brain", "Oxford University Press"),
])
def test_a_house_nobody_refused_stays_searchable_with_its_warning(journal, publisher):
    result = journal_policy.search_verdict(journal, None, publisher)
    assert result["verdict"] == journal_policy.OUTSIDE_LIST and result["warning"]


def test_a_listed_journal_is_never_refused_by_its_own_house():
    """Scientific Reports and Nature share a publisher; only the title is refused."""
    assert journal_policy.search_verdict("Nature Genetics", None, "Nature Portfolio")["verdict"] \
        == journal_policy.ALLOWED
    assert journal_policy.search_verdict("Scientific Reports", None, "Nature Portfolio")["verdict"] \
        == journal_policy.FORBIDDEN
    assert journal_policy.search_verdict("Cell Genomics", None, "Elsevier BV")["verdict"] \
        == journal_policy.ALLOWED
    assert journal_policy.search_verdict("iScience", None, "Elsevier BV")["verdict"] \
        == journal_policy.FORBIDDEN


@pytest.mark.parametrize("journal, issn, source_id, family", [
    ("arXiv (Cornell University)", None, None, "Preprint server"),
    ("Journal of Data Science", "1680-743X", None, "Byeori reading list"),
    ("Some drifted title", None, "https://openalex.org/S2764553900", "Byeori reading list"),
    ("International Conference on Learning Representations", None, "S4306419637", "Machine learning conference"),
    ("Neural Information Processing Systems", None, "S4306420609", "Machine learning conference"),
    ("neural information processing systems", "1049-5258", "S4363606243", "Machine learning conference"),
    ("International Conference on Machine Learning", None, "S4306419644", "Machine learning conference"),
    ("Proceedings of Machine Learning Research", None, None, "Machine learning conference"),
    ("Some drifted title", "2640-3498", None, "Machine learning conference"),
    ("International Conference on Artificial Intelligence and Statistics", None, "S4306419146", "Machine learning conference"),
    ("Conference on Learning Theory", None, "S4306418075", "Machine learning conference"),
])
def test_the_lab_s_own_additions_are_collectable(journal, issn, source_id, family):
    """arXiv, Journal of Data Science, ICLR, NeurIPS, ICML and PMLR, added by the user on 2026-09-23."""
    result = journal_verdict(journal, issn, source_id)
    assert (result["verdict"], result["family"]) == ("include", family)


def test_a_look_alike_conference_is_not_let_in_by_its_name():
    assert journal_verdict("International Conference on Machine Learning and Applications", None, "S4306419645")["verdict"] == "exclude"


def test_the_discovery_filter_carries_the_additions_and_stays_under_openalex_s_limit():
    assert {"S2764553900", "S4306419637", "S4306420609", "S4363606243", "S4306419644",
            "S4363608721", "S4306419146", "S4306418075"} <= set(journal_policy.SOURCE_IDS)
    assert len(journal_policy.SOURCE_IDS) <= 100


@pytest.mark.parametrize("journal, publisher, verdict", [
    ("Human Mutation", "Wiley", "include"),                       # outside the discovery list, chosen by the lab
    ("Journal of Data Science", None, "include"),
    ("Nutrients", "Multidisciplinary Digital Publishing Institute", "exclude"),
    ("Frontiers in Psychiatry", "Frontiers Media SA", "exclude"),
    ("Scientific Reports", "Springer Nature", "exclude"),
    (None, None, "include"),                                       # nothing to refuse it on
])
def test_a_paper_the_lab_uploaded_is_refused_only_for_a_denied_house_or_title(journal, publisher, verdict):
    """Uploads are what the lab chose to read, whatever the discovery list says (user, 2026-09-23)."""
    assert journal_policy.upload_verdict(journal, None, publisher)["verdict"] == verdict


def test_apply_upload_policy_records_why_on_the_work():
    work = journal_policy.apply_upload_policy({"source": "Human Mutation", "source_publisher": "Wiley"})
    assert work["corpus"]["journal_verdict"] == "include"
    assert "chosen by the lab" in work["corpus"]["journal_reason"]
    listed = journal_policy.apply_upload_policy({"source": "Nature Genetics"})
    assert listed["corpus"]["journal_reason"].startswith("on the collection list")
