"""infra/asset_worker.py: naming an asset, finding its caption, and the text an answer reads.

The decisions here are the ones that were made once on the user's Mac for 11,554 papers and now
have to be made again for every paper that arrives. They are pure functions of a marker layout, so
they are tested without marker, S3 or a PDF.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load_worker():
    """Import the worker with the environment it reads at module level."""
    import os

    os.environ.setdefault("BUCKET_NAME", "bucket")
    os.environ.setdefault("JOB_KEY", "jobs/none.json")
    spec = importlib.util.spec_from_file_location("asset_worker", ROOT / "infra/asset_worker.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["asset_worker"] = module
    spec.loader.exec_module(module)
    return module


worker = load_worker()


# ---------------------------------------------------------------------------------------------
# What an asset is called
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("caption, key, label", [
    ("Figure 1. Odds ratio by ancestry.", "figure1", "Figure 1"),
    ("Fig. 3 Cortical volume.", "figure3", "Figure 3"),
    ("Table 2 Sample characteristics", "table2", "Table 2"),
    ("Extended Data Fig. 4 | Replication.", "extdatafig4", "Extended Data Fig. 4"),
    ("Extended Data Table 1 Cohorts", "extdatatable1", "Extended Data Table 1"),
    ("Supplementary Figure 7. Controls.", "suppfig7", "Supplementary Fig. 7"),
    ("Supplementary Table 3", "supptable3", "Supplementary Table 3"),
])
def test_a_caption_names_its_asset(caption, key, label):
    assert worker.asset_name(caption) == (key, label)


@pytest.mark.parametrize("caption", ["A figure with no number", "", "Discussion"])
def test_a_caption_with_no_number_names_nothing(caption):
    """A crop nobody can refer to is worse than no crop: it cannot be cited or found again."""
    assert worker.asset_name(caption) == (None, None)


# ---------------------------------------------------------------------------------------------
# Which caption belongs to which figure
# ---------------------------------------------------------------------------------------------

def block(kind, box, html=""):
    return kind, {"bbox": box, "html": html}


def test_the_nearest_caption_below_the_figure_wins():
    items = [block("fig", [100, 100, 500, 400]),
             block("cap", [100, 405, 500, 430], "Figure 1. The right one."),
             block("cap", [100, 700, 500, 730], "Figure 2. Further down the page.")]
    assert "The right one" in worker.nearest_caption(items, 0)


def test_a_caption_above_the_figure_is_penalised_but_still_usable():
    items = [block("fig", [100, 400, 500, 700]),
             block("cap", [100, 360, 500, 390], "Figure 1. Above the figure.")]
    assert "Above the figure" in worker.nearest_caption(items, 0)


def test_a_body_block_serves_when_the_paper_tags_no_caption():
    """Some publishers emit the caption as ordinary text; a crop without one cannot be named."""
    items = [block("fig", [100, 100, 500, 400]),
             block("txt", [100, 405, 500, 440], "Figure 1. Written as body text.")]
    assert "Written as body text" in worker.nearest_caption(items, 0)


def test_a_figure_with_nothing_nearby_gets_no_caption():
    assert worker.nearest_caption([block("fig", [0, 0, 10, 10])], 0) == ""


# ---------------------------------------------------------------------------------------------
# The sentences that mention the asset: the part an answer actually cites
# ---------------------------------------------------------------------------------------------

TEXT = (
    "The effect held in every ancestry group (Fig. 1a). "
    "Group sizes are given in Fig. 1b and in Table 2. "
    "Replication is shown in Extended Data Fig. 1. "
    "Supplementary Figure 1 lists the excluded samples. "
    "Short. "
    "We saw nothing in Figures 3 and 4."
)


def test_only_the_sentences_about_this_asset_come_back():
    mentions = worker.mentions_for(TEXT, "figure1")
    assert len(mentions) == 2 and all("Fig. 1" in m for m in mentions)
    assert not any("Extended Data" in m or "Supplementary" in m for m in mentions)


def test_an_extended_data_figure_is_not_the_plain_one():
    """Fig. 1 and Extended Data Fig. 1 are different figures with the same number."""
    assert worker.mentions_for(TEXT, "extdatafig1") == ["Replication is shown in Extended Data Fig. 1."]
    assert worker.mentions_for(TEXT, "suppfig1") == ["Supplementary Figure 1 lists the excluded samples."]


def test_a_table_is_not_a_figure_of_the_same_number():
    assert worker.mentions_for(TEXT, "table2") == ["Group sizes are given in Fig. 1b and in Table 2."]
    assert worker.mentions_for(TEXT, "figure2") == []


def test_a_list_names_each_figure_in_it():
    assert worker.mentions_for(TEXT, "figure3") == ["We saw nothing in Figures 3 and 4."]
    assert worker.mentions_for(TEXT, "figure4") == ["We saw nothing in Figures 3 and 4."]


def test_a_range_gives_its_ends_and_does_not_invent_the_middle():
    assert worker._numbers("1-3") == {"1", "3"}
    assert worker.mentions_for("This is shown across Figures 1-3 of the paper.", "figure2") == []


def test_a_fragment_too_short_to_be_a_sentence_is_not_a_mention():
    assert "Short." not in worker.mentions_for(TEXT, "figure1")


# ---------------------------------------------------------------------------------------------
# Panels and co-citations
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("caption, panels", [
    ("Figure 1. (a) One. (b) Two. (c) Three.", ["a", "b", "c"]),
    ("Figure 2 a, Design. b, Result.", ["a", "b"]),
    # No punctuation after the letter, either case. Common in Nature and Springer titles.
    ("Fig. 1 | Effects. A Top: traces. B Summary of the groups.", ["a", "b"]),
    ("Fig. 5 Design.  a Abeta burden, b Tau burden, c Thickness.", ["a", "b", "c"]),
    # A letter that merely follows a word is not a panel marker, so this caption yields nothing
    # rather than a list starting at "b". Its panel letters are on the image, which is stage two.
    ("Fig. 5 Correlations of a Abeta burden, b Tau burden, c Thickness.", []),
    ("Figure 3. No panels here.", []),
    ("Figure 4. (a) One. (c) Three.", []),      # stops at the letter the caption skips, so one
    ("Figure 6. A single letter a. proves nothing.", []),
])
def test_panels_are_read_from_the_caption(caption, panels):
    assert worker.panels_from_caption(caption) == panels


def test_the_label_is_not_mistaken_for_a_panel():
    """"Fig. 1 | a, Design" must give [a, b], not the "a" hidden in "Data"."""
    assert worker.panels_from_caption("Extended Data Fig. 1 | a, Design. b, Result.") == ["a", "b"]


# ---------------------------------------------------------------------------------------------
# A caption split across layout blocks
# ---------------------------------------------------------------------------------------------

def test_an_unfinished_caption_takes_in_the_block_that_continues_it():
    """A two-column paper breaks a long caption mid-word; the halves are one caption."""
    items = [block("fig", [100, 100, 500, 400]),
             block("cap", [100, 405, 500, 430], "Figure 1. Correlation of a Abeta and b Tau bur-"),
             block("txt", [100, 435, 500, 460], "den measured in the temporal neocortex (n = 52).")]

    caption = worker.nearest_caption(items, 0)

    assert caption == ("Figure 1. Correlation of a Abeta and b Tau bur- "
                       "den measured in the temporal neocortex (n = 52).")


def test_a_finished_caption_is_left_alone():
    items = [block("fig", [100, 100, 500, 400]),
             block("cap", [100, 405, 500, 430], "Figure 1. Odds ratio by ancestry."),
             block("txt", [100, 435, 500, 460], "We next asked whether the effect held in adults.")]

    assert worker.nearest_caption(items, 0) == "Figure 1. Odds ratio by ancestry."


@pytest.mark.parametrize("following", [
    "Figure 2. The next figure begins something else entirely.",
    "Methods We recruited participants from three sites over four years.",
])
def test_the_continuation_stops_at_whatever_begins_something_else(following):
    items = [block("fig", [100, 100, 500, 400]),
             block("cap", [100, 405, 500, 430], "Figure 1. Correlation of a Abeta and b Tau bur-"),
             block("txt", [100, 435, 500, 470], following)]

    assert worker.nearest_caption(items, 0).endswith("bur-")


def test_the_assets_cited_in_the_same_sentences_are_listed_without_the_asset_itself():
    related = worker.related_keys(["Group sizes are given in Fig. 1b and in Table 2."], "figure1")
    assert related == ["table2"]


# ---------------------------------------------------------------------------------------------
# The Markdown an answer reads
# ---------------------------------------------------------------------------------------------

def entry(**overrides):
    base = {"key": "figure1", "label": "Figure 1", "kind": "figure", "page": 4, "image": "figure1.png",
            "caption": "Figure 1. Odds ratio by ancestry.", "mentions": ["The effect held (Fig. 1a)."],
            "related": ["table2"], "panels": ["a", "b"], "bbox": [10.0, 20.0, 30.0, 40.0], "dpi": 350,
            "table_text": None}
    return {**base, **overrides}


def test_the_markdown_carries_what_an_answer_needs_and_where_it_came_from():
    text = worker.assets_markdown("paper-one", [entry()])
    for expected in ("# paper-one", "## Figure 1", "![Figure 1](figure1.png)", "### Caption",
                     "Odds ratio by ancestry", "### Panels", "a, b", "### Mentioned in the text",
                     "- The effect held (Fig. 1a).", "### Cited alongside", "table2",
                     "page 4", "bbox (pt) [10.0, 20.0, 30.0, 40.0]", "350 dpi"):
        assert expected in text, expected


def test_a_figure_with_no_caption_says_so_rather_than_inventing_one():
    text = worker.assets_markdown("paper-one", [entry(caption="", mentions=[], related=[], panels=[])])
    assert "(no caption found)" in text
    assert "### Panels" not in text and "### Mentioned in the text" not in text


def test_a_table_carries_its_extracted_text():
    text = worker.assets_markdown("paper-one", [entry(key="table1", label="Table 1", kind="table",
                                                      table_text="n = 312 | OR 2.4")])
    assert "### Table (extracted)" in text and "n = 312 | OR 2.4" in text


def test_the_header_counts_what_the_paper_has():
    text = worker.assets_markdown("paper-one", [entry(), entry(key="table1", label="Table 1", kind="table")])
    assert "2 items — Figures 1, Tables 1" in text
