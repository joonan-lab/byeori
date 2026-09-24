"""byeori.wiki_question_links: one standing line from an indexed page down to its questions."""
from __future__ import annotations

import pytest

from byeori import wiki_question_links as links
from byeori.lab_pages import page_link_line
from lab_fakes import MemoryS3

NOTE = '---\ntitle: "A note"\ncategory: "asd-ndd"\n---\n\n## Results\n\nx\n\n## Related pages\n\n- y\n'


class ListingS3(MemoryS3):
    """``MemoryS3`` plus the paginator ``iter_pages`` uses."""

    def get_paginator(self, _name):
        objects = self.objects

        class _Paginator:
            @staticmethod
            def paginate(*, Bucket, Prefix):  # noqa: N803 - boto3's own spelling
                yield {"Contents": [{"Key": key} for key in sorted(objects) if key.startswith(Prefix)]}

        return _Paginator()


def world(**pages) -> ListingS3:
    return ListingS3({key.replace("__", "/"): body for key, body in pages.items()})


def test_the_line_is_appended_under_its_own_heading_and_nothing_else_moves():
    out = links.add_line(NOTE, "sources/paper-one")

    assert out.startswith(NOTE.rstrip("\n"))
    assert out.endswith(f"\n\n{links.HEADING}\n\n{page_link_line('sources/paper-one')}\n")
    assert "## Results" in out and "## Related pages" in out


def test_a_page_that_already_carries_its_line_is_left_alone():
    once = links.add_line(NOTE, "sources/paper-one")

    assert links.needs_line(NOTE, "sources/paper-one") is True
    assert links.needs_line(once, "sources/paper-one") is False


def test_a_dry_run_reads_every_indexed_layer_and_writes_nothing():
    s3 = world(**{
        "wiki__sources__a.md": NOTE,
        "wiki__concepts__b.md": NOTE,
        "wiki__overviews__c.md": NOTE,
        "wiki__questions__d.md": NOTE,
        "wiki__lab-questions__2026-09__j1.md": NOTE,          # never edited
        "wiki__sources__failed__e.md": NOTE,                   # a failed note is not a page
        "wiki__sources__f.txt": "not markdown",
    })

    report = links.link_pages(s3, "bucket")

    assert report == {"dry_run": True, "scanned": 4, "already_linked": 0, "linked": 4, "conflicts": [],
                      "errors": [], "samples": report["samples"]}
    assert s3.writes == []
    # Samples follow the folder order the tool scans in, and stop at three.
    assert [sample["key"] for sample in report["samples"]] == ["wiki/sources/a.md", "wiki/concepts/b.md",
                                                               "wiki/overviews/c.md"]


def test_apply_writes_each_page_once_and_a_second_run_changes_nothing():
    s3 = world(**{"wiki__sources__a.md": NOTE, "wiki__sources__b.md": NOTE})

    first = links.link_pages(s3, "bucket", dry_run=False)
    second = links.link_pages(s3, "bucket", dry_run=False)

    assert (first["linked"], first["already_linked"]) == (2, 0)
    assert (second["linked"], second["already_linked"]) == (0, 2)
    assert [key for key, _ in s3.writes] == ["wiki/sources/a.md", "wiki/sources/b.md"]
    body = s3.objects["wiki/sources/a.md"].decode("utf-8")
    assert body.count(page_link_line("sources/a")) == 1 and "## Results" in body


def test_a_page_that_changed_since_it_was_read_is_reported_not_overwritten():
    s3 = world(**{"wiki__sources__a.md": NOTE})
    s3.conflict_keys.add("wiki/sources/a.md")

    report = links.link_pages(s3, "bucket", dry_run=False)

    assert report["linked"] == 0 and [c["key"] for c in report["conflicts"]] == ["wiki/sources/a.md"]
    assert s3.objects["wiki/sources/a.md"].decode("utf-8") == NOTE


def test_a_trial_run_stops_at_the_limit():
    s3 = world(**{f"wiki__sources__{name}.md": NOTE for name in "abcde"})

    report = links.link_pages(s3, "bucket", dry_run=False, limit=2)

    assert report["scanned"] == 2 and report["linked"] == 2 and len(s3.writes) == 2


@pytest.mark.parametrize("folder", ["wiki/lab-questions/", "papers/", "index/", "runs/lab-questions/"])
def test_the_tool_never_lists_a_folder_outside_the_indexed_layers(folder):
    assert not any(indexed.startswith(folder) for indexed in links.INDEXED_FOLDERS)
