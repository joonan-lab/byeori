"""byeori.supplementary: kept supplementary files beside the original, one section in the note."""
from __future__ import annotations

import hashlib
import json

import pytest
from botocore.exceptions import ClientError

from byeori import supplementary as supp
from byeori.wiki_question_links import HEADING as LAB_HEADING
from lab_fakes import MemoryS3

STEM = "doe-2024-a-paper"
DOI = "10.1038/s41586-024-00001-2"
NOTE = '---\ntitle: "A note"\n---\n\n## Results\n\nx\n'
NOTE_WITH_LAB = NOTE + f"\n{LAB_HEADING}\n\n- 이 페이지를 근거로 답한 랩 질문: [[lab-questions/by-page/sources/{STEM}]]\n"
SECTION = (f"{supp.NOTE_HEADING}\n\n- Supplementary Data 1 (`t1.xlsx`): DEGs per cell type, HGNC symbols, FDR\n"
           f"- File guide: {supp.guide_key(STEM)}\n")
GUIDE = f"# Supplementary files\n\nEvidence note: [[sources/{STEM}]]\n"


class MetaS3(MemoryS3):
    """``MemoryS3`` that keeps object metadata and answers the listing ``iter_manifest_keys`` uses."""

    def __init__(self, objects=None, **kwargs):
        self.metadata: dict[str, dict[str, str]] = {}
        super().__init__(objects, **kwargs)

    def put_object(self, *, Bucket, Key, Body, ContentType=None, Metadata=None, **conditions):
        result = super().put_object(Bucket=Bucket, Key=Key, Body=Body, ContentType=ContentType, **conditions)
        self.metadata[Key] = dict(Metadata or {})
        return result

    def head_object(self, *, Bucket, Key, **_):
        head = super().head_object(Bucket=Bucket, Key=Key)
        head["Metadata"] = self.metadata.get(Key, {})
        return head

    def get_paginator(self, _name):
        objects = self.objects

        class _Paginator:
            @staticmethod
            def paginate(*, Bucket, Prefix):  # noqa: N803 - boto3's own spelling
                yield {"Contents": [{"Key": key} for key in sorted(objects) if key.startswith(Prefix)]}

        return _Paginator()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def folder(tmp_path, files: dict[str, bytes]):
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return tmp_path


def triage(identity="confirmed", **files):
    entries = []
    for name, (data, decision) in files.items():
        entries.append({"file": name.replace("__", "/"), "sha256": sha(data), "bytes": len(data),
                        "kind": "data_table" if decision == "upload" else "reporting_summary",
                        "decision": decision, "reason": "r", "uses": ["gene_sets"] if decision == "upload" else []})
    return {"stem": STEM, "doi": DOI, "identity": identity, "files": entries}


def bucket_with_paper(**extra) -> MetaS3:
    objects = {f"papers/{STEM}/meta.json": json.dumps({"doi": DOI}), f"wiki/sources/{STEM}.md": NOTE}
    objects.update(extra)
    return MetaS3(objects)


def publish(s3, tmp_path, record, *, apply=True, lab_doi=None, section=SECTION):
    return supp.publish_paper(s3, "bucket", STEM, triage=record, guide=GUIDE, note_section=section,
                              files_root=tmp_path, lab_manifest_doi=lab_doi, apply=apply)


# --- keys and identity -------------------------------------------------------------------------

def test_kept_files_keep_their_publisher_name_under_the_supplementary_folder():
    assert supp.member_key(STEM, "mmc2.xlsx") == f"papers/{STEM}/supplementary/mmc2.xlsx"
    assert supp.member_key(STEM, "Supplementary/Table S1.xlsx") == f"papers/{STEM}/supplementary/Supplementary/Table S1.xlsx"


@pytest.mark.parametrize("bad", ["", "../meta.json", "/etc/passwd", "a/../../b", "a\\b", "README.md", "manifest.json"])
def test_a_path_that_could_leave_the_folder_or_replace_our_files_is_refused(bad):
    with pytest.raises(ValueError):
        supp.member_key(STEM, bad)


def test_identity_needs_a_match_in_the_files_or_the_same_doi_in_the_lab_manifest():
    assert supp.check_identity({"identity": "confirmed"}, DOI, None) == "confirmed"
    assert supp.check_identity({"identity": "probable"}, DOI, "https://doi.org/" + DOI.upper()) == "probable_doi_match"
    assert supp.check_identity({"identity": "probable"}, DOI, "10.1/other") is None
    assert supp.check_identity({"identity": "probable"}, "", "") is None
    assert supp.check_identity({"identity": "mismatch"}, DOI, DOI) is None


# --- publishing --------------------------------------------------------------------------------

def test_a_dry_run_checks_everything_and_writes_nothing(tmp_path):
    root = folder(tmp_path, {"t1.xlsx": b"table", "rs.pdf": b"form"})
    s3 = bucket_with_paper()

    report = publish(s3, root, triage(**{"t1.xlsx": (b"table", "upload"), "rs.pdf": (b"form", "skip")}),
                     apply=False)

    assert report["outcome"] == "would_publish" and report["files"] == 1 and report["bytes"] == 5
    assert s3.writes == []


def test_kept_files_guide_and_manifest_are_stored_and_skipped_files_only_recorded(tmp_path):
    root = folder(tmp_path, {"t1.xlsx": b"table", "sub/t2.csv": b"a,b", "rs.pdf": b"form"})
    s3 = bucket_with_paper()
    record = triage(**{"t1.xlsx": (b"table", "upload"), "sub__t2.csv": (b"a,b", "upload"), "rs.pdf": (b"form", "skip")})

    report = publish(s3, root, record)

    assert report["outcome"] == "published" and report["stored"] == 2
    base = supp.prefix(STEM)
    assert s3.objects[base + "t1.xlsx"] == b"table" and s3.objects[base + "sub/t2.csv"] == b"a,b"
    assert base + "rs.pdf" not in s3.objects
    assert s3.metadata[base + "t1.xlsx"] == {"sha256": sha(b"table")}
    assert all(conditions == {"IfNoneMatch": "*"} for key, conditions in s3.writes if key.endswith((".xlsx", ".csv")))
    manifest = json.loads(s3.objects[supp.manifest_key(STEM)])
    assert manifest["files_kept"] == 2 and manifest["files_not_kept"] == 1
    assert manifest["note_section"] == SECTION
    assert [f.get("key") for f in manifest["files"]] == [base + "t1.xlsx", base + "sub/t2.csv", None]
    assert s3.objects[supp.guide_key(STEM)].decode() == GUIDE
    # the original, its extraction and its metadata are untouched
    assert {key for key, _ in s3.writes} <= {k for k in s3.objects if k.startswith(base)}


def test_a_rerun_counts_stored_files_and_never_rewrites_them(tmp_path):
    root = folder(tmp_path, {"t1.xlsx": b"table"})
    s3 = bucket_with_paper()
    record = triage(**{"t1.xlsx": (b"table", "upload")})
    publish(s3, root, record)
    writes = len(s3.writes)

    again = publish(s3, root, record)

    assert again["outcome"] == "published" and again["stored"] == 0 and again["already"] == 1
    assert not any(key.endswith("t1.xlsx") for key, _ in s3.writes[writes:])


def test_different_bytes_already_at_the_key_are_a_conflict_and_nothing_is_described(tmp_path):
    root = folder(tmp_path, {"t1.xlsx": b"table"})
    s3 = bucket_with_paper(**{supp.prefix(STEM) + "t1.xlsx": b"someone else's"})

    report = publish(s3, root, triage(**{"t1.xlsx": (b"table", "upload")}))

    assert report["outcome"] == "conflict" and report["conflicts"] == [supp.prefix(STEM) + "t1.xlsx"]
    assert s3.objects[supp.prefix(STEM) + "t1.xlsx"] == b"someone else's"
    assert supp.manifest_key(STEM) not in s3.objects and supp.guide_key(STEM) not in s3.objects


def test_a_file_changed_since_it_was_read_stops_the_paper(tmp_path):
    root = folder(tmp_path, {"t1.xlsx": b"edited"})
    s3 = bucket_with_paper()

    report = publish(s3, root, triage(**{"t1.xlsx": (b"table", "upload")}))

    assert report == {"stem": STEM, "outcome": "changed_since_read", "file": "t1.xlsx"}
    assert s3.writes == []


@pytest.mark.parametrize("identity, lab_doi, outcome", [
    ("probable", "10.1/other", "identity_unconfirmed"),
    ("mismatch", DOI, "identity_unconfirmed"),
    ("probable", DOI, "published"),
])
def test_only_a_confirmed_identity_is_published(tmp_path, identity, lab_doi, outcome):
    root = folder(tmp_path, {"t1.xlsx": b"table"})
    s3 = bucket_with_paper()

    report = publish(s3, root, triage(identity, **{"t1.xlsx": (b"table", "upload")}), lab_doi=lab_doi)

    assert report["outcome"] == outcome


def test_a_paper_byeori_does_not_hold_or_with_nothing_kept_is_left_alone(tmp_path):
    root = folder(tmp_path, {"t1.xlsx": b"table", "rs.pdf": b"form"})

    assert publish(MetaS3(), root, triage(**{"t1.xlsx": (b"table", "upload")}))["outcome"] == "unmatched_paper"
    assert publish(bucket_with_paper(), root, triage(**{"rs.pdf": (b"form", "skip")}))["outcome"] == "nothing_to_keep"


def test_a_note_section_without_the_guide_or_too_long_is_refused(tmp_path):
    root = folder(tmp_path, {"t1.xlsx": b"table"})
    record = triage(**{"t1.xlsx": (b"table", "upload")})

    no_guide = publish(bucket_with_paper(), root, record, section=f"{supp.NOTE_HEADING}\n\n- a table\n")
    too_long = publish(bucket_with_paper(), root, record, section=SECTION + "- x" * 600)

    assert no_guide["outcome"] == too_long["outcome"] == "note_section_invalid"


# --- the note ----------------------------------------------------------------------------------

def test_the_section_is_appended_once_and_nothing_else_moves():
    out = supp.insert_section(NOTE, SECTION)

    assert out.startswith(NOTE.rstrip("\n")) and out.endswith(SECTION)
    assert supp.insert_section(out, SECTION) == out


def test_the_lab_question_section_stays_last():
    out = supp.insert_section(NOTE_WITH_LAB, SECTION)

    assert out.index(supp.NOTE_HEADING) < out.index(LAB_HEADING)
    assert out.rstrip("\n").endswith(f"[[lab-questions/by-page/sources/{STEM}]]")
    assert out.replace(SECTION.strip() + "\n\n", "") == NOTE_WITH_LAB


def test_linking_reads_only_s3_and_edits_each_note_once(tmp_path):
    root = folder(tmp_path, {"t1.xlsx": b"table"})
    s3 = bucket_with_paper()
    publish(s3, root, triage(**{"t1.xlsx": (b"table", "upload")}))

    dry = supp.link_notes(s3, "bucket")
    assert dry["linked"] == 1 and s3.objects[f"wiki/sources/{STEM}.md"] == NOTE.encode()

    done = supp.link_notes(s3, "bucket", dry_run=False)
    again = supp.link_notes(s3, "bucket", dry_run=False)

    assert done["linked"] == 1 and again["already_linked"] == 1 and again["linked"] == 0
    assert s3.objects[f"wiki/sources/{STEM}.md"].decode() == supp.insert_section(NOTE, SECTION)


def test_a_note_changed_between_read_and_write_is_reported_not_overwritten(tmp_path):
    root = folder(tmp_path, {"t1.xlsx": b"table"})
    s3 = bucket_with_paper()
    publish(s3, root, triage(**{"t1.xlsx": (b"table", "upload")}))
    note_key = f"wiki/sources/{STEM}.md"
    s3.conflict_keys.add(note_key)

    report = supp.link_notes(s3, "bucket", stems=[STEM], dry_run=False)

    assert report["conflicts"] == [{"stem": STEM, "error": "ConditionalRequestConflict"}]
    assert s3.objects[note_key] == NOTE.encode()


def test_a_published_paper_without_a_note_is_listed():
    s3 = MetaS3({supp.manifest_key(STEM): json.dumps({"note_section": SECTION})})

    report = supp.link_notes(s3, "bucket", dry_run=False)

    assert report["no_note"] == [STEM] and report["linked"] == 0


def test_client_errors_other_than_absence_are_not_hidden(tmp_path):
    class Denied(MetaS3):
        def head_object(self, **_):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")

    root = folder(tmp_path, {"t1.xlsx": b"table"})
    s3 = Denied({f"papers/{STEM}/meta.json": json.dumps({"doi": DOI})})

    with pytest.raises(ClientError):
        publish(s3, root, triage(**{"t1.xlsx": (b"table", "upload")}))
