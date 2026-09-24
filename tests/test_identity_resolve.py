"""An upload settles its own identity in AWS as soon as its text is stored (user, 2026-09-23)."""
import json
import re

import pytest

from byeori import identity_resolve

STEM = "samocha-2014-framework-interpretati"
TITLE = "A framework for the interpretation of de novo mutation in human disease"
TEI = f"""<TEI><teiHeader><fileDesc><titleStmt><title level="a" type="main">{TITLE}</title></titleStmt>
<sourceDesc><biblStruct><analytic><author><persName><forename>Kaitlin</forename><surname>Samocha</surname></persName></author>
<idno type="DOI">10.1038/ng.3050</idno></analytic>
<monogr><imprint><date type="published" when="2014-08">2014</date></imprint></monogr></biblStruct>
</sourceDesc></fileDesc></teiHeader></TEI>"""
WORK = {"id": "https://openalex.org/W2101", "doi": "https://doi.org/10.1038/ng.3050",
        "display_name": TITLE, "publication_year": 2014, "type": "article",
        "ids": {"openalex": "https://openalex.org/W2101", "pmid": "https://pubmed.ncbi.nlm.nih.gov/25086666"},
        "authorships": [{"author": {"display_name": "Kaitlin E. Samocha"}},
                        {"author": {"display_name": "Mark J. Daly"}}],
        "primary_location": {"source": {"display_name": "Nature Genetics", "type": "journal"}}}


class Table:
    def __init__(self, item):
        self.items = {item["work_id"]: dict(item)} if item else {}
        self.updates = []

    def get_item(self, Key):  # noqa: N803 - boto3's own spelling
        item = self.items.get(Key["work_id"])
        return {"Item": dict(item)} if item else {}

    def update_item(self, **call):
        stem = call["Key"]["work_id"]
        item = self.items.setdefault(stem, {"work_id": stem})
        self.updates.append(call)
        if "ConditionExpression" in call:
            from botocore.exceptions import ClientError
            if item.get("ingest_status") != "fulltext_ready_unclassified":
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
            item["ingest_status"] = "fulltext_ready"
            return
        names, values = call["ExpressionAttributeNames"], call["ExpressionAttributeValues"]
        # `if_not_exists(#f3, :v3)` carries a comma of its own, so the assignments are matched
        # rather than split on ", ".
        for name, expression in re.findall(r"(#\w+) = (if_not_exists\([^)]*\)|:\w+)",
                                           call["UpdateExpression"]):
            key = names[name]
            if expression.startswith("if_not_exists"):
                item.setdefault(key, values[expression.rstrip(")").split(", ")[1]])
            else:
                item[key] = values[expression]

    def query(self, **_call):
        return {"Items": []}


class S3:
    def __init__(self, objects):
        self.objects = dict(objects)

    def get_object(self, Bucket, Key, Range=None):  # noqa: N803
        from botocore.exceptions import ClientError
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body = self.objects[Key]
        return {"Body": type("B", (), {"read": staticmethod(lambda b=body: b)})()}

    def put_object(self, Bucket, Key, Body, **_rest):  # noqa: N803
        self.objects[Key] = Body


@pytest.fixture
def parked():
    table = Table({"work_id": STEM, "source": "to-s3", "ingest_status": "fulltext_ready_unclassified"})
    s3 = S3({f"papers/{STEM}/grobid.tei.xml": TEI.encode(),
             f"papers/{STEM}/clean.md": f"# {TITLE}\n".encode(),
             f"papers/{STEM}/meta.json": json.dumps({"stem": STEM, "source": "to-s3"}).encode()})
    return s3, table


def run(parked, *, works=None, results=None, apply=True):
    s3, table = parked
    works = {"10.1038/ng.3050": WORK} if works is None else works
    calls = []

    def fetch_work(doi):
        calls.append(("fetch", doi))
        return works.get(str(doi).rsplit("/", 2)[-2] + "/" + str(doi).rsplit("/", 1)[-1]
                         if str(doi).startswith("http") else str(doi))

    def search(query, year):
        calls.append(("search", query, year))
        return list(results or [])

    result = identity_resolve.resolve(STEM, s3=s3, table=table, bucket="bucket",
                                      fetch_work=fetch_work, search=search, apply=apply)
    return result, calls


def test_the_doi_the_pdf_prints_settles_the_paper_and_releases_it(parked):
    result, calls = run(parked)
    s3, table = parked
    assert (result["state"], result["how"]) == ("verified", "pdf_doi")
    assert result["journal_verdict"] == "include" and result["released_for_notes"] is True
    assert table.items[STEM]["ingest_status"] == "fulltext_ready"
    assert table.items[STEM]["identity"]["method"] == "aws_pdf_doi"
    assert table.items[STEM]["doi"] == "10.1038/ng.3050"
    meta = json.loads(s3.objects[f"papers/{STEM}/meta.json"])
    assert meta["title"] == TITLE and meta["authors"] == "Kaitlin E. Samocha, Mark J. Daly"
    assert meta["journal"] == "Nature Genetics" and meta["pmid"] == "25086666"
    assert not [c for c in calls if c[0] == "search"], "a DOI that agrees needs no search"


def test_a_dry_run_reports_the_verdict_and_writes_nothing(parked):
    result, _ = run(parked, apply=False)
    s3, table = parked
    assert result["state"] == "would_verify" and result["journal_verdict"] == "include"
    assert table.updates == [] and "title" not in json.loads(s3.objects[f"papers/{STEM}/meta.json"])


def test_a_doi_that_names_another_paper_sends_the_title_to_search(parked):
    other = {**WORK, "display_name": "Soil bacteria of the Atacama",
             "authorships": [{"author": {"display_name": "A. Villicana"}}]}
    found = {**WORK, "doi": "https://doi.org/10.1038/ng.9999"}
    result, calls = run(parked, works={"10.1038/ng.3050": other, "10.1038/ng.9999": found},
                        results=[found])
    assert result["state"] == "verified" and result["how"] == "search"
    assert [c for c in calls if c[0] == "search"], "the search has to run"


def test_a_paper_nothing_agrees_with_keeps_its_parked_status(parked):
    result, _ = run(parked, works={}, results=[])
    _s3, table = parked
    assert result["state"] == "not_found"
    assert table.items[STEM]["ingest_status"] == "fulltext_ready_unclassified"
    assert "identity_status" not in table.items[STEM]


def test_a_paper_the_user_turned_away_is_never_judged(parked):
    _s3, table = parked
    table.items[STEM]["ingest_status"] = "not_a_paper"
    result, calls = run(parked)
    assert result["state"] == "skipped" and calls == [] and table.updates == []


def test_a_paper_already_verified_is_left_alone(parked):
    _s3, table = parked
    table.items[STEM]["identity_status"] = "verified"
    result, calls = run(parked)
    assert result["state"] == "already_verified" and calls == [] and table.updates == []


def test_a_refused_journal_is_recorded_but_not_released(parked):
    refused = {**WORK, "primary_location": {"source": {"display_name": "Scientific Reports",
                                                       "type": "journal",
                                                       "host_organization_name": "Springer Nature"}}}
    result, _ = run(parked, works={"10.1038/ng.3050": refused})
    _s3, table = parked
    assert result["state"] == "verified" and result["journal_verdict"] == "exclude"
    assert "released_for_notes" not in result
    assert table.items[STEM]["ingest_status"] == "fulltext_ready_unclassified"


def test_a_journal_outside_the_discovery_list_is_still_the_lab_s_to_read(parked):
    """An upload is refused only for a refused house or title (user, 2026-09-23)."""
    sibling = {**WORK, "primary_location": {"source": {"display_name": "Cell Reports", "type": "journal"}}}
    result, _ = run(parked, works={"10.1038/ng.3050": sibling})
    assert result["journal_verdict"] == "include" and result["released_for_notes"] is True


def test_a_doi_another_row_already_holds_is_not_written_onto_this_one(parked, monkeypatch):
    monkeypatch.setattr(identity_resolve, "doi_holder", lambda table, doi, stem: "W9999")
    result, _ = run(parked)
    _s3, table = parked
    assert result["doi_conflict"] == "W9999" and "doi" not in table.items[STEM]


def test_nothing_the_extraction_owns_is_written(parked):
    run(parked)
    _s3, table = parked
    written = {name for call in table.updates for name in (call.get("ExpressionAttributeNames") or {}).values()}
    assert not written & set(identity_resolve.UNTOUCHABLE)


@pytest.mark.parametrize("key, stem", [
    ("papers/kim-2026-a-paper/clean.md", "kim-2026-a-paper"),
    ("papers/kim-2026-a-paper/original.pdf", None),
    ("wiki/sources/kim-2026-a-paper.md", None),
    ("papers/a/b/clean.md", None),
])
def test_only_a_stored_extraction_names_a_paper(key, stem):
    assert identity_resolve.stem_of(key) == stem


# ---------------------------------------------------------------- what starts it in AWS

from pathlib import Path  # noqa: E402 - read below, beside the assertions that use it

TEMPLATE = (Path(__file__).parents[1] / "infra" / "template.yaml").read_text()


def test_a_stored_extraction_starts_the_resolve_and_carries_its_key():
    """Without the transformer the ingest function reads the S3 event as an `ingest` call."""
    rule = TEMPLATE[TEMPLATE.index("  IdentityTriggerRule:"):TEMPLATE.index("  IdentityTriggerPermission:")]
    assert 'suffix: "/clean.md"' in rule and "Object Created" in rule
    assert '{"action": "resolve_identity", "key": <key>}' in rule
    assert 'key: "$.detail.object.key"' in rule
    assert "!GetAtt IngestFunction.Arn" in rule


def test_the_rule_may_invoke_the_ingest_function():
    permission = TEMPLATE[TEMPLATE.index("  IdentityTriggerPermission:"):]
    assert "events.amazonaws.com" in permission
    assert "!GetAtt IdentityTriggerRule.Arn" in permission


def test_the_lambda_answers_the_action_the_rule_sends():
    source = (Path(__file__).parents[1] / "src" / "byeori" / "ingest_lambda.py").read_text()
    assert 'if action == "resolve_identity":' in source
    assert "action must be search, get, ingest, resolve_identity" in source


def test_the_ingest_function_may_ask_the_doi_index_who_holds_a_doi():
    """A refused Query reads as "nobody holds it", which is how one paper gets two rows."""
    role = TEMPLATE[TEMPLATE.index("  IngestFunctionRole:"):TEMPLATE.index("  IngestFunction:")]
    assert "dynamodb:Query" in role
    assert '!Sub "${CatalogTable.Arn}/index/*"' in role


@pytest.fixture
def cloud(cloud_catalog):
    """The same parked upload in the shared in-memory AWS, which evaluates conditional writes."""
    cloud_catalog.items[STEM] = {"work_id": STEM, "source": "to-s3",
                                 "ingest_status": "fulltext_ready_unclassified",
                                 "pdf_key": f"papers/{STEM}/original.pdf"}
    cloud_catalog.objects.update({f"papers/{STEM}/grobid.tei.xml": TEI.encode(),
                                  f"papers/{STEM}/clean.md": f"# {TITLE}\n".encode(),
                                  f"papers/{STEM}/meta.json": json.dumps({"stem": STEM}).encode()})
    return cloud_catalog


def resolve_in(cloud):
    return identity_resolve.resolve(STEM, s3=cloud, table=cloud, bucket="bucket",
                                    fetch_work=lambda doi: WORK, search=lambda query, year: [])


def test_an_upload_of_a_paper_already_noted_is_marked_a_duplicate_and_not_released(cloud):
    """Releasing it would write a second note for one paper (Ahanger, 2026-09-23)."""
    cloud.items["samocha-2014-noted"] = {"work_id": "samocha-2014-noted", "doi": "10.1038/ng.3050",
                                         "source_note_key": "wiki/sources/samocha-2014-noted.md"}
    result = resolve_in(cloud)
    item = cloud.items[STEM]
    assert result["doi_holder_kind"] == "noted" and result["marked_duplicate"]
    assert item["ingest_status"] == "duplicate_of_noted_paper"
    assert item["duplicate_of"] == "samocha-2014-noted" and "doi" not in item
    assert "superseded_by" not in cloud.items["samocha-2014-noted"]


def test_the_bare_discovery_of_the_same_work_is_retired_to_the_upload(cloud):
    cloud.items["W2101"] = {"work_id": "W2101", "doi": "10.1038/ng.3050", "status": "candidate"}
    result = resolve_in(cloud)
    assert result["doi_holder_kind"] == "discovery" and result["released_for_notes"]
    assert cloud.items["W2101"]["doi"] == "openalex:W2101"
    assert cloud.items["W2101"]["superseded_by"] == STEM
    assert cloud.items[STEM]["doi"] == "10.1038/ng.3050"
    assert cloud.items[STEM]["ingest_status"] == "fulltext_ready"


@pytest.mark.parametrize("holder", [
    {"work_id": "W2101", "doi": "10.1038/ng.3050", "pdf_key": "papers/W2101/original.pdf"},
    {"work_id": "W7777", "doi": "10.1038/ng.3050"},
])
def test_any_other_holder_leaves_the_upload_parked_for_a_person(cloud, holder):
    cloud.items[holder["work_id"]] = dict(holder)
    result = resolve_in(cloud)
    assert result["doi_holder_kind"] == "other" and result["released_for_notes"] is False
    assert cloud.items[STEM]["ingest_status"] == "fulltext_ready_unclassified"
    assert "doi" not in cloud.items[STEM]
    assert "superseded_by" not in cloud.items[holder["work_id"]]
