import io

import pytest

from byeori import synthesis, synthesis_support as support


def test_reference_intake_is_validated_and_runs_in_worker(monkeypatch):
    header = "symbol\tname\talias_symbol\tprev_symbol\tunused\n"
    raw = header + "".join(f"G{i}\tGene {i}\tA{i}\tP{i}\tx\n" for i in range(1000))
    monkeypatch.setattr(support.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(raw.encode()))
    writes = []
    class S3:
        def put_object(self, **kwargs):
            writes.append(kwargs)
            return {"VersionId": "version"}
    result = support.reference({}, s3=S3(), bucket="b")
    assert result["genes"] == 1000 and result["version_id"] == "version"
    assert b"unused" not in writes[0]["Body"]
    assert writes[0]["Metadata"]["sha256"] == result["sha256"]
    from byeori.synthesis_terms import HgncTable, line_key
    parsed = HgncTable.from_tsv(writes[0]["Body"].decode())
    assert parsed.names["G0"] == "Gene 0"
    assert line_key(["G0", "Gene 0"], parsed) == ("g0", "gene", "G0")
    assert line_key(["A0", "differentially methylated region"], parsed)[1] == "term"
    with pytest.raises(ValueError, match="configured HGNC"):
        support.reference({"url": "https://example.com/untrusted.tsv"}, s3=S3(), bucket="b")


def test_empty_or_short_hgnc_never_overwrites_reference(monkeypatch):
    class S3:
        def put_object(self, **kwargs):
            raise AssertionError("Invalid reference must not publish")
    for raw in (b"", b"symbol\talias_symbol\tprev_symbol\nG\tA\tP\n"):
        monkeypatch.setattr(support.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(raw))
        with pytest.raises(ValueError):
            support.reference({}, s3=S3(), bucket="b")


def test_failure_report_counts_all_pages_but_bounds_response():
    class Table:
        def scan(self, **kwargs):
            if "ExclusiveStartKey" not in kwargs:
                return {"Items": [{"work_id": "concept#b", "id_kind": "concept", "problems": ["bad"]}],
                        "LastEvaluatedKey": {"work_id": "concept#b"}}
            return {"Items": [{"work_id": "concept#a", "id_kind": "concept", "problems": ["bad"]}]}
    result = support.failures({"limit": 1, "verbose": True}, table=Table())
    assert result["failed"] == 2 and result["by_reason"] == {"bad": 2}
    assert result["pages"][0]["id"] == "concept#a" and result["next_offset"] == 1


def test_reference_and_failures_client_only_send_requests(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(synthesis, "invoke_synthesis", lambda settings, payload: calls.append(payload) or {})
    synthesis.upload_hgnc(None)
    synthesis.failed_synthesis(None, verbose=True, offset=10)
    assert [call["action"] for call in calls] == ["reference", "failures"]
    assert calls[1]["offset"] == 10
    assert not list(tmp_path.iterdir())
