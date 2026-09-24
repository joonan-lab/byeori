from __future__ import annotations

import io
import json
from pathlib import Path

from byeori.aws_store import AwsStore
from byeori.config import Settings


class FakeLambda:
    def invoke(self, **kwargs: object) -> dict[str, object]:
        payload = json.loads(kwargs["Payload"])
        assert kwargs["FunctionName"] == "workshop-ingest"
        if payload["action"] == "read_text":
            from byeori.wiki_ops import dispatch
            result = dispatch(payload, s3=FakeS3(), table=None, bucket="workshop-bucket", index=None)
            return {"Payload": io.BytesIO(json.dumps(result).encode())}
        if payload["action"] == "search":
            return {
                "Payload": io.BytesIO(
                    json.dumps(
                        {
                            "results": [
                                {
                                    "id": "https://openalex.org/W123",
                                    "display_name": "A useful paper",
                                }
                            ]
                        }
                    ).encode()
                )
            }
        if payload["action"] == "get":
            assert payload["work_id"] == "W123"
            return {
                "Payload": io.BytesIO(
                    json.dumps(
                        {
                            "work": {
                                "id": "https://openalex.org/W123",
                                "display_name": "A useful paper",
                            }
                        }
                    ).encode()
                )
            }
        if payload["action"] == "page":
            assert payload == {"action": "page", "work_id": "W123"}
            return {"Payload": io.BytesIO(json.dumps({"work_id": "W123", "status": "model_page"}).encode())}
        if payload["action"] == "synthesize":
            assert payload == {"action": "synthesize", "topic": "de-novo", "title": "De novo", "work_ids": ["W123"]}
            return {"Payload": io.BytesIO(json.dumps({"topic": "de-novo", "status": "model_topic"}).encode())}
        assert payload == {"action": "ingest", "work_id": "W123"}
        return {
            "Payload": io.BytesIO(
                json.dumps({"work_id": "W123", "status": "fulltext_ready"}).encode()
            )
        }


class FakeS3:
    def get_object(self, **kwargs: object) -> dict[str, object]:
        assert kwargs == {"Bucket": "workshop-bucket", "Key": "sources/W123.md"}
        return {"Body": io.BytesIO(b"0123456789")}


class FakeSession:
    def client(self, service: str, **_: object) -> object:
        if service == "lambda":
            return FakeLambda()
        if service == "s3":
            return FakeS3()
        raise AssertionError(service)


def settings_for(root: Path) -> Settings:
    return Settings(
        root=root,
        data_dir=root / "data",
        state_dir=root / "state",
        openalex_api_key="test-key",
        aws_region="us-east-1",
        aws_bucket="workshop-bucket",
        aws_table="workshop-table",
        aws_ingest_function="workshop-ingest",
    )


def test_ingest_work_invokes_configured_lambda(tmp_path: Path) -> None:
    result = AwsStore(settings_for(tmp_path), session=FakeSession()).ingest_work("W123")
    assert result["status"] == "fulltext_ready"


def test_read_text_supports_pagination(tmp_path: Path) -> None:
    result = AwsStore(settings_for(tmp_path), session=FakeSession()).read_text(
        "sources/W123.md",
        start=3,
        max_chars=4,
    )
    assert result["text"] == "3456"
    assert result["next_start"] == 7
    assert result["total_chars"] == 10


def test_search_openalex_uses_lambda_proxy(tmp_path: Path) -> None:
    result = AwsStore(settings_for(tmp_path), session=FakeSession()).search_openalex(
        "brain organoid",
        oa_only=True,
        fulltext_only=True,
    )
    assert result[0]["work_id"] == "W123"


def test_get_openalex_work_uses_lambda_proxy(tmp_path: Path) -> None:
    result = AwsStore(settings_for(tmp_path), session=FakeSession()).get_openalex_work("W123")
    assert result["title"] == "A useful paper"


def test_synthesize_invokes_its_lambda_action(tmp_path: Path) -> None:
    store = AwsStore(settings_for(tmp_path), session=FakeSession())  # type: ignore[arg-type]
    assert store.synthesize_topic("de-novo", "De novo", ["W123"])["status"] == "model_topic"
