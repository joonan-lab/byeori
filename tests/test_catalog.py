from __future__ import annotations

from pathlib import Path

import pytest

from byeori.catalog import Catalog
from byeori.config import Settings


pytestmark = pytest.mark.usefixtures("cloud_catalog")

def settings_for(root: Path) -> Settings:
    return Settings(
        root=root,
        data_dir=root / "data",
        state_dir=root / "state",
        openalex_api_key=None,
        aws_region="us-east-1",
        aws_bucket="bucket",
        aws_table="table",
    )


def sample_work() -> dict[str, object]:
    return {
        "work_id": "W123",
        "doi": "10.1000/test",
        "title": "A useful paper",
        "publication_year": 2026,
        "authors": ["Ada Lovelace"],
    }


def test_save_and_resolve_candidate_by_doi(tmp_path: Path) -> None:
    with Catalog(settings_for(tmp_path)) as catalog:
        saved = catalog.save_candidate(sample_work())
        resolved = catalog.get_candidate("https://doi.org/10.1000/TEST")
    assert saved["work_id"] == "W123"
    assert resolved["stem"].startswith("lovelace-2026")
    assert not (tmp_path / "data").exists()


def test_attach_pdf_uploads_and_hashes(tmp_path: Path, cloud_catalog) -> None:
    source = tmp_path / "outside.pdf"
    source.write_bytes(b"%PDF-1.7\noriginal")
    with Catalog(settings_for(tmp_path)) as catalog:
        catalog.save_candidate(sample_work())
        result = catalog.attach_pdf("W123", source)
    assert cloud_catalog.objects[result["path"].split("/",3)[3]] == source.read_bytes()
    assert not (tmp_path / "data").exists()
    assert result["status"] == "pdf_attached"


def test_attach_pdf_rejects_non_pdf(tmp_path: Path) -> None:
    source = tmp_path / "not.pdf"
    source.write_text("not a pdf")
    with Catalog(settings_for(tmp_path)) as catalog:
        catalog.save_candidate(sample_work())
        with pytest.raises(ValueError, match="not a PDF"):
            catalog.attach_pdf("W123", source)


def test_duplicate_doi_with_different_work_id_is_rejected(tmp_path: Path) -> None:
    duplicate = sample_work() | {"work_id": "W999"}
    with Catalog(settings_for(tmp_path)) as catalog:
        catalog.save_candidate(sample_work())
        with pytest.raises(ValueError, match="already stored"):
            catalog.save_candidate(duplicate)


def test_dynamodb_numbers_survive_metadata_refresh(tmp_path, cloud_catalog):
    from decimal import Decimal
    with Catalog(settings_for(tmp_path)) as catalog:
        catalog.save_candidate(sample_work())
        stored = cloud_catalog.items['W123']
        stored['publication_year'] = Decimal('2026')
        stored['record']['publication_year'] = Decimal('2026')
        stored['record']['score'] = Decimal('0.5')
        by_doi = catalog.get_candidate('10.1000/test')
        assert type(by_doi['record']['publication_year']) is int
        assert type(by_doi['record']['score']) is float
        for item in catalog.list_candidates():
            catalog.save_candidate(item['record'])
    assert not (tmp_path / 'data').exists()


def test_two_aws_clients_cannot_claim_one_doi_while_gsi_lags(tmp_path, cloud_catalog, monkeypatch):
    # A successful write is not yet visible through the asynchronous DOI index.
    monkeypatch.setattr(cloud_catalog, 'query', lambda **kwargs: {'Items': []})
    first = Catalog(settings_for(tmp_path))
    second = Catalog(settings_for(tmp_path))
    first.save_candidate(sample_work())
    with pytest.raises(ValueError, match='already stored as W123'):
        second.save_candidate(sample_work() | {'work_id':'W999'})
    first.save_candidate(sample_work())  # Same owner may retry after any partial failure.
    assert 'W999' not in cloud_catalog.items
