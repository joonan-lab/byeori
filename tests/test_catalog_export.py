"""The CSV export lists stored PDFs without touching S3, the index, or the Lambda."""
from decimal import Decimal
from pathlib import Path

import pytest

from byeori.catalog_export import COLUMNS, export_papers_csv
from byeori.cli import build_parser, command_papers_export
from byeori.config import Settings


def settings_for(root):
    return Settings(root, root / 'data', root / 'state', None, 'ap-northeast-2', 'bucket', 'table', 'function')


def paper(work_id, **fields):
    return {'work_id': work_id, 'pdf_key': f'papers/{work_id}/original.pdf', **fields}


@pytest.fixture
def catalogue(cloud_catalog):
    cloud_catalog.items = {
        'ohta-1973-slightly-deleterious': paper(
            'ohta-1973-slightly-deleterious', id_kind='stem', title='Slightly deleterious substitutions',
            authors='Tomoko Ohta', year='1973', journal='Nature', doi='10.1038/246096a0', pmid='4585855',
            openalex_id='W2007325254', ingest_status='fulltext_ready', source_note_status='source_ready',
            pdf_bytes=Decimal('5425627'), pdf_sha256='26639cca', uploaded_at='2026-09-18T09:46:49+00:00'),
        'anand-2022-organoid-symmetry': paper(
            'anand-2022-organoid-symmetry', id_kind='stem', title='Controlling organoid symmetry breaking',
            authors=['Giridhar M. Anand', 'Heitor C. Megale'], year='2022', openalex_venue='Cell',
            doi='10.1016/j.cell.2022.12.043', ingest_status='fulltext_ready'),
        'W4317567753': paper('W4317567753', title='An OpenAlex-hosted ingest', doi='10.1000/openalex',
                             record={'authors': ['Ada Lovelace'], 'publication_year': Decimal('2023'),
                                     'source': 'Nature Communications', 'type': 'article'}),
        'metadata-only-candidate': {'work_id': 'metadata-only-candidate', 'title': 'No PDF was ever stored',
                                    'doi': '10.1000/none', 'record': {'corpus': {'id': 'autism-genomics'}}},
    }
    return cloud_catalog


def rows_of(path):
    import csv
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def test_only_papers_with_a_stored_pdf_are_listed(tmp_path, catalogue):
    receipt = export_papers_csv(settings_for(tmp_path), tmp_path / 'papers.csv')
    rows = rows_of(receipt['output'])
    assert [row['work_id'] for row in rows] == ['W4317567753', 'anand-2022-organoid-symmetry',
                                                'ohta-1973-slightly-deleterious']
    assert receipt['rows'] == 3 and receipt['basis'].startswith('catalogue rows with a pdf_key')
    assert list(rows[0]) == list(COLUMNS)


def test_cells_carry_identification_fields_and_blank_missing_ones(tmp_path, catalogue):
    rows = {row['work_id']: row for row in
            rows_of(export_papers_csv(settings_for(tmp_path), tmp_path / 'papers.csv')['output'])}
    ohta = rows['ohta-1973-slightly-deleterious']
    assert (ohta['doi'], ohta['pmid'], ohta['journal']) == ('10.1038/246096a0', '4585855', 'Nature')
    assert ohta['pdf_bytes'] == '5425627' and ohta['pmcid'] == ''
    assert ohta['pdf_key'] == 'papers/ohta-1973-slightly-deleterious/original.pdf'
    anand = rows['anand-2022-organoid-symmetry']
    assert anand['authors'] == 'Giridhar M. Anand; Heitor C. Megale'
    assert (anand['journal'], anand['openalex_venue']) == ('', 'Cell')


def test_openalex_path_rows_fall_back_to_the_stored_record(tmp_path, catalogue):
    rows = {row['work_id']: row for row in
            rows_of(export_papers_csv(settings_for(tmp_path), tmp_path / 'papers.csv')['output'])}
    ingested = rows['W4317567753']
    assert (ingested['authors'], ingested['year']) == ('Ada Lovelace', '2023')
    assert (ingested['journal'], ingested['document_type']) == ('Nature Communications', 'article')
    # A row that carries its own attributes keeps them.
    assert rows['ohta-1973-slightly-deleterious']['journal'] == 'Nature'


def test_export_reads_no_object_and_writes_only_the_requested_file(tmp_path, catalogue):
    catalogue.objects = {}
    export_papers_csv(settings_for(tmp_path), tmp_path / 'out' / 'papers.csv')
    assert catalogue.objects == {} and catalogue.invocations == []
    assert [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()] == ['out/papers.csv']


def test_existing_file_is_kept_unless_force_is_given(tmp_path, catalogue):
    destination = tmp_path / 'papers.csv'
    destination.write_text('keep me', encoding='utf-8')
    with pytest.raises(FileExistsError):
        export_papers_csv(settings_for(tmp_path), destination)
    assert destination.read_text(encoding='utf-8') == 'keep me'
    assert export_papers_csv(settings_for(tmp_path), destination, force=True)['rows'] == 3


def test_cli_requires_an_output_path_and_prints_the_receipt(tmp_path, catalogue, capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['papers-export'])
    args = parser.parse_args(['papers-export', '--output', str(tmp_path / 'papers.csv')])
    assert args.handler is command_papers_export
    assert command_papers_export(settings_for(tmp_path), args) == 0
    assert '"rows": 3' in capsys.readouterr().out
