"""Prevent client operations from recreating retired cloud-data mirrors."""
from pathlib import Path

import pytest

from byeori import pipeline, synthesis
from byeori.aws_store import AwsStore
from byeori.cli import build_parser, command_aws_build_index
from byeori.config import Settings


def settings_for(root):
    return Settings(root, root / 'data', root / 'state', None, 'us-east-1', 'bucket', 'table', 'function')


@pytest.mark.parametrize('step', ['synthesize', 'source_note'])
def test_successful_generation_retains_only_operational_ledger(tmp_path, step):
    class Store:
        def get_text(self, key):
            raise AssertionError('Generated pages must stay in S3')

        def synthesize_topic(self, *args):
            return {'status': 'model_topic', 'overview_key': 'wiki/overviews/topic.md'}

        def source_note(self, *args):
            return {'status': 'source_ready', 'source_note_key': 'wiki/sources/a-2020-paper.md'}

    settings = settings_for(tmp_path)
    args = ('topic', 'Topic', ['W1']) if step == 'synthesize' else ('W1',)
    getattr(pipeline, step + '_step')(settings, Store(), *args)
    assert not settings.data_dir.exists()
    assert [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()] == ['state/cost-ledger.jsonl']


def test_manifest_inspection_returns_aws_summary_without_files_or_full_download(monkeypatch, tmp_path):
    monkeypatch.setattr(synthesis.boto3, 'Session', lambda *a, **kw: pytest.fail('Manifest computation must run in AWS'))
    sent = []
    monkeypatch.setattr(synthesis, 'invoke_synthesis', lambda settings, payload:
                        sent.append(payload) or {'concepts': {'count': 1}, 'categories': {}})
    result = synthesis.pull_manifests(settings_for(tmp_path), scope='autism')
    assert sent == [{'action': 'manifest_summary', 'scope': 'autism'}]
    assert 'manifests' not in result
    assert result['concepts']['count'] == 1
    assert not list(tmp_path.iterdir())


def test_init_directories_and_index_aliases_cannot_create_mirror(tmp_path):
    settings_for(tmp_path).ensure_directories()
    assert [p.name for p in tmp_path.iterdir()] == ['state']
    parser = build_parser()
    for name in ('build-index', 'build-search-index', 'aws-build-index'):
        assert parser.parse_args([name]).handler is command_aws_build_index
    for args in (['sync-wiki'], ['wiki-search', 'gene', '--backend', 'local'],
                 ['wiki-backlinks', 'note', 'gene', '--backend', 'local']):
        with pytest.raises(SystemExit):
            parser.parse_args(args)


def test_s3_text_publication_refuses_to_overwrite_existing_content(tmp_path, cloud_catalog):
    store = AwsStore(settings_for(tmp_path))
    store.put_text('wiki/questions/q.md', 'Original', create_only=True)
    with pytest.raises(FileExistsError):
        store.put_text('wiki/questions/q.md', 'Replacement', create_only=True)
    assert cloud_catalog.objects['wiki/questions/q.md'] == b'Original'
    assert not list(tmp_path.iterdir())


def test_categories_include_all_document_types_without_disk_index(tmp_path, cloud_catalog):
    import sqlite3
    from byeori.search import categories
    database = sqlite3.connect(':memory:')
    database.execute('CREATE TABLE docs (category TEXT, doc_type TEXT)')
    database.executemany('INSERT INTO docs VALUES (?,?)', [
        ('asd-ndd','note'), ('asd-ndd','note'), ('asd-ndd','paper'),
        ('asd-ndd','overview'), ('questions','question'), ('concepts','concept')])
    cloud_catalog.objects['index/wiki-index.sqlite3'] = database.serialize()
    database.close()
    result = categories(settings_for(tmp_path))
    assert result['total'] == 6
    assert result['categories']['asd-ndd'] == {'note':2,'paper':1,'overview':1}
    assert result['categories']['questions'] == {'question':1}
    assert not list(tmp_path.iterdir())
