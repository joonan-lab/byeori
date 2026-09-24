"""Exercise the AWS boundary with a Lambda-only client and a separate worker store."""
import hashlib
import io
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from byeori.aws_store import AwsStore
from byeori.config import Settings
from byeori.promote import promote_draft
from byeori.search import categories, read_page
from byeori.validation import validate
from byeori.wiki_ops import dispatch, MAX_RESPONSE_BYTES
from conftest import MemoryAws


class LambdaOnlySession:
    def __init__(self, cloud):
        self.cloud = cloud

    def client(self, name, **kwargs):
        assert name == 'lambda', 'Client must not read S3 or the index'
        return self.cloud


@pytest.fixture
def remote(tmp_path):
    cloud = MemoryAws()
    cloud.objects['wiki/sources/example.md'] = (
        '---\ntitle: "Example"\ndoi: "10.1/example"\nprivate_field: "Hidden"\n---\n'
        '# Example\n## Results\n' + '근거😀 CHD8 1,234 10% [[concepts/chd8]] ' * 1000 +
        '\n## Limits\nSmall cohort.\n').encode()
    cloud.objects['sources/W1.md'] = ('αβγδ' * 10000).encode()
    con = sqlite3.connect(':memory:')
    con.execute('CREATE TABLE docs (category TEXT, doc_type TEXT, doc_id TEXT, path TEXT)')
    con.executemany('INSERT INTO docs VALUES (?,?,?,?)', [
        ('asd-ndd','note','example','data/sources/example.md'),
        ('asd-ndd','paper','legacy','data/wiki/asd-ndd/legacy.md'),
        ('concepts','concept','chd8','data/wiki/concepts/chd8.md')])
    cloud.objects['index/wiki-index.sqlite3'] = con.serialize()
    con.close()
    settings = Settings(tmp_path,tmp_path/'data',tmp_path/'state',None,'us-east-1','bucket','table','function')
    store = AwsStore(settings, session=LambdaOnlySession(cloud))
    return settings, store, cloud


def test_outline_and_sections_only_cross_lambda_boundary(remote):
    settings, store, cloud = remote
    outline = read_page(settings, 'note', 'example', store=store)
    assert outline['text'] == '' and outline['mode'] == 'outline'
    assert outline['sections'] == ['Results', 'Limits']
    assert 'Hidden' not in json.dumps(outline) and 'Small cohort' not in json.dumps(outline)
    first = read_page(settings, 'note', 'example', section='Results', max_chars=5, store=store)
    second = read_page(settings, 'note', 'example', section='Results', start=first['next_start'], max_chars=5, store=store)
    assert first['text'] + second['text'] == '근거😀 CHD8 1'
    assert first['has_more'] and second['next_start'] == 10
    assert first['sha256'] == hashlib.sha256(cloud.objects['wiki/sources/example.md']).hexdigest()
    assert all(call['action'] == 'wiki_read' for call in cloud.invocations)
    assert not list(settings.root.iterdir())


def test_aggregate_metrics_and_validation_return_results_only(remote):
    settings, store, cloud = remote
    counts = categories(settings, store=store)
    assert counts['total'] == 3 and counts['execution'] == 'aws'
    assert counts['categories']['asd-ndd'] == {'note':1,'paper':1}
    metrics = store.wiki_metrics('wiki/sources/example.md', compare=True)
    assert metrics['metrics']['chars'] > 30000 and metrics['genes'] == ['CHD8']
    assert set(metrics['numbers']) == {'1234','10%'} and 'text' not in metrics
    for i in range(24):
        cloud.objects[f'wiki/questions/q{i:02}.md'] = b'# Incomplete'
    errors = validate(settings, store=store)
    assert len([c for c in cloud.invocations if c['action'] == 'wiki_validate']) == 3
    assert any('q23.md' in e for e in errors)
    assert all('근거' not in e for e in errors)
    assert not list(settings.root.iterdir())


def test_raw_source_is_sliced_by_worker(remote):
    settings, store, cloud = remote
    result = store.read_text('sources/W1.md', start=2, max_chars=3)
    assert result['text'] == 'γδα' and result['next_start'] == 5 and result['has_more']
    assert result['execution'] == 'aws'


def test_comparison_preserves_x_suffix_normalization(remote):
    _, store, cloud = remote
    cloud.objects['wiki/sources/example.md'] = b'## Results\n10x coverage, 20x depth, 1,234 samples.'
    result = store.wiki_metrics('wiki/sources/example.md', compare=True)
    assert result['numbers'] == ['10','1234','20']
    assert result['metrics']['numbers'] == 1


@pytest.mark.parametrize('key', ['index/wiki-index.sqlite3','papers/W1.pdf','runs/x.json',
                               'wiki/../secret.md','/wiki/sources/a.md'])
def test_source_read_cannot_expose_index_or_arbitrary_object(remote, key):
    _, store, _ = remote
    with pytest.raises(ValueError, match='Markdown key'):
        store.read_text(key)


@pytest.mark.parametrize('window', [{'max_chars':8001}, {'max_chars':0}, {'max_chars':True},
                                   {'max_chars':'20'}, {'start':-1}, {'start':1.1}])
def test_worker_rejects_unbounded_or_invalid_windows(remote, window):
    _, store, _ = remote
    with pytest.raises(ValueError, match='max_chars'):
        store.wiki_read('note','example',section='Results',**window)


@pytest.mark.parametrize('key', ['wiki/sources/example.md','sources/W1.md','index/wiki-index.sqlite3'])
def test_client_full_object_reader_is_disabled(remote, key):
    _, store, _ = remote
    with pytest.raises(ValueError, match='downloads are disabled'):
        store.get_text(key)


def test_missing_section_and_stale_index_fail_without_fallback(remote):
    _, store, cloud = remote
    with pytest.raises(ValueError, match='not found'):
        store.wiki_read('note','example',section='Missing')
    with pytest.raises(FileNotFoundError, match='index may be stale'):
        store.wiki_read('paper','legacy')
    cloud.objects['wiki/asd-ndd/legacy.md'] = b'## Summary\nLegacy text.'
    assert store.wiki_read('paper','legacy',section='Summary')['text'] == 'Legacy text.'


def test_nested_overview_and_response_budget(remote):
    _, store, cloud = remote
    cloud.objects['wiki/overviews/asd-ndd/index.md'] = ('## Landscape\n' + '😀' * 20000).encode()
    result = store.wiki_read('overview','asd-ndd/index',section='Landscape',max_chars=8000)
    assert len(result['text']) == 8000 and len(json.dumps(result).encode()) <= MAX_RESPONSE_BYTES


@pytest.mark.parametrize('suffix', ['', '\n## Results\nOther text.'])
def test_opening_text_can_be_read_without_downloading_body(remote, suffix):
    _, store, cloud = remote
    cloud.objects['wiki/sources/example.md'] = ('# Title\nOpening evidence.' + suffix).encode()
    outline = store.wiki_read('note','example')
    assert outline['text'] == '' and outline['sections'][0] == '(opening)'
    excerpt = store.wiki_read('note','example',section='(opening)',max_chars=7)
    assert excerpt['text'] == 'Opening' and excerpt['has_more']


def test_cli_preserves_outline_and_continuation(remote, monkeypatch, capsys):
    from byeori.cli import build_parser, command_wiki_read
    import byeori.search as search_module
    settings, store, _ = remote
    monkeypatch.setattr(search_module, 'AwsStore', lambda settings: store)
    args = build_parser().parse_args(['wiki-read','note','example'])
    assert command_wiki_read(settings,args) == 0
    assert json.loads(capsys.readouterr().out)['sections'] == ['Results','Limits']
    args = build_parser().parse_args(['wiki-read','note','example','--section','Results','--max-chars','3'])
    assert command_wiki_read(settings,args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['text'] == '근거😀' and result['next_start'] == 3 and result['has_more']


def test_promotion_body_checks_run_in_worker(remote):
    from test_promote import frontmatter, BODY, item_for
    settings, store, cloud = remote
    text = frontmatter() + BODY
    cloud.objects['wiki/drafts/W1.md'] = text.encode()
    cloud.items['W1'] = item_for(hashlib.sha256(text.encode()).hexdigest())
    result = promote_draft(settings, 'W1', reviewer='Reviewer', dry_run=True, store=store)
    assert result['execution'] == 'aws' and result['dry_run']
    assert 'wiki/sources/W1.md' not in cloud.objects
    assert BODY not in json.dumps(result)
    result = promote_draft(settings, 'W1', reviewer='Reviewer', store=store)
    assert result['execution'] == 'aws'
    assert cloud.items['W1']['review_status'] == 'reviewed'
    assert BODY in cloud.objects['wiki/sources/W1.md'].decode()
    assert not list(settings.root.iterdir())


def test_parallel_readers_keep_page_state_in_worker(remote):
    _, store, _ = remote
    with ThreadPoolExecutor(max_workers=25) as pool:
        results = list(pool.map(lambda start: store.wiki_read('note','example',section='Results',
                                                            start=start,max_chars=50), range(25)))
    assert [r['start'] for r in results] == list(range(25))
    assert all(len(r['text']) == 50 and r['execution'] == 'aws' for r in results)


def test_synthesis_coverage_shows_which_categories_have_no_synthesis():
    """llm-wiki reports this per category and holds it near 100%; Byeori had one whole-corpus
    number and no way to see where the gap is (user, 2026-09-23)."""
    from lab_fakes import build_index, index_connection
    from byeori.wiki_ops import dispatch

    def note(category, title):
        return f'---\ntitle: "{title}"\ncategory: "{category}"\n---\n\n## Results\nMeasured.\n'

    pages = {
        "wiki/sources/cited-one.md": note("liver", "Cited one"),
        "wiki/sources/cited-two.md": note("liver", "Cited two"),
        "wiki/sources/orphan-one.md": note("liver", "Orphan one"),
        "wiki/sources/orphan-two.md": note("gwas", "Orphan two"),
        "wiki/concepts/zonation.md": "# Zonation\n\nSee [[sources/cited-one]] and [[sources/cited-two]].\n",
        # A question citing a note is not synthesis: it must not count as coverage.
        "wiki/questions/asked.md": "# Asked\n\nSee [[sources/orphan-two]].\n",
    }
    raw = build_index(pages)
    result = dispatch({"action": "synthesis_coverage"}, s3=None, table=None, bucket="bucket",
                      index=lambda: (index_connection(raw), "etag"))
    by_category = {c["category"]: c for c in result["categories"]}
    assert by_category["liver"] == {"category": "liver", "notes": 3, "connected": 2, "orphans": 1,
                                    "coverage": 0.667}
    assert by_category["gwas"]["orphans"] == 1 and by_category["gwas"]["coverage"] == 0.0
    assert result["notes"] == 4 and result["connected"] == 2 and result["orphans"] == 2
    assert result["execution"] == "aws"


def test_a_category_listing_pages_and_carries_no_titles():
    """The whole of `other` was 1,336 rows; with titles the response passed the 128 KiB budget."""
    from lab_fakes import build_index, index_connection
    from byeori.wiki_ops import dispatch

    pages = {f"wiki/sources/n{i:03}.md": f'---\ntitle: "Paper {i}"\ncategory: "other"\n---\n\n## Results\nX.\n'
             for i in range(5)}
    raw = build_index(pages)
    index = lambda: (index_connection(raw), "etag")  # noqa: E731 - one line, used twice below
    first = dispatch({"action": "notes_in_category", "category": "other", "limit": 2},
                     s3=None, table=None, bucket="bucket", index=index)
    assert first["stems"] == ["n000", "n001"] and first["total"] == 5
    assert first["next_offset"] == 2 and "title" not in str(first["stems"])
    rest = dispatch({"action": "notes_in_category", "category": "other", "offset": 3},
                    s3=None, table=None, bucket="bucket", index=index)
    assert rest["stems"] == ["n003", "n004"] and rest["next_offset"] is None
