from pathlib import Path
from dataclasses import replace

import pytest

from byeori.config import Settings
from byeori.wiki_ops import dispatch
from conftest import MemoryAws
from byeori.search import search, read_page, backlinks, save_question


class Store:
    def __init__(self):
        self.objects = {"wiki/sources/example.md": "## Results\nCurrent AWS evidence.\n"}
        self.reads = []
        self.queries = []

    def wiki_read(self, kind, ident, **options):
        self.reads.append((kind, ident, options))
        cloud = MemoryAws()
        cloud.objects = {k: v.encode() for k,v in self.objects.items()}
        if hasattr(self, 'get_item'):
            cloud.items[ident] = self.get_item(ident)
        return dispatch({'action':'wiki_read', 'doc_type':kind, 'doc_id':ident, **options},
                        s3=cloud, table=cloud, bucket='bucket', index=cloud.index)

    def wiki_search(self, query, **options):
        self.queries.append((query, options))
        return {"results": [{"doc_type": "note", "doc_id": "example"}]}

    def wiki_backlinks(self, doc_type, doc_id):
        return {"cited_by": [{"doc_type": "concept", "doc_id": "example-concept"}]}

    def put_text(self, key, text, *, create_only=False):
        assert create_only
        if key in self.objects:
            raise FileExistsError(key)
        self.objects[key] = text
        return {"key": key, "bucket": "bucket"}

    def caller_name(self):
        return "reviewer"


def settings_for(root):
    return Settings(root, root/"data", root/"state", None, "ap-northeast-2", "bucket", "table", "function")


def test_read_uses_aws_even_with_stale_local_copy(tmp_path):
    settings = settings_for(tmp_path)
    stale = settings.data_dir / "sources/example.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("STALE LOCAL COPY")
    store = Store()
    result = read_page(settings, "note", "example", backend="aws", section="Results", store=store)
    assert "Current AWS evidence" in result["text"]
    assert result["path"] == "s3://bucket/wiki/sources/example.md"
    assert "cited_by" not in result
    assert stale.read_text() == "STALE LOCAL COPY"


def test_read_and_search_create_no_local_files(tmp_path):
    settings = settings_for(tmp_path); store = Store()
    page = read_page(settings, "note", "example", section="results", store=store)
    assert page["text"] == "Current AWS evidence."
    hits = search(settings, "example", doc_type="note", category="asd-ndd", store=store)
    assert hits["backend"] == "aws"
    assert store.queries == [("example", {"limit":10,"doc_type":"note","category":"asd-ndd"})]
    assert not settings.data_dir.exists()
    assert not settings.state_dir.exists()


@pytest.mark.parametrize("call", [
    lambda s: read_page(s,"note","example",store=Store()),
    lambda s: search(s,"example",store=Store()),
    lambda s: backlinks(s,"note","example",store=Store()),
])
def test_missing_aws_config_does_not_fall_back(tmp_path,call):
    with pytest.raises(RuntimeError,match="no local fallback"):
        call(replace(settings_for(tmp_path),aws_bucket=None))


def test_read_is_available_when_backlink_index_is_unavailable(tmp_path):
    store=Store()
    def broken(*args): raise RuntimeError("index not ready")
    store.wiki_backlinks=broken
    result=read_page(settings_for(tmp_path),"note","example",store=store)
    assert result["text"] == "" and result["sections"] == ["Results"]
    assert "cited_by" not in result


@pytest.mark.parametrize("kind,ident", [("bogus","example"),("note","../outside")])
def test_read_rejects_invalid_target(tmp_path,kind,ident):
    with pytest.raises(ValueError): read_page(settings_for(tmp_path),kind,ident,store=Store())


def test_local_backend_is_rejected(tmp_path):
    with pytest.raises(ValueError,match="removed"):
        search(settings_for(tmp_path),"example",backend="local",store=Store())


def test_question_publishes_only_to_s3_and_remote_conflicts_are_detected(tmp_path):
    settings=settings_for(tmp_path); store=Store()
    args=dict(title="Which evidence supports this?",question="The question.",sharper_followup="Which comparison?",
              holdings="The available note.",tentative_answer="A bounded answer.",related=["sources/example"],tags=[],store=store)
    saved=save_question(settings,**args)
    assert saved["path"].startswith("s3://bucket/wiki/questions/")
    assert 'author: "reviewer"' in store.objects[saved["s3"]["key"]]
    assert not settings.data_dir.exists()
    with pytest.raises(FileExistsError): save_question(settings,**args)


def test_legacy_category_paper_resolves_aws_recorded_key(tmp_path):
    store = Store()
    store.get_item = lambda ident: {'page_key': 'wiki/asd-ndd/example.md'}
    store.objects['wiki/asd-ndd/example.md'] = '## Summary\nLegacy category page.\n'
    result = read_page(settings_for(tmp_path), 'paper', 'example', section='Summary', store=store)
    assert result['path'] == 's3://bucket/wiki/asd-ndd/example.md'
    assert 'Legacy category page' in result['text']
    assert not settings_for(tmp_path).data_dir.exists()


def test_aws_search_paths_are_cloud_locations(tmp_path):
    store = Store()
    store.wiki_search = lambda *args, **kwargs: {'results':[
        {'path':'data/sources/example.md'}, {'path':'data/wiki/asd-ndd/example.md'}]}
    result = search(settings_for(tmp_path), 'example', store=store)
    assert [r['path'] for r in result['results']] == [
        's3://bucket/wiki/sources/example.md', 's3://bucket/wiki/asd-ndd/example.md']


def test_imported_category_paper_without_page_key_remains_readable(tmp_path):
    store = Store()
    store.get_item = lambda ident: {'category': 'asd-ndd', 'stem': ident}
    store.objects['wiki/asd-ndd/example.md'] = '## Summary\nImported page.\n'
    result = read_page(settings_for(tmp_path), 'paper', 'example', section='Summary', store=store)
    assert result['path'] == 's3://bucket/wiki/asd-ndd/example.md'
    assert not settings_for(tmp_path).data_dir.exists()
