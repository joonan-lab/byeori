from byeori.config import Settings
from byeori.validation import validate, SOURCE_SECTIONS, LLM_WIKI_SOURCE_SECTIONS
from conftest import MemoryAws
from byeori.wiki_ops import dispatch


class Store:
    def __init__(self,objects):
        self.cloud = MemoryAws()
        self.cloud.objects = {k: v.encode() for k, v in objects.items()}
    def validate_wiki(self, *, cursor=None):
        return dispatch({'action':'wiki_validate', 'cursor':cursor},
                        s3=self.cloud, table=None, bucket='bucket', index=None)


def settings(root): return Settings(root,root/"data",root/"state",None,"us-east-1","bucket","table")


def test_validate_reports_missing_sections_from_s3(tmp_path):
    errors=validate(settings(tmp_path),store=Store({"wiki/sources/a.md":"# Paper"}))
    assert "s3://bucket/wiki/sources/a.md: missing ## Methods" in errors
    assert not (tmp_path/"data").exists()


def test_validate_accepts_both_source_schemas(tmp_path):
    assert validate(settings(tmp_path),store=Store({"wiki/sources/a.md":"\n".join(SOURCE_SECTIONS),
        "wiki/sources/b.md":"\n".join(LLM_WIKI_SOURCE_SECTIONS)}))==[]


def test_validate_checks_synthesis_kind(tmp_path):
    errors=validate(settings(tmp_path),store=Store({
        "wiki/overviews/asd-ndd/index.md":'---\nkind: "category"\n---\n## Landscape\nx',
        "wiki/concepts/gene.md":"## Definition\nx"}))
    assert any("missing ## Subtopics" in e for e in errors)
    assert any("missing ## What the notes show" in e for e in errors)


def test_empty_remote_inventory_does_not_pass(tmp_path):
    assert validate(settings(tmp_path),store=Store({}))==["No published wiki Markdown found in the configured S3 wiki/ prefix"]
