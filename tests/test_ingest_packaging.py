"""Verify the deployed entry point works without client-only dependencies."""
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def test_zip_handler_serves_outline_and_metadata_audit_without_client_dependencies(tmp_path):
    source = Path(__file__).parents[1] / 'src'
    package = tmp_path / 'ingest.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        for path in list(source.rglob('*.py')) + list(source.rglob('*.json')):
            archive.write(path, path.relative_to(source))
    code = r'''
import sys, io, json, importlib.abc, zipfile
from pathlib import Path
import boto3
from unittest.mock import MagicMock
class BlockClientDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'httpx', 'mcp', 'fitz'} or fullname == 'byeori.aws_store':
            raise AssertionError('Worker imported a client dependency: ' + fullname)
sys.meta_path.insert(0, BlockClientDependencies())
cloud = MagicMock()
cloud.get_object.return_value = {'Body': io.BytesIO(b'## Results\nEvidence stays in AWS.')}
boto3.client = lambda *args, **kwargs: cloud
boto3.resource = lambda *args, **kwargs: cloud
extract_dir = Path(sys.argv[1]).parent / 'extracted'
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(extract_dir)
sys.path.insert(0, str(extract_dir))
from index import handler
result = handler({'action':'wiki_read','doc_type':'note','doc_id':'example'}, None)
assert result['text'] == '' and result['sections'] == ['Results']
assert result['execution'] == 'aws'
cloud.Table.return_value.scan.return_value = {'Items': [
    {'work_id':'example','id_kind':'stem','doi':'10.1234/example',
     'openalex_id':'W123','openalex_status':'matched'}]}
audit = handler({'action':'openalex_audit','sample_limit':1}, None)
assert audit['read_only'] and audit['scopes']['stem']['status']['matched'] == 1
assert audit['total_rows'] == 1
cloud.Table.return_value.update_item.assert_not_called()
# Every request is logged in AWS, reads included, so put_object is expected - but only for
# the log. A read must still not touch the wiki, the index, or a paper.
written = [call.kwargs.get('Key', '') for call in cloud.put_object.call_args_list]
assert written and all(k.startswith('runs/requests/') for k in written), written
cloud.get_parameter.assert_not_called()
from botocore.exceptions import ClientError
objects = {}
def get_object(**kwargs):
    if kwargs['Key'] not in objects:
        raise ClientError({'Error':{'Code':'NoSuchKey'}}, 'GetObject')
    return {'Body':io.BytesIO(objects[kwargs['Key']])}
def put_object(**kwargs):
    assert kwargs['Key'].startswith('runs/metadata-review/')
    assert kwargs['IfNoneMatch'] == '*'
    objects[kwargs['Key']] = kwargs['Body']
cloud.get_object.side_effect = get_object
cloud.put_object.side_effect = put_object
planned = handler({'action':'metadata_review_plan','run_id':'packaged-review'}, None)
assert planned['planned_papers'] == 1 and planned['canonical_writes'] is False
summary = handler({'action':'metadata_review_plan_summary','run_id':'packaged-review'}, None)
assert summary['planned_papers'] == 1
inspection = handler({'action':'metadata_review_plan_inspect','run_id':'packaged-review','limit':1}, None)
assert len(inspection['items']) == 1
progress = handler({'action':'metadata_review_progress','run_id':'packaged-review'}, None)
assert progress['status'] == 'not_started'
review = handler({'action':'metadata_review_results','run_id':'packaged-review'}, None)
assert review['rows'] == []
cloud.Table.return_value.update_item.assert_not_called()
print(json.dumps(result))
'''
    env = {**os.environ, 'BUCKET_NAME':'bucket','TABLE_NAME':'table',
           'OPENALEX_API_KEY_PARAMETER':'parameter','AWS_DEFAULT_REGION':'us-east-1'}
    result = subprocess.run([sys.executable, '-c', code, str(package)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'Evidence stays' not in result.stdout


def test_zip_handler_settles_an_upload_identity_with_only_the_lambda_runtime(tmp_path):
    """The clean.md rule failed 128 times on 2026-09-23 with `No module named 'httpx'`: identity
    reached httpx through corpus and aws_store, and the local tests had httpx installed. Here the
    client packages are missing the way they are in the Lambda, and the client store is refused."""
    source = Path(__file__).parents[1] / 'src'
    package = tmp_path / 'ingest.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        for path in list(source.rglob('*.py')) + list(source.rglob('*.json')):
            archive.write(path, path.relative_to(source))
    code = r'''
import sys, io, json, importlib.abc, zipfile
from pathlib import Path
import boto3
from unittest.mock import MagicMock
from botocore.exceptions import ClientError
class LambdaRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'httpx', 'mcp', 'fitz'}:
            raise ModuleNotFoundError('No module named ' + repr(fullname))
        if fullname in {'byeori.aws_store', 'byeori.corpus'}:
            raise AssertionError('Worker imported the client store: ' + fullname)
sys.meta_path.insert(0, LambdaRuntime())
stem = 'samocha-2014-framework-interpretati'
title = 'A framework for the interpretation of de novo mutation in human disease'
tei = ('<TEI><teiHeader><fileDesc><titleStmt><title level="a" type="main">' + title + '</title></titleStmt>'
       '<sourceDesc><biblStruct><analytic><author><persName><forename>Kaitlin</forename>'
       '<surname>Samocha</surname></persName></author><idno type="DOI">10.1038/ng.3050</idno></analytic>'
       '<monogr><imprint><date type="published" when="2014-08">2014</date></imprint></monogr>'
       '</biblStruct></sourceDesc></fileDesc></teiHeader></TEI>')
objects = {'papers/%s/grobid.tei.xml' % stem: tei.encode(), 'papers/%s/clean.md' % stem: ('# ' + title).encode(),
           'papers/%s/meta.json' % stem: json.dumps({'stem': stem}).encode()}
cloud = MagicMock()
def get_object(**kwargs):
    if kwargs['Key'] not in objects:
        raise ClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')
    return {'Body': io.BytesIO(objects[kwargs['Key']])}
cloud.get_object.side_effect = get_object
catalog = cloud.Table.return_value
catalog.get_item.return_value = {'Item': {'work_id': stem, 'source': 'to-s3',
                                          'ingest_status': 'fulltext_ready_unclassified'}}
catalog.query.return_value = {'Items': []}
boto3.client = lambda *args, **kwargs: cloud
boto3.resource = lambda *args, **kwargs: cloud
extract_dir = Path(sys.argv[1]).parent / 'extracted'
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(extract_dir)
sys.path.insert(0, str(extract_dir))
from index import handler
import byeori.ingest_lambda as ingest
work = {'id': 'https://openalex.org/W2101', 'doi': 'https://doi.org/10.1038/ng.3050', 'display_name': title,
        'publication_year': 2014, 'type': 'article', 'ids': {'openalex': 'https://openalex.org/W2101'},
        'authorships': [{'author': {'display_name': 'Kaitlin E. Samocha'}}],
        'primary_location': {'source': {'display_name': 'Nature Genetics', 'type': 'journal'}}}
ingest._fetch_json = lambda url, api_key: work
result = handler({'action': 'resolve_identity', 'stem': stem}, None)
assert result['state'] == 'verified' and result['released_for_notes'] is True, result
assert result['how'] == 'pdf_doi'
written = [call.kwargs['UpdateExpression'] for call in catalog.update_item.call_args_list]
assert any('ingest_status' in expression for expression in written), written
assert 'byeori.aws_store' not in sys.modules and 'byeori.corpus' not in sys.modules
print('ok')
'''
    env = {**os.environ, 'BUCKET_NAME': 'bucket', 'TABLE_NAME': 'table',
           'OPENALEX_API_KEY_PARAMETER': 'parameter', 'AWS_DEFAULT_REGION': 'us-east-1'}
    result = subprocess.run([sys.executable, '-c', code, str(package)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_zip_handler_files_other_notes_into_the_folders_the_wiki_uses(tmp_path):
    """A paper uploaded to the shared folder has no llm-wiki folder to inherit, so 1,336 notes sat
    in `other` on 2026-09-23 and a per-field catalog over them would be a catalog of nothing."""
    source = Path(__file__).parents[1] / 'src'
    package = tmp_path / 'ingest.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        for path in list(source.rglob('*.py')) + list(source.rglob('*.json')):
            archive.write(path, path.relative_to(source))
    code = r'''
import sys, io, json, sqlite3, zipfile
from pathlib import Path
import boto3
from unittest.mock import MagicMock
from botocore.exceptions import ClientError
note = lambda title: '---\ntitle: "' + title + '"\ncategory: "other"\n---\n\n## Results\nMeasured.\n'
objects = {'wiki/sources/a.md': note('Astroglial Kir4.1 drives neuronal hyperexcitability').encode(),
           'wiki/sources/b.md': note('Long-read metagenomics of the microbial tree of life').encode(),
           'papers/a/meta.json': json.dumps({'stem': 'a', 'category': 'other'}).encode()}
etags = {k: '"%d"' % i for i, k in enumerate(objects)}
cloud = MagicMock()
def get_object(**kw):
    if kw['Key'] not in objects:
        raise ClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')
    return {'Body': io.BytesIO(objects[kw['Key']]), 'ETag': etags[kw['Key']]}
def put_object(**kw):
    if kw.get('IfMatch') and kw['IfMatch'] != etags.get(kw['Key']):
        raise ClientError({'Error': {'Code': 'PreconditionFailed'}}, 'PutObject')
    body = kw['Body']
    objects[kw['Key']] = body if isinstance(body, bytes) else body.read()
    etags[kw['Key']] = '"written"'
cloud.get_object.side_effect = get_object
cloud.put_object.side_effect = put_object
boto3.client = lambda *a, **k: cloud
boto3.resource = lambda *a, **k: cloud
extract_dir = Path(sys.argv[1]).parent / 'extracted'
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(extract_dir)
sys.path.insert(0, str(extract_dir))
from index import handler
import byeori.ingest_lambda as ingest
con = sqlite3.connect(':memory:')
con.execute("CREATE TABLE docs (doc_type TEXT, doc_id TEXT, title TEXT, path TEXT, year TEXT, "
            "journal TEXT, doi TEXT, work_ids TEXT, category TEXT, key TEXT)")
for i in range(25):
    con.execute("INSERT INTO docs VALUES ('note', ?, '', '', '', '', '', '', 'glia', '')", ('g%d' % i,))
    con.execute("INSERT INTO docs VALUES ('note', ?, '', '', '', '', '', '', 'long-read', '')", ('l%d' % i,))
raw = con.serialize()
con.close()
def fresh_index():
    # production hands out a new connection per call and the reader closes it
    fresh = sqlite3.connect(':memory:')
    fresh.deserialize(raw)
    return fresh, 'etag'
ingest._wiki_index = fresh_index
ingest._converse_with_backoff = lambda client, request: (
    {'output': {'message': {'content': [{'text': '1: glia\n2: long-read\n'}]}},
     'usage': {'inputTokens': 700, 'outputTokens': 20}, 'stopReason': 'end_turn'}, 0, 1)
plan = handler({'action': 'classify_notes', 'stems': ['a', 'b']}, None)
assert [p['category'] for p in plan['papers']] == ['glia', 'long-read'], plan
assert 'title' not in plan['papers'][0], 'the title is what the model read, not a result'
assert plan['states'] == {'would_file': 2} and plan['categories'] == 2
assert b'category: "other"' in objects['wiki/sources/a.md']
done = handler({'action': 'classify_notes', 'stems': ['a', 'b'], 'apply': True}, None)
assert done['states'] == {'filed': 2}, done
assert b'category: "glia"' in objects['wiki/sources/a.md']
assert b'## Results' in objects['wiki/sources/a.md']
assert json.loads(objects['papers/a/meta.json'])['category'] == 'glia'
assert done['usage']['inputTokens'] == 700
print('ok')
'''
    env = {**os.environ, 'BUCKET_NAME': 'bucket', 'TABLE_NAME': 'table',
           'OPENALEX_API_KEY_PARAMETER': 'parameter', 'AWS_DEFAULT_REGION': 'us-east-1'}
    result = subprocess.run([sys.executable, '-c', code, str(package)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_zip_handler_opens_a_field_no_note_carries_yet(tmp_path):
    """`_known_categories` learns the folder list from the notes and keeps only folders of 20 or
    more, so a field nobody has filed into can never start. A person naming one breaks that circle
    (user, 2026-09-24): the folder reaches the prompt with its scope line, and a note may be filed
    into it."""
    source = Path(__file__).parents[1] / 'src'
    package = tmp_path / 'ingest.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        for path in list(source.rglob('*.py')) + list(source.rglob('*.json')):
            archive.write(path, path.relative_to(source))
    code = r'''
import sys, io, json, sqlite3, zipfile
from pathlib import Path
import boto3
from unittest.mock import MagicMock
from botocore.exceptions import ClientError
note = lambda title: '---\ntitle: "' + title + '"\ncategory: "single-cell-dl"\n---\n\n## Results\nMeasured.\n'
objects = {'wiki/sources/a.md': note('Slide-seq resolves cortical layers in situ').encode(),
           'papers/a/meta.json': json.dumps({'stem': 'a', 'category': 'single-cell-dl'}).encode()}
etags = {k: '"%d"' % i for i, k in enumerate(objects)}
cloud = MagicMock()
def get_object(**kw):
    if kw['Key'] not in objects:
        raise ClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')
    return {'Body': io.BytesIO(objects[kw['Key']]), 'ETag': etags[kw['Key']]}
def put_object(**kw):
    if kw.get('IfNoneMatch') and kw['Key'] in objects:
        raise ClientError({'Error': {'Code': 'PreconditionFailed'}}, 'PutObject')
    if kw.get('IfMatch') and kw['IfMatch'] != etags.get(kw['Key']):
        raise ClientError({'Error': {'Code': 'PreconditionFailed'}}, 'PutObject')
    body = kw['Body']
    objects[kw['Key']] = body if isinstance(body, bytes) else body.read()
    etags[kw['Key']] = '"written"'
cloud.get_object.side_effect = get_object
cloud.put_object.side_effect = put_object
boto3.client = lambda *a, **k: cloud
boto3.resource = lambda *a, **k: cloud
extract_dir = Path(sys.argv[1]).parent / 'extracted'
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(extract_dir)
sys.path.insert(0, str(extract_dir))
from index import handler
import byeori.ingest_lambda as ingest
con = sqlite3.connect(':memory:')
con.execute("CREATE TABLE docs (doc_type TEXT, doc_id TEXT, title TEXT, path TEXT, year TEXT, "
            "journal TEXT, doi TEXT, work_ids TEXT, category TEXT, key TEXT)")
for i in range(25):
    con.execute("INSERT INTO docs VALUES ('note', ?, '', '', '', '', '', '', 'single-cell-dl', '')", ('s%d' % i,))
raw = con.serialize()
con.close()
def fresh_index():
    fresh = sqlite3.connect(':memory:')
    fresh.deserialize(raw)
    return fresh, 'etag'
ingest._wiki_index = fresh_index
seen = []
def converse(client, request):
    seen.append(request['messages'][0]['content'][0]['text'])
    return ({'output': {'message': {'content': [{'text': '1: spatial-seq\n'}]}},
             'usage': {'inputTokens': 700, 'outputTokens': 20}, 'stopReason': 'end_turn'}, 0, 1)
ingest._converse_with_backoff = converse

# before the field is opened the same answer is refused, which is what made it impossible
blocked = handler({'action': 'classify_notes', 'stems': ['a']}, None)
assert blocked['states'] == {'undecided': 1}, blocked
assert b'category: "single-cell-dl"' in objects['wiki/sources/a.md']

opening = [{'name': 'spatial-seq', 'scope': 'spatial transcriptomics assays and their analysis tools'},
           {'name': 'spatial-seq-foundation-model', 'scope': 'models pretrained on spatial data'}]
done = handler({'action': 'classify_notes', 'stems': ['a'], 'apply': True, 'new_folders': opening}, None)
assert done['states'] == {'filed': 1}, done
assert done['papers'][0]['category'] == 'spatial-seq', done
assert b'category: "spatial-seq"' in objects['wiki/sources/a.md']
assert json.loads(objects['papers/a/meta.json'])['category'] == 'spatial-seq'
# both new folders reach the prompt, each with the scope line that separates them
assert '- spatial-seq: spatial transcriptomics assays' in seen[-1], seen[-1]
assert '- spatial-seq-foundation-model: models pretrained on spatial data' in seen[-1], seen[-1]
assert '- single-cell-dl\n' in seen[-1], seen[-1]

# applying remembers the field, so the next paper for it is filed there rather than nearby
stored = json.loads(objects['wiki/indexes/open-fields.json'])['fields']
assert sorted(stored) == ['spatial-seq', 'spatial-seq-foundation-model'], stored
assert stored['spatial-seq']['scope'].startswith('spatial transcriptomics assays'), stored
ingest._converse_with_backoff = converse
later = handler({'action': 'classify_notes', 'stems': ['a']}, None)
assert '- spatial-seq: spatial transcriptomics assays' in seen[-1], seen[-1]
assert later['states'] == {'would_file': 1}, later
assert later['open_fields'] == ['spatial-seq', 'spatial-seq-foundation-model'], later

# opening a field must not re-file the rest of the wiki
ingest._converse_with_backoff = lambda client, request: (
    {'output': {'message': {'content': [{'text': '1: glia\n'}]}},
     'usage': {'inputTokens': 700, 'outputTokens': 20}, 'stopReason': 'end_turn'}, 0, 1)
held = handler({'action': 'classify_notes', 'stems': ['a'], 'apply': True,
                'only_new_folders': True,
                'new_folders': [{'name': 'mpra', 'scope': 'reporter assays'}]}, None)
assert held['states'] == {'undecided': 1}, held
assert b'category: "spatial-seq"' in objects['wiki/sources/a.md']

# the field also reaches the catalogue, which is what the synthesis planner partitions
assert cloud.Table.return_value.update_item.called, 'the catalogue was never told the field'
call = cloud.Table.return_value.update_item.call_args
assert call.kwargs['Key'] == {'work_id': 'a'}, call.kwargs
assert call.kwargs['ExpressionAttributeValues'] == {':c': 'spatial-seq'}, call.kwargs

# `only_into` gathers a subtopic into a field the wiki already has, and moves nothing else
ingest._converse_with_backoff = lambda client, request: (
    {'output': {'message': {'content': [{'text': '1: single-cell-dl\n'}]}},
     'usage': {'inputTokens': 700, 'outputTokens': 20}, 'stopReason': 'end_turn'}, 0, 1)
gathered = handler({'action': 'classify_notes', 'stems': ['a'], 'only_into': ['single-cell-dl']}, None)
assert gathered['states'] == {'would_file': 1}, gathered
elsewhere = handler({'action': 'classify_notes', 'stems': ['a'], 'only_into': ['glia']}, None)
assert elsewhere['states'] == {'undecided': 1}, elsewhere

# a person may put named notes in a named field directly, which is what moves the few papers a
# field boundary cuts across; the field must exist, so a typo cannot scatter notes
moved = handler({'action': 'file_notes', 'stems': ['a'], 'category': 'single-cell-dl', 'apply': True}, None)
assert moved['states'] == {'filed': 1}, moved
assert b'category: "single-cell-dl"' in objects['wiki/sources/a.md']
opened = handler({'action': 'file_notes', 'stems': ['a'], 'category': 'spatial-seq', 'apply': True}, None)
assert opened['states'] == {'filed': 1}, opened
try:
    handler({'action': 'file_notes', 'stems': ['a'], 'category': 'spatal-seq'}, None)
except ValueError:
    pass
else:
    raise AssertionError('filed into a field this wiki does not have')

# a page's own kind is never a field, however many notes carry it as one
try:
    handler({'action': 'file_notes', 'stems': ['a'], 'category': 'note'}, None)
except ValueError:
    pass
else:
    raise AssertionError('filed into `note`, which is a page kind and not a field')

for bad in ('Spatial Seq', 'spatial_seq', ''):
    try:
        handler({'action': 'classify_notes', 'stems': ['a'], 'new_folders': [{'name': bad}]}, None)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted a folder name that is not a slug: ' + repr(bad))
print('ok')
'''
    env = {**os.environ, 'BUCKET_NAME': 'bucket', 'TABLE_NAME': 'table',
           'OPENALEX_API_KEY_PARAMETER': 'parameter', 'AWS_DEFAULT_REGION': 'us-east-1'}
    result = subprocess.run([sys.executable, '-c', code, str(package)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_zip_handler_writes_one_browse_catalog_per_field(tmp_path):
    """llm-wiki keeps one catalog per category and its agents browse it before searching; Byeori
    had one flat sources.md of 12,878 lines over every field at once (user, 2026-09-23)."""
    source = Path(__file__).parents[1] / 'src'
    package = tmp_path / 'ingest.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        for path in list(source.rglob('*.py')) + list(source.rglob('*.json')):
            archive.write(path, path.relative_to(source))
    code = r'''
import sys, io, json, sqlite3, zipfile
from pathlib import Path
import boto3
from unittest.mock import MagicMock
written = {}
cloud = MagicMock()
cloud.put_object.side_effect = lambda **kw: written.__setitem__(kw['Key'], kw['Body'])
boto3.client = lambda *a, **k: cloud
boto3.resource = lambda *a, **k: cloud
extract_dir = Path(sys.argv[1]).parent / 'extracted'
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(extract_dir)
sys.path.insert(0, str(extract_dir))
from index import handler
import byeori.ingest_lambda as ingest
con = sqlite3.connect(':memory:')
con.execute("CREATE TABLE docs (doc_type TEXT, doc_id TEXT, title TEXT, path TEXT, year TEXT, "
            "journal TEXT, doi TEXT, work_ids TEXT, category TEXT, s3_key TEXT, summary TEXT)")
con.execute("CREATE TABLE links (from_type TEXT, from_id TEXT, to_type TEXT, to_id TEXT)")
con.execute("INSERT INTO docs VALUES ('note','cited','Zonation of the liver','','2024','Nature','10.1/a','','liver','','Bile flows one way.')")
con.execute("INSERT INTO docs VALUES ('note','lonely','A liver paper nobody cites','','2023','Cell','10.1/b','','liver','','')")
con.execute("INSERT INTO docs VALUES ('note','g1','A glia paper','','2022','','','','glia','','')")
con.execute("INSERT INTO links VALUES ('overview','zon','note','cited')")
raw = con.serialize(); con.close()
def fresh():
    f = sqlite3.connect(':memory:'); f.deserialize(raw); return f, 'etag'
ingest._wiki_index = fresh
result = handler({'action': 'build_category_catalogs'}, None)
assert result['catalogs'] == 2 and result['notes'] == 3 and result['connected'] == 1, result
liver = written['wiki/indexes/categories/liver.md'].decode()
assert '[[sources/cited|Zonation of the liver]]' in liver
assert '2024 · Nature · 10.1/a' in liver and 'Bile flows one way.' in liver
assert '*(orphan)*' in liver and '1/2 connected (50%), 1 orphans' in liver
root = written['wiki/indexes/categories.md'].decode()
assert '| liver | 2 | 50% (1 orphan) | [[indexes/categories/liver]] |' in root
assert '| glia | 1 | 0% (1 orphan) |' in root
# A catalog is a browse path, never evidence: wiki/indexes/ is skipped by the index builder.
assert all(k.startswith('wiki/indexes/') for k in written), written
print('ok')
'''
    env = {**os.environ, 'BUCKET_NAME': 'bucket', 'TABLE_NAME': 'table',
           'OPENALEX_API_KEY_PARAMETER': 'parameter', 'AWS_DEFAULT_REGION': 'us-east-1'}
    result = subprocess.run([sys.executable, '-c', code, str(package)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_zip_handler_reads_an_extraction_and_publishes_a_note_written_outside_aws(tmp_path):
    """The user asks for a note to be written locally for a paper that has just arrived
    (2026-09-24). The body comes from outside; everything that makes it a wiki object is built
    here, and the frontmatter says what actually wrote it."""
    source = Path(__file__).parents[1] / 'src'
    package = tmp_path / 'ingest.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        for path in list(source.rglob('*.py')) + list(source.rglob('*.json')):
            archive.write(path, path.relative_to(source))
    code = r'''
import sys, io, json, zipfile
from pathlib import Path
import boto3
from unittest.mock import MagicMock
from botocore.exceptions import ClientError
stem = 'beer-2026-splicing-in-five-prime-utrs'
objects = {'papers/%s/clean.md' % stem: b'# Paper\n\n## Abstract\n\nSkipping a uORF exon raises protein.\n',
           'papers/%s/meta.json' % stem: json.dumps({'stem': stem, 'title': 'Modulating splicing',
                                                     'authors': 'Beer Wells', 'year': '2026',
                                                     'doi': '10.1/gm', 'category': 'asd-ndd'}).encode()}
written = {}
cloud = MagicMock()
def get_object(**kw):
    if kw['Key'] not in objects:
        raise ClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')
    return {'Body': io.BytesIO(objects[kw['Key']]), 'ETag': '"e"'}
def put_object(**kw):
    body = kw['Body']
    data = body if isinstance(body, bytes) else body.read()
    written[kw['Key']] = data
    objects[kw['Key']] = data
    return {'ETag': '"written"'}
cloud.get_object.side_effect = get_object
cloud.put_object.side_effect = put_object
catalog = cloud.Table.return_value
catalog.get_item.return_value = {'Item': {'work_id': stem, 'id_kind': 'stem',
                                          'ingest_status': 'fulltext_ready',
                                          'source_key': 'papers/%s/clean.md' % stem,
                                          'text_extractor': 'grobid-0.8.2'}}
boto3.client = lambda *a, **k: cloud
boto3.resource = lambda *a, **k: cloud
extract_dir = Path(sys.argv[1]).parent / 'extracted'
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(extract_dir)
sys.path.insert(0, str(extract_dir))
from index import handler
window = handler({'action': 'read_extraction', 'stem': stem, 'max_chars': 20}, None)
assert window['text'] == '# Paper\n\n## Abstract' and window['next_start'] == 20, window
assert window['total_chars'] > 20 and window['execution'] == 'aws'
sections = ['## One-line Summary', '## 2. Key Contributions', '## 3. Methodology and Architecture',
            '## 4. Key Results and Benchmarks', '## 5. Limitations and Future Work',
            '## 6. Related Work', '## 7. Glossary']
body = '\n\n'.join(s + '\n\n' + ('Skipping a uORF exon raised protein 1.4 to 5.5-fold. ' * 8) for s in sections)
bad = handler({'action': 'publish_source_note', 'stem': stem, 'markdown': '## One-line Summary\n\nToo short.',
               'model_id': 'claude-opus-5'}, None)
assert bad['status'] == 'source_failed' and bad['published'] is False and bad['problems']
assert 'wiki/sources/%s.md' % stem not in written
ok = handler({'action': 'publish_source_note', 'stem': stem, 'markdown': body,
              'model_id': 'claude-opus-5'}, None)
assert ok['published'] and ok['status'] == 'source_ready', ok
page = written['wiki/sources/%s.md' % stem].decode()
assert 'ingest_harness: "claude-code"' in page and 'ingest_agent: "byeori-note-local"' in page
assert 'aws-bedrock' not in page
assert 'ingest_model: "opus"' in page and 'ingest_model_id: "claude-opus-5"' in page
assert '## 1. Document Information' in page and 'Modulating splicing' in page
expression = catalog.update_item.call_args.kwargs['UpdateExpression']
assert 'REMOVE source_note_input_tokens' in expression, 'no token count may be invented'
assert 'source_note_written_by' in expression
# A paper that already has a note is not this path's to rewrite.
catalog.get_item.return_value = {'Item': {'work_id': stem, 'id_kind': 'stem',
                                          'ingest_status': 'fulltext_ready',
                                          'source_key': 'papers/%s/clean.md' % stem,
                                          'source_note_key': 'wiki/sources/%s.md' % stem,
                                          'source_note_status': 'source_ready'}}
try:
    handler({'action': 'publish_source_note', 'stem': stem, 'markdown': body, 'model_id': 'claude-opus-5'}, None)
    raise AssertionError('a paper with a note must be refused')
except ValueError as exc:
    assert 'first note' in str(exc), exc
print('ok')
'''
    env = {**os.environ, 'BUCKET_NAME': 'bucket', 'TABLE_NAME': 'table',
           'OPENALEX_API_KEY_PARAMETER': 'parameter', 'AWS_DEFAULT_REGION': 'us-east-1'}
    result = subprocess.run([sys.executable, '-c', code, str(package)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
