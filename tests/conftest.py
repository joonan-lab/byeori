"""In-memory AWS surfaces for candidate and publication regression tests."""
import os

# A fresh clone's tests must pass on a machine with no AWS configuration: two modules create
# boto3 clients at import time and need a region name, and nothing here reaches AWS.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

import copy
import io
import json
import re
import sqlite3

import boto3
import pytest
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import AttributeBase, ConditionBase


def matches(condition, item):
    def value(v):
        if isinstance(v, ConditionBase): return matches(v,item)
        if isinstance(v, AttributeBase):
            cur=item
            for part in v.name.split('.'):
                cur=cur.get(part) if isinstance(cur,dict) else None
            return cur
        return v
    expr=condition.get_expression(); op=expr['operator']; vals=[value(v) for v in expr['values']]
    if op=='AND': return all(vals)
    if op=='OR': return any(vals)
    if op=='attribute_exists': return vals[0] is not None
    if op=='attribute_not_exists': return vals[0] is None
    if op=='=': return vals[0]==vals[1]
    if op=='IN': return vals[0] in vals[1]
    if op=='contains': return vals[0] is not None and vals[1] in vals[0]
    if op=='>=': return vals[0] is not None and vals[0]>=vals[1]
    if op=='<=': return vals[0] is not None and vals[0]<=vals[1]
    raise AssertionError(op)


class MemoryAws:
    def __init__(self):
        self.items={}; self.objects={}; self.invocations=[]
    def resource(self, service):
        assert service=='dynamodb'; return self
    def Table(self,name): return self
    def client(self,service,**kwargs):
        assert service in {'s3', 'lambda'}, f'Unexpected unmocked service: {service}'
        return self
    def invoke(self, **request):
        from byeori.wiki_ops import dispatch
        event = json.loads(request['Payload'])
        self.invocations.append(event)
        result = dispatch(event, s3=self, table=self, bucket='bucket', index=self.index)
        return {'Payload': io.BytesIO(json.dumps(result).encode())}
    def index(self):
        con = sqlite3.connect(':memory:')
        con.deserialize(self.objects['index/wiki-index.sqlite3'])
        return con, 'index-version'
    def list_objects_v2(self, **request):
        keys = sorted(k for k in self.objects if k.startswith(request['Prefix']))
        start = int(request.get('ContinuationToken', 0))
        end = start + request['MaxKeys']
        return {'Contents': [{'Key': k} for k in keys[start:end]],
                **({'NextContinuationToken': str(end)} if end < len(keys) else {})}
    def get_item(self,Key): return {'Item':copy.deepcopy(self.items.get(Key['work_id'],{}))}
    def scan(self,**request):
        # A scan without a filter is legal and returns everything, as it does in DynamoDB.
        condition=request.get('FilterExpression')
        rows=[copy.deepcopy(v) for v in self.items.values() if condition is None or matches(condition,v)]
        return {'Items':rows,'ScannedCount':len(self.items)}
    def query(self,**request):
        return {'Items':[copy.deepcopy(v) for v in self.items.values() if matches(request['KeyConditionExpression'],v)]}
    def update_item(self,**request):
        item=self.items.setdefault(request['Key']['work_id'],dict(request['Key']))
        # A conditional update that cannot fail in tests is not the update production runs, so a
        # boto3 condition object is evaluated for real. A condition written as a DynamoDB
        # expression string is NOT evaluated here: those use parentheses, OR and <>, and parsing
        # them would mean writing an expression parser. Tests that need a conditional write checked
        # should pass Attr(...)/Key(...) rather than a string.
        condition=request.get('ConditionExpression')
        if isinstance(condition,ConditionBase) and not matches(condition,item):
            raise ClientError({'Error':{'Code':'ConditionalCheckFailedException'}},'UpdateItem')
        names=request.get('ExpressionAttributeNames',{}); values=request['ExpressionAttributeValues']
        expression=request['UpdateExpression']
        for lhs,rhs in re.findall(r'([#\w]+)\s*=\s*(if_not_exists\([^)]*\)|:\w+)',expression):
            key=names.get(lhs,lhs)
            if rhs.startswith('if_not_exists'):
                if key in item: continue
                rhs=rhs.rsplit(',',1)[1].rstrip(')').strip()
            item[key]=copy.deepcopy(values[rhs])
        remove=re.search(r'\bREMOVE\s+(.*?)(?:\bSET\b|\bADD\b|\bDELETE\b|$)',expression,re.S)
        if remove:
            for token in remove.group(1).split(','):
                key=names.get(token.strip(),token.strip())
                item.pop(key,None)
        return {}
    def put_object(self,**request):
        key=request['Key']
        if request.get('IfNoneMatch')=='*' and key in self.objects:
            raise ClientError({'Error':{'Code':'PreconditionFailed'}},'PutObject')
        body=request['Body'];self.objects[key]=body.read() if hasattr(body,'read') else bytes(body)
        return {}
    def get_object(self,**request):
        if request['Key'] not in self.objects:
            raise ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject')
        return {'Body':io.BytesIO(self.objects[request['Key']])}


@pytest.fixture
def cloud_catalog(monkeypatch):
    cloud=MemoryAws()
    monkeypatch.setattr(boto3,'Session',lambda *args,**kwargs:cloud)
    return cloud
