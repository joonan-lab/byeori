"""Keep deployed intrinsic notation while using AWS CLI's uploaded Lambda artifacts.

CloudFormation can treat GetAtt scalar-to-list conversion as a resource change,
including unnecessary ECS task-definition replacement. Only Code needs packaging.
"""
from pathlib import Path
import re
import sys


def preserve(source: str, packaged: str) -> str:
    blocks = re.findall(r"^      Code:\n(?:        [^\n]+\n)+", packaged, re.MULTILINE)
    count = source.count("      Code: ../src\n")
    if not count or len(blocks) != count:
        raise ValueError("Expected one packaged artifact for each source Lambda Code entry")
    if any(not re.fullmatch(r"      Code:\n        S3Bucket: [^\n]+\n        S3Key: [^\n]+\n", block)
           for block in blocks):
        raise ValueError("Unexpected packaged Code shape; inspect before deploying")
    for block in blocks:
        source = source.replace("      Code: ../src\n", block, 1)
    return source


if __name__ == "__main__":
    source_path, packaged_path = map(Path, sys.argv[1:])
    packaged_path.write_text(preserve(source_path.read_text(), packaged_path.read_text()))
