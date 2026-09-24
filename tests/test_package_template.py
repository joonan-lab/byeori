import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("packaging_syntax", Path(__file__).parents[1] / "scripts/preserve_template_syntax.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_only_uploaded_code_changes_and_intrinsic_notation_is_preserved():
    source = "Resources:\n  Worker:\n      Role: !GetAtt ExistingRole.Arn\n      Code: ../src\n"
    packaged = "Resources:\n  Worker:\n      Role:\n        Fn::GetAtt: [ExistingRole, Arn]\n      Code:\n        S3Bucket: archive\n        S3Key: cfn/verified-code\n"
    result = module.preserve(source, packaged)
    assert result.replace("      Code:\n        S3Bucket: archive\n        S3Key: cfn/verified-code\n", "      Code: ../src\n") == source


@pytest.mark.parametrize("packaged", ["", "      Code:\n        ZipFile: unexpected\n"])
def test_unexpected_artifact_shape_stops_before_deployment(packaged):
    with pytest.raises(ValueError):
        module.preserve("      Code: ../src\n", packaged)
