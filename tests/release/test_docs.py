# tests/release/test_docs.py
"""The documents an installer reads exist and say what the design requires."""
from pathlib import Path
import re

ROOT = Path(__file__).parents[2]
SERVICES = ["IAM", "S3", "DynamoDB", "Lambda", "Fargate", "ECR", "Step Functions", "Bedrock",
            "Parameter Store", "KMS", "CloudFormation", "CloudTrail", "EventBridge"]


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_readme_explains_every_service_and_both_install_paths():
    text = read("README.md")
    for service in SERVICES:
        assert re.search(rf"^#{{2,4}} .*{re.escape(service)}|\*\*{re.escape(service)}", text, re.M), service
    assert "byeori init" in text and "/byeori-install" in text
    assert "docs/INSTALL.md" in text and "docs/COST.md" in text
    assert "beta" in text.lower()


def test_install_has_ten_numbered_steps_in_the_design_order():
    text = read("docs/INSTALL.md")
    headings = re.findall(r"^## (\d+)\. ", text, re.M)
    assert headings == [str(i) for i in range(1, 11)]
    for command in ("byeori init", "byeori deploy", "byeori build-workers", "byeori doctor", "byeori deploy-lab", "byeori grant-client"):
        assert command in text, command
    assert "CreateVpc" in text or "create its own" in text
    assert "aws ssm put-parameter" in text


def test_aws_services_document_covers_the_list_with_console_pointers():
    text = read("docs/AWS-SERVICES.md")
    for service in SERVICES:
        assert f"## {service}" in text, service
    assert "console" in text.lower()


def test_jev_document_covers_key_acquisition_storage_and_rotation():
    text = read("docs/JEV.md")
    for phrase in ("typesafe.ai", "api.typesafe.ai/v1/systemone", "SecureString", "/byeori/jev/api-key",
                   "ParameterKmsKeyArn", "JevApiKeyParameter"):
        assert phrase in text, phrase
    assert "rotate" in text.lower() and "revoke" in text.lower()


def test_cost_document_states_the_three_numbers():
    text = read("docs/COST.md")
    for phrase in ("per paper", "per question", "idle"):
        assert phrase in text.lower(), phrase


def test_no_lab_residue_in_documents():
    # Character classes keep the residue words themselves out of this file, which the export scans too.
    residue = re.compile(r"\b\d{12}\b|vpc-[0-9a-f]{8}|/Use[r]s/|Dropbo[x]|llm-wik[i]")
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "CLAUDE.md"]:
        assert not residue.search(path.read_text(encoding="utf-8")), path.name
