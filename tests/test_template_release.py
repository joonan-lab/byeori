"""Nothing in the templates or deploy scripts names this lab's account (spec 2026-09-24)."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


class CfnLoader(yaml.SafeLoader):
    """Read intrinsic tags as plain values so the document parses."""


def _tag(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return {tag_suffix: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {tag_suffix: loader.construct_sequence(node)}
    return {tag_suffix: loader.construct_mapping(node)}


CfnLoader.add_multi_constructor("!", _tag)
TEMPLATE = yaml.load((ROOT / "infra/template.yaml").read_text(encoding="utf-8"), Loader=CfnLoader)


def test_no_account_literal_in_templates_or_scripts():
    residue = re.compile(r"vpc-[0-9a-f]{8}|subnet-[0-9a-f]{8}|\b\d{12}\b|ap-northeast-2")
    for path in ["infra/template.yaml", "infra/lab-template.json", "infra/jev-eval-template.json",
                 "scripts/deploy.sh", "scripts/deploy_lab.sh", "scripts/deploy_jev_eval.sh",
                 "scripts/build_asset_image.sh"]:
        text = (ROOT / path).read_text(encoding="utf-8")
        assert not residue.search(text), f"{path} carries an account literal: {residue.search(text).group()}"


def test_vpc_parameters_are_empty_and_explicit():
    parameters = TEMPLATE["Parameters"]
    assert parameters["VpcId"] == {"Type": "String", "Default": "", "Description": parameters["VpcId"]["Description"]}
    assert parameters["ExtractionSubnets"]["Type"] == "CommaDelimitedList"
    assert parameters["ExtractionSubnets"]["Default"] == ""
    assert parameters["CreateVpc"]["Default"] == "false"
    assert parameters["CreateVpc"]["AllowedValues"] == ["true", "false"]
    assert "Default" not in parameters["OpenAlexApiKeyParameterName"]
    assert parameters["ContactEmail"]["Default"] == ""


def test_rule_refuses_an_empty_vpc_unless_the_stack_creates_one():
    rule = TEMPLATE["Rules"]["VpcGivenOrCreated"]
    assert rule["RuleCondition"] == {"Equals": [{"Ref": "CreateVpc"}, "false"]}
    assert rule["Assertions"][0]["Assert"] == {"Not": [{"Equals": [{"Ref": "VpcId"}, ""]}]}
    assert rule["Assertions"][1]["Assert"] == {"Not": [{"Contains": [{"Ref": "ExtractionSubnets"}, ""]}]}


def test_own_vpc_resources_exist_only_when_asked():
    resources = TEMPLATE["Resources"]
    for name in ["ByeoriVpc", "ByeoriInternetGateway", "ByeoriGatewayAttachment", "ByeoriPublicSubnetA",
                 "ByeoriPublicSubnetB", "ByeoriRouteTable", "ByeoriDefaultRoute", "ByeoriSubnetRouteA",
                 "ByeoriSubnetRouteB"]:
        assert resources[name]["Condition"] == "CreateOwnVpc", name
    assert resources["ExtractionSecurityGroup"]["Properties"]["VpcId"] == {
        "If": ["CreateOwnVpc", {"Ref": "ByeoriVpc"}, {"Ref": "VpcId"}]}
    subnets = {"If": ["CreateOwnVpc", [{"Ref": "ByeoriPublicSubnetA"}, {"Ref": "ByeoriPublicSubnetB"}],
                      {"Ref": "ExtractionSubnets"}]}
    assert TEMPLATE["Outputs"]["ExtractionSubnetIds"]["Value"] == {"Join": [",", subnets]}
    assert TEMPLATE["Outputs"]["ExtractionExecutionRoleArn"]["Value"] == {"GetAtt": ["ExtractionExecutionRole", "Arn"]}
    assert TEMPLATE["Outputs"]["ExtractionTaskRoleArn"]["Value"] == {"GetAtt": ["ExtractionTaskRole", "Arn"]}


def test_deploy_scripts_pass_the_values_from_the_environment():
    deploy = (ROOT / "scripts/deploy.sh").read_text(encoding="utf-8")
    for line in ['VpcId="${KIRO_WIKI_VPC_ID:-}"', 'ExtractionSubnets="${KIRO_WIKI_SUBNET_IDS:-}"',
                 'CreateVpc="${KIRO_WIKI_CREATE_VPC:-false}"', 'ContactEmail="${KIRO_WIKI_CONTACT_EMAIL:-}"']:
        assert line in deploy
    build = (ROOT / "scripts/build_asset_image.sh").read_text(encoding="utf-8")
    assert 'REGION="${AWS_REGION:?set AWS_REGION' in build
    assert "docker build --platform linux/arm64" in build


def test_deploy_checks_origin_main_and_contact_only_in_lab_mode():
    deploy = (ROOT / "scripts/deploy.sh").read_text(encoding="utf-8")
    guard = deploy.index('if [ "${KIRO_WIKI_REQUIRE_ORIGIN_MAIN:-}" = "1" ]; then')
    end = deploy.index("\nfi\n", guard)
    block = deploy[guard:end]
    assert "git merge-base --is-ancestor origin/main HEAD" in block
    assert ': "${KIRO_WIKI_CONTACT_EMAIL:?' in block
    assert deploy.count("git fetch") == 1 and deploy.index("git fetch") > guard
