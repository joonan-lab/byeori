"""Member registry writes, IAM policy attachment and the operator CLI (design section 3).

Everything runs against ``lab_fakes.MemoryTable`` and two small recording fakes for the IAM and
CloudFormation clients. No AWS call, no network. The gateway's own ``verify_identity`` is used
to confirm that the rows this module writes are the rows the gateway accepts and refuses.
"""
from __future__ import annotations

import ast
import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from byeori import lab_members
from byeori.lab_gateway import GatewayError, verify_identity
from byeori.lab_policy import POLICY_REVISION
from byeori.lab_store import Delete, Put, Update, keys
from lab_fakes import MemoryTable, iam_event

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/lab_members.py"
ACCOUNT = "123456789012"
NOW = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)
LATER = datetime(2026, 9, 22, 9, 0, tzinfo=UTC)
OUTPUTS = {
    "StudentAccessPolicyArn": f"arn:aws:iam::{ACCOUNT}:policy/byeori-lab-StudentAccessPolicy-ABC",
    "AdminAccessPolicyArn": f"arn:aws:iam::{ACCOUNT}:policy/byeori-lab-AdminAccessPolicy-DEF",
    "ControlTableName": "byeori-lab-ControlTable-XYZ",
    "GatewayUrl": "https://abc123.lambda-url.ap-northeast-2.on.aws/",
    "AnswerQueueUrl": "https://sqs.ap-northeast-2.amazonaws.com/123456789012/answers",
}


def arn(name: str) -> str:
    return f"arn:aws:iam::{ACCOUNT}:user/{name}"


class FakeIam:
    """``get_user`` and ``attach_user_policy`` only; anything that would mint credentials fails."""

    def __init__(self, users: dict[str, tuple[str, str]]):
        self.users = dict(users)
        self.calls: list[tuple[str, dict]] = []
        self.attached: list[tuple[str, str]] = []

    def get_user(self, *, UserName):
        self.calls.append(("get_user", {"UserName": UserName}))
        if UserName not in self.users:
            raise ClientError({"Error": {"Code": "NoSuchEntity", "Message": f"The user {UserName} cannot be found."}},
                              "GetUser")
        user_id, user_arn = self.users[UserName]
        return {"User": {"Path": "/", "UserName": UserName, "UserId": user_id, "Arn": user_arn,
                         "CreateDate": datetime(2026, 9, 20, tzinfo=UTC)}}

    def attach_user_policy(self, *, UserName, PolicyArn):
        self.calls.append(("attach_user_policy", {"UserName": UserName, "PolicyArn": PolicyArn}))
        self.attached.append((UserName, PolicyArn))
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def create_user(self, **_):
        raise AssertionError("the registry must never create IAM users")

    def create_access_key(self, **_):
        raise AssertionError("the registry must never create access keys")


class FakeCloudFormation:
    def __init__(self, outputs: dict[str, str] | None = OUTPUTS, *, exists: bool = True):
        self.outputs, self.exists = outputs, exists
        self.calls: list[str] = []

    def describe_stacks(self, *, StackName):
        self.calls.append(StackName)
        if not self.exists:
            raise ClientError({"Error": {"Code": "ValidationError", "Message": f"Stack with id {StackName} does not exist"}},
                              "DescribeStacks")
        stack = {"StackName": StackName, "StackStatus": "CREATE_COMPLETE"}
        if self.outputs is not None:
            stack["Outputs"] = [{"OutputKey": key, "OutputValue": value} for key, value in self.outputs.items()]
        return {"Stacks": [stack]}


@pytest.fixture
def table():
    return MemoryTable()


@pytest.fixture
def iam():
    return FakeIam({"kim.minji": ("AIDAKIMMINJI0000000001", arn("kim.minji")),
                    "admin.a": ("AIDAADMINA000000000001", arn("admin.a"))})


def register(table, iam, member_id="kim.minji", user="kim.minji", role="student", **kwargs):
    return lab_members.register(table, iam, member_id=member_id, iam_user_name=user, role=role, now=NOW, **kwargs)


def identity(table, user_id, user_arn):
    return verify_identity(iam_event("ask", user_id=user_id, user_arn=user_arn), table, account_id=ACCOUNT)


# ---------------------------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------------------------

def test_register_writes_profile_pointer_and_index_in_one_transaction(table, iam):
    result = register(table, iam, display_name="  Kim   Minji ")

    assert result["outcome"] == "registered" and result["principal_matches"] is True
    assert len(table.transactions) == 1
    assert [type(op) for op in table.transactions[0]] == [Put, Put, Put]

    profile = table.get(*keys.member("kim.minji"))
    assert profile == result["member"]
    assert profile["revision"] == 1
    assert profile["member_id"] == "kim.minji"
    assert profile["principal_id"] == "AIDAKIMMINJI0000000001"
    assert profile["principal_arn"] == arn("kim.minji")
    assert profile["iam_user_name"] == "kim.minji"
    assert profile["role"] == "student" and profile["active"] is True
    assert profile["policy_revision"] == POLICY_REVISION
    assert profile["display_name"] == "Kim Minji"
    assert profile["registered_at"] == profile["created_at"] == "2026-09-21T09:00:00.000000+00:00"

    pointer = table.get(*keys.principal("AIDAKIMMINJI0000000001"))
    assert pointer["member_id"] == "kim.minji" and pointer["revision"] == 1
    index_row = table.get(*lab_members.members_index_key("kim.minji"))
    assert index_row["member_id"] == "kim.minji"
    assert iam.calls == [("get_user", {"UserName": "kim.minji"})]
    assert iam.attached == []


def test_registered_member_passes_the_gateway_identity_check(table, iam):
    register(table, iam)
    member = identity(table, "AIDAKIMMINJI0000000001", arn("kim.minji"))
    assert (member.member_id, member.role, member.policy_revision) == ("kim.minji", "student", POLICY_REVISION)


def test_register_admin_role_and_omits_display_name_when_blank(table, iam):
    result = register(table, iam, member_id="admin.a", user="admin.a", role="admin", display_name="   ")
    assert result["member"]["role"] == "admin"
    assert "display_name" not in result["member"]
    assert identity(table, "AIDAADMINA000000000001", arn("admin.a")).role == "admin"


def test_duplicate_register_returns_existing_member_without_writing(table, iam):
    first = register(table, iam, display_name="Kim Minji")
    again = lab_members.register(table, iam, member_id="kim.minji", iam_user_name="kim.minji", role="student",
                                 now=LATER, display_name="Different Name")

    assert again["outcome"] == "unchanged" and again["principal_matches"] is True
    assert again["member"] == first["member"]
    assert len(table.transactions) == 1
    assert table.get(*keys.member("kim.minji"))["display_name"] == "Kim Minji"


def test_recreated_iam_user_is_not_relinked_without_replace_principal(table, iam):
    register(table, iam)
    iam.users["kim.minji"] = ("AIDAKIMMINJI0000000002", arn("kim.minji"))  # deleted and re-created, same name

    result = lab_members.register(table, iam, member_id="kim.minji", iam_user_name="kim.minji", role="student", now=LATER)

    assert result["outcome"] == "unchanged" and result["principal_matches"] is False
    assert len(table.transactions) == 1
    assert table.get(*keys.member("kim.minji"))["principal_id"] == "AIDAKIMMINJI0000000001"
    assert table.get(*keys.principal("AIDAKIMMINJI0000000002")) is None
    with pytest.raises(GatewayError) as refused:
        identity(table, "AIDAKIMMINJI0000000002", arn("kim.minji"))
    assert refused.value.code == "forbidden"


def test_replace_principal_moves_the_pointer_and_updates_the_profile_in_one_transaction(table, iam):
    register(table, iam)
    iam.users["kim.minji"] = ("AIDAKIMMINJI0000000002", arn("kim.minji"))

    result = lab_members.register(table, iam, member_id="kim.minji", iam_user_name="kim.minji", role="student",
                                  now=LATER, replace_principal=True)

    assert result["outcome"] == "principal_replaced" and result["principal_matches"] is True
    assert result["previous_principal_id"] == "AIDAKIMMINJI0000000001"
    assert len(table.transactions) == 2
    operations = table.transactions[-1]
    assert [type(op) for op in operations] == [Update, Delete, Put]
    assert (operations[0].pk, operations[0].sk, operations[0].expected_revision) == ("MEMBER#kim.minji", "PROFILE", 1)
    assert (operations[1].pk, operations[1].sk) == ("PRINCIPAL#AIDAKIMMINJI0000000001", "MEMBER")
    assert operations[2].item["pk"] == "PRINCIPAL#AIDAKIMMINJI0000000002"

    profile = table.get(*keys.member("kim.minji"))
    assert profile == result["member"]
    assert profile["revision"] == 2
    assert profile["principal_id"] == "AIDAKIMMINJI0000000002"
    assert profile["previous_principal_id"] == "AIDAKIMMINJI0000000001"
    assert profile["principal_replaced_at"] == "2026-09-22T09:00:00.000000+00:00"
    assert profile["role"] == "student" and profile["active"] is True and profile["registered_at"] == profile["created_at"]
    assert table.get(*keys.principal("AIDAKIMMINJI0000000001")) is None
    assert table.get(*keys.principal("AIDAKIMMINJI0000000002"))["member_id"] == "kim.minji"

    assert identity(table, "AIDAKIMMINJI0000000002", arn("kim.minji")).member_id == "kim.minji"
    with pytest.raises(GatewayError):
        identity(table, "AIDAKIMMINJI0000000001", arn("kim.minji"))


def test_replace_principal_for_a_renamed_user_updates_only_the_profile(table, iam):
    register(table, iam)
    iam.users["kim.minji"] = ("AIDAKIMMINJI0000000001", arn("minji.kim"))  # rename keeps the UserId

    result = lab_members.register(table, iam, member_id="kim.minji", iam_user_name="kim.minji", role="student",
                                  now=LATER, replace_principal=True)

    assert result["outcome"] == "principal_replaced"
    assert [type(op) for op in table.transactions[-1]] == [Update]
    profile = table.get(*keys.member("kim.minji"))
    assert profile["principal_arn"] == arn("minji.kim") and profile["principal_id"] == "AIDAKIMMINJI0000000001"
    assert table.get(*keys.principal("AIDAKIMMINJI0000000001"))["member_id"] == "kim.minji"
    assert identity(table, "AIDAKIMMINJI0000000001", arn("minji.kim")).member_id == "kim.minji"


def test_replace_principal_with_matching_principal_is_a_no_op(table, iam):
    register(table, iam)
    result = lab_members.register(table, iam, member_id="kim.minji", iam_user_name="kim.minji", role="student",
                                  now=LATER, replace_principal=True)
    assert result["outcome"] == "unchanged" and result["principal_matches"] is True
    assert len(table.transactions) == 1


def test_register_refuses_a_principal_that_belongs_to_another_member(table, iam):
    register(table, iam)
    register(table, iam, member_id="admin.a", user="admin.a", role="admin")

    # A second member id for an IAM user that is already someone's principal.
    with pytest.raises(ValueError, match="already registered as member 'kim.minji'"):
        register(table, iam, member_id="second", user="kim.minji")

    # replace_principal must not steal another member's principal either.
    iam.users["admin.a"] = ("AIDAKIMMINJI0000000001", arn("admin.a"))
    with pytest.raises(ValueError, match="already registered as member 'kim.minji'"):
        lab_members.register(table, iam, member_id="admin.a", iam_user_name="admin.a", role="admin", now=LATER,
                             replace_principal=True)

    assert len(table.transactions) == 2  # only the two clean registrations were written
    assert table.get(*keys.member("admin.a"))["principal_id"] == "AIDAADMINA000000000001"


def test_register_refuses_a_role_outside_student_and_admin(table, iam):
    with pytest.raises(ValueError, match="role must be one of: admin, student"):
        register(table, iam, role="professor")
    assert iam.calls == [] and table.items == {}


@pytest.mark.parametrize("member_id", ["", "kim#minji", "a b", "x" * 129, None])
def test_register_refuses_a_malformed_member_id(table, iam, member_id):
    with pytest.raises(ValueError, match="member_id"):
        register(table, iam, member_id=member_id)
    assert iam.calls == [] and table.items == {}


def test_register_refuses_a_malformed_iam_user_name(table, iam):
    with pytest.raises(ValueError, match="iam_user_name"):
        register(table, iam, user="kim minji")
    assert iam.calls == [] and table.items == {}


def test_unknown_iam_user_propagates_and_writes_nothing(table, iam):
    with pytest.raises(ClientError):
        register(table, iam, member_id="ghost", user="ghost")
    assert table.items == {}


def test_register_requires_iam_to_return_user_id_and_arn(table):
    class BrokenIam:
        def get_user(self, *, UserName):
            return {"User": {"UserName": UserName}}

    with pytest.raises(ValueError, match="UserId and Arn"):
        lab_members.register(table, BrokenIam(), member_id="kim.minji", iam_user_name="kim.minji", role="student", now=NOW)
    assert table.items == {}


# ---------------------------------------------------------------------------------------------
# deactivate / activate / list
# ---------------------------------------------------------------------------------------------

def test_deactivate_and_activate_toggle_active_with_revision_checks(table, iam):
    register(table, iam)

    off = lab_members.deactivate(table, member_id="kim.minji", now=LATER)
    assert off["outcome"] == "deactivated"
    profile = table.get(*keys.member("kim.minji"))
    assert profile == off["member"]
    assert profile["active"] is False and profile["revision"] == 2
    assert profile["deactivated_at"] == "2026-09-22T09:00:00.000000+00:00"
    assert [type(op) for op in table.transactions[-1]] == [Update]
    with pytest.raises(GatewayError) as refused:
        identity(table, "AIDAKIMMINJI0000000001", arn("kim.minji"))
    assert refused.value.code == "inactive_member"

    same = lab_members.deactivate(table, member_id="kim.minji", now=LATER)
    assert same["outcome"] == "unchanged" and len(table.transactions) == 2

    on = lab_members.activate(table, member_id="kim.minji", now=LATER)
    assert on["outcome"] == "activated"
    profile = table.get(*keys.member("kim.minji"))
    assert profile["active"] is True and profile["revision"] == 3 and "reactivated_at" in profile
    assert identity(table, "AIDAKIMMINJI0000000001", arn("kim.minji")).member_id == "kim.minji"
    assert table.get(*keys.principal("AIDAKIMMINJI0000000001"))["member_id"] == "kim.minji"


def test_deactivate_unknown_member_raises_lookup_error(table):
    with pytest.raises(LookupError, match="not registered"):
        lab_members.deactivate(table, member_id="nobody")
    with pytest.raises(LookupError):
        lab_members.activate(table, member_id="nobody")
    assert table.items == {}


def test_list_members_reads_the_members_partition_across_pages(table, iam):
    iam.users.update({"lee.a": ("AIDALEEA00000000000001", arn("lee.a")), "park.b": ("AIDAPARKB0000000000001", arn("park.b"))})
    register(table, iam, member_id="admin.a", user="admin.a", role="admin", display_name="Admin A")
    register(table, iam, member_id="park.b", user="park.b")
    register(table, iam, member_id="kim.minji", user="kim.minji")
    register(table, iam, member_id="lee.a", user="lee.a")
    lab_members.deactivate(table, member_id="park.b", now=LATER)

    members = lab_members.list_members(table, page_size=1)

    assert [m["member_id"] for m in members] == ["admin.a", "kim.minji", "lee.a", "park.b"]
    assert [m["role"] for m in members] == ["admin", "student", "student", "student"]
    assert [m["active"] for m in members] == [True, True, True, False]
    assert members[0]["display_name"] == "Admin A"
    assert lab_members.list_members(table) == members
    assert lab_members.list_members(MemoryTable()) == []


def test_list_members_reports_an_index_row_without_a_profile(table, iam):
    register(table, iam)
    table.delete(*keys.member("kim.minji"), 1)
    assert lab_members.list_members(table) == [{"member_id": "kim.minji", "missing_profile": True}]


def test_summary_keeps_identity_and_state_fields_only(table, iam):
    profile = register(table, iam, display_name="Kim Minji")["member"]
    view = lab_members.summary(profile)
    assert view == {"member_id": "kim.minji", "display_name": "Kim Minji", "iam_user_name": "kim.minji",
                    "principal_id": "AIDAKIMMINJI0000000001", "principal_arn": arn("kim.minji"), "role": "student",
                    "active": True, "policy_revision": POLICY_REVISION,
                    "registered_at": "2026-09-21T09:00:00.000000+00:00", "revision": 1}


# ---------------------------------------------------------------------------------------------
# attach_policy / stack_policy_arns
# ---------------------------------------------------------------------------------------------

def test_attach_policy_calls_attach_user_policy_with_the_role_arn(iam):
    outputs = lab_members.stack_policy_arns(FakeCloudFormation())
    result = lab_members.attach_policy(iam, iam_user_name="kim.minji",
                                       policy_arn=lab_members.policy_arn_for_role(outputs, "student"))
    assert result == {"iam_user_name": "kim.minji", "policy_arn": OUTPUTS["StudentAccessPolicyArn"]}
    assert iam.attached == [("kim.minji", OUTPUTS["StudentAccessPolicyArn"])]
    assert lab_members.policy_arn_for_role(outputs, "admin") == OUTPUTS["AdminAccessPolicyArn"]
    with pytest.raises(ValueError):
        lab_members.policy_arn_for_role(outputs, "professor")


@pytest.mark.parametrize("policy_arn", ["", "StudentAccessPolicy", "arn:aws:iam::123456789012:user/kim.minji",
                                        "arn:aws:s3:::bucket", None])
def test_attach_policy_refuses_anything_but_a_managed_policy_arn(iam, policy_arn):
    with pytest.raises(ValueError, match="policy_arn"):
        lab_members.attach_policy(iam, iam_user_name="kim.minji", policy_arn=policy_arn)
    assert iam.attached == [] and iam.calls == []


def test_stack_policy_arns_maps_the_four_outputs():
    cloudformation = FakeCloudFormation()
    assert lab_members.stack_policy_arns(cloudformation) == {
        "student": OUTPUTS["StudentAccessPolicyArn"], "admin": OUTPUTS["AdminAccessPolicyArn"],
        "table": OUTPUTS["ControlTableName"], "gateway_url": OUTPUTS["GatewayUrl"]}
    assert cloudformation.calls == ["byeori-lab"]
    lab_members.stack_policy_arns(cloudformation, stack_name="byeori-lab-test")
    assert cloudformation.calls[-1] == "byeori-lab-test"


def test_stack_policy_arns_refuses_a_stack_missing_an_output():
    partial = {key: value for key, value in OUTPUTS.items() if key != "AdminAccessPolicyArn"}
    with pytest.raises(LookupError, match="AdminAccessPolicyArn"):
        lab_members.stack_policy_arns(FakeCloudFormation(partial))
    with pytest.raises(LookupError):
        lab_members.stack_policy_arns(FakeCloudFormation(None))
    with pytest.raises(ClientError):
        lab_members.stack_policy_arns(FakeCloudFormation(exists=False))
    with pytest.raises(ValueError):
        lab_members.stack_policy_arns(FakeCloudFormation(), stack_name="")


# ---------------------------------------------------------------------------------------------
# The operator CLI (parse and dispatch only; boto3 clients are built in main and never here)
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("lab_members_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_parser_accepts_every_subcommand(cli):
    parser = cli.build_parser()
    args = parser.parse_args(["register", "--member-id", "kim.minji", "--iam-user", "kim.minji", "--role", "student",
                              "--display-name", "Kim Minji", "--attach-policy", "--replace-principal"])
    assert (args.command, args.member_id, args.iam_user, args.role) == ("register", "kim.minji", "kim.minji", "student")
    assert args.display_name == "Kim Minji" and args.attach_policy is True and args.replace_principal is True
    plain = parser.parse_args(["register", "--member-id", "x", "--iam-user", "x", "--role", "admin"])
    assert plain.display_name is None and plain.attach_policy is False and plain.replace_principal is False
    assert parser.parse_args(["deactivate", "--member-id", "kim.minji"]).command == "deactivate"
    assert parser.parse_args(["activate", "--member-id", "kim.minji"]).command == "activate"
    assert parser.parse_args(["list"]).command == "list"
    assert parser.parse_args(["outputs"]).command == "outputs"
    assert set(cli.SUBCOMMANDS) == {"register", "deactivate", "activate", "list", "outputs"}


@pytest.mark.parametrize("argv", [[], ["register"], ["register", "--member-id", "x", "--iam-user", "x"],
                                  ["register", "--member-id", "x", "--iam-user", "x", "--role", "professor"],
                                  ["deactivate"], ["nonsense"]])
def test_cli_parser_rejects_incomplete_or_unknown_commands(cli, argv):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv)


def test_cli_run_registers_and_attaches_the_policy_with_injected_clients(cli, table, iam):
    outputs = lab_members.stack_policy_arns(FakeCloudFormation())
    args = cli.build_parser().parse_args(["register", "--member-id", "kim.minji", "--iam-user", "kim.minji",
                                          "--role", "student", "--attach-policy"])
    report = cli.run(args, table=table, iam=iam, outputs=outputs)
    assert report["ok"] is True and report["outcome"] == "registered"
    assert report["member"]["member_id"] == "kim.minji"
    assert report["attached_policy"]["policy_arn"] == OUTPUTS["StudentAccessPolicyArn"]
    assert iam.attached == [("kim.minji", OUTPUTS["StudentAccessPolicyArn"])]

    iam.users["kim.minji"] = ("AIDAKIMMINJI0000000002", arn("kim.minji"))
    again = cli.run(cli.build_parser().parse_args(["register", "--member-id", "kim.minji", "--iam-user", "kim.minji",
                                                   "--role", "student", "--attach-policy"]), table=table, iam=iam,
                    outputs=outputs)
    assert again["outcome"] == "unchanged" and again["principal_matches"] is False and "replace-principal" in again["hint"]
    assert again["attached_policy"] is None and "not this member's registered principal" in again["attach_policy_skipped"]
    assert iam.attached == [("kim.minji", OUTPUTS["StudentAccessPolicyArn"])]  # no grant to the unregistered principal

    listing = cli.run(cli.build_parser().parse_args(["list"]), table=table, iam=iam, outputs=outputs)
    assert listing["count"] == 1 and listing["members"][0]["member_id"] == "kim.minji"
    off = cli.run(cli.build_parser().parse_args(["deactivate", "--member-id", "kim.minji"]), table=table, iam=iam, outputs=outputs)
    assert off["outcome"] == "deactivated" and off["member"]["active"] is False
    shown = cli.run(cli.build_parser().parse_args(["outputs"]), table=table, iam=iam, outputs=outputs)
    assert shown["student_environment"]["LAB_FUNCTION_URL"] == OUTPUTS["GatewayUrl"]
    assert shown["mcp_command"] == "python -m byeori.lab_mcp_server"


def test_cli_and_module_never_create_users_or_keys_and_build_clients_only_in_main():
    for path in (SCRIPT, ROOT / "src/byeori/lab_members.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        forbidden = {"create_user", "create_access_key", "create_login_profile", "update_access_key",
                     "delete_user", "delete_access_key", "detach_user_policy", "put_user_policy"}
        assert not attributes & forbidden, f"{path.name} touches {sorted(attributes & forbidden)}"
        allowed = {"get_user", "attach_user_policy", "describe_stacks"}
        aws_calls = attributes & {"get_user", "attach_user_policy", "describe_stacks", "put_item", "update_item",
                                  "delete_item", "transact_write_items", "scan"}
        assert aws_calls <= allowed, sorted(aws_calls)
        text = path.read_text(encoding="utf-8")
        assert "SecretAccessKey" not in text and "AccessKeyId" not in text

    module_tree = ast.parse((ROOT / "src/byeori/lab_members.py").read_text(encoding="utf-8"))
    module_imports = {alias.name.split(".")[0] for node in ast.walk(module_tree)
                      if isinstance(node, ast.Import) for alias in node.names}
    module_imports |= {(node.module or "").split(".")[0] for node in ast.walk(module_tree) if isinstance(node, ast.ImportFrom)}
    assert "boto3" not in module_imports and "botocore" not in module_imports

    script_tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    boto3_calls_outside_main = []
    functions = {node.name: node for node in script_tree.body if isinstance(node, ast.FunctionDef)}
    for name, function in functions.items():
        uses_boto3 = any(isinstance(n, ast.Name) and n.id == "boto3" for n in ast.walk(function))
        if uses_boto3 and name != "main":
            boto3_calls_outside_main.append(name)
    assert boto3_calls_outside_main == []
    top_level_boto3 = [n for n in script_tree.body if not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))
                       and any(isinstance(m, ast.Name) and m.id == "boto3" for m in ast.walk(n))]
    assert top_level_boto3 == []


def test_lab_stack_name_prefers_lab_stack_then_the_main_stack_then_the_default():
    assert lab_members.lab_stack_name({"LAB_STACK": "x-lab", "KIRO_WIKI_STACK": "y"}) == "x-lab"
    assert lab_members.lab_stack_name({"KIRO_WIKI_STACK": "mylab"}) == "mylab-lab"
    assert lab_members.lab_stack_name({"LAB_STACK": "", "KIRO_WIKI_STACK": ""}) == "byeori-lab"
    assert lab_members.lab_stack_name({}) == lab_members.DEFAULT_STACK == "byeori-lab"
