"""Register laboratory members for the byeori-lab student service (design section 3).

Onboarding order for one student:

1. The professor creates the student's IAM user and one access key in the lab account, in the
   console or with ``aws iam create-user --user-name NAME`` and
   ``aws iam create-access-key --user-name NAME``, and hands the key pair to the student over a
   private channel. The student configures an AWS profile with it
   (``aws configure --profile byeori-lab``). This script never creates users or access keys and
   never prints credentials.
2. Register the member here and attach the stack's managed policy in the same run::

       uv run python scripts/lab_members.py register --member-id kim.minji --iam-user kim.minji \\
           --role student --display-name "Kim Minji" --attach-policy

   The profile ``MEMBER#{id}/PROFILE``, the pointer ``PRINCIPAL#{UserId}/MEMBER`` and the index
   row ``MEMBERS/{id}`` are written in one DynamoDB transaction. A member that already exists is
   returned unchanged. A re-created IAM user with the same name has a new UserId and is linked
   only with an explicit ``--replace-principal``; the old pointer is deleted in that transaction.
3. Give the student the ``GatewayUrl`` output (``outputs`` subcommand) as ``LAB_FUNCTION_URL``
   and the MCP command ``python -m byeori.lab_mcp_server``, run with ``AWS_PROFILE`` set
   to the profile from step 1. Nothing else is configured on the student's machine.

Later: ``deactivate --member-id ID`` makes the gateway refuse the member without deleting the
rows; ``activate`` reverses it; ``list`` prints every registered member.

Environment: ``AWS_REGION`` (required; source the env file first) and ``LAB_STACK`` (else
``{KIRO_WIKI_STACK}-lab`` when ``KIRO_WIKI_STACK`` is set, else byeori-lab). The control table name and the two policy ARNs come from the stack outputs. Every
subcommand prints one JSON summary to stdout; failures print ``{"ok": false, ...}`` to stderr and
exit 2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from byeori import lab_members
from byeori.lab_store import ConditionFailed, DynamoTable

SUBCOMMANDS = ("register", "deactivate", "activate", "list", "outputs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab_members.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    register = commands.add_parser("register", help="register an existing IAM user as a lab member")
    register.add_argument("--member-id", required=True, help="stable internal member id, e.g. kim.minji")
    register.add_argument("--iam-user", required=True, help="IAM user name created beforehand by the professor")
    register.add_argument("--role", required=True, choices=sorted(lab_members.ROLES))
    register.add_argument("--display-name", default=None, help="name shown in operator listings")
    register.add_argument("--attach-policy", action="store_true",
                          help="also attach the stack's Student/AdminAccessPolicy to the IAM user")
    register.add_argument("--replace-principal", action="store_true",
                          help="adopt a re-created or renamed IAM user for an existing member")

    for name, text in (("deactivate", "make the gateway refuse the member; rows are kept"),
                       ("activate", "re-enable a deactivated member")):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("--member-id", required=True)

    commands.add_parser("list", help="list registered members from the MEMBERS index")
    commands.add_parser("outputs", help="print the stack outputs the clients need")
    return parser


def run(args: argparse.Namespace, *, table: Any, iam: Any, outputs: dict[str, str]) -> dict[str, Any]:
    """Execute one parsed subcommand against already-built clients; returns the JSON summary."""
    if args.command == "outputs":
        return {"ok": True, "command": "outputs", "outputs": outputs,
                "student_environment": {"LAB_FUNCTION_URL": outputs["gateway_url"],
                                        "AWS_PROFILE": "<the student's own profile>"},
                "mcp_command": "python -m byeori.lab_mcp_server"}
    if args.command == "list":
        members = [lab_members.summary(profile) for profile in lab_members.list_members(table)]
        return {"ok": True, "command": "list", "count": len(members), "members": members}
    if args.command in {"deactivate", "activate"}:
        action = lab_members.deactivate if args.command == "deactivate" else lab_members.activate
        result = action(table, member_id=args.member_id)
        return {"ok": True, "command": args.command, "outcome": result["outcome"],
                "member": lab_members.summary(result["member"])}
    result = lab_members.register(table, iam, member_id=args.member_id, iam_user_name=args.iam_user, role=args.role,
                                  display_name=args.display_name, replace_principal=args.replace_principal)
    report: dict[str, Any] = {"ok": True, "command": "register", "outcome": result["outcome"],
                              "principal_matches": result["principal_matches"],
                              "member": lab_members.summary(result["member"])}
    if result["outcome"] == "principal_replaced":
        report["previous_principal_id"] = result["previous_principal_id"]
        report["previous_principal_arn"] = result["previous_principal_arn"]
    if result["outcome"] == "unchanged" and not result["principal_matches"]:
        report["hint"] = ("The IAM user's UserId/ARN differ from the registered principal (re-created or renamed "
                          "user). Nothing was changed; re-run with --replace-principal to adopt it deliberately.")
    if args.attach_policy:
        if result["principal_matches"]:
            role = result["member"]["role"]
            report["attached_policy"] = lab_members.attach_policy(
                iam, iam_user_name=args.iam_user, policy_arn=lab_members.policy_arn_for_role(outputs, role))
        else:
            # The IAM user is not the registered principal; granting it the access policy would
            # let an unregistered identity reach the gateway (which still refuses it). Skip.
            report["attached_policy"] = None
            report["attach_policy_skipped"] = "the IAM user is not this member's registered principal"
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    region = os.environ.get("AWS_REGION")
    if not region:
        print("error: set AWS_REGION (source the env file first)", file=sys.stderr)
        return 2
    stack = lab_members.lab_stack_name(os.environ)
    try:
        session = boto3.Session(region_name=region)
        outputs = lab_members.stack_policy_arns(session.client("cloudformation"), stack_name=stack)
        table = DynamoTable(session.client("dynamodb"), outputs["table"])
        iam = session.client("iam")
        report = run(args, table=table, iam=iam, outputs=outputs)
    except (ValueError, LookupError, ConditionFailed, ClientError, BotoCoreError) as exc:
        failure = {"ok": False, "command": args.command, "error": type(exc).__name__, "message": str(exc),
                   "region": region, "stack": stack}
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    report.update(region=region, stack=stack)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
