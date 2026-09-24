"""Member registry for the student question workflow (docs/LAB-QUESTION-WORKFLOW.md, section 3).

The gateway authenticates a request by the IAM principal the Function URL reports and looks it
up in the control table: ``PRINCIPAL#{UserId}/MEMBER`` points at ``MEMBER#{member_id}/PROFILE``,
and ``lab_gateway.verify_identity`` accepts the caller only when the profile carries the same
principal id **and** ARN, an allowed role and ``active`` set to true. This module writes those
rows from an IAM user name, so the professor registers each member deliberately.

Two rules follow from the design. A member that is already registered is returned unchanged:
a re-created IAM user with the old name has a new ``UserId``, and it is linked only when the
operator asks for ``replace_principal``. Registration, replacement and deactivation are
revision-checked ``lab_store`` operations, so two operators cannot silently overwrite each
other. A third row, ``MEMBERS/{member_id}``, is written with the profile so the registry can be
listed with one partition query instead of a table scan.

This module never creates IAM users or access keys and never handles credentials. It talks to
the table through ``lab_store.TablePort`` and to IAM and CloudFormation through the two read or
attach calls named below; the CLI in ``scripts/lab_members.py`` builds the boto3 clients.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from .lab_jobs import ROLES
from .lab_policy import POLICY_REVISION
from .lab_store import Delete, Put, TablePort, Update, keys, new_item, now_iso

DEFAULT_STACK = "byeori-lab"


def lab_stack_name(environ: Mapping[str, str]) -> str:
    """``LAB_STACK``, else ``{KIRO_WIKI_STACK}-lab`` (what ``deploy-lab`` names it), else byeori-lab."""
    if environ.get("LAB_STACK"):
        return environ["LAB_STACK"]
    if environ.get("KIRO_WIKI_STACK"):
        return f"{environ['KIRO_WIKI_STACK']}-lab"
    return DEFAULT_STACK

# Partition that lists every registered member: one row per member, sk = member_id.
MEMBERS_INDEX = "MEMBERS"

# Stack outputs the CLI needs, by the short name the rest of this module uses.
OUTPUT_KEYS = {
    "student": "StudentAccessPolicyArn",
    "admin": "AdminAccessPolicyArn",
    "table": "ControlTableName",
    "gateway_url": "GatewayUrl",
}

# The same identifier shape the gateway and the job module accept for member_id.
_MEMBER_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
# IAM user names: https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_iam-quotas.html
_IAM_USER_NAME = re.compile(r"^[\w+=,.@-]{1,64}$")
_POLICY_ARN = re.compile(r"^arn:aws[a-z-]*:iam::(?:\d{12}|aws):policy/.+$")
_ROLE_TEXT = ", ".join(sorted(ROLES))


def members_index_key(member_id: str) -> tuple[str, str]:
    """The registry index row that lists ``member_id`` (``MEMBERS`` / ``{member_id}``)."""
    return MEMBERS_INDEX, member_id


# ---------------------------------------------------------------------------------------------
# IAM and CloudFormation lookups (read or attach only)
# ---------------------------------------------------------------------------------------------

def lookup_principal(iam: Any, iam_user_name: str) -> dict[str, str]:
    """The IAM user's stable ``UserId`` and ``Arn``; ``iam.get_user`` is the only call made."""
    name = _iam_user_name(iam_user_name)
    user = iam.get_user(UserName=name).get("User")
    principal_id = user.get("UserId") if isinstance(user, Mapping) else None
    principal_arn = user.get("Arn") if isinstance(user, Mapping) else None
    if not isinstance(principal_id, str) or not principal_id or not isinstance(principal_arn, str) \
            or not principal_arn.startswith("arn:"):
        raise ValueError(f"IAM did not return a UserId and Arn for user {name!r}")
    return {"principal_id": principal_id, "principal_arn": principal_arn,
            "iam_user_name": user.get("UserName") if isinstance(user.get("UserName"), str) else name}


def stack_policy_arns(cloudformation: Any, stack_name: str = DEFAULT_STACK) -> dict[str, str]:
    """``{'student', 'admin', 'table', 'gateway_url'}`` from the lab stack's outputs.

    Raises ``LookupError`` when the stack has no outputs or one of the four is missing, so a
    half-deployed stack is never used to attach a policy or write the table.
    """
    if not stack_name or not isinstance(stack_name, str):
        raise ValueError("stack_name must be a non-empty string")
    stacks = cloudformation.describe_stacks(StackName=stack_name).get("Stacks") or []
    if not stacks:
        raise LookupError(f"CloudFormation stack {stack_name!r} was not found")
    outputs = {entry.get("OutputKey"): entry.get("OutputValue")
               for entry in stacks[0].get("Outputs") or [] if isinstance(entry, Mapping)}
    missing = [key for key in OUTPUT_KEYS.values() if not isinstance(outputs.get(key), str) or not outputs.get(key)]
    if missing:
        raise LookupError(f"stack {stack_name!r} lacks outputs {missing}; deploy scripts/deploy_lab.sh first")
    return {name: outputs[key] for name, key in OUTPUT_KEYS.items()}


def policy_arn_for_role(outputs: Mapping[str, str], role: str) -> str:
    """The managed policy ARN a member of ``role`` receives (student or admin access policy)."""
    if role not in ROLES:
        raise ValueError(f"role must be one of: {_ROLE_TEXT}")
    return outputs[role]


def attach_policy(iam: Any, *, iam_user_name: str, policy_arn: str) -> dict[str, str]:
    """Attach the stack's StudentAccessPolicy or AdminAccessPolicy to the IAM user.

    ``attach_user_policy`` is idempotent, so running the CLI twice is harmless. The ARN must be
    a customer managed IAM policy ARN; nothing else is attached from here.
    """
    name = _iam_user_name(iam_user_name)
    if not isinstance(policy_arn, str) or not _POLICY_ARN.match(policy_arn):
        raise ValueError(f"policy_arn must be an IAM managed policy ARN, got {policy_arn!r}")
    iam.attach_user_policy(UserName=name, PolicyArn=policy_arn)
    return {"iam_user_name": name, "policy_arn": policy_arn}


# ---------------------------------------------------------------------------------------------
# Registry writes
# ---------------------------------------------------------------------------------------------

def register(table: TablePort, iam: Any, *, member_id: str, iam_user_name: str, role: str,
             now: datetime | None = None, display_name: str | None = None,
             replace_principal: bool = False) -> dict[str, Any]:
    """Register ``member_id`` for the IAM user, or report the existing member.

    A new member is three rows in one transaction: the profile (revision 1, ``active`` true,
    the current ``POLICY_REVISION``), the principal pointer and the ``MEMBERS`` index row. If
    the profile already exists the call writes nothing and returns it with ``outcome``
    ``unchanged`` and ``principal_matches`` saying whether the IAM user still has the registered
    ``UserId``/ARN. With ``replace_principal`` a differing principal is adopted in one
    transaction: the profile is updated at its revision, the old pointer is deleted and the new
    one is created. A principal that already points at another member is refused.
    """
    member_id = _member_id(member_id)
    if role not in ROLES:
        raise ValueError(f"role must be one of: {_ROLE_TEXT}")
    display = _display_name(display_name)
    principal = lookup_principal(iam, iam_user_name)
    stamp = now_iso(now)
    existing = table.get(*keys.member(member_id))
    if existing is not None:
        return _existing_member(table, existing, principal, stamp, replace_principal=replace_principal)
    _refuse_foreign_pointer(table, principal["principal_id"], member_id)
    attributes = {"member_id": member_id, "principal_id": principal["principal_id"],
                  "principal_arn": principal["principal_arn"], "iam_user_name": principal["iam_user_name"],
                  "role": role, "active": True, "policy_revision": POLICY_REVISION, "registered_at": stamp}
    if display:
        attributes["display_name"] = display
    profile = new_item(*keys.member(member_id), stamp, **attributes)
    pointer = new_item(*keys.principal(principal["principal_id"]), stamp, member_id=member_id)
    index_row = new_item(*members_index_key(member_id), stamp, member_id=member_id)
    table.transact([Put(profile), Put(pointer), Put(index_row)])
    return {"outcome": "registered", "member": profile, "principal_matches": True}


def _existing_member(table: TablePort, existing: dict[str, Any], principal: Mapping[str, str], stamp: str,
                     *, replace_principal: bool) -> dict[str, Any]:
    member_id = existing["member_id"]
    same_id = existing.get("principal_id") == principal["principal_id"]
    same_arn = existing.get("principal_arn") == principal["principal_arn"]
    if (same_id and same_arn) or not replace_principal:
        return {"outcome": "unchanged", "member": existing, "principal_matches": same_id and same_arn}
    changes: dict[str, Any] = {"principal_id": principal["principal_id"], "principal_arn": principal["principal_arn"],
                               "iam_user_name": principal["iam_user_name"], "principal_replaced_at": stamp,
                               "previous_principal_id": existing.get("principal_id"),
                               "previous_principal_arn": existing.get("principal_arn")}
    operations: list[Any] = [Update(*keys.member(member_id), existing["revision"], changes)]
    if not same_id:
        # A new UserId: the pointer moves. Same UserId with a new ARN is a renamed user; the
        # pointer row is keyed by UserId and stays where it is.
        _refuse_foreign_pointer(table, principal["principal_id"], member_id)
        old_pointer = table.get(*keys.principal(existing["principal_id"])) if existing.get("principal_id") else None
        if old_pointer is not None:
            if old_pointer.get("member_id") != member_id:
                raise ValueError(f"pointer for principal {existing['principal_id']!r} belongs to member "
                                 f"{old_pointer.get('member_id')!r}, not {member_id!r}; fix the registry by hand")
            operations.append(Delete(old_pointer["pk"], old_pointer["sk"], old_pointer["revision"]))
        operations.append(Put(new_item(*keys.principal(principal["principal_id"]), stamp, member_id=member_id)))
    table.transact(operations)
    return {"outcome": "principal_replaced", "member": table.get(*keys.member(member_id)), "principal_matches": True,
            "previous_principal_id": existing.get("principal_id"), "previous_principal_arn": existing.get("principal_arn")}


def _refuse_foreign_pointer(table: TablePort, principal_id: str, member_id: str) -> None:
    pointer = table.get(*keys.principal(principal_id))
    if pointer is not None:
        owner = pointer.get("member_id")
        if owner == member_id:
            raise ValueError(f"principal {principal_id!r} already points at member {member_id!r} but the profile "
                             "does not match; fix the registry by hand")
        raise ValueError(f"principal {principal_id!r} is already registered as member {owner!r}")


def deactivate(table: TablePort, *, member_id: str, now: datetime | None = None) -> dict[str, Any]:
    """Set ``active`` false so ``verify_identity`` refuses the member; the rows are kept."""
    return _set_active(table, member_id, False, now)


def activate(table: TablePort, *, member_id: str, now: datetime | None = None) -> dict[str, Any]:
    """Set ``active`` true again on an existing member."""
    return _set_active(table, member_id, True, now)


def _set_active(table: TablePort, member_id: str, active: bool, now: datetime | None) -> dict[str, Any]:
    member_id = _member_id(member_id)
    profile = table.get(*keys.member(member_id))
    if profile is None:
        raise LookupError(f"member {member_id!r} is not registered")
    if profile.get("active") is active:
        return {"outcome": "unchanged", "member": profile}
    stamp = now_iso(now)
    changes = {"active": active, "reactivated_at" if active else "deactivated_at": stamp}
    updated = table.update(*keys.member(member_id), profile["revision"], changes)
    return {"outcome": "activated" if active else "deactivated", "member": updated}


def list_members(table: TablePort, *, page_size: int = 100) -> list[dict[str, Any]]:
    """Every registered member's profile, by member_id, from the ``MEMBERS`` partition.

    The index rows are read with a paginated partition query (no scan); each profile is then
    fetched by key so the listing shows the current role and ``active`` state. An index row
    whose profile is missing is reported as ``{"member_id": ..., "missing_profile": True}``
    rather than dropped.
    """
    pointers: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        page, cursor = table.query(MEMBERS_INDEX, limit=page_size, start_after=cursor)
        pointers.extend(page)
        if cursor is None:
            break
    members = []
    for pointer in pointers:
        member_id = pointer.get("member_id") or pointer["sk"]
        profile = table.get(*keys.member(member_id))
        members.append(profile if profile is not None else {"member_id": member_id, "missing_profile": True})
    return sorted(members, key=lambda item: item["member_id"])


def summary(profile: Mapping[str, Any]) -> dict[str, Any]:
    """The operator-facing view of a profile: identity, role, state and stamps, nothing else."""
    fields = ("member_id", "display_name", "iam_user_name", "principal_id", "principal_arn", "role", "active",
              "policy_revision", "registered_at", "deactivated_at", "reactivated_at", "principal_replaced_at",
              "revision", "missing_profile")
    return {name: profile[name] for name in fields if name in profile}


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------

def _member_id(value: Any) -> str:
    if not isinstance(value, str) or not _MEMBER_ID.match(value):
        raise ValueError("member_id must be 1-128 characters of letters, digits, '.', '_' or '-'")
    return value


def _iam_user_name(value: Any) -> str:
    if not isinstance(value, str) or not _IAM_USER_NAME.match(value):
        raise ValueError("iam_user_name must be an IAM user name (1-64 characters of letters, digits, +=,.@-_)")
    return value


def _display_name(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("display_name must be text")
    text = " ".join(value.split())
    if len(text) > 200:
        raise ValueError("display_name must be at most 200 characters")
    return text or None
