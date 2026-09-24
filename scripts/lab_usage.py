"""Print the lab's monthly Byeori spend per member.

    LAB_FUNCTION_URL=https://<id>.lambda-url.<region>.on.aws/ AWS_PROFILE=<admin> \
        uv run python scripts/lab_usage.py [--period 2026-09] [--member <member_id>] [--json]

The gateway reads the budget ledger the answer and research workers already write; this script only
formats it. Amounts are the application's estimate from its own price table, not an AWS invoice, and
Jev triage calls are not settled into the ledger. No limit is enforced anywhere: this is accounting.
"""
from __future__ import annotations

import argparse
import json
import sys

from byeori import lab_mcp_server as client


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lab_usage", description="Monthly Byeori spend per member.")
    parser.add_argument("--period", default=None, help="YYYY-MM; defaults to the current month in AWS")
    parser.add_argument("--member", default=None, help="limit the report to one member id")
    parser.add_argument("--json", action="store_true", help="print the raw response")
    args = parser.parse_args(argv)

    body: dict[str, object] = {}
    if args.period:
        body["period"] = args.period
    if args.member:
        body["member_id"] = args.member
    try:
        result = client.client().call("usage_report", body)
    except client.LabConfigError as exc:
        print(f"lab_usage: {exc}", file=sys.stderr)
        return 2
    if not result.get("ok"):
        print(f"lab_usage: {result.get('error')}: {result.get('message')}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    rows = result.get("members") or []
    print(f"{result['period']} 벼리 사용량 (추정치, 청구서 아님)")
    print(f"{'member':<20}{'role':<10}{'$':>10}{'answers':>10}{'research':>10}")
    for row in rows:
        print(f"{row['member_id']:<20}{str(row.get('role') or ''):<10}{row['usd']:>10.2f}"
              f"{row.get('answers', 0):>10}{row.get('research', 0):>10}")
    lab = result.get("lab") or {}
    print(f"{'lab total':<30}{lab.get('usd', 0):>10.2f}")
    unattributed = result.get("unattributed_micros")
    if unattributed:
        print(f"(등록이 해제된 멤버 몫 {unattributed / 1e6:.2f} 포함)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
