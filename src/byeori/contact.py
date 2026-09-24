"""The contact address Byeori sends to OpenAlex, Crossref and NCBI, from configuration only."""
from __future__ import annotations

import os

ENV_NAME = "BYEORI_CONTACT_EMAIL"


def contact_email() -> str:
    """The configured address, or "" when none is set; never a built-in default."""
    return os.environ.get(ENV_NAME, "").strip()


def mailto_parameter() -> dict[str, str]:
    """`{"mailto": address}` for a query string, or {} when no address is configured."""
    address = contact_email()
    return {"mailto": address} if address else {}


def user_agent(product: str) -> str:
    """`product (mailto:address)` when configured, else just `product`."""
    address = contact_email()
    return f"{product} (mailto:{address})" if address else product
