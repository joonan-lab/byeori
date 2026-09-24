"""Place bounded Converse cache checkpoints without changing conversation content.

AWS processes the prefix in tools, system, messages order and allows four Claude
checkpoints: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
"""
from __future__ import annotations

from copy import deepcopy


def cached_request(request: dict) -> dict:
    """Copy a request and cache stable instructions plus the latest two user turns."""
    result = deepcopy(request)
    if "system" in result:
        result["system"] = [block for block in result["system"] if "cachePoint" not in block]
        if result["system"]:
            result["system"].append({"cachePoint": {"type": "default"}})
    if "toolConfig" in result and "tools" in result["toolConfig"]:
        tools = [block for block in result["toolConfig"]["tools"] if "cachePoint" not in block]
        result["toolConfig"]["tools"] = tools
        if tools:
            tools.append({"cachePoint": {"type": "default"}})
    user_turns = []
    for message in result.get("messages", []):
        message["content"] = [block for block in message["content"] if "cachePoint" not in block]
        if message["role"] == "user" and message["content"]:
            user_turns.append(message)
    for message in user_turns[-2:]:
        message["content"].append({"cachePoint": {"type": "default"}})
    return result
