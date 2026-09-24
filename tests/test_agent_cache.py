from copy import deepcopy

from byeori.agent_cache import cached_request


def test_cache_points_preserve_inference_options_and_signed_tool_history():
    request = {"modelId": "global.anthropic.claude-opus-5", "system": [{"text": "Stable instructions"}],
               "toolConfig": {"tools": [{"toolSpec": {"name": "read", "inputSchema": {"json": {"type": "object"}}}}],
                              "toolChoice": {"auto": {}}},
               "inferenceConfig": {"maxTokens": 32000},
               "additionalModelRequestFields": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
               "messages": [{"role": "user", "content": [{"text": "Question"}]},
                            {"role": "assistant", "content": [
                                {"reasoningContent": {"reasoningText": {"text": "Reasoning", "signature": "unaltered-signature"}}},
                                {"toolUse": {"toolUseId": "one", "name": "read", "input": {"key": "wiki/sources/a.md"}}}]},
                            {"role": "user", "content": [{"toolResult": {"toolUseId": "one", "content": [{"text": "Original evidence"}]}}]}]}
    original = deepcopy(request)
    cached = cached_request(request)
    assert request == original
    assert cached["inferenceConfig"] == original["inferenceConfig"]
    assert cached["additionalModelRequestFields"] == original["additionalModelRequestFields"]
    assert cached["toolConfig"]["toolChoice"] == original["toolConfig"]["toolChoice"]
    assert cached["messages"][1] == original["messages"][1]
    assert cached["messages"][2]["content"][0] == original["messages"][2]["content"][0]
    assert cached["system"][-1] == {"cachePoint": {"type": "default"}}
    assert cached["toolConfig"]["tools"][-1] == {"cachePoint": {"type": "default"}}
    assert cached["messages"][0]["content"][-1] == {"cachePoint": {"type": "default"}}
    assert cached["messages"][2]["content"][-1] == {"cachePoint": {"type": "default"}}
    cached["messages"][2]["content"][0]["toolResult"]["content"][0]["text"] = "Changed copy"
    assert request == original


def test_long_history_repositions_old_checkpoints_and_caches_only_latest_two_user_turns():
    point = {"cachePoint": {"type": "default"}}
    request = {"system": [{"text": "Stable"}, point],
               "toolConfig": {"tools": [{"toolSpec": {"name": "read"}}, point]},
               "messages": [{"role": "user", "content": [{"text": str(index)}, point]} for index in range(10)]}
    cached = cached_request(request)
    assert all(message["content"] == [{"text": str(index)}] for index, message in enumerate(cached["messages"][:8]))
    assert cached["messages"][-2]["content"] == [{"text": "8"}, point]
    assert cached["messages"][-1]["content"] == [{"text": "9"}, point]
    all_blocks = cached["system"] + cached["toolConfig"]["tools"] + [block for message in cached["messages"] for block in message["content"]]
    assert sum("cachePoint" in block for block in all_blocks) == 4
    assert cached_request(cached) == cached


def test_minimal_request_does_not_gain_system_or_tool_configuration():
    request = {"messages": [{"role": "user", "content": [{"text": "Question"}]}]}
    cached = cached_request(request)
    assert set(cached) == {"messages"}
    assert cached["messages"][0]["content"] == [{"text": "Question"}, {"cachePoint": {"type": "default"}}]
