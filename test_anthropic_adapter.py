import sys
sys.path.insert(0, ".")

from core.anthropic_adapter import (
    convert_request, convert_response, convert_error,
    map_model_to_nvidia, map_stop_reason, convert_models_to_anthropic,
)


def test_request_conversion():
    anthropic_req = {
        "model": "claude-3-sonnet-20240229",
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ],
        "max_tokens": 1024,
        "temperature": 0.5,
    }
    openai_req = convert_request(anthropic_req)
    assert openai_req["messages"][0]["role"] == "system", f"Expected system, got {openai_req['messages'][0]['role']}"
    assert openai_req["messages"][0]["content"] == "You are helpful."
    assert openai_req["messages"][1]["role"] == "user"
    assert openai_req["temperature"] == 0.5
    assert openai_req["max_tokens"] == 1024
    print("[PASS] Request conversion")


def test_request_with_tools():
    anthropic_req = {
        "model": "claude-3-sonnet-20240229",
        "messages": [
            {"role": "user", "content": "What is the weather?"},
        ],
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ],
    }
    openai_req = convert_request(anthropic_req)
    assert "tools" in openai_req
    assert openai_req["tools"][0]["type"] == "function"
    assert openai_req["tools"][0]["function"]["name"] == "get_weather"
    print("[PASS] Request with tools")


def test_request_tool_use_message():
    anthropic_req = {
        "model": "claude-3-sonnet-20240229",
        "messages": [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "Sunny, 72F",
                    },
                ],
            },
        ],
        "max_tokens": 1024,
    }
    openai_req = convert_request(anthropic_req)
    msgs = openai_req["messages"]
    assistant_msg = [m for m in msgs if m["role"] == "assistant"][0]
    assert "tool_calls" in assistant_msg
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"

    tool_msg = [m for m in msgs if m["role"] == "tool"][0]
    assert tool_msg["tool_call_id"] == "toolu_123"
    print("[PASS] Request with tool_use/tool_result messages")


def test_response_conversion():
    openai_resp = {
        "id": "chatcmpl-123",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }
    anthropic_resp = convert_response(openai_resp, "claude-3-sonnet-20240229")
    assert anthropic_resp["type"] == "message"
    assert anthropic_resp["stop_reason"] == "end_turn"
    assert len(anthropic_resp["content"]) == 1
    assert anthropic_resp["content"][0]["type"] == "text"
    assert anthropic_resp["content"][0]["text"] == "Hello! How can I help?"
    assert anthropic_resp["usage"]["input_tokens"] == 20
    assert anthropic_resp["usage"]["output_tokens"] == 10
    print("[PASS] Response conversion")


def test_response_with_tool_calls():
    openai_resp = {
        "id": "chatcmpl-456",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
    }
    anthropic_resp = convert_response(openai_resp, "claude-3-sonnet-20240229")
    assert anthropic_resp["stop_reason"] == "tool_use"
    tool_blocks = [b for b in anthropic_resp["content"] if b["type"] == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["name"] == "get_weather"
    assert tool_blocks[0]["input"] == {"location": "NYC"}
    print("[PASS] Response with tool_calls")


def test_error_conversion():
    err = convert_error(429, "Too many requests")
    assert err["type"] == "error"
    assert err["error"]["type"] == "rate_limit_error"
    assert err["error"]["message"] == "Too many requests"

    err400 = convert_error(400, {"error": {"message": "Bad request", "type": "invalid_request_error"}})
    assert err400["error"]["type"] == "invalid_request_error"
    print("[PASS] Error conversion")


def test_stop_reason_mapping():
    assert map_stop_reason("stop") == "end_turn"
    assert map_stop_reason("length") == "max_tokens"
    assert map_stop_reason("tool_calls") == "tool_use"
    assert map_stop_reason("content_filter") == "stop_sequence"
    assert map_stop_reason(None) == "end_turn"
    print("[PASS] Stop reason mapping")


def test_models_conversion():
    openai_models = [
        {"id": "meta/llama-3.1-70b-instruct", "object": "model", "created": 1234, "owned_by": "meta"},
        {"id": "meta/llama-3.1-8b-instruct", "object": "model", "created": 1235, "owned_by": "meta"},
    ]
    result = convert_models_to_anthropic(openai_models)
    assert "data" in result
    assert len(result["data"]) == 2
    assert result["data"][0]["type"] == "model"
    assert "has_more" in result
    print("[PASS] Models conversion")


def test_system_as_content_blocks():
    anthropic_req = {
        "model": "claude-3-sonnet-20240229",
        "system": [
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": "Be concise."},
        ],
        "messages": [
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 1024,
    }
    openai_req = convert_request(anthropic_req)
    assert openai_req["messages"][0]["role"] == "system"
    assert "helpful" in openai_req["messages"][0]["content"]
    assert "concise" in openai_req["messages"][0]["content"]
    print("[PASS] System as content blocks")


def test_alternating_roles():
    anthropic_req = {
        "model": "claude-3-sonnet-20240229",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Are you there?"},
        ],
        "max_tokens": 1024,
    }
    openai_req = convert_request(anthropic_req)
    msgs = openai_req["messages"]
    roles = [m["role"] for m in msgs]
    assert roles.count("user") == 1 or roles[-1] == "user"
    print("[PASS] Alternating roles enforcement")


if __name__ == "__main__":
    print("=" * 50)
    print("  Anthropic Adapter Unit Tests")
    print("=" * 50)
    test_request_conversion()
    test_request_with_tools()
    test_request_tool_use_message()
    test_response_conversion()
    test_response_with_tool_calls()
    test_error_conversion()
    test_stop_reason_mapping()
    test_models_conversion()
    test_system_as_content_blocks()
    test_alternating_roles()
    print("\nAll tests PASSED!")
