"""
Anthropic ↔ OpenAI 格式双向转换适配器
负责：
  1. Anthropic Messages API 请求 → OpenAI Chat Completions 请求
  2. OpenAI Chat Completions 响应 → Anthropic Messages API 响应
  3. OpenAI SSE 流 → Anthropic SSE 流
  4. 错误格式转换
  5. 模型名称映射
"""

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from loguru import logger

from core.config import cfg


_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "stop_sequence",
}

_STOP_REASON_MAP = {v: k for k, v in _FINISH_REASON_MAP.items()}


def _generate_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def map_model_to_nvidia(anthropic_model: str) -> str:
    mapping = cfg.anthropic_model_mapping
    if anthropic_model in mapping:
        return mapping[anthropic_model]
    return anthropic_model


def map_model_to_anthropic(nvidia_model: str) -> str:
    mapping = cfg.anthropic_model_mapping
    reverse = {v: k for k, v in mapping.items()}
    if nvidia_model in reverse:
        return reverse[nvidia_model]
    return nvidia_model


def map_stop_reason(finish_reason: Optional[str]) -> str:
    if not finish_reason:
        return "end_turn"
    return _FINISH_REASON_MAP.get(finish_reason, "end_turn")


def map_finish_reason(stop_reason: Optional[str]) -> str:
    if not stop_reason:
        return "stop"
    return _STOP_REASON_MAP.get(stop_reason, "stop")


def convert_request(anthropic_req: Dict[str, Any]) -> Dict[str, Any]:
    messages = []

    system = anthropic_req.get("system")
    if system is not None:
        system_content = _convert_system_content(system)
        messages.append({"role": "system", "content": system_content})

    raw_messages = anthropic_req.get("messages", [])
    converted = _convert_messages(raw_messages)
    messages.extend(converted)

    openai_req: Dict[str, Any] = {
        "model": map_model_to_nvidia(anthropic_req.get("model", "")),
        "messages": messages,
        "max_tokens": anthropic_req.get("max_tokens", 4096),
    }

    if "temperature" in anthropic_req and anthropic_req["temperature"] is not None:
        openai_req["temperature"] = anthropic_req["temperature"]
    if "top_p" in anthropic_req and anthropic_req["top_p"] is not None:
        openai_req["top_p"] = anthropic_req["top_p"]
    if anthropic_req.get("stream") is not None:
        openai_req["stream"] = anthropic_req["stream"]
    if "stop_sequences" in anthropic_req and anthropic_req["stop_sequences"]:
        openai_req["stop"] = anthropic_req["stop_sequences"]

    tools = anthropic_req.get("tools")
    if tools:
        openai_tools = _convert_anthropic_tools(tools)
        if openai_tools:
            openai_req["tools"] = openai_tools

    extra = {}
    for key in ("metadata",):
        if key in anthropic_req:
            extra[key] = anthropic_req[key]

    return openai_req


def _convert_system_content(system: Union[str, List[Dict]]) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(system)


def _convert_messages(anthropic_messages: List[Dict]) -> List[Dict]:
    result = []
    for msg in anthropic_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            result.append({"role": "system", "content": _extract_text(content)})
            continue

        if isinstance(content, str):
            result.append({"role": role, "content": content})
        elif isinstance(content, list):
            has_tool_use = any(
                isinstance(b, dict) and b.get("type") == "tool_use"
                for b in content
            )
            has_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
            )

            if has_tool_use and role == "assistant":
                result.append(_convert_tool_use_message(content))
            elif has_tool_result and role == "user":
                result.append(_convert_tool_result_message(content))
            else:
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "image":
                            text_parts.append("[image]")
                        elif block.get("type") == "tool_result":
                            text_parts.append(_extract_tool_result_text(block))
                        elif block.get("type") == "thinking":
                            pass
                        else:
                            text_parts.append(json.dumps(block, ensure_ascii=False))
                    else:
                        text_parts.append(str(block))
                result.append({"role": role, "content": "\n".join(text_parts)})
        else:
            result.append({"role": role, "content": str(content)})

    result = _ensure_alternating_roles(result)
    return result


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, dict):
                parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _extract_tool_result_text(block: Dict) -> str:
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            else:
                parts.append(json.dumps(b, ensure_ascii=False))
        return "\n".join(parts)
    return str(content)


def _convert_tool_use_message(content: List[Dict]) -> Dict:
    text_parts = []
    tool_calls = []
    for block in content:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                },
            })
        elif block.get("type") == "thinking":
            pass

    msg: Dict[str, Any] = {"role": "assistant"}
    if text_parts:
        msg["content"] = "\n".join(text_parts)
    else:
        msg["content"] = None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _convert_tool_result_message(content: List[Dict]) -> Dict:
    tool_results = []
    for block in content:
        if block.get("type") == "tool_result":
            tool_id = block.get("tool_use_id", "")
            result_text = _extract_tool_result_text(block)
            tool_results.append({
                "tool_id": tool_id,
                "content": result_text,
            })

    if len(tool_results) == 1:
        return {
            "role": "tool",
            "tool_call_id": tool_results[0]["tool_id"],
            "content": tool_results[0]["content"],
        }
    elif len(tool_results) > 1:
        combined = []
        for tr in tool_results:
            combined.append(f"[tool_result:{tr['tool_id']}] {tr['content']}")
        return {
            "role": "tool",
            "tool_call_id": tool_results[0]["tool_id"],
            "content": "\n".join(combined),
        }
    return {"role": "user", "content": ""}


def _convert_anthropic_tools(tools: List[Dict]) -> List[Dict]:
    openai_tools = []
    for tool in tools:
        name = tool.get("name", "")
        description = tool.get("description", "")
        input_schema = tool.get("input_schema", {})
        openai_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_schema,
            },
        })
    return openai_tools


def _ensure_alternating_roles(messages: List[Dict]) -> List[Dict]:
    if not messages:
        return messages

    result = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            result.append(msg)
            continue

        if result:
            last_role = None
            for i in range(len(result) - 1, -1, -1):
                if result[i].get("role") != "system":
                    last_role = result[i].get("role")
                    break

            if last_role == role and role in ("user", "assistant"):
                last_msg = result[-1]
                if isinstance(last_msg.get("content"), str) and isinstance(msg.get("content"), str):
                    last_msg["content"] += "\n" + msg["content"]
                    continue

        result.append(msg)

    first_non_system = None
    for i, msg in enumerate(result):
        if msg.get("role") != "system":
            first_non_system = i
            break

    if first_non_system is not None and result[first_non_system].get("role") != "user":
        result.insert(first_non_system, {"role": "user", "content": "."})

    return result


def convert_response(openai_response: Dict[str, Any], original_model: str) -> Dict[str, Any]:
    choices = openai_response.get("choices", [])
    usage = openai_response.get("usage", {})

    content_blocks = []
    stop_reason = "end_turn"

    if choices:
        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")

        stop_reason = map_stop_reason(finish_reason)

        msg_content = message.get("content")
        tool_calls = message.get("tool_calls")

        if tool_calls:
            if msg_content:
                content_blocks.append({"type": "text", "text": msg_content})
            for tc in tool_calls:
                fn = tc.get("function", {})
                try:
                    input_data = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    input_data = {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                    "name": fn.get("name", ""),
                    "input": input_data,
                })
            stop_reason = "tool_use"
        elif msg_content:
            content_blocks.append({"type": "text", "text": msg_content})
        else:
            content_blocks.append({"type": "text", "text": ""})

    return {
        "id": openai_response.get("id", _generate_msg_id()),
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


async def convert_stream(
    openai_stream: AsyncGenerator[str, None],
    model: str,
) -> AsyncGenerator[str, None]:
    msg_id = _generate_msg_id()
    input_tokens = 0
    output_tokens = 0
    content_block_started = False
    content_block_index = 0
    message_started = False
    model_name = model
    stream_ended = False

    async for raw_chunk in openai_stream:
        if not raw_chunk or not raw_chunk.strip():
            continue

        if raw_chunk.strip() == "data: [DONE]":
            if not stream_ended:
                if content_block_started:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}}, ensure_ascii=False)}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            break

        if stream_ended:
            continue

        if not raw_chunk.startswith("data: "):
            continue

        data_str = raw_chunk[6:].strip()
        if not data_str:
            continue

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        chunk_model = chunk.get("model")
        if chunk_model:
            model_name = chunk_model

        if not message_started:
            if chunk.get("usage"):
                input_tokens = chunk["usage"].get("prompt_tokens", 0)
            message_started = True
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': input_tokens, 'output_tokens': 0}}}, ensure_ascii=False)}\n\n"

        choices = chunk.get("choices", [])
        if not choices:
            if chunk.get("usage"):
                u = chunk["usage"]
                input_tokens = u.get("prompt_tokens", input_tokens)
                output_tokens = u.get("completion_tokens", output_tokens)
            continue

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        tool_calls = delta.get("tool_calls")
        if tool_calls:
            if not content_block_started:
                content_block_started = True
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'tool_use', 'id': tool_calls[0].get('id', f'toolu_{uuid.uuid4().hex[:8]}'), 'name': tool_calls[0].get('function', {}).get('name', ''), 'input': {}}}, ensure_ascii=False)}\n\n"

            for tc in tool_calls:
                fn = tc.get("function", {})
                partial_args = fn.get("arguments", "")
                if partial_args:
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_index, 'delta': {'type': 'input_json_delta', 'partial_json': partial_args}}, ensure_ascii=False)}\n\n"
            continue

        delta_content = delta.get("content")

        if delta_content is not None:
            if not content_block_started:
                content_block_started = True
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'text', 'text': ''}}, ensure_ascii=False)}\n\n"

            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_index, 'delta': {'type': 'text_delta', 'text': delta_content}}, ensure_ascii=False)}\n\n"
            output_tokens += 1

        if finish_reason is not None:
            if content_block_started:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                content_block_started = False
                content_block_index += 1

            stop_reason = map_stop_reason(finish_reason)
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}}, ensure_ascii=False)}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            stream_ended = True

    if not message_started:
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}}, ensure_ascii=False)}\n\n"
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}}, ensure_ascii=False)}\n\n"
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': ''}}, ensure_ascii=False)}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}}, ensure_ascii=False)}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


def convert_error(status_code: int, detail: Any) -> Dict[str, Any]:
    error_type_map = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
        500: "api_error",
        503: "overloaded_error",
    }

    error_type = error_type_map.get(status_code, "api_error")

    message = ""
    if isinstance(detail, dict):
        if "error" in detail:
            err = detail["error"]
            if isinstance(err, dict):
                message = err.get("message", str(err))
            else:
                message = str(err)
        elif "detail" in detail:
            message = str(detail["detail"])
        else:
            message = str(detail)
    else:
        message = str(detail)

    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def convert_models_to_anthropic(openai_models: List[Dict[str, Any]]) -> Dict[str, Any]:
    mapping = cfg.anthropic_model_mapping
    reverse = {v: k for k, v in mapping.items()}

    data = []
    for m in openai_models:
        model_id = m.get("id", "")
        display_id = reverse.get(model_id, model_id)
        data.append({
            "id": display_id,
            "object": "model",
            "created": m.get("created", int(time.time())),
            "owned_by": m.get("owned_by", "anthropic"),
        })

    return {"object": "list", "data": data}
