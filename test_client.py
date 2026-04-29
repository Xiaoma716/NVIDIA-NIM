"""
测试脚本：验证本地代理是否正常工作（OpenAI + Anthropic 格式）
运行前请先启动 main.py
"""

import json
import time
import httpx
from openai import OpenAI

from core.config import cfg

cfg.load("config.yaml")

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

MODEL = cfg.default_model
ANTHROPIC_MODEL = cfg.anthropic_default_model
BASE_URL = "http://localhost:8000"


def test_basic():
    print("\n[测试1] OpenAI 基础对话...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "用一句话介绍你自己"}],
        max_tokens=100,
    )
    print(f"✅ 回复: {response.choices[0].message.content}")


def test_stream():
    print("\n[测试2] OpenAI 流式输出...")
    print("流式回复: ", end="", flush=True)
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "数一下1到5"}],
        max_tokens=50,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print("\n✅ 流式测试完成")


def test_concurrent():
    import concurrent.futures
    print("\n[测试3] 并发请求（5个）...")

    def single_request(i):
        start = time.time()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": f"请用一个词回答：{i}+{i}=？"}],
            max_tokens=20,
        )
        elapsed = round(time.time() - start, 2)
        return i, response.choices[0].message.content.strip(), elapsed

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(single_request, i) for i in range(1, 6)]
        for future in concurrent.futures.as_completed(futures):
            idx, content, elapsed = future.result()
            print(f"  请求{idx}: {content} | 耗时: {elapsed}s")

    print("✅ 并发测试完成")


def test_anthropic_basic():
    print(f"\n[测试4] Anthropic 非流式对话 (model={ANTHROPIC_MODEL})...")
    resp = httpx.post(
        f"{BASE_URL}/v1/messages",
        headers={
            "x-api-key": "not-needed",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "用一句话介绍你自己"}
            ],
        },
        timeout=60.0,
    )
    if resp.status_code != 200:
        print(f"❌ 请求失败: HTTP {resp.status_code} | {resp.text}")
        return
    data = resp.json()
    text_blocks = [b for b in data.get("content", []) if b.get("type") == "text"]
    text = text_blocks[0]["text"] if text_blocks else "(无文本内容)"
    print(f"✅ 回复: {text}")
    print(f"   stop_reason: {data.get('stop_reason')}")
    print(f"   usage: {data.get('usage')}")


def test_anthropic_stream():
    print(f"\n[测试5] Anthropic 流式对话 (model={ANTHROPIC_MODEL})...")
    print("流式回复: ", end="", flush=True)

    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/messages",
        headers={
            "x-api-key": "not-needed",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 100,
            "stream": True,
            "messages": [
                {"role": "user", "content": "数一下1到5"}
            ],
        },
        timeout=60.0,
    ) as resp:
        if resp.status_code != 200:
            print(f"❌ 请求失败: HTTP {resp.status_code}")
            return
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                continue
            if line.startswith("data:"):
                data_str = line[5:].strip()
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                event_type = data.get("type", "")
                if event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        print(delta.get("text", ""), end="", flush=True)
                elif event_type == "message_stop":
                    break
    print("\n✅ Anthropic 流式测试完成")


def test_anthropic_system():
    print(f"\n[测试6] Anthropic 带 system 参数...")
    resp = httpx.post(
        f"{BASE_URL}/v1/messages",
        headers={
            "x-api-key": "not-needed",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 50,
            "system": "你是一个海盗，请用海盗的语气说话。",
            "messages": [
                {"role": "user", "content": "你好"}
            ],
        },
        timeout=60.0,
    )
    if resp.status_code != 200:
        print(f"❌ 请求失败: HTTP {resp.status_code} | {resp.text}")
        return
    data = resp.json()
    text_blocks = [b for b in data.get("content", []) if b.get("type") == "text"]
    text = text_blocks[0]["text"] if text_blocks else "(无文本内容)"
    print(f"✅ 回复: {text}")


if __name__ == "__main__":
    print("=" * 55)
    print(f"  NVIDIA NIM Load Balancer 测试 | 模型: {MODEL}")
    print(f"  Anthropic 模型: {ANTHROPIC_MODEL}")
    print("=" * 55)
    try:
        test_basic()
        test_stream()
        test_concurrent()
        test_anthropic_basic()
        test_anthropic_stream()
        test_anthropic_system()
        print("\n🎉 所有测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("请确认 main.py 已启动，且 config.yaml 中的Key有效")