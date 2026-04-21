"""
测试脚本：验证本地代理是否正常工作
运行前请先启动 main.py
"""

import time
from openai import OpenAI

from core.config import cfg

cfg.load("config.yaml")

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

MODEL = cfg.default_model


def test_basic():
    """测试基础对话"""
    print("\n[测试1] 基础对话...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "用一句话介绍你自己"}],
        max_tokens=100,
    )
    print(f"✅ 回复: {response.choices[0].message.content}")


def test_stream():
    """测试流式输出"""
    print("\n[测试2] 流式输出...")
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
    """简单并发测试"""
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


if __name__ == "__main__":
    print("=" * 50)
    print(f"  NVIDIA NIM Load Balancer 测试 | 模型: {MODEL}")
    print("=" * 50)
    try:
        test_basic()
        test_stream()
        test_concurrent()
        print("\n🎉 所有测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("请确认 main.py 已启动，且 config.yaml 中的Key有效")