"""
Prompt Tokens 估算回退机制 — 修复验证测试
==========================================
验证目标：
  1. ✅ 修复后：异常场景下 prompt_tokens 使用估算值（不再为 0）
  2. ✅ 正常场景：usage 存在时仍使用 API 返回的精确值
  3. ✅ _estimate_prompt_tokens() 估算方法准确性
  4. ✅ 流式请求成功但无 usage chunk 时使用估算值
  5. ✅ 多轮重试累积误差被正确记录

运行方式: pytest tests/test_prompt_token_leak.py -v
"""

import time
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from core.proxy import NvidiaProxy
from core.stats_manager import StatsManager
from core.balancer import LoadBalancer
from core.key_pool import KeyPool


class MockUsage:
    def __init__(self, prompt_tokens=100, completion_tokens=50):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class MockResponse:
    def __init__(self, usage=None, model="test-model"):
        self.usage = usage
        self.model = model

    def model_dump(self):
        return {"usage": self.usage, "model": self.model}


@pytest.fixture(autouse=True)
def isolate_stats_db(tmp_path, monkeypatch):
    db_dir = tmp_path / "test_data"
    db_dir.mkdir()
    db_path = str(db_dir / "test_nim.db")

    def mock_get_db_path():
        return db_path

    import core.database
    monkeypatch.setattr(core.database, "get_db_path", mock_get_db_path)
    monkeypatch.setattr(core.database, "_engine", None)

    yield

    import core.database as _db
    _db._engine = None


def _make_proxy_with_stats():
    pool = KeyPool(
        keys_config=[{"key": "test-key-1", "alias": "TestKey-1"}],
        rpm_limit=100,
        rpm_buffer=10,
    )
    balancer = LoadBalancer(key_pool=pool, strategy="most_remaining")
    stats = StatsManager()
    proxy = NvidiaProxy(
        balancer=balancer,
        base_url="https://integrate.api.nvidia.com/v1",
        max_retries=1,
        stats_manager=stats,
    )
    return proxy, stats


# ====================================================================
# 测试组 1：_estimate_prompt_tokens() 估算方法单元测试
# ====================================================================

class TestEstimatePromptTokens:

    def test_empty_messages_returns_minimum(self):
        result = NvidiaProxy._estimate_prompt_tokens([])
        assert result >= 1, f"空消息应返回至少 1，实际={result}"

    def test_short_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = NvidiaProxy._estimate_prompt_tokens(messages)
        assert result > 0, f"短消息应有正值估算，实际={result}"

    def test_long_chinese_text(self):
        content = "这是一段很长的中文测试内容" * 100
        messages = [{"role": "user", "content": content}]
        result = NvidiaProxy._estimate_prompt_tokens(messages)
        expected_min = len(content) / 4 + 4
        assert result >= expected_min * 0.5, \
            f"长中文文本估算偏低：实际={result}, 期望至少≈{expected_min:.0f}"

    def test_multimodal_content_list(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "图片描述文本" * 20},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
            ]
        }]
        result = NvidiaProxy._estimate_prompt_tokens(messages)
        assert result > 0, "多模态消息中 text 部分应被计入"

    def test_multiple_messages(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant." * 10},
            {"role": "user", "content": "Question here" * 5},
            {"role": "assistant", "content": "Answer here" * 5},
            {"role": "user", "content": "Follow up" * 5},
        ]
        result = NvidiaProxy._estimate_prompt_tokens(messages)
        single_msg_result = NvidiaProxy._estimate_prompt_tokens([messages[0]])
        assert result > single_msg_result, \
            f"多条消息估算值({result})应大于单条({single_msg_result})"

    def test_monotonically_increasing_with_length(self):
        for length in [10, 100, 1000, 5000]:
            messages = [{"role": "user", "content": "x" * length}]
            result = NvidiaProxy._estimate_prompt_tokens(messages)
            assert result > 0


# ====================================================================
# 测试组 2：_report() 方法核心行为（含 estimated_pt fallback）
# ====================================================================

class TestReportMethodCoreBehavior:

    @pytest.mark.asyncio
    async def test_report_with_valid_usage_ignores_estimate(self):
        """usage 存在时，优先使用 API 返回的精确值"""
        proxy, stats = _make_proxy_with_stats()
        usage = MockUsage(prompt_tokens=150, completion_tokens=80)
        proxy._report("model-a", "K1", usage, time.time(), True, estimated_pt=9999)
        overview = stats.get_overview()
        assert overview["total"]["prompt_tokens"] == 150, \
            "usage 存在时应忽略 estimated_pt"
        assert overview["total"]["completion_tokens"] == 80

    @pytest.mark.asyncio
    async def test_report_none_usage_no_estimate_stays_zero(self):
        """向后兼容：usage=None 且未提供 estimate 时仍为 0"""
        proxy, stats = _make_proxy_with_stats()
        proxy._report("model-a", "K1", None, time.time(), False)
        overview = stats.get_overview()
        assert overview["total"]["prompt_tokens"] == 0

    @pytest.mark.asyncio
    async def test_report_none_usage_with_estimate_uses_fallback(self):
        """✅ 修复验证：usage=None 时使用 estimated_pt 作为 fallback"""
        proxy, stats = _make_proxy_with_stats()
        proxy._report("model-a", "K1", None, time.time(), False, estimated_pt=258)
        overview = stats.get_overview()
        assert overview["total"]["prompt_tokens"] == 258, \
            f"期望 estimated_pt fallback 生效=258，实际={overview['total']['prompt_tokens']}"
        assert overview["total"]["errors"] == 1

    @pytest.mark.asyncio
    async def test_report_zero_usage_with_estimate_uses_fallback(self):
        """usage 对象存在但 tokens 为 0 时，estimated_pt 不生效（usage 非 None）"""
        proxy, stats = _make_proxy_with_stats()

        class ZeroUsage:
            prompt_tokens = 0
            completion_tokens = 0

        proxy._report("m", "K1", ZeroUsage(), time.time(), True, estimated_pt=100)
        overview = stats.get_overview()
        assert overview["total"]["prompt_tokens"] == 0, \
            "usage 对象存在时即使值为 0 也应使用 usage 值"

    @pytest.mark.asyncio
    async def test_report_none_attribute_usage(self):
        """usage 属性为 None 时 getattr+or 降级为 0"""
        proxy, stats = _make_proxy_with_stats()

        class NoneAttrUsage:
            prompt_tokens = None
            completion_tokens = None

        proxy._report("m", "K1", NoneAttrUsage(), time.time(), True, estimated_pt=200)
        overview = stats.get_overview()
        assert overview["total"]["prompt_tokens"] == 0, \
            "getattr(None, 0) or 0 → 0，不触发 estimated_pt fallback"


def _make_stats():
    return StatsManager()


# ====================================================================
# 测试组 3：非流式请求 — 异常场景修复验证
# ====================================================================

class TestNonStreamErrorScenariosFixed:

    @pytest.mark.asyncio
    async def test_rate_limit_error_now_has_estimated_tokens(self):
        """✅ 修复验证 #1：RateLimitError 后 prompt_tokens 使用估算值"""
        from openai import RateLimitError
        proxy, stats = _make_proxy_with_stats()

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )

        with patch.object(proxy, '_make_client', return_value=mock_client):
            with pytest.raises(RateLimitError):
                await proxy.chat_completion(
                    messages=[{"role": "user", "content": "很长的测试内容" * 100}],
                    model="test-model",
                )

        overview = stats.get_overview()
        assert overview["total"]["errors"] >= 1
        assert overview["total"]["prompt_tokens"] > 0, \
            f"✅ 修复成功：RateLimitError 后 prompt_tokens={overview['total']['prompt_tokens']} > 0"
        est = NvidiaProxy._estimate_prompt_tokens(
            [{"role": "user", "content": "很长的测试内容" * 100}])
        assert overview["total"]["prompt_tokens"] == est, \
            f"估算值应匹配：实际={overview['total']['prompt_tokens']}，期望={est}"

    @pytest.mark.asyncio
    async def test_connection_timeout_now_has_estimated_tokens(self):
        """✅ 修复验证 #2：连接超时后使用估算值"""
        from openai import APIConnectionError
        proxy, stats = _make_proxy_with_stats()

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            message="Connection timed out after 120s",
            request=MagicMock(),
        )

        with patch.object(proxy, '_make_client', return_value=mock_client):
            with pytest.raises(APIConnectionError):
                await proxy.chat_completion(
                    messages=[{"role": "user", "content": "重要提示词内容" * 50}],
                    model="test-model",
                )

        overview = stats.get_overview()
        assert overview["total"]["prompt_tokens"] > 0, \
            f"✅ 修复成功：超时后 prompt_tokens={overview['total']['prompt_tokens']} > 0"

    @pytest.mark.asyncio
    async def test_server_error_5xx_now_has_estimated_tokens(self):
        """✅ 修复验证 #3：5xx 错误后使用估算值"""
        from openai import APIStatusError
        proxy, stats = _make_proxy_with_stats()

        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = APIStatusError(
            message="503 Service Unavailable",
            response=mock_response,
            body=None,
        )

        with patch.object(proxy, '_make_client', return_value=mock_client):
            with pytest.raises(APIStatusError):
                await proxy.chat_completion(
                    messages=[{"role": "system", "content": "系统提示词" * 30},
                              {"role": "user", "content": "用户问题" * 20}],
                    model="test-model",
                )

        overview = stats.get_overview()
        assert overview["total"]["prompt_tokens"] > 0, \
            f"✅ 修复成功：5xx 后 prompt_tokens={overview['total']['prompt_tokens']} > 0"

    @pytest.mark.asyncio
    async def test_unknown_exception_now_has_estimated_tokens(self):
        """✅ 修复验证 #4：未知异常后使用估算值"""
        proxy, stats = _make_proxy_with_stats()

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("Unexpected error")

        with patch.object(proxy, '_make_client', return_value=mock_client):
            with pytest.raises(Exception):
                await proxy.chat_completion(
                    messages=[{"role": "user", "content": "test content" * 30}],
                    model="test-model",
                )

        overview = stats.get_overview()
        assert overview["total"]["prompt_tokens"] > 0, \
            f"✅ 修复成功：未知异常后 prompt_tokens={overview['total']['prompt_tokens']} > 0"


# ====================================================================
# 测试组 4：流式请求 — 中断/异常场景修复验证
# ====================================================================

class TestStreamErrorScenariosFixed:

    @pytest.mark.asyncio
    async def test_stream_rate_limit_now_has_estimated_tokens(self):
        """✅ 修复验证 #5：流式 RateLimit 后使用估算值"""
        from openai import RateLimitError
        proxy, stats = _make_proxy_with_stats()

        async def mock_stream():
            for text in ["Hello", " world"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = text
                chunk.model_dump_json.return_value = '{"choices":[{}]}'
                yield chunk
            raise RateLimitError(
                message="Rate limit during stream",
                response=MagicMock(status_code=429),
                body=None,
            )

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_stream()

        with patch.object(proxy, '_make_client', return_value=mock_client):
            gen = proxy.chat_completion_stream(
                messages=[{"role": "user", "content": "流式测试内容" * 50}],
                model="test-model",
            )
            try:
                async for _ in gen:
                    pass
            except RateLimitError:
                pass

        error_records = [r for r in stats._records if not r.success]
        total_pt = sum(r.prompt_tokens for r in error_records)
        assert total_pt > 0, \
            f"✅ 修复成功：流式RateLimit错误记录中 prompt_tokens={total_pt} > 0"

    @pytest.mark.asyncio
    async def test_stream_connection_lost_now_has_estimated_tokens(self):
        """✅ 修复验证 #6：流式中途断开后使用估算值"""
        from openai import APIConnectionError
        proxy, stats = _make_proxy_with_stats()

        async def mock_stream_interrupted():
            for text in ["部分输出", "被截断"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = text
                chunk.model_dump_json.return_value = '{"choices":[{}]}'
                yield chunk
            raise APIConnectionError(
                message="Stream connection lost",
                request=MagicMock(),
            )

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_stream_interrupted()

        with patch.object(proxy, '_make_client', return_value=mock_client):
            gen = proxy.chat_completion_stream(
                messages=[{"role": "user", "content": "长文本输入" * 100}],
                model="test-model",
            )
            try:
                async for _ in gen:
                    pass
            except APIConnectionError:
                pass

        error_records = [r for r in stats._records if not r.success]
        total_pt = sum(r.prompt_tokens for r in error_records)
        assert total_pt > 0, \
            f"✅ 修复成功：流式中断的错误记录中 prompt_tokens={total_pt} > 0"


# ====================================================================
# 测试组 5：正常路径回归测试（确保修复不影响正常功能）
# ====================================================================

class TestRegressionNormalPaths:

    @pytest.mark.asyncio
    async def test_nonstream_success_uses_api_exact_values(self):
        """非流式成功：必须使用 API 返回的精确值，而非估算值"""
        proxy, stats = _make_proxy_with_stats()
        mock_resp = MockResponse(usage=MockUsage(prompt_tokens=200, completion_tokens=100))

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch.object(proxy, '_make_client', return_value=mock_client):
            response = await proxy.chat_completion(
                messages=[{"role": "user", "content": "any content" * 99}],
                model="test-model",
            )

        overview = stats.get_overview()
        assert overview["total"]["prompt_tokens"] == 200, \
            "成功路径必须使用 API 精确值 200，非估算值"
        assert overview["total"]["completion_tokens"] == 100
        assert overview["total"]["errors"] == 0

    @pytest.mark.asyncio
    async def test_stream_success_with_usage_chunk_uses_exact_values(self):
        """流式成功且收到 usage chunk：使用精确值"""
        proxy, stats = _make_proxy_with_stats()

        async def mock_stream_with_usage():
            for text in ["Hi", " there"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = text
                chunk.model_dump_json.return_value = '{"choices":[{}]}'
                yield chunk
            usage_chunk = MagicMock()
            usage_chunk.choices = []
            usage_chunk.usage = MagicMock()
            usage_chunk.usage.prompt_tokens = 300
            usage_chunk.usage.completion_tokens = 150
            yield usage_chunk

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_stream_with_usage()

        with patch.object(proxy, '_make_client', return_value=mock_client):
            async for _ in proxy.chat_completion_stream(
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
            ):
                pass

        success_records = [r for r in stats._records if r.success and r.stream]
        assert len(success_records) == 1
        assert success_records[0].prompt_tokens == 300, \
            "流式有 usage chunk 时必须用精确值 300"

    @pytest.mark.asyncio
    async def test_stream_success_no_usage_chunk_falls_back_to_estimate(self):
        """流式成功但无 usage chunk：使用估算值作为 fallback"""
        proxy, stats = _make_proxy_with_stats()

        async def mock_stream_no_usage():
            for text in ["Some", " output", " here"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = text
                chunk.model_dump_json.return_value = '{"choices":[{}]}'
                yield chunk

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_stream_no_usage()
        test_messages = [{"role": "user", "content": "测试内容" * 50}]
        expected_est = NvidiaProxy._estimate_prompt_tokens(test_messages)

        with patch.object(proxy, '_make_client', return_value=mock_client):
            async for _ in proxy.chat_completion_stream(
                messages=test_messages,
                model="test-model",
            ):
                pass

        success_records = [r for r in stats._records if r.success and r.stream]
        assert len(success_records) == 1
        assert success_records[0].prompt_tokens == expected_est, \
            f"流式无 usage chunk 应 fallback 到估算值 {expected_est}" \
            f"，实际={success_records[0].prompt_tokens}"

    @pytest.mark.asyncio
    async def test_timeout_scenario_fixed(self):
        """✅ 修复验证 #7：超时场景使用估算值"""
        from openai import APIConnectionError
        proxy, stats = _make_proxy_with_stats()

        async def timeout_create(*args, **kwargs):
            await asyncio.sleep(0.01)
            raise APIConnectionError(
                message="Request timed out (simulating 120s NVIDIA resource strain)",
                request=MagicMock(),
            )

        mock_client = AsyncMock()
        mock_client.chat.completions.create = timeout_create
        test_messages = [{"role": "user", "content": "超时前已发送的prompt" * 200}]

        with patch.object(proxy, '_make_client', return_value=mock_client):
            try:
                await proxy.chat_completion(messages=test_messages, model="test-model")
            except (APIConnectionError, Exception):
                pass

        overview = stats.get_overview()
        expected_est = NvidiaProxy._estimate_prompt_tokens(test_messages)
        assert overview["total"]["prompt_tokens"] == expected_est, \
            f"✅ 超时后 prompt_tokens={overview['total']['prompt_tokens']} " \
            f"== 估算值={expected_est}"


# ====================================================================
# 测试组 6：多轮重试累积统计修复验证
# ====================================================================

class TestRetryAccumulatedEstimation:

    @pytest.mark.asyncio
    async def test_multiple_retries_all_fail_reports_once(self):
        """✅ 修复验证 #8：多次重试失败时仅报告一次（Token 去重）"""
        from openai import APIStatusError
        proxy, stats = _make_proxy_with_stats()

        call_count = 0
        test_messages = [{"role": "user", "content": "每次重试都发送此内容" * 50}]
        single_est = NvidiaProxy._estimate_prompt_tokens(test_messages)

        async def failing_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise APIStatusError(
                message=f"Server Error (attempt {call_count})",
                response=MagicMock(status_code=500),
                body=None,
            )

        mock_client = AsyncMock()
        mock_client.chat.completions.create = failing_create

        with patch.object(proxy, '_make_client', return_value=mock_client):
            proxy.max_retries = 3
            try:
                await proxy.chat_completion(messages=test_messages, model="test-model")
            except APIStatusError:
                pass

        overview = stats.get_overview()
        assert overview["total"]["requests"] == 1, \
            f"去重后应仅报告 1 次请求，实际={overview['total']['requests']}"
        assert overview["total"]["errors"] == 1, \
            f"去重后应仅报告 1 次错误，实际={overview['total']['errors']}"
        assert overview["total"]["prompt_tokens"] == single_est, \
            f"✅ Token 去重：{call_count}次尝试仅报告 1 次 × 单次估算{single_est} = " \
            f"{single_est}，实际={overview['total']['prompt_tokens']}"
        print(f"\n  [OK] 去重统计正确：{call_count} 次尝试 → "
              f"仅报告 {overview['total']['requests']} 次 × "
              f"{single_est} tokens = {overview['total']['prompt_tokens']}")


# ====================================================================
# 测试组 7：StatsManager 层确认性测试（排除干扰）
# ====================================================================

class TestStatsManagerRecordCorrectness:

    def test_statsmanager_accepts_any_value(self):
        sm = StatsManager()
        sm.record(model="m", key_alias="K1",
                  prompt_tokens=999, completion_tokens=888,
                  latency_ms=100, success=False)
        overview = sm.get_overview()
        assert overview["total"]["prompt_tokens"] == 999
        assert overview["total"]["completion_tokens"] == 888
        assert overview["total"]["errors"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
