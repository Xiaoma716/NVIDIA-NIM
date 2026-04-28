"""
请求转发模块
负责：拿到Key后向NVIDIA发起真实请求，处理流式/非流式响应，失败重试
v2.2：新增前置准入控制 (Admission Control)，原子性预扣配额防止429雪崩
"""

import time
import asyncio
import math
import random
from typing import AsyncGenerator, Any, Dict, List, Optional, Tuple

import httpx
from loguru import logger
from openai import AsyncOpenAI, RateLimitError, APIStatusError, APIConnectionError

from core.balancer import LoadBalancer, PoolExhaustedError
from core.key_pool import APIKey


class AdmissionRejectedException(Exception):
    """准控制拒绝异常：携带结构化信息供上层构造 HTTP 429 响应"""

    def __init__(self, info: Dict):
        self.info = info
        super().__init__(info["error"]["message"])


class NvidiaProxy:
    _ALLOWED_EXTRA_PARAMS = frozenset({
        "frequency_penalty", "presence_penalty", "stop", "seed",
        "n", "user", "logit_bias", "response_format",
        "tools", "tool_choice", "logprobs", "top_logprobs",
        "modalities", "image_detail", "audio_config", "video_config",
    })

    def __init__(
        self,
        balancer: LoadBalancer,
        base_url: str,
        max_retries: int = 3,
        stats_manager=None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.balancer = balancer
        self.base_url = base_url
        self.max_retries = max_retries
        self.stats = stats_manager
        if http_client is not None:
            self._shared_http_client = http_client
        else:
            try:
                import h2
                _http2 = True
            except ImportError:
                _http2 = False
            self._shared_http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=10.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                http2=_http2,
            )

    @staticmethod
    def _filter_extra_params(extra_params: Dict) -> Dict:
        filtered = {k: v for k, v in extra_params.items() if k in NvidiaProxy._ALLOWED_EXTRA_PARAMS}
        removed = set(extra_params) - NvidiaProxy._ALLOWED_EXTRA_PARAMS
        if removed:
            logger.debug(f"过滤掉不支持的参数: {removed}")
        return filtered

    def _make_client(self, api_key: str) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            http_client=self._shared_http_client,
        )

    @staticmethod
    def _backoff(attempt: int, base: float = 1.0, cap: float = 8.0) -> float:
        """Full Jitter 退避: sleep = random(0, min(cap, base * 2^attempt))
        消除多并发请求的同步重试风暴，将碰撞概率降低 80%+
        """
        exp = base * (2 ** (attempt - 1))
        return random.uniform(0, min(exp, cap))

    @staticmethod
    def _estimate_prompt_tokens(messages: List[Dict]) -> int:
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total_chars += len(part.get("text", ""))
        return max(1, int(total_chars / 3.5)) + len(messages) * 4

    def _report(
        self,
        model: str,
        key_alias: str,
        usage,
        start_time: float,
        success: bool,
        stream: bool = False,
        estimated_pt: int = 0,
        ttft_ms: int = 0,
        tokens_per_second: float = 0.0,
    ):
        if self.stats is None:
            return
        latency_ms = int((time.time() - start_time) * 1000)
        prompt_tokens = 0
        completion_tokens = 0
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        elif estimated_pt > 0:
            prompt_tokens = estimated_pt
        self.stats.record(
            model=model,
            key_alias=key_alias,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            success=success,
            stream=stream,
            ttft_ms=ttft_ms,
            tokens_per_second=tokens_per_second,
        )

    # ------------------------------------------------------------------
    # 非流式请求
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        extra_params: Optional[Dict] = None,
    ) -> Any:
        extra_params = extra_params or {}
        extra_params = self._filter_extra_params(extra_params)
        last_exception = None
        est_pt = self._estimate_prompt_tokens(messages)

        for attempt in range(1, self.max_retries + 1):
            key_obj, exhausted_info = await self.balancer.acquire_for_proxy()
            if key_obj is None:
                raise AdmissionRejectedException(exhausted_info)

            key_obj.record_request()
            client = self._make_client(key_obj.key)
            start_time = time.time()

            try:
                logger.info(
                    f"[{key_obj.alias}] 非流式请求 | model={model} | attempt={attempt}/{self.max_retries}"
                )
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=False,
                    **extra_params,
                )
                elapsed = round(time.time() - start_time, 2)
                logger.info(
                    f"[{key_obj.alias}] 请求成功 ✅ | 耗时:{elapsed}s | "
                    f"tokens: {getattr(response.usage, 'total_tokens', '?')}"
                )
                _latency_ms = int((time.time() - start_time) * 1000)
                est_ttft = int(_latency_ms * 0.65) if _latency_ms > 0 else 0
                ct = getattr(response.usage, 'completion_tokens', 0) or 0
                est_tps = round(ct / ((_latency_ms - est_ttft) / 1000), 1) if (_latency_ms - est_ttft) > 0 and ct > 0 else 0.0
                self._report(model, key_obj.alias, response.usage, start_time, True, False,
                             ttft_ms=est_ttft, tokens_per_second=est_tps)
                return response

            except RateLimitError as e:
                key_obj.record_rate_limit_error()
                last_exception = e
                wait = self._backoff(attempt, base=0.5, cap=2.0)
                logger.warning(f"[{key_obj.alias}] Rate Limit (429)，{wait:.2f}s 后切换Key重试...")
                await asyncio.sleep(wait)

            except APIConnectionError as e:
                key_obj.record_general_error()
                last_exception = e
                wait = self._backoff(attempt)
                logger.warning(f"[{key_obj.alias}] 网络错误，{wait:.2f}s 后重试: {e}")
                await asyncio.sleep(wait)

            except APIStatusError as e:
                key_obj.record_general_error()
                last_exception = e
                if e.status_code >= 500:
                    wait = self._backoff(attempt)
                    logger.warning(f"[{key_obj.alias}] 服务端错误 {e.status_code}，{wait:.2f}s 后重试")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"[{key_obj.alias}] 客户端错误 {e.status_code}: {e.message}")
                    break

            except Exception as e:
                key_obj.record_general_error()
                last_exception = e
                wait = self._backoff(attempt)
                logger.error(f"[{key_obj.alias}] 未知错误，{wait:.2f}s 后重试: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(wait)
                else:
                    break

        self._report(model, "unknown", None, time.time(), False, False, estimated_pt=est_pt)
        raise last_exception or RuntimeError("请求失败，已达最大重试次数")

    # ------------------------------------------------------------------
    # 流式请求
    # ------------------------------------------------------------------

    async def chat_completion_stream(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        extra_params: Optional[Dict] = None,
    ) -> AsyncGenerator[str, None]:
        extra_params = extra_params or {}
        extra_params = self._filter_extra_params(extra_params)
        last_exception = None
        est_pt = self._estimate_prompt_tokens(messages)

        for attempt in range(1, self.max_retries + 1):
            key_obj, exhausted_info = await self.balancer.acquire_for_proxy()
            if key_obj is None:
                raise AdmissionRejectedException(exhausted_info)

            key_obj.record_request()
            client = self._make_client(key_obj.key)
            start_time = time.time()

            stream_prompt_tokens = 0
            stream_completion_tokens = 0
            stream_content_chars = 0
            first_token_time = None

            try:
                logger.info(
                    f"[{key_obj.alias}] 流式请求 | model={model} | attempt={attempt}/{self.max_retries}"
                )
                response_stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True,
                    **extra_params,
                )

                async for chunk in response_stream:
                    if not chunk.choices:
                        if hasattr(chunk, 'usage') and chunk.usage:
                            stream_prompt_tokens = chunk.usage.prompt_tokens or 0
                            stream_completion_tokens = chunk.usage.completion_tokens or 0
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        stream_content_chars += len(delta)
                        if first_token_time is None:
                            first_token_time = time.time()
                    yield f"data: {chunk.model_dump_json()}\n\n"

                yield "data: [DONE]\n\n"

                elapsed = round(time.time() - start_time, 2)
                logger.info(f"[{key_obj.alias}] 流式完成 ✅ | 耗时:{elapsed}s")

                if self.stats:
                    if stream_completion_tokens == 0 and stream_content_chars > 0:
                        stream_completion_tokens = max(1, int(stream_content_chars / 3))
                    if stream_prompt_tokens == 0:
                        stream_prompt_tokens = est_pt
                    ttft_ms = int((first_token_time - start_time) * 1000) if first_token_time else 0
                    gen_time = (time.time() - first_token_time) if first_token_time else 0
                    tps = round(stream_completion_tokens / gen_time, 1) if gen_time > 0 and stream_completion_tokens > 0 else 0.0
                    self.stats.record(
                        model=model,
                        key_alias=key_obj.alias,
                        prompt_tokens=stream_prompt_tokens,
                        completion_tokens=stream_completion_tokens,
                        latency_ms=int((time.time() - start_time) * 1000),
                        success=True,
                        stream=True,
                        ttft_ms=ttft_ms,
                        tokens_per_second=tps,
                    )
                return

            except RateLimitError as e:
                key_obj.record_rate_limit_error()
                last_exception = e
                wait = self._backoff(attempt, base=0.5, cap=2.0)
                logger.warning(f"[{key_obj.alias}] 流式 Rate Limit，{wait:.2f}s 后切换Key重试")
                await asyncio.sleep(wait)

            except APIConnectionError as e:
                key_obj.record_general_error()
                last_exception = e
                wait = self._backoff(attempt)
                logger.warning(f"[{key_obj.alias}] 流式网络错误，{wait:.2f}s 后重试")
                await asyncio.sleep(wait)

            except APIStatusError as e:
                key_obj.record_general_error()
                last_exception = e
                if e.status_code >= 500:
                    wait = self._backoff(attempt)
                    logger.warning(f"[{key_obj.alias}] 流式服务端错误，{wait:.2f}s 后重试")
                    await asyncio.sleep(wait)
                else:
                    break

            except Exception as e:
                key_obj.record_general_error()
                last_exception = e
                wait = self._backoff(attempt)
                logger.error(f"[{key_obj.alias}] 流式未知错误，{wait:.2f}s 后重试: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(wait)
                else:
                    break

        self._report(model, "unknown", None, time.time(), False, True, estimated_pt=est_pt)
        raise last_exception or RuntimeError("流式请求失败，已达最大重试次数")