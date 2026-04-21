"""
请求转发模块
负责：拿到Key后向NVIDIA发起真实请求，处理流式/非流式响应，失败重试
v2.1：新增 Token 统计上报
"""

import time
import asyncio
import math
from typing import AsyncGenerator, Any, Dict, List, Optional

from loguru import logger
from openai import AsyncOpenAI, RateLimitError, APIStatusError, APIConnectionError

from core.balancer import LoadBalancer
from core.key_pool import APIKey


class NvidiaProxy:
    _ALLOWED_EXTRA_PARAMS = frozenset({
        "frequency_penalty", "presence_penalty", "stop", "seed",
        "n", "user", "logit_bias", "response_format",
        "tools", "tool_choice", "logprobs", "top_logprobs",
    })

    def __init__(
        self,
        balancer: LoadBalancer,
        base_url: str,
        max_retries: int = 3,
        stats_manager=None,
    ):
        self.balancer = balancer
        self.base_url = base_url
        self.max_retries = max_retries
        self.stats = stats_manager

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
            timeout=120.0,
        )

    def _report(
        self,
        model: str,
        key_alias: str,
        usage,              # openai Usage 对象或 None
        start_time: float,
        success: bool,
        stream: bool = False,
    ):
        """将Token统计上报给 StatsManager"""
        if self.stats is None:
            return
        latency_ms = int((time.time() - start_time) * 1000)
        prompt_tokens = 0
        completion_tokens = 0
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.stats.record(
            model=model,
            key_alias=key_alias,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            success=success,
            stream=stream,
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

        for attempt in range(1, self.max_retries + 1):
            key_obj: Optional[APIKey] = await self.balancer.get_key_or_wait_async()
            if key_obj is None:
                raise RuntimeError("所有API Key均不可用，请检查配置或等待解封")

            key_obj.record_request()
            client = self._make_client(key_obj.key)
            start_time = time.time()

            stream_prompt_tokens = 0
            stream_completion_tokens = 0
            stream_content_chars = 0

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
                # ✅ 上报统计
                self._report(model, key_obj.alias, response.usage, start_time, True, False)
                return response

            except RateLimitError as e:
                key_obj.record_rate_limit_error()
                self._report(model, key_obj.alias, None, start_time, False)
                last_exception = e
                logger.warning(f"[{key_obj.alias}] Rate Limit (429)，切换Key重试...")
                continue

            except APIConnectionError as e:
                key_obj.record_general_error()
                self._report(model, key_obj.alias, None, start_time, False)
                last_exception = e
                wait = 2 ** (attempt - 1)
                logger.warning(f"[{key_obj.alias}] 网络错误，{wait}s 后重试: {e}")
                await asyncio.sleep(wait)

            except APIStatusError as e:
                key_obj.record_general_error()
                self._report(model, key_obj.alias, None, start_time, False)
                last_exception = e
                if e.status_code >= 500:
                    wait = 2 ** (attempt - 1)
                    logger.warning(f"[{key_obj.alias}] 服务端错误 {e.status_code}，{wait}s 后重试")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"[{key_obj.alias}] 客户端错误 {e.status_code}: {e.message}")
                    raise

            except Exception as e:
                key_obj.record_general_error()
                self._report(model, key_obj.alias, None, start_time, False)
                logger.error(f"[{key_obj.alias}] 未知错误: {e}")
                raise

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

        for attempt in range(1, self.max_retries + 1):
            key_obj: Optional[APIKey] = await self.balancer.get_key_or_wait_async()
            if key_obj is None:
                raise RuntimeError("所有API Key均不可用，请检查配置或等待解封")

            key_obj.record_request()
            client = self._make_client(key_obj.key)
            start_time = time.time()

            stream_prompt_tokens = 0
            stream_completion_tokens = 0
            stream_content_chars = 0

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
                    yield f"data: {chunk.model_dump_json()}\n\n"

                yield "data: [DONE]\n\n"

                elapsed = round(time.time() - start_time, 2)
                logger.info(f"[{key_obj.alias}] 流式完成 ✅ | 耗时:{elapsed}s")

                if self.stats:
                    if stream_completion_tokens == 0 and stream_content_chars > 0:
                        stream_completion_tokens = max(1, int(stream_content_chars / 3))
                    self.stats.record(
                        model=model,
                        key_alias=key_obj.alias,
                        prompt_tokens=stream_prompt_tokens,
                        completion_tokens=stream_completion_tokens,
                        latency_ms=int((time.time() - start_time) * 1000),
                        success=True,
                        stream=True,
                    )
                return

            except RateLimitError as e:
                key_obj.record_rate_limit_error()
                self._report(model, key_obj.alias, None, start_time, False, True)
                last_exception = e
                logger.warning(f"[{key_obj.alias}] 流式 Rate Limit，切换Key重试...")
                continue

            except APIConnectionError as e:
                key_obj.record_general_error()
                self._report(model, key_obj.alias, None, start_time, False, True)
                last_exception = e
                wait = 2 ** (attempt - 1)
                logger.warning(f"[{key_obj.alias}] 流式网络错误，{wait}s 后重试")
                await asyncio.sleep(wait)

            except APIStatusError as e:
                key_obj.record_general_error()
                self._report(model, key_obj.alias, None, start_time, False, True)
                last_exception = e
                if e.status_code >= 500:
                    wait = 2 ** (attempt - 1)
                    logger.warning(f"[{key_obj.alias}] 流式服务端错误，{wait}s 后重试")
                    await asyncio.sleep(wait)
                else:
                    raise

            except Exception as e:
                key_obj.record_general_error()
                self._report(model, key_obj.alias, None, start_time, False, True)
                last_exception = e
                logger.error(f"[{key_obj.alias}] 流式未知错误: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** (attempt - 1))
                else:
                    raise

        raise last_exception or RuntimeError("流式请求失败，已达最大重试次数")