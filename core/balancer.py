"""
负载均衡模块
支持三种策略：most_remaining / round_robin / least_used
新增：前置准入控制 (Admission Control) — 原子性预扣配额，防止429雪崩
"""

import time
import asyncio
import threading
from typing import Optional, List, Tuple
from loguru import logger
from core.key_pool import APIKey, KeyPool


class PoolExhaustedError(Exception):
    """所有 Key 配额已耗尽，请求被准入控制拒绝"""
    pass


class LoadBalancer:
    """
    负载均衡器
    根据配置策略从Key池中选出最优Key
    支持两种准入模式：
      - reject_fast: 配额不足时立即拒绝（返回 None / 抛出异常）
      - queue_wait:  配额不足时智能等待，直到有配额释放或超时
    """

    STRATEGIES = ("most_remaining", "round_robin", "least_used")
    ADMISSION_MODES = ("reject_fast", "queue_wait")

    def __init__(
        self,
        key_pool: KeyPool,
        strategy: str = "most_remaining",
        wait_timeout: float = 65.0,
        admission_mode: str = "reject_fast",
        queue_wait_timeout: float = 30.0,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"不支持的策略: {strategy}，可选: {self.STRATEGIES}")
        if admission_mode not in self.ADMISSION_MODES:
            raise ValueError(f"不支持的准入模式: {admission_mode}，可选: {self.ADMISSION_MODES}")

        self.key_pool = key_pool
        self.strategy = strategy
        self.wait_timeout = wait_timeout
        self.admission_mode = admission_mode
        self.queue_wait_timeout = queue_wait_timeout

        # round_robin 专用索引，需要线程安全
        self._rr_index = 0
        self._rr_lock = threading.Lock()

        logger.info(
            f"负载均衡器启动 | 策略: [{strategy}] | 等待超时: {wait_timeout}s | "
            f"准入模式: [{admission_mode}] | 排队超时: {queue_wait_timeout}s"
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get_key(self) -> Optional[APIKey]:
        """
        立即获取一个可用Key，如果没有则返回 None
        """
        available = self.key_pool.get_available_keys()
        if not available:
            return None
        return self._select(available)

    def get_key_or_wait(self) -> Optional[APIKey]:
        """
        获取可用Key，如果当前没有则阻塞等待，超时返回 None
        适合同步场景
        """
        start = time.time()
        while True:
            key = self.get_key()
            if key:
                return key

            elapsed = time.time() - start
            if elapsed >= self.wait_timeout:
                logger.error(f"等待可用Key超时（{self.wait_timeout}s），所有Key均不可用")
                return None

            remaining_times = [
                k.get_ban_remaining_seconds()
                for k in self.key_pool.keys
                if k.get_ban_remaining_seconds() > 0
            ]
            wait_secs = min(remaining_times) if remaining_times else 5.0
            wait_secs = min(wait_secs, 5.0)
            logger.info(f"暂无可用Key，{wait_secs:.1f}s 后重试...")
            time.sleep(wait_secs)

    async def get_key_or_wait_async(self) -> Optional[APIKey]:
        """
        异步版本：获取可用Key（兼容旧接口）
        内部已接入前置准入控制，行为由 admission_mode 决定
        """
        if self.admission_mode == "reject_fast":
            return await self._acquire_fast_async()
        else:
            return await self._acquire_or_wait_async()

    async def _acquire_fast_async(self) -> Optional[APIKey]:
        """快速失败模式：原子性预扣配额，不足立即返回 None"""
        key = await asyncio.to_thread(self.key_pool.try_acquire)
        if key is None:
            total = self.key_pool.get_total_remaining()
            logger.warning(
                f"[准入拒绝] 所有 Key 配额已满 (总剩余: {total})，请求被快速拒绝"
            )
        return key

    async def _acquire_or_wait_async(self) -> Optional[APIKey]:
        """
        排队等待模式：先尝试预扣，失败则智能等待到有配额释放
        超时后返回 None
        """
        start = time.time()

        while True:
            key = await asyncio.to_thread(self.key_pool.try_acquire)
            if key:
                return key

            elapsed = time.time() - start
            if elapsed >= self.queue_wait_timeout:
                total = self.key_pool.get_total_remaining()
                logger.warning(
                    f"[排队超时] 等待 {self.queue_wait_timeout}s 仍无可用配额 "
                    f"(总剩余: {total})"
                )
                return None

            wait_secs = self._calculate_smart_wait()
            logger.info(f"配额已满，{wait_secs:.1f}s 后重试 (已等待 {elapsed:.1f}s)")
            await asyncio.sleep(wait_secs)

    def _calculate_smart_wait(self) -> float:
        """
        计算智能等待时间：取所有 Key 中最近一个时间戳即将过期的时间点
        这样可以精确等到有配额释放的时刻，避免盲目轮询
        """
        now = time.time()
        earliest_release = None
        for k in self.key_pool.keys:
            with k._lock:
                if k._timestamps:
                    ts = k._timestamps[0]
                    release_at = ts + 60
                    if release_at > now:
                        if earliest_release is None or release_at < earliest_release:
                            earliest_release = release_at
        if earliest_release is not None:
            wait = earliest_release - now + 0.05
            return max(0.1, min(wait, 5.0))
        return min(5.0, self.queue_wait_timeout / 10)

    async def acquire_for_proxy(self):
        """
        Proxy 层专用入口：统一处理 Key 获取 + 准入控制
        返回 (key_obj, exhausted_info) 元组：
          - 成功时: (APIKey, None)
          - 失败时: (None, dict) 包含拒绝详情供上层构造 HTTP 响应
        """
        key = await self.get_key_or_wait_async()
        if key is not None:
            return (key, None)

        total = self.key_pool.get_total_remaining()
        info = {
            "error": {
                "message": "所有 API Key 当前配额已满，请稍后重试",
                "type": "rate_limit_error",
                "code": "pool_exhausted",
                "total_remaining": total,
                "admission_mode": self.admission_mode,
            }
        }
        return (None, info)

    # ------------------------------------------------------------------
    # 内部策略实现
    # ------------------------------------------------------------------

    def _select(self, available: List[APIKey]) -> APIKey:
        """根据策略分派到对应方法"""
        if self.strategy == "most_remaining":
            return self._most_remaining(available)
        elif self.strategy == "round_robin":
            return self._round_robin(available)
        elif self.strategy == "least_used":
            return self._least_used(available)
        return available[0]

    def _most_remaining(self, available: List[APIKey]) -> APIKey:
        """
        剩余配额最多优先
        优点：尽量均衡消耗所有Key，避免某个Key过快耗尽
        推荐个人使用场景
        """
        selected = max(available, key=lambda k: k.get_remaining_quota())
        logger.debug(
            f"[most_remaining] 选中 [{selected.alias}]，"
            f"剩余配额: {selected.get_remaining_quota()}"
        )
        return selected

    def _round_robin(self, available: List[APIKey]) -> APIKey:
        """
        轮询策略
        优点：简单公平，请求均匀分布
        """
        with self._rr_lock:
            all_keys = self.key_pool.keys
            n = len(all_keys)
            # 从当前索引往后找第一个可用Key
            for _ in range(n):
                self._rr_index = (self._rr_index + 1) % n
                key = all_keys[self._rr_index]
                if key.is_available():
                    logger.debug(f"[round_robin] 选中 [{key.alias}]")
                    return key
        # fallback
        return available[0]

    def _least_used(self, available: List[APIKey]) -> APIKey:
        """
        历史使用最少优先
        优点：尽量保持每个Key总用量平均
        """
        def _safe_total_requests(k: APIKey) -> int:
            with k._lock:
                return k._total_requests

        selected = min(available, key=_safe_total_requests)
        logger.debug(
            f"[least_used] 选中 [{selected.alias}]，"
            f"历史请求数: {selected._total_requests}"
        )
        return selected