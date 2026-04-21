"""
负载均衡模块
支持三种策略：most_remaining / round_robin / least_used
"""

import time
import asyncio
import threading
from typing import Optional, List
from loguru import logger
from core.key_pool import APIKey, KeyPool


class LoadBalancer:
    """
    负载均衡器
    根据配置策略从Key池中选出最优Key
    """

    STRATEGIES = ("most_remaining", "round_robin", "least_used")

    def __init__(self, key_pool: KeyPool, strategy: str = "most_remaining", wait_timeout: float = 65.0):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"不支持的策略: {strategy}，可选: {self.STRATEGIES}")

        self.key_pool = key_pool
        self.strategy = strategy
        self.wait_timeout = wait_timeout

        # round_robin 专用索引，需要线程安全
        self._rr_index = 0
        self._rr_lock = threading.Lock()

        logger.info(f"负载均衡器启动，策略: [{strategy}]，等待超时: {wait_timeout}s")

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
        异步版本：获取可用Key，如果当前没有则异步等待，超时返回 None
        使用 asyncio.sleep() 避免阻塞事件循环
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
            await asyncio.sleep(wait_secs)

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