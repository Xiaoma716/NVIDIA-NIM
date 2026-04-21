"""
Key状态管理模块
负责：滑动窗口计数、可用性判断、封禁/解封、统计数据、主动健康检查
"""

import time
import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import httpx
from loguru import logger


@dataclass
class APIKey:
    """
    单个 API Key 的完整状态封装
    使用滑动窗口统计过去60秒内的请求数量
    """
    key: str
    alias: str
    rpm_limit: int = 40
    rpm_buffer: int = 5

    # ---- 运行时状态（不参与dataclass比较）----
    # 存储过去60秒内每次请求的时间戳
    _timestamps: deque = field(default_factory=deque, compare=False, repr=False)
    # 是否被封禁（触发429时启用）
    _is_banned: bool = field(default=False, compare=False)
    # 封禁到期时间戳
    _ban_until: float = field(default=0.0, compare=False)
    # 统计数据
    _total_requests: int = field(default=0, compare=False)
    _total_errors: int = field(default=0, compare=False)
    _total_rate_limit_errors: int = field(default=0, compare=False)
    # 线程锁，保证并发安全
    _lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)
    # 服务启动时间
    _start_time: float = field(default_factory=time.time, compare=False)
    # 是否被健康检查标记为永久不可用（Key过期/撤销等）
    _is_disabled: bool = field(default=False, compare=False)
    # 禁用原因
    _disable_reason: str = field(default="", compare=False)

    def _clean_old_timestamps(self):
        """
        清理60秒前的过期时间戳（内部调用，调用前需持锁）
        """
        cutoff = time.time() - 60
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def is_available(self) -> bool:
        """
        判断当前Key是否可用
        条件：未被永久禁用 && 未被封禁 && 过去60秒请求数未超过限额
        """
        with self._lock:
            if self._is_disabled:
                return False

            now = time.time()

            # 检查封禁状态
            if self._is_banned:
                if now < self._ban_until:
                    return False
                else:
                    self._is_banned = False
                    logger.info(f"[{self.alias}] 封禁已解除，恢复可用 ✅")

            self._clean_old_timestamps()
            effective_limit = self.rpm_limit - self.rpm_buffer
            return len(self._timestamps) < effective_limit

    def disable(self, reason: str = ""):
        """永久禁用此Key（健康检查发现无效时调用）"""
        with self._lock:
            self._is_disabled = True
            self._disable_reason = reason
        logger.warning(f"[{self.alias}] 已被永久禁用 | 原因: {reason}")

    def enable(self):
        """重新启用被禁用的Key（手动恢复）"""
        with self._lock:
            was_disabled = self._is_disabled
            self._is_disabled = False
            self._disable_reason = ""
        if was_disabled:
            logger.info(f"[{self.alias}] 已手动恢复启用 ✅")

    def get_current_usage(self) -> int:
        """获取过去60秒的实际请求数"""
        with self._lock:
            self._clean_old_timestamps()
            return len(self._timestamps)

    def get_remaining_quota(self) -> int:
        """获取当前剩余可用配额"""
        with self._lock:
            self._clean_old_timestamps()
            effective_limit = self.rpm_limit - self.rpm_buffer
            return max(0, effective_limit - len(self._timestamps))

    def get_ban_remaining_seconds(self) -> float:
        """获取封禁剩余秒数，未封禁返回0"""
        with self._lock:
            if self._is_banned and time.time() < self._ban_until:
                return round(self._ban_until - time.time(), 1)
            return 0.0

    def record_request(self):
        """记录一次请求（请求发出前调用）"""
        with self._lock:
            self._timestamps.append(time.time())
            self._total_requests += 1

    def record_rate_limit_error(self):
        """
        记录一次 429 Rate Limit 错误
        触发封禁：60秒内禁用此Key
        """
        with self._lock:
            self._total_errors += 1
            self._total_rate_limit_errors += 1
            self._is_banned = True
            self._ban_until = time.time() + 60
            logger.warning(
                f"[{self.alias}] 触发 Rate Limit！封禁60秒 "
                f"(累计限速次数: {self._total_rate_limit_errors})"
            )

    def record_general_error(self):
        with self._lock:
            self._total_errors += 1

    def set_historical_totals(self, requests: int, errors: int, rate_limit_errors: int):
        with self._lock:
            self._total_requests = requests
            self._total_errors = errors
            self._total_rate_limit_errors = rate_limit_errors

    def get_stats(self) -> Dict[str, Any]:
        """返回该Key的完整统计信息（供监控面板使用）"""
        with self._lock:
            self._clean_old_timestamps()
            now = time.time()

            # 计算错误率
            error_rate = 0.0
            if self._total_requests > 0:
                error_rate = round(self._total_errors / self._total_requests * 100, 2)

            # 封禁剩余时间
            ban_remaining = 0.0
            if self._is_banned and now < self._ban_until:
                ban_remaining = round(self._ban_until - now, 1)
            elif self._is_banned and now >= self._ban_until:
                # 顺带解封（双重保险）
                self._is_banned = False

            available = not self._is_disabled and not self._is_banned and \
                        len(self._timestamps) < (self.rpm_limit - self.rpm_buffer)

            return {
                "alias": self.alias,
                "available": available,
                "is_disabled": self._is_disabled,
                "disable_reason": self._disable_reason,
                "is_banned": self._is_banned,
                "ban_remaining_seconds": ban_remaining,
                "current_usage": len(self._timestamps),
                "rpm_limit": self.rpm_limit,
                "rpm_effective_limit": self.rpm_limit - self.rpm_buffer,
                "remaining_quota": max(0, (self.rpm_limit - self.rpm_buffer) - len(self._timestamps)),
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "total_rate_limit_errors": self._total_rate_limit_errors,
                "error_rate_percent": error_rate,
                "key_preview": f"{self.key[:10]}...{self.key[-4:]}",
            }


class KeyPool:
    """
    Key池：管理所有 APIKey 对象
    提供统一的查询、统计接口
    """

    def __init__(self, keys_config: List[Dict], rpm_limit: int, rpm_buffer: int):
        self.keys: List[APIKey] = []
        for cfg in keys_config:
            api_key = APIKey(
                key=cfg["key"],
                alias=cfg.get("alias", f"Key-{len(self.keys)+1}"),
                rpm_limit=rpm_limit,
                rpm_buffer=rpm_buffer,
            )
            self.keys.append(api_key)
        logger.info(f"Key池初始化完成，共加载 {len(self.keys)} 个Key")

    def get_available_keys(self) -> List[APIKey]:
        """返回当前所有可用的Key"""
        return [k for k in self.keys if k.is_available()]

    def get_all_stats(self) -> List[Dict[str, Any]]:
        """返回所有Key的统计信息"""
        return [k.get_stats() for k in self.keys]

    def get_pool_summary(self) -> Dict[str, Any]:
        """返回整个Key池的汇总信息"""
        stats = self.get_all_stats()
        available_count = sum(1 for s in stats if s["available"])
        total_remaining = sum(s["remaining_quota"] for s in stats if s["available"])
        total_requests = sum(s["total_requests"] for s in stats)
        total_errors = sum(s["total_errors"] for s in stats)

        return {
            "total_keys": len(self.keys),
            "available_keys": available_count,
            "unavailable_keys": len(self.keys) - available_count,
            "total_remaining_quota": total_remaining,
            "total_requests": total_requests,
            "total_errors": total_errors,
        }

    def enable_key(self, alias: str) -> bool:
        """手动重新启用一个被禁用的Key"""
        for k in self.keys:
            if k.alias == alias:
                k.enable()
                return True
        return False

    def disable_key(self, alias: str, reason: str = "") -> bool:
        """手动禁用指定Key"""
        for k in self.keys:
            if k.alias == alias:
                k.disable(reason)
                return True
        return False

    def restore_from_db(self, key_stats: Dict[str, Dict[str, int]]):
        for k in self.keys:
            if k.alias in key_stats:
                s = key_stats[k.alias]
                k.set_historical_totals(
                    requests=s.get("requests", 0),
                    errors=s.get("errors", 0),
                    rate_limit_errors=s.get("rate_limit_errors", 0),
                )
        restored = sum(1 for k in self.keys if k.alias in key_stats)
        logger.info(f"Key历史数据已恢复 | 恢复: {restored}/{len(self.keys)}")


class KeyHealthChecker:
    """
    Key 主动健康检查器
    定期向 NVIDIA API 发送轻量级请求，验证每个 Key 的有效性
    发现无效 Key（401/403）时自动标记为永久禁用
    """

    CHECK_INTERVAL = 3600       # 检查间隔：1小时
    CHECK_TIMEOUT = 10.0       # 单次检查超时：10秒

    def __init__(self, key_pool: KeyPool, base_url: str):
        self.key_pool = key_pool
        self.base_url = base_url.rstrip("/")
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """启动后台健康检查任务"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        logger.info(f"🩺 Key健康检查器已启动 (间隔: {self.CHECK_INTERVAL}s)")

    async def stop(self):
        """停止健康检查任务"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Key健康检查器已停止")

    async def _check_loop(self):
        """后台循环：定期执行健康检查"""
        try:
            while self._running:
                await self.check_all_keys()
                await asyncio.sleep(self.CHECK_INTERVAL)
        except asyncio.CancelledError:
            logger.debug("健康检查循环被取消")
        except Exception as e:
            logger.error(f"健康检查循环异常: {e}")

    async def check_all_keys(self):
        """对所有 Key 执行一次健康检查"""
        url = f"{self.base_url}/models"
        results = {"ok": 0, "disabled": 0, "error": 0}

        for api_key in self.key_pool.keys:
            if api_key._is_disabled:
                continue

            try:
                async with httpx.AsyncClient(timeout=self.CHECK_TIMEOUT) as client:
                    response = await client.get(
                        url,
                        headers={
                            "Authorization": f"Bearer {api_key.key}",
                            "Content-Type": "application/json",
                        },
                    )

                if response.status_code in (401, 403):
                    api_key.disable(
                        f"HTTP {response.status_code} (Key无效/过期/撤销)"
                    )
                    results["disabled"] += 1
                elif response.status_code == 429:
                    logger.debug(f"[{api_key.alias}] 健康检查触发限速(429)，暂不处理")
                    results["ok"] += 1
                elif response.status_code == 200:
                    results["ok"] += 1
                else:
                    logger.warning(f"[{api_key.alias}] 健康检查返回 HTTP {response.status_code}")
                    results["error"] += 1

            except Exception as e:
                logger.warning(f"[{api_key.alias}] 健康检查异常: {e}")
                results["error"] += 1

        if results["disabled"] > 0:
            logger.warning(
                f"🩺 健康检查完成 | ✅{results['ok']} | ❌禁用{results['disabled']} | ⚠️异常{results['error']}"
            )
        else:
            logger.info(
                f"🩺 健康检查完成 | ✅{results['ok']} | ❌禁用{results['disabled']} | ⚠️异常{results['error']}"
            )