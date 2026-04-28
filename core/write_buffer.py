import asyncio
from collections import deque
from typing import List, Optional

from loguru import logger

from core.stats_manager import RequestRecord


class WriteBuffer:
    def __init__(
        self,
        flush_interval: float = 5.0,
        max_buffer_size: int = 100,
    ):
        self._buffer: deque = deque()
        self._flush_interval = flush_interval
        self._max_buffer_size = max_buffer_size
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._flush_callback = None
        self._flush_lock = asyncio.Lock()

    def set_flush_callback(self, callback):
        self._flush_callback = callback

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._flush_loop())
        logger.info(
            f"写入缓冲区已启动 | 间隔:{self._flush_interval}s | 容量:{self._max_buffer_size}"
        )

    async def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._flush_all()
        logger.info("写入缓冲区已停止")

    def append(self, record: RequestRecord):
        self._buffer.append(record)
        if len(self._buffer) >= self._max_buffer_size:
            asyncio.ensure_future(self._flush_all())

    async def _flush_loop(self):
        try:
            while self._running:
                await asyncio.sleep(self._flush_interval)
                if self._buffer:
                    await self._flush_all()
        except asyncio.CancelledError:
            pass

    async def _flush_all(self):
        if not self._buffer or not self._flush_callback:
            return
        async with self._flush_lock:
            if not self._buffer:
                return
            batch: List[RequestRecord] = list(self._buffer)
            self._buffer.clear()
            try:
                await self._flush_callback(batch)
            except Exception as e:
                for record in batch:
                    self._buffer.append(record)
                logger.error(f"批量写入数据库失败 ({len(batch)}条)，数据已回退到缓冲区: {e}")

    @property
    def pending_count(self) -> int:
        return len(self._buffer)
