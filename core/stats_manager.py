"""
统计管理模块
负责：
  - 记录每次请求的Token消耗（prompt/completion/total）
  - 维护24小时滑动窗口时序数据（精度：分钟）
  - 按模型、按Key维度统计
  - 提供图表所需的数据结构
  - ★ 新增：SQLite持久化（内存缓存 + 异步批量刷写）
"""

import time
import math
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from loguru import logger

from sqlalchemy import text


@dataclass
class RequestRecord:
    timestamp: float
    model: str
    key_alias: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    success: bool
    stream: bool
    error_type: Optional[str] = None
    error_msg: Optional[str] = None


@dataclass
class TimeSlot:
    timestamp: float
    requests: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0

    @property
    def avg_latency_ms(self) -> int:
        if self.requests == 0:
            return 0
        return self.total_latency_ms // self.requests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "requests": self.requests,
            "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_latency_ms": self.avg_latency_ms,
        }


class StatsManager:
    WINDOW_MINUTES = 24 * 60
    MAX_RECORDS = 500
    CLEANUP_INTERVAL = 3600

    def __init__(self, write_buffer=None):
        from core.database import get_engine
        self._engine = get_engine()
        self._write_buffer = write_buffer

        if self._write_buffer:
            self._write_buffer.set_flush_callback(self._flush_to_db)

        self._lock = threading.Lock()

        self._timeline: Dict[float, TimeSlot] = {}
        self._model_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "requests": 0, "errors": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        })
        self._key_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "requests": 0, "errors": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        })
        self._total = {
            "requests": 0, "errors": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }
        self._records: deque = deque(maxlen=self.MAX_RECORDS)

        self._start_time = time.time()
        self._last_cleanup_time = self._start_time
        self._model_last_seen: Dict[str, float] = {}
        self._key_last_seen: Dict[str, float] = {}

        self._load_from_db()

    def _load_from_db(self):
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT model_id, total_requests, total_errors, "
                    "prompt_tokens, completion_tokens, total_tokens, last_seen_at "
                    "FROM model_stats"
                )).fetchall()
                loaded_models = 0
                for r in rows:
                    self._model_stats[r[0]] = {
                        "requests": r[1], "errors": r[2],
                        "prompt_tokens": r[3], "completion_tokens": r[4],
                        "total_tokens": r[5],
                    }
                    self._model_last_seen[r[0]] = r[6]
                    loaded_models += 1

                rows = conn.execute(text(
                    "SELECT key_alias, total_requests, total_errors, "
                    "prompt_tokens, completion_tokens, total_tokens, last_seen_at "
                    "FROM key_stats"
                )).fetchall()
                loaded_keys = 0
                for r in rows:
                    self._key_stats[r[0]] = {
                        "requests": r[1], "errors": r[2],
                        "prompt_tokens": r[3], "completion_tokens": r[4],
                        "total_tokens": r[5],
                    }
                    self._key_last_seen[r[0]] = r[6]
                    loaded_keys += 1

                row = conn.execute(text(
                    "SELECT COUNT(*), "
                    "SUM(CASE WHEN success=0 THEN 1 ELSE 0 END), "
                    "SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens) "
                    "FROM request_logs"
                )).one_or_none()
                if row and row[0]:
                    self._total = {
                        "requests": int(row[0]), "errors": int(row[1] or 0),
                        "prompt_tokens": int(row[2] or 0),
                        "completion_tokens": int(row[3] or 0),
                        "total_tokens": int(row[4] or 0),
                    }

                rows = conn.execute(text(
                    "SELECT slot_time, requests, errors, "
                    "prompt_tokens, completion_tokens, total_tokens, total_latency_ms "
                    "FROM timeline_slots ORDER BY slot_time"
                )).fetchall()
                cutoff = time.time() - self.WINDOW_MINUTES * 60
                loaded_slots = 0
                for r in rows:
                    if r[0] >= cutoff:
                        self._timeline[r[0]] = TimeSlot(
                            timestamp=r[0], requests=r[1], errors=r[2],
                            prompt_tokens=r[3], completion_tokens=r[4],
                            total_tokens=r[5], total_latency_ms=r[6],
                        )
                        loaded_slots += 1

            logger.info(
                f"从数据库恢复统计数据 | 模型:{loaded_models} | Key:{loaded_keys} | 时序槽:{loaded_slots}"
            )
        except Exception as e:
            logger.warning(f"从数据库加载统计数据失败（将使用空数据启动）: {e}")

    async def _flush_to_db(self, records: List[RequestRecord]):
        if not records:
            return
        try:
            with self._engine.begin() as conn:
                for r in records:
                    minute_ts = math.floor(r.timestamp / 60) * 60
                    success_int = 1 if r.success else 0
                    stream_int = 1 if r.stream else 0
                    errors_val = 0 if r.success else 1

                    conn.execute(text(
                        "INSERT INTO request_logs "
                        "(timestamp, model, key_alias, prompt_tokens, completion_tokens, "
                        "total_tokens, latency_ms, success, stream, error_type, error_msg) "
                        "VALUES (:ts, :m, :ka, :pt, :ct, :tt, :lm, :suc, :strm, :et, :em)"
                    ), {
                        "ts": r.timestamp, "m": r.model, "ka": r.key_alias,
                        "pt": r.prompt_tokens, "ct": r.completion_tokens,
                        "tt": r.total_tokens, "lm": r.latency_ms,
                        "suc": success_int, "strm": stream_int,
                        "et": r.error_type, "em": r.error_msg,
                    })

                    conn.execute(text(
                        "INSERT INTO model_stats "
                        "(model_id, total_requests, total_errors, prompt_tokens, "
                        "completion_tokens, total_tokens, last_seen_at, first_seen_at) "
                        "VALUES (:mid, 1, :te, :pt, :ct, :tt, :ls, :fs) "
                        "ON CONFLICT(model_id) DO UPDATE SET "
                        "total_requests=total_requests+1, "
                        "total_errors=total_errors+:te, "
                        "prompt_tokens=prompt_tokens+:pt, "
                        "completion_tokens=completion_tokens+:ct, "
                        "total_tokens=total_tokens+:tt, "
                        "last_seen_at=:ls"
                    ), {
                        "mid": r.model, "te": errors_val,
                        "pt": r.prompt_tokens, "ct": r.completion_tokens,
                        "tt": r.total_tokens, "ls": r.timestamp, "fs": r.timestamp,
                    })

                    conn.execute(text(
                        "INSERT INTO key_stats "
                        "(key_alias, total_requests, total_errors, rate_limit_errors, "
                        "prompt_tokens, completion_tokens, total_tokens, "
                        "last_seen_at, disabled, disable_reason, updated_at) "
                        "VALUES (:ka, 1, :te, 0, :pt, :ct, :tt, :ls, 0, '', :ua) "
                        "ON CONFLICT(key_alias) DO UPDATE SET "
                        "total_requests=total_requests+1, "
                        "total_errors=total_errors+:te, "
                        "prompt_tokens=prompt_tokens+:pt, "
                        "completion_tokens=completion_tokens+:ct, "
                        "total_tokens=total_tokens+:tt, "
                        "last_seen_at=:ls, updated_at=:ua"
                    ), {
                        "ka": r.key_alias, "te": errors_val,
                        "pt": r.prompt_tokens, "ct": r.completion_tokens,
                        "tt": r.total_tokens, "ls": r.timestamp, "ua": r.timestamp,
                    })

                    conn.execute(text(
                        "INSERT INTO timeline_slots "
                        "(slot_time, requests, errors, prompt_tokens, "
                        "completion_tokens, total_tokens, total_latency_ms) "
                        "VALUES (:st, 1, :e, :pt, :ct, :tt, :tlm) "
                        "ON CONFLICT(slot_time) DO UPDATE SET "
                        "requests=requests+1, errors=errors+:e, "
                        "prompt_tokens=prompt_tokens+:pt, "
                        "completion_tokens=completion_tokens+:ct, "
                        "total_tokens=total_tokens+:tt, "
                        "total_latency_ms=total_latency_ms+:tlm"
                    ), {
                        "st": minute_ts, "e": errors_val,
                        "pt": r.prompt_tokens, "ct": r.completion_tokens,
                        "tt": r.total_tokens, "tlm": r.latency_ms,
                    })

                cutoff = time.time() - self.WINDOW_MINUTES * 60
                conn.execute(
                    text("DELETE FROM timeline_slots WHERE slot_time < :cutoff"),
                    {"cutoff": cutoff},
                )

        except Exception as e:
            logger.error(f"批量写入数据库失败 ({len(records)}条): {e}")
            raise

    # ------------------------------------------------------------------
    # 核心写入接口
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        key_alias: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        success: bool = True,
        stream: bool = False,
    ):
        total_tokens = prompt_tokens + completion_tokens
        now = time.time()
        minute_ts = float(int(now // 60) * 60)

        record = RequestRecord(
            timestamp=now,
            model=model,
            key_alias=key_alias,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            success=success,
            stream=stream,
        )

        with self._lock:
            if minute_ts not in self._timeline:
                self._timeline[minute_ts] = TimeSlot(timestamp=minute_ts)
                self._evict_old_slots()

            slot = self._timeline[minute_ts]
            slot.requests += 1
            slot.prompt_tokens += prompt_tokens
            slot.completion_tokens += completion_tokens
            slot.total_tokens += total_tokens
            slot.total_latency_ms += latency_ms
            if not success:
                slot.errors += 1

            self._model_last_seen[model] = now
            ms = self._model_stats[model]
            ms["requests"] += 1
            ms["prompt_tokens"] += prompt_tokens
            ms["completion_tokens"] += completion_tokens
            ms["total_tokens"] += total_tokens
            if not success:
                ms["errors"] += 1

            self._key_last_seen[key_alias] = now
            ks = self._key_stats[key_alias]
            ks["requests"] += 1
            ks["prompt_tokens"] += prompt_tokens
            ks["completion_tokens"] += completion_tokens
            ks["total_tokens"] += total_tokens
            if not success:
                ks["errors"] += 1

            if now - self._last_cleanup_time > self.CLEANUP_INTERVAL:
                self._evict_stale_stats(now)
                self._last_cleanup_time = now

            self._total["requests"] += 1
            self._total["prompt_tokens"] += prompt_tokens
            self._total["completion_tokens"] += completion_tokens
            self._total["total_tokens"] += total_tokens
            if not success:
                self._total["errors"] += 1

            self._records.append(record)

        if self._write_buffer:
            self._write_buffer.append(record)

    def _evict_old_slots(self):
        cutoff = time.time() - self.WINDOW_MINUTES * 60
        old_keys = [ts for ts in self._timeline if ts < cutoff]
        for k in old_keys:
            del self._timeline[k]

    def _evict_stale_stats(self, now: float):
        cutoff = now - self.WINDOW_MINUTES * 60

        stale_models = [
            m for m, last_seen in self._model_last_seen.items()
            if last_seen < cutoff
        ]
        for m in stale_models:
            del self._model_stats[m]
            del self._model_last_seen[m]
        if stale_models:
            logger.debug(f"清理过期模型统计: {len(stale_models)} 个")

        stale_keys = [
            k for k, last_seen in self._key_last_seen.items()
            if last_seen < cutoff
        ]
        for k in stale_keys:
            del self._key_stats[k]
            del self._key_last_seen[k]
        if stale_keys:
            logger.debug(f"清理过期Key统计: {len(stale_keys)} 个")

    # ------------------------------------------------------------------
    # 查询接口（供 Router 调用）
    # ------------------------------------------------------------------

    def get_timeline(self, minutes: int = 60) -> List[Dict]:
        minutes = min(minutes, self.WINDOW_MINUTES)
        now = time.time()
        start_ts = float(int((now - minutes * 60) // 60) * 60)

        result = []
        with self._lock:
            ts = start_ts
            while ts <= float(int(now // 60) * 60):
                slot = self._timeline.get(ts)
                if slot:
                    result.append(slot.to_dict())
                else:
                    result.append(TimeSlot(timestamp=ts).to_dict())
                ts += 60

        return result

    def get_model_stats(self) -> List[Dict]:
        with self._lock:
            return [
                {"model": model, **stats}
                for model, stats in self._model_stats.items()
            ]

    def get_key_stats(self) -> List[Dict]:
        with self._lock:
            return [
                {"key_alias": alias, **stats}
                for alias, stats in self._key_stats.items()
            ]

    def get_key_stats_raw(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return dict(self._key_stats)

    def get_overview(self) -> Dict[str, Any]:
        with self._lock:
            uptime_seconds = int(time.time() - self._start_time)
            hours, remainder = divmod(uptime_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            one_hour_ago = time.time() - 3600
            recent_requests = sum(
                s.requests for ts, s in self._timeline.items()
                if ts >= one_hour_ago
            )
            recent_tokens = sum(
                s.total_tokens for ts, s in self._timeline.items()
                if ts >= one_hour_ago
            )

            return {
                "total": dict(self._total),
                "recent_1h_requests": recent_requests,
                "recent_1h_tokens": recent_tokens,
                "uptime": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
                "uptime_seconds": uptime_seconds,
                "start_time": self._start_time,
            }

    def get_recent_records(self, limit: int = 50) -> List[Dict]:
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT timestamp, model, key_alias, prompt_tokens, "
                    "completion_tokens, total_tokens, latency_ms, success, stream "
                    "FROM request_logs ORDER BY id DESC LIMIT :limit"
                ), {"limit": limit}).fetchall()

                return [
                    {
                        "time": r[0],
                        "model": r[1],
                        "key_alias": r[2],
                        "prompt_tokens": r[3],
                        "completion_tokens": r[4],
                        "total_tokens": r[5],
                        "latency_ms": r[6],
                        "success": bool(r[7]),
                        "stream": bool(r[8]),
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.warning(f"从数据库查询最近记录失败，回退到内存: {e}")
            with self._lock:
                records = list(self._records)[-limit:]
                records.reverse()
                return [
                    {
                        "time": r.timestamp,
                        "model": r.model,
                        "key_alias": r.key_alias,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "total_tokens": r.total_tokens,
                        "latency_ms": r.latency_ms,
                        "success": r.success,
                        "stream": r.stream,
                    }
                    for r in records
                ]
