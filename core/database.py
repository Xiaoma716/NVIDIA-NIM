"""
数据库引擎模块
负责：ORM模型定义、引擎配置、建表初始化
"""

from pathlib import Path

from sqlalchemy import (
    create_engine, MetaData, Index,
    Column, Integer, Text, Float,
)
from sqlalchemy.orm import DeclarativeBase
from loguru import logger


class Base(DeclarativeBase):
    pass


class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Float, nullable=False)
    model = Column(Text, nullable=False)
    key_alias = Column(Text, nullable=False)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    latency_ms = Column(Integer, nullable=False, default=0)
    success = Column(Integer, nullable=False, default=1)
    stream = Column(Integer, nullable=False, default=0)
    error_type = Column(Text, nullable=True)
    error_msg = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_rl_timestamp", "timestamp"),
        Index("idx_rl_model", "model"),
        Index("idx_rl_key_alias", "key_alias"),
        Index("idx_rl_ts_model", "timestamp", "model"),
    )


class ModelStat(Base):
    __tablename__ = "model_stats"

    model_id = Column(Text, primary_key=True)
    total_requests = Column(Integer, nullable=False, default=0)
    total_errors = Column(Integer, nullable=False, default=0)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    last_seen_at = Column(Float, nullable=False, default=0.0)
    first_seen_at = Column(Float, nullable=False, default=0.0)


class KeyStat(Base):
    __tablename__ = "key_stats"

    key_alias = Column(Text, primary_key=True)
    total_requests = Column(Integer, nullable=False, default=0)
    total_errors = Column(Integer, nullable=False, default=0)
    rate_limit_errors = Column(Integer, nullable=False, default=0)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    last_seen_at = Column(Float, nullable=False, default=0.0)
    disabled = Column(Integer, nullable=False, default=0)
    disable_reason = Column(Text, nullable=True)
    updated_at = Column(Float, nullable=False, default=0.0)


class TimelineSlot(Base):
    __tablename__ = "timeline_slots"

    slot_time = Column(Float, primary_key=True)
    requests = Column(Integer, nullable=False, default=0)
    errors = Column(Integer, nullable=False, default=0)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    total_latency_ms = Column(Integer, nullable=False, default=0)


class ModelState(Base):
    __tablename__ = "model_states"

    model_id = Column(Text, primary_key=True)
    owned_by = Column(Text, nullable=False, default="")
    enabled = Column(Integer, nullable=False, default=1)
    created_at = Column(Float, nullable=False, default=0.0)
    updated_at = Column(Float, nullable=False, default=0.0)


class SystemMeta(Base):
    __tablename__ = "system_meta"

    key = Column(Text, primary_key=True)
    value = Column(Text, nullable=False, default="")


_engine = None


def get_db_path() -> str:
    db_dir = Path(__file__).resolve().parent.parent / "data"
    db_dir.mkdir(exist_ok=True)
    return str(db_dir / "nim.db")


def get_engine():
    global _engine
    if _engine is None:
        db_path = get_db_path()
        _engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            future=True,
        )
        from sqlalchemy import text
        with _engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA synchronous=NORMAL"))
            conn.execute(text("PRAGMA busy_timeout=5000"))
            conn.execute(text("PRAGMA cache_size=-64000"))
            conn.execute(text("PRAGMA temp_store=MEMORY"))
            conn.commit()
        logger.info(f"数据库引擎已初始化 | 路径: {db_path} | 模式: WAL")
    return _engine


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)

    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT OR IGNORE INTO system_meta (key, value) VALUES ('db_schema_version', '1')"
        ))
        conn.execute(text(
            "INSERT OR IGNORE INTO system_meta (key, value) VALUES ('total_startup_count', '0')"
        ))

    startup_count_row = _get_meta("total_startup_count") or "0"
    new_count = int(startup_count_row) + 1
    _set_meta("total_startup_count", str(new_count))
    _set_meta("service_start_time", str(__import__("time").time()))

    logger.info(f"数据库表已就绪 | 启动次数: {new_count}")
    return engine


def _get_meta(key: str) -> str | None:
    from sqlalchemy import text
    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT value FROM system_meta WHERE key = :k"), {"k": key}
        ).one_or_none()
        return row[0] if row else None


def _set_meta(key: str, value: str):
    from sqlalchemy import text
    with get_engine().begin() as conn:
        conn.execute(
            text("INSERT OR REPLACE INTO system_meta (key, value) VALUES (:k, :v)"),
            {"k": key, "v": value},
        )
