"""
启动入口 v2.2
新增：SQLite持久化 + WriteBuffer异步批量写入
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from core.config import cfg
from core.key_pool import KeyPool, KeyHealthChecker
from core.balancer import LoadBalancer
from core.proxy import NvidiaProxy
from core.model_manager import ModelManager
from core.stats_manager import StatsManager
from core.database import init_db
from core.write_buffer import WriteBuffer
from api.router import router, init_app_state

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger


def validate_config(config: dict):
    """校验配置文件的结构和字段，给出明确的错误提示"""
    errors = []

    if "keys" not in config or not isinstance(config["keys"], list):
        errors.append("缺少 'keys' 字段或格式错误（应为列表）")
    else:
        for i, k in enumerate(config["keys"]):
            if not isinstance(k, dict):
                errors.append(f"keys[{i}] 应为字典对象")
                continue
            if "key" not in k or not k["key"]:
                errors.append(f"keys[{i}] 缺少 'key' 字段（API Key 不能为空）")

    if "nvidia" not in config or not isinstance(config["nvidia"], dict):
        errors.append("缺少 'nvidia' 节点")
    else:
        nv = config["nvidia"]
        if "base_url" in nv and not nv["base_url"].startswith(("http://", "https://")):
            errors.append(f"nvidia.base_url 格式无效: {nv['base_url']}")
        for int_field in ("rpm_limit", "rpm_buffer"):
            if int_field in nv and (not isinstance(nv[int_field], int) or nv[int_field] <= 0):
                errors.append(f"nvidia.{int_field} 必须为正整数，当前值: {nv.get(int_field)}")

    if "balancer" not in config or not isinstance(config["balancer"], dict):
        errors.append("缺少 'balancer' 节点")
    else:
        bl = config["balancer"]
        valid_strategies = ("most_remaining", "round_robin", "least_used")
        if "strategy" in bl and bl["strategy"] not in valid_strategies:
            errors.append(f"balancer.strategy 无效: '{bl['strategy']}'，可选: {valid_strategies}")
        for float_field in ("wait_timeout",):
            if float_field in bl and (not isinstance(bl[float_field], (int, float)) or bl[float_field] <= 0):
                errors.append(f"balancer.{float_field} 必须为正数，当前值: {bl.get(float_field)}")
        if "max_retries" in bl and (not isinstance(bl["max_retries"], int) or bl["max_retries"] < 0):
            errors.append(f"balancer.max_retries 必须为非负整数，当前值: {bl.get('max_retries')}")
        valid_admission_modes = ("reject_fast", "queue_wait")
        if "admission_mode" in bl and bl["admission_mode"] not in valid_admission_modes:
            errors.append(f"balancer.admission_mode 无效: '{bl['admission_mode']}'，可选: {valid_admission_modes}")
        if "queue_wait_timeout" in bl and (not isinstance(bl["queue_wait_timeout"], (int, float)) or bl["queue_wait_timeout"] <= 0):
            errors.append(f"balancer.queue_wait_timeout 必须为正数，当前值: {bl.get('queue_wait_timeout')}")

    if "server" in config:
        sv = config["server"]
        if "port" in sv and (not isinstance(sv["port"], int) or sv["port"] < 1 or sv["port"] > 65535):
            errors.append(f"server.port 必须为 1-65535 的整数，当前值: {sv.get('port')}")

    if errors:
        print("配置文件校验失败：")
        for e in errors:
            print(f"  ❌ {e}")
        sys.exit(1)


def setup_logging(log_config: dict):
    Path("logs").mkdir(exist_ok=True)
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_config.get("level", "INFO"),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "{message}"
        ),
        colorize=True,
    )
    logger.add(
        log_config.get("file", "logs/app.log"),
        level=log_config.get("level", "INFO"),
        rotation=log_config.get("rotation", "10 MB"),
        retention=log_config.get("retention", "7 days"),
        encoding="utf-8",
    )


def create_app() -> FastAPI:
    init_db()

    write_buffer = WriteBuffer(flush_interval=5.0, max_buffer_size=100)

    try:
        import h2
        _http2 = True
    except ImportError:
        _http2 = False
    shared_http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        http2=_http2,
    )

    key_pool = KeyPool(
        keys_config=cfg.keys,
        rpm_limit=cfg.rpm_limit,
        rpm_buffer=cfg.rpm_buffer,
    )
    balancer = LoadBalancer(
        key_pool=key_pool,
        strategy=cfg.balancer.get("strategy", "most_remaining"),
        wait_timeout=cfg.balancer.get("wait_timeout", 65.0),
        admission_mode=cfg.balancer.get("admission_mode", "reject_fast"),
        queue_wait_timeout=cfg.balancer.get("queue_wait_timeout", 30.0),
    )
    stats_manager = StatsManager(write_buffer=write_buffer)

    key_pool.restore_from_db(stats_manager.get_key_stats_raw())

    proxy = NvidiaProxy(
        balancer=balancer,
        base_url=cfg.base_url,
        max_retries=cfg.balancer.get("max_retries", 3),
        stats_manager=stats_manager,
        http_client=shared_http_client,
    )
    model_manager = ModelManager(
        base_url=cfg.base_url,
        api_keys=[k["key"] for k in cfg.keys],
        fallback_list=cfg.models.get("fallback_list", []),
        auto_fetch=cfg.models.get("auto_fetch", True),
    )

    init_app_state(
        proxy=proxy,
        key_pool=key_pool,
        model_manager=model_manager,
        stats_manager=stats_manager,
    )

    health_checker = KeyHealthChecker(
        key_pool=key_pool,
        base_url=cfg.base_url,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await model_manager.initialize()
        await health_checker.start()
        await write_buffer.start()
        summary = key_pool.get_pool_summary()
        model_stats = model_manager.get_stats()
        server_cfg = cfg.server
        host = server_cfg.get("host", "127.0.0.1")
        port = server_cfg.get("port", 8000)
        logger.info("=" * 55)
        logger.info("  🚀 NVIDIA NIM Load Balancer v2.3 启动成功！")
        logger.info("=" * 55)
        logger.info(f"  📡 代理地址  : http://{host}:{port}/v1")
        logger.info(f"  📊 监控面板  : http://{host}:{port}/dashboard")
        logger.info(f"  📖 API文档   : http://{host}:{port}/docs")
        logger.info(f"  🔑 Key总数   : {summary['total_keys']} 个")
        logger.info(f"  🤖 模型总数  : {model_stats['total_models']} 个")
        logger.info(f"  📈 统计追踪  : 已启用（24h窗口 + SQLite持久化）")
        logger.info(f"  [DB] 数据库    : SQLite (WAL模式, 批量写入5s)")
        logger.info(f"  [HC] 健康检查  : 已启用（每1小时）")
        anthropic_default = cfg.anthropic_default_model
        anthropic_mapping_count = len(cfg.anthropic_model_mapping)
        logger.info(f"  [AT] Anthropic : /v1/messages (映射 {anthropic_mapping_count} 个模型)")
        logger.info(f"  [AT] 默认模型  : {anthropic_default}")
        logger.info("=" * 55)
        yield
        await write_buffer.stop()
        await health_checker.stop()
        await shared_http_client.aclose()
        logger.info("服务已关闭")

    app = FastAPI(
        title="NVIDIA NIM Load Balancer",
        version="2.3.0",
        docs_url="/docs",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        if request.url.path.startswith("/v1/"):
            logger.info(
                f"[INBOUND] {request.method} {request.url.path} | "
                f"User-Agent: {request.headers.get('user-agent', 'unknown')[:80]}"
            )
        response = await call_next(request)
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


if __name__ == "__main__":
    cfg.load("config.yaml")
    validate_config(cfg.data)
    setup_logging(cfg.logging)
    app = create_app()
    server_cfg = cfg.server
    uvicorn.run(
        app,
        host=server_cfg.get("host", "127.0.0.1"),
        port=server_cfg.get("port", 8000),
        log_level="warning",
        access_log=False,
    )