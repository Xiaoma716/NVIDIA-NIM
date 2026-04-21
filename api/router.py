"""
API路由层
完全兼容 OpenAI 格式，新增模型管理接口
新增：/api/stats/* 统计数据接口
重构：使用 FastAPI 依赖注入替代模块级全局变量
"""

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from loguru import logger

from api.dashboard import DASHBOARD_HTML
from core.proxy import NvidiaProxy
from core.key_pool import KeyPool
from core.model_manager import ModelManager
from core.stats_manager import StatsManager
from core.config import cfg


class AppState:
    """应用共享状态容器，通过 FastAPI Depends 注入"""

    def __init__(
        self,
        proxy: NvidiaProxy,
        key_pool: KeyPool,
        model_manager: ModelManager,
        stats_manager: StatsManager,
    ):
        self.proxy = proxy
        self.key_pool = key_pool
        self.model_manager = model_manager
        self.stats_manager = stats_manager

    @property
    def default_model(self) -> str:
        return cfg.default_model


_app_state: Optional[AppState] = None


def get_app_state() -> AppState:
    """FastAPI 依赖：获取应用状态"""
    if _app_state is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    return _app_state


def init_app_state(
    proxy: NvidiaProxy,
    key_pool: KeyPool,
    model_manager: ModelManager,
    stats_manager: StatsManager,
):
    """初始化应用状态（在 create_app 中调用一次）"""
    global _app_state
    _app_state = AppState(
        proxy=proxy,
        key_pool=key_pool,
        model_manager=model_manager,
        stats_manager=stats_manager,
    )


router = APIRouter()


# ------------------------------------------------------------------
# Pydantic 模型
# ------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    model_config = {"extra": "allow"}


# ------------------------------------------------------------------
# 页面路由
# ------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return HTMLResponse(content='<script>location.href="/dashboard"</script>')


@router.get("/dashboard", response_class=HTMLResponse, tags=["监控"])
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


# ------------------------------------------------------------------
# Key池监控接口
# ------------------------------------------------------------------

@router.get("/health", tags=["监控"])
async def health(state: AppState = Depends(get_app_state)):
    summary = state.key_pool.get_pool_summary()
    return {
        "status": "healthy" if summary["available_keys"] > 0 else "degraded",
        "timestamp": time.time(),
        "summary": summary,
    }


@router.get("/stats", tags=["监控"])
async def key_stats(state: AppState = Depends(get_app_state)):
    return {
        "summary": state.key_pool.get_pool_summary(),
        "keys": state.key_pool.get_all_stats(),
        "timestamp": time.time(),
    }


# ------------------------------------------------------------------
# 统计分析接口
# ------------------------------------------------------------------

@router.get("/api/stats/overview", tags=["统计分析"])
async def stats_overview(state: AppState = Depends(get_app_state)):
    """总体统计概览：累计请求、Token、运行时长等"""
    return state.stats_manager.get_overview()


@router.get("/api/stats/timeline", tags=["统计分析"])
async def stats_timeline(
    minutes: int = Query(default=60, ge=10, le=1440, description="查询最近N分钟，最大1440（24h）"),
    state: AppState = Depends(get_app_state),
):
    """时序图数据：请求量、Token消耗（分钟粒度）"""
    return {
        "minutes": minutes,
        "data": state.stats_manager.get_timeline(minutes),
    }


@router.get("/api/stats/models", tags=["统计分析"])
async def stats_models(state: AppState = Depends(get_app_state)):
    """各模型调用量、Token消耗统计（用于饼图/排行）"""
    return {"data": state.stats_manager.get_model_stats()}


@router.get("/api/stats/keys", tags=["统计分析"])
async def stats_keys(state: AppState = Depends(get_app_state)):
    """各Key的Token消耗统计"""
    return {"data": state.stats_manager.get_key_stats()}


@router.get("/api/stats/records", tags=["统计分析"])
async def stats_records(
    limit: int = Query(default=50, ge=1, le=500, description="返回最近N条记录"),
    state: AppState = Depends(get_app_state),
):
    """最近N条请求详情记录"""
    return {"data": state.stats_manager.get_recent_records(limit)}


# ------------------------------------------------------------------
# 模型管理接口
# ------------------------------------------------------------------

@router.get("/api/models", tags=["模型管理"])
async def get_all_models(state: AppState = Depends(get_app_state)):
    models = state.model_manager.get_all_models()
    groups: Dict[str, List] = {}
    for m in models:
        owner = m.owned_by
        if owner not in groups:
            groups[owner] = []
        groups[owner].append(m.to_dict())
    return {
        "stats": state.model_manager.get_stats(),
        "groups": groups,
        "models": [m.to_dict() for m in models],
    }


@router.post("/api/models/{model_id:path}/enable", tags=["模型管理"])
async def enable_model(model_id: str, state: AppState = Depends(get_app_state)):
    success = state.model_manager.enable_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
    return {"success": True, "model_id": model_id, "enabled": True}


@router.post("/api/models/{model_id:path}/disable", tags=["模型管理"])
async def disable_model(model_id: str, state: AppState = Depends(get_app_state)):
    success = state.model_manager.disable_model(model_id)
    if not success:
        enabled_count = state.model_manager.get_enabled_count()
        if enabled_count <= 1:
            raise HTTPException(status_code=400, detail="至少需要保留1个启用的模型")
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
    return {"success": True, "model_id": model_id, "enabled": False}


@router.post("/api/models/{model_id:path}/toggle", tags=["模型管理"])
async def toggle_model(model_id: str, state: AppState = Depends(get_app_state)):
    result = state.model_manager.toggle_model(model_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
    return {"success": True, "model_id": model_id, "enabled": result}


@router.post("/api/models/enable-all", tags=["模型管理"])
async def enable_all_models(state: AppState = Depends(get_app_state)):
    state.model_manager.enable_all()
    return {"success": True, "message": "所有模型已启用"}


@router.post("/api/models/disable-all", tags=["模型管理"])
async def disable_all_models(state: AppState = Depends(get_app_state)):
    state.model_manager.disable_all_except_default()
    return {"success": True, "message": f"已禁用所有模型（保留默认模型: {state.default_model}）"}


@router.post("/api/models/fetch", tags=["模型管理"])
async def fetch_models(state: AppState = Depends(get_app_state)):
    success = await state.model_manager.fetch_from_nvidia()
    return {
        "success": success,
        "message": "拉取成功" if success else "拉取失败，已保留原有列表",
        "stats": state.model_manager.get_stats(),
    }


# ------------------------------------------------------------------
# OpenAI 兼容接口
# ------------------------------------------------------------------

@router.get("/v1/models", tags=["OpenAI兼容"])
async def list_models(state: AppState = Depends(get_app_state)):
    enabled_models = state.model_manager.get_enabled_models()
    return {"object": "list", "data": [m.to_dict() for m in enabled_models]}


@router.post("/v1/chat/completions", tags=["OpenAI兼容"])
async def chat_completions(request: ChatCompletionRequest, state: AppState = Depends(get_app_state)):
    model = request.model or state.default_model

    if state.model_manager and not state.model_manager.is_model_enabled(model):
        raise HTTPException(
            status_code=400,
            detail=f"模型 '{model}' 未启用或不存在，请在 Dashboard 中启用后使用"
        )

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    extra_params = dict(request.model_extra or {})

    logger.info(f"收到请求 | model={model} | stream={request.stream} | messages={len(messages)}条")

    try:
        if request.stream:
            async def stream_generator():
                async for chunk in state.proxy.chat_completion_stream(
                    messages=messages, model=model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    extra_params=extra_params,
                ):
                    yield chunk

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            response = await state.proxy.chat_completion(
                messages=messages, model=model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                extra_params=extra_params,
            )
            return JSONResponse(content=response.model_dump())

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")
