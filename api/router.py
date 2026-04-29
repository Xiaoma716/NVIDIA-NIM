"""
API路由层
完全兼容 OpenAI 格式 + Anthropic Messages API 格式
新增：/api/stats/* 统计数据接口
新增：/v1/messages Anthropic 兼容端点
重构：使用 FastAPI 依赖注入替代模块级全局变量
"""

import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from loguru import logger

from api.dashboard import DASHBOARD_HTML
from core.proxy import NvidiaProxy, AdmissionRejectedException
from core.key_pool import KeyPool
from core.model_manager import ModelManager
from core.stats_manager import StatsManager
from core.config import cfg
from core import anthropic_adapter


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


def normalize_content(content: Union[str, List, Dict]) -> Union[str, List]:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return content
    elif isinstance(content, dict):
        if "type" in content:
            return [content]
        else:
            text = content.get("text", str(content))
            parts = []
            if text:
                parts.append({"type": "text", "text": text})
            for key in ("image_url", "image", "video_url", "video", "audio_url", "audio"):
                if key in content and content[key]:
                    media_type = key.replace("_url", "")
                    parts.append({
                        "type": f"{media_type}_url" if "_url" in key else media_type,
                        f"{media_type}_url" if "_url" in key else media_type: content[key] if isinstance(content[key], dict) else {"url": content[key]}
                    })
            return parts if parts else str(content)
    else:
        return str(content)


# ------------------------------------------------------------------
# Pydantic 模型
# ------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict], Dict]
    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    model_config = {"extra": "allow"}


class AnthropicContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    source: Optional[Dict] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict] = None
    tool_use_id: Optional[str] = None
    content: Optional[Any] = None
    model_config = {"extra": "allow"}


class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[AnthropicContentBlock]]
    model_config = {"extra": "allow"}


class AnthropicToolInputSchema(BaseModel):
    type: str = "object"
    properties: Optional[Dict] = None
    required: Optional[List[str]] = None
    model_config = {"extra": "allow"}


class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: AnthropicToolInputSchema
    model_config = {"extra": "allow"}


class AnthropicRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int = 4096
    system: Optional[Union[str, List[AnthropicContentBlock]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[AnthropicTool]] = None
    metadata: Optional[Dict] = None
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
    """各模型调用量、Token消耗统计（用于饼图/排行），含性能指标"""
    return {"data": state.stats_manager.get_model_stats()}


@router.get("/api/stats/perf", tags=["统计分析"])
async def stats_perf(state: AppState = Depends(get_app_state)):
    """各模型性能指标：TTFT P50/P95、Tokens/s P50/P95（轻量接口，供模型卡片轮询）"""
    return {"data": state.stats_manager.get_all_model_perf()}


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
# OpenAI + Anthropic 兼容接口
# ------------------------------------------------------------------


@router.post("/v1/chat/completions", tags=["OpenAI兼容"])
async def chat_completions(request: ChatCompletionRequest, state: AppState = Depends(get_app_state)):
    model = request.model or state.default_model

    if state.model_manager and not state.model_manager.is_model_enabled(model):
        raise HTTPException(
            status_code=400,
            detail=f"模型 '{model}' 未启用或不存在，请在 Dashboard 中启用后使用"
        )

    messages = []
    for m in request.messages:
        normalized = normalize_content(m.content)
        messages.append({"role": m.role, "content": normalized})
        if isinstance(m.content, dict) and "type" not in m.content:
            logger.debug(f"检测到非标准content格式，已自动转换: {m.content} -> {normalized}")
        elif isinstance(normalized, list) and len(normalized) > 1:
            logger.debug(f"多模态消息: role={m.role}, content_parts={len(normalized)}")

    extra_params = dict(request.model_extra or {})

    logger.info(f"收到请求 | model={model} | stream={request.stream} | messages={len(messages)}条")
    logger.debug(f"完整请求体: {request.model_dump()}")

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

    except AdmissionRejectedException as e:
        logger.warning(f"准入控制拒绝: {e.info['error']}")
        raise HTTPException(
            status_code=429,
            detail=e.info,
            headers={"Retry-After": "5"},
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")


# ------------------------------------------------------------------
# Anthropic 兼容接口
# ------------------------------------------------------------------

@router.get("/v1/models", tags=["模型列表"])
async def list_models(request: Request, state: AppState = Depends(get_app_state)):
    enabled_models = state.model_manager.get_enabled_models()
    openai_data = [m.to_dict() for m in enabled_models]

    anthropic_version = request.headers.get("anthropic-version", "")
    if anthropic_version:
        return anthropic_adapter.convert_models_to_anthropic(openai_data)

    return {"object": "list", "data": openai_data}


@router.post("/v1/messages", tags=["Anthropic兼容"])
async def anthropic_messages(request: AnthropicRequest, state: AppState = Depends(get_app_state)):
    original_model = request.model
    nvidia_model = anthropic_adapter.map_model_to_nvidia(original_model)

    if state.model_manager and not state.model_manager.is_model_enabled(nvidia_model):
        err = anthropic_adapter.convert_error(
            400, f"模型 '{original_model}' (映射为 '{nvidia_model}') 未启用或不存在"
        )
        raise HTTPException(status_code=400, detail=err)

    anthropic_dict = request.model_dump()
    openai_req = anthropic_adapter.convert_request(anthropic_dict)

    messages = openai_req["messages"]
    model = openai_req["model"]
    temperature = openai_req.get("temperature", 0.7)
    max_tokens = openai_req.get("max_tokens", 4096)
    top_p = openai_req.get("top_p", 1.0)
    extra_params = {}
    if "stop" in openai_req:
        extra_params["stop"] = openai_req["stop"]
    if "tools" in openai_req:
        extra_params["tools"] = openai_req["tools"]

    logger.info(
        f"[Anthropic] 收到请求 | model={original_model} -> {model} | "
        f"stream={request.stream} | messages={len(messages)}条"
    )

    try:
        if request.stream:
            raw_stream = state.proxy.chat_completion_raw_stream(
                messages=messages, model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_params=extra_params,
            )

            async def anthropic_stream_generator():
                async for anthropic_chunk in anthropic_adapter.convert_stream(
                    raw_stream, original_model
                ):
                    yield anthropic_chunk

            return StreamingResponse(
                anthropic_stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            response = await state.proxy.chat_completion(
                messages=messages, model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_params=extra_params,
            )
            anthropic_resp = anthropic_adapter.convert_response(
                response.model_dump(), original_model
            )
            return JSONResponse(content=anthropic_resp)

    except AdmissionRejectedException as e:
        logger.warning(f"[Anthropic] 准入控制拒绝: {e.info['error']}")
        err = anthropic_adapter.convert_error(429, e.info)
        raise HTTPException(status_code=429, detail=err, headers={"Retry-After": "5"})
    except RuntimeError as e:
        err = anthropic_adapter.convert_error(503, str(e))
        raise HTTPException(status_code=503, detail=err)
    except Exception as e:
        logger.error(f"[Anthropic] 处理请求时发生错误: {e}")
        err = anthropic_adapter.convert_error(500, str(e))
        raise HTTPException(status_code=500, detail=err)
