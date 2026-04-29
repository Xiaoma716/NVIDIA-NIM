# NVIDIA NIM Load Balancer — Anthropic 格式适配开发文档

> 版本: v1.0 | 日期: 2026-04-29  
> 目标: 在现有 OpenAI 兼容层基础上，新增 Anthropic Messages API 兼容层，使项目可接入 Claude Code 等工具

---

## 一、背景与可行性分析

### 1.1 现状

当前项目仅暴露 OpenAI 格式端点：

| 端点 | 用途 |
|------|------|
| `POST /v1/chat/completions` | 对话补全（流式/非流式） |
| `GET /v1/models` | 模型列表 |

这使项目可以接入 OpenCode、ChatBox 等支持 OpenAI 格式的客户端，但 **无法接入 Claude Code**，因为 Claude Code 使用 Anthropic Messages API 格式。

### 1.2 可行性结论：✅ 完全可行

**核心理由：**

1. **后端不变** — 项目底层通过 `openai` SDK 调用 NVIDIA NIM API（OpenAI 兼容），这部分无需修改
2. **纯协议转换** — 只需在 API 层新增 Anthropic 格式的请求解析 + 响应转换层，将 Anthropic 格式 ↔ OpenAI 格式进行双向转换
3. **架构天然支持** — 项目已有清晰的分层：Router（协议层）→ Proxy（转发层）→ Balancer（调度层），只需在 Router 层扩展

**架构示意：**

```
Claude Code ──Anthropic格式──▶ Router (/v1/messages)
                                    │
OpenCode ────OpenAI格式────▶ Router (/v1/chat/completions)
                                    │
                                    ▼
                              NvidiaProxy（不变）
                                    │
                                    ▼
                           NVIDIA NIM API（不变）
```

---

## 二、Anthropic vs OpenAI 格式差异详解

### 2.1 端点差异

| 功能 | OpenAI | Anthropic |
|------|--------|-----------|
| 对话补全 | `POST /v1/chat/completions` | `POST /v1/messages` |
| 模型列表 | `GET /v1/models` | 无标准端点（需自定义） |

### 2.2 请求格式差异

**OpenAI 请求：**
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "top_p": 1.0,
  "stream": false
}
```

**Anthropic 请求：**
```json
{
  "model": "claude-3-opus-20240229",
  "system": "You are helpful.",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 1024,
  "temperature": 0.7,
  "top_p": 1.0,
  "stream": false
}
```

**关键差异：**
- Anthropic 的 `system` 是顶层参数（字符串或 content blocks 数组），不在 `messages` 中
- Anthropic 的 `max_tokens` 是**必填**参数
- Anthropic 的 `messages` 不允许出现 `role: "system"`
- Anthropic 的 `messages` 必须以 `role: "user"` 开头，且 user/assistant 严格交替
- Anthropic 支持 `thinking` 类型的 content block（扩展思考功能）
- Anthropic 支持 `tool_use` / `tool_result` 类型的 content block

### 2.3 响应格式差异

**OpenAI 非流式响应：**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hi there!"},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
```

**Anthropic 非流式响应：**
```json
{
  "id": "msg_xxx",
  "type": "message",
  "role": "assistant",
  "model": "claude-3-opus-20240229",
  "content": [
    {"type": "text", "text": "Hi there!"}
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 10,
    "output_tokens": 5
  }
}
```

**关键差异：**
- Anthropic 的 `content` 是数组（支持多个 content block），OpenAI 的 `content` 是字符串
- Anthropic 的 `stop_reason` 对应 OpenAI 的 `finish_reason`（值映射不同，见下表）
- Anthropic 的 `usage` 用 `input_tokens` / `output_tokens`，OpenAI 用 `prompt_tokens` / `completion_tokens`

**stop_reason / finish_reason 映射：**

| OpenAI `finish_reason` | Anthropic `stop_reason` |
|------------------------|-------------------------|
| `stop` | `end_turn` |
| `length` | `max_tokens` |
| `tool_calls` | `tool_use` |
| `content_filter` | `stop_sequence` |

### 2.4 流式响应差异（最复杂的部分）

**OpenAI SSE 格式：**
```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Hi"}}]}\n\n
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":" there"}}]}\n\n
data: {"id":"chatcmpl-xxx","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n
data: [DONE]\n\n
```

**Anthropic SSE 格式：**
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_xxx","type":"message","role":"assistant","model":"...","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}\n\n

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}\n\n

event: content_block_stop
data: {"type":"content_block_stop","index":0}\n\n

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n

event: message_stop
data: {"type":"message_stop"}\n\n
```

**关键差异：**
- Anthropic 使用 `event:` 行标识事件类型，OpenAI 不使用
- Anthropic 有明确的生命周期：`message_start` → `content_block_start` → `content_block_delta`(多次) → `content_block_stop` → `message_delta` → `message_stop`
- Anthropic 的 `message_start` 携带完整消息元信息
- Anthropic 的 `message_delta` 携带 stop_reason 和 output_tokens

### 2.5 认证头差异

| 客户端 | 认证方式 |
|--------|----------|
| OpenAI 格式 | `Authorization: Bearer <api_key>` |
| Anthropic 格式 | `x-api-key: <api_key>` + `anthropic-version: 2023-06-01` |

### 2.6 错误格式差异

**OpenAI 错误：**
```json
{
  "error": {
    "message": "...",
    "type": "...",
    "code": "..."
  }
}
```

**Anthropic 错误：**
```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "..."
  }
}
```

---

## 三、详细设计方案

### 3.1 新增模块：`core/anthropic_adapter.py`

这是整个适配的核心模块，负责 Anthropic ↔ OpenAI 格式的双向转换。

```
core/
  anthropic_adapter.py    ← 新增：格式转换适配器
```

**核心类设计：**

```python
class AnthropicAdapter:
    """Anthropic ↔ OpenAI 格式双向转换器"""

    # --- 请求转换 ---
    @staticmethod
    def convert_request(anthropic_req: dict) -> dict:
        """Anthropic 请求 → OpenAI 请求格式
        - 提取顶层 system 参数，转为 system message 插入 messages 头部
        - 处理 content blocks（text / image / tool_use / tool_result）
        - 映射参数名差异
        """

    @staticmethod
    def convert_messages(anthropic_messages: list, system: str | list | None) -> list:
        """转换消息列表
        - system → system message
        - content blocks → OpenAI 兼容格式
        - tool_use / tool_result → OpenAI tool_calls / tool message
        - 确保 user/assistant 交替（必要时合并相邻同 role 消息）
        """

    # --- 非流式响应转换 ---
    @staticmethod
    def convert_response(openai_response: dict, original_model: str) -> dict:
        """OpenAI 响应 → Anthropic 响应格式
        - choices[0].message.content → content blocks 数组
        - finish_reason → stop_reason 映射
        - usage 字段映射
        """

    # --- 流式响应转换 ---
    @staticmethod
    async def convert_stream(openai_stream, model: str) -> AsyncGenerator[str, None]:
        """OpenAI SSE 流 → Anthropic SSE 流
        - 生成 message_start 事件
        - 生成 content_block_start 事件
        - 将每个 delta.content 转为 content_block_delta 事件
        - 检测 finish_reason 生成 content_block_stop + message_delta + message_stop
        """

    # --- 错误转换 ---
    @staticmethod
    def convert_error(status_code: int, detail: any) -> dict:
        """OpenAI/HTTP 错误 → Anthropic 错误格式"""
```

### 3.2 修改模块：`api/router.py`

新增 Anthropic 兼容端点：

```python
# --- 新增 Pydantic 模型 ---

class AnthropicMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: Union[str, List[Dict]]

class AnthropicRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int = 1024          # Anthropic 必填
    system: Optional[Union[str, List[Dict]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    model_config = {"extra": "allow"}

# --- 新增端点 ---

@router.post("/v1/messages", tags=["Anthropic兼容"])
async def anthropic_messages(request: AnthropicRequest, ...):
    """Anthropic Messages API 兼容端点"""

@router.get("/v1/models", tags=["模型列表"])
# 需要增强：根据请求头判断返回 OpenAI 还是 Anthropic 格式
```

### 3.3 修改模块：`core/proxy.py`

**无需修改核心逻辑**，但需要暴露一个更底层的接口，让 Anthropic 适配层可以获取原始的 OpenAI 流式数据用于格式转换：

```python
# 现有 chat_completion_stream 返回的是已格式化的 SSE 字符串
# 需要新增一个方法返回原始的 OpenAI stream 对象，供适配层消费

async def chat_completion_raw_stream(self, ...) -> AsyncGenerator[dict, None]:
    """返回原始 OpenAI chunk dict（而非 SSE 字符串），供格式转换层使用"""
```

### 3.4 认证头处理

Claude Code 发送请求时使用 `x-api-key` 头，项目需要在中间件或端点中兼容此认证方式：

```python
# 在 router 端点中，从 x-api-key 头提取 API Key（可选，用于透传）
# 当前项目的 Key 管理是服务端的，客户端 Key 仅做透传校验
# Claude Code 需要配置 API Key，我们可以接受任意值或从 x-api-key 提取
```

### 3.5 模型名称映射

Claude Code 期望使用 `claude-3-opus-20240229` 等模型名，但后端是 NVIDIA NIM 模型。需要提供映射机制：

**方案：配置文件映射 + 透传**

```yaml
# config.yaml 新增
anthropic:
  model_mapping:
    "claude-opus-4-7": "qwen/qwen3.5-122b-a10b"
    "claude-sonnet-4-6": "moonshotai/kimi-k2-instruct-0905"
    "claude-haiku-4-5-20251001": "openai/gpt-oss-120b"
  default_model: "claude-sonnet-4-6"
```

### 3.6 流式转换详细流程

这是最复杂的部分，详细设计如下：

```
OpenAI Stream                Anthropic Stream (输出)
─────────────                ──────────────────────
(连接建立)          →  event: message_start
                              data: {type:"message_start", message:{...}}

(首个 chunk)        →  event: content_block_start
                              data: {type:"content_block_start", index:0, ...}

chunk.delta.content →  event: content_block_delta
"Hello"                       data: {type:"content_block_delta", index:0, delta:{text:"Hello"}}

chunk.delta.content →  event: content_block_delta
" world"                      data: {type:"content_block_delta", index:0, delta:{text:" world"}}

chunk.finish_reason  →  event: content_block_stop
= "stop"                      data: {type:"content_block_stop", index:0}

                       event: message_delta
                              data: {type:"message_delta", delta:{stop_reason:"end_turn"}, ...}

                       event: message_stop
                              data: {type:"message_stop"}
```

**特殊处理：**
- OpenAI 的 `chunk.choices` 为空时（usage chunk），提取 token 信息用于 `message_delta` 的 `usage`
- 如果 OpenAI 流中包含 tool_calls，需要转为 Anthropic 的 `tool_use` content block
- 需要处理 OpenAI 流中 `role` 字段（首次出现时用于 `message_start`）

---

## 四、文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `core/anthropic_adapter.py` | **新增** | Anthropic ↔ OpenAI 格式转换核心逻辑 |
| `api/router.py` | **修改** | 新增 `/v1/messages` 端点、Anthropic Pydantic 模型、增强 `/v1/models` |
| `core/proxy.py` | **修改** | 新增 `chat_completion_raw_stream` 方法，返回原始 chunk dict |
| `core/config.py` | **修改** | 新增 `anthropic` 配置属性、模型映射 |
| `config.yaml` | **修改** | 新增 `anthropic` 配置节 |
| `main.py` | **微调** | 启动日志中新增 Anthropic 端点信息 |
| `test_client.py` | **修改** | 新增 Anthropic 格式测试用例 |

---

## 五、开发步骤（建议顺序）

### Step 1: 新增 `core/anthropic_adapter.py`

实现完整的格式转换逻辑，包括：
- `convert_request()`: Anthropic 请求 → OpenAI 请求
- `convert_response()`: OpenAI 响应 → Anthropic 响应
- `convert_stream()`: OpenAI SSE → Anthropic SSE（最复杂）
- `convert_error()`: 错误格式转换
- `map_model_name()`: 模型名称映射
- `map_stop_reason()`: stop_reason / finish_reason 映射

### Step 2: 修改 `core/proxy.py`

新增 `chat_completion_raw_stream()` 方法：
- 与现有 `chat_completion_stream()` 逻辑相同
- 但 yield 原始的 `chunk.model_dump()` dict 而非 SSE 字符串
- 供 Anthropic 适配层消费后重新格式化

### Step 3: 修改 `core/config.py`

新增 `anthropic` 配置属性：
- `model_mapping`: 模型名称映射字典
- `default_model`: Anthropic 默认模型名

### Step 4: 修改 `api/router.py`

- 新增 `AnthropicRequest` / `AnthropicMessage` Pydantic 模型
- 新增 `POST /v1/messages` 端点
- 增强 `GET /v1/models`：根据 `anthropic-version` 请求头判断返回格式
- 新增 `GET /v1/models` 的 Anthropic 格式变体

### Step 5: 更新 `config.yaml` 示例

新增 `anthropic` 配置节。

### Step 6: 更新 `main.py` 启动日志

新增 Anthropic 端点信息展示。

### Step 7: 新增测试

- 新增 Anthropic 格式的请求/响应转换单元测试
- 新增 Anthropic 端点集成测试
- 新增流式转换测试

---

## 六、风险与注意事项

### 6.1 Tool Use 支持

Claude Code 大量使用 tool_use（函数调用），这是核心功能。需要确保：
- Anthropic 的 `tool_use` content block 正确转换为 OpenAI 的 `tool_calls`
- Anthropic 的 `tool_result` content block 正确转换为 OpenAI 的 `tool` role message
- 流式场景下 `tool_use` 的 `content_block_start/delta/stop` 正确映射

### 6.2 多模态内容

Anthropic 的图片格式与 OpenAI 不同：
- Anthropic: `{"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}`
- OpenAI: `{"type": "image_url", "image_url": {"url": "data:...;base64,..."}}`

需要做格式转换。但 NVIDIA NIM 后端对多模态的支持取决于具体模型。

### 6.3 Thinking / Extended Thinking

Anthropic 的 `thinking` content block（扩展思考）是 Claude 特有功能，NVIDIA NIM 后端不支持。建议：
- 请求中忽略 `thinking` 相关参数
- 响应中不生成 `thinking` content block

### 6.4 消息交替规则

Anthropic 要求 messages 中 user/assistant 严格交替，且必须以 user 开头。在转换为 OpenAI 格式时需要：
- 合并相邻同 role 的消息
- 如果首条不是 user，需自动添加占位 user 消息

### 6.5 Token 计数差异

Anthropic 的 token 计数方式与 OpenAI 不同，转换后的 `input_tokens` / `output_tokens` 是从 OpenAI 的 `prompt_tokens` / `completion_tokens` 直接映射，数值可能不完全准确，但对 Claude Code 功能无影响。

### 6.6 并发与性能

新增的格式转换层是纯 CPU 计算，开销极小，不会成为性能瓶颈。流式转换是逐 chunk 处理，延迟增加可忽略（< 1ms/chunk）。

---

## 七、Claude Code 接入配置示例

升级完成后，Claude Code 的配置方式：

```bash
# 设置环境变量
export ANTHROPIC_API_KEY=any-value-works
export ANTHROPIC_BASE_URL=http://localhost:8000

# 或在 Claude Code 配置文件中
# api_base_url: http://localhost:8000
# api_key: any-value-works
```

Claude Code 将发送请求到 `http://localhost:8000/v1/messages`，项目自动完成格式转换后转发到 NVIDIA NIM。

---

## 八、预期工作量

| 步骤 | 复杂度 | 说明 |
|------|--------|------|
| Step 1: anthropic_adapter.py | ⭐⭐⭐⭐ | 核心模块，流式转换最复杂 |
| Step 2: proxy.py 修改 | ⭐⭐ | 新增一个方法，逻辑复用 |
| Step 3: config.py 修改 | ⭐ | 简单配置扩展 |
| Step 4: router.py 修改 | ⭐⭐⭐ | 新增端点 + Pydantic 模型 |
| Step 5-6: 配置 + 日志 | ⭐ | 简单 |
| Step 7: 测试 | ⭐⭐⭐ | 需覆盖各种边界情况 |

**整体评估：中等复杂度，核心难点在流式 SSE 格式转换和 Tool Use 映射。**
