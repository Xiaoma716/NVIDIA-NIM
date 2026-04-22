# NVIDIA NIM 多 Key 负载均衡代理服务

> 本地部署的 NVIDIA NIM API 代理网关，支持多 API Key 智能负载均衡、自动故障转移、实时监控面板，完全兼容 OpenAI 接口格式。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🔑 **多 Key 负载均衡** | 支持配置多个 NVIDIA API Key，自动分配请求，突破单 Key RPM 限制 |
| ⚖️ **三种均衡策略** | `most_remaining`（剩余最多优先）/ `round_robin`（轮询）/ `least_used`（最少使用） |
| 📊 **滑动窗口限流** | 精确追踪每个 Key 的 RPM 使用量，基于时间窗口动态计算剩余配额 |
| 🔄 **自动故障转移** | Key 被限流时自动切换，支持可配置的重试次数与超时等待 |
| 🩺 **健康检查** | 后台周期性检测 Key 状态，自动封禁异常 Key 并在恢复后解禁 |
| 🌐 **OpenAI 兼容** | 完全兼容 `/v1/chat/completions`、`/v1/models` 等 OpenAI 接口格式，可直接接入 ChatBox / LobeChat 等客户端 |
| 📡 **流式响应** | 完整支持 SSE 流式输出（Streaming） |
| 📈 **可视化监控面板** | 内置 Web Dashboard，实时展示请求趋势、Token 统计、Key 健康状态、调用记录 |
| 💾 **SQLite 持久化** | 异步批量写入（WriteBuffer），统计数据不丢，重启后历史数据保留 |

---

## 🏗️ 架构概览

```
客户端 (curl / ChatBox / LobeChat / 自定义脚本)
        │
        │  HTTP POST (OpenAI 格式)
        ▼
┌───────────────────────────────────────────────┐
│           FastAPI 本地代理服务                  │
│           http://localhost:8000                │
│                                               │
│   ┌─────────────┐  ┌──────────────────────┐   │
│   │ /v1/chat/*  │  │  /dashboard (Web面板) │   │
│   │ /v1/models  │  │  /api/stats/*         │   │
│   │ /health     │  └──────────────────────┘   │
│   └──────┬──────┘                             │
│          ▼                                    │
│   ┌──────────────────────────────┐            │
│   │       LoadBalancer           │            │
│   │                              │            │
│   │  ┌──────┐ ┌──────┐ ┌──────┐ │            │
│   │  │ Key1 │ │ Key2 │ │ Key3 │ │ ...        │
│   │  │ 35/40│ │ 封禁 │ │ 38/40│ │            │
│   │  └──────┘ └──────┘ └──────┘ │            │
│   └──────────────┬───────────────┘            │
└──────────────────│────────────────────────────┘
                   │  真实 HTTPS 请求
                   ▼
        ┌─────────────────────┐
        │  NVIDIA NIM API     │
        │  integrate.api.     │
        │  nvidia.com/v1      │
        └─────────────────────┘
```

---

## 📁 项目结构

```
NVIDIA-NIM/
├── main.py                 # 启动入口
├── requirements.txt        # Python 依赖
│
├── core/                   # 核心逻辑层
│   ├── config.py           # 集中化配置管理（单例模式）
│   ├── key_pool.py         # Key 池管理（滑动窗口计数、状态机）
│   ├── balancer.py         # 负载均衡器（3 种策略）
│   ├── proxy.py            # 请求转发（重试、流式、Token 提取）
│   ├── model_manager.py    # 模型列表管理与缓存
│   ├── stats_manager.py    # 统计数据核心（时序/Token/记录）
│   ├── database.py         # SQLite 初始化与 ORM
│   └── write_buffer.py     # 异步批量写入缓冲区
│
├── api/                    # HTTP 接口层
│   ├── router.py           # 路由定义（OpenAI 兼容 + 统计接口）
│   └── dashboard.py        # 监控面板 HTML（内嵌 ECharts）
│
├── static/
│   └── dashboard.html      # 静态资源
│
├── tests/                  # 测试
│   └── test_core.py
│
├── data/                   # 运行时数据（SQLite 数据库）
└── logs/                   # 运行日志
```

---

## 🚀 快速开始

### 环境要求

- **Python** >= 3.10
- **NVIDIA API Key**（一个或多个）

### 安装

```bash
# 克隆项目
git clone https://github.com/Xiaoma716/NVIDIA-NIM.git
cd NVIDIA-NIM

# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate      # Linux/macOS

# 安装依赖
pip install -r requirements.txt
```

### 配置

创建 `config.yaml` 文件（项目根目录），参考以下模板：

```yaml
keys:
  - key: "nvapi-xxxxxxxxxxxx"    # 你的 NVIDIA API Key
  - key: "nvapi-yyyyyyyyyyyy"    # 可添加多个 Key
  - key: "nvapi-zzzzzzzzzzzz"

nvidia:
  base_url: "https://integrate.api.nvidia.com/v1"
  default_model: "meta/llama-3.1-70b-instruct"
  rpm_limit: 40                    # 每 Key 每分钟请求上限
  rpm_buffer: 5                     # 安全余量

balancer:
  strategy: "most_remaining"        # 可选: most_remaining / round_robin / least_used
  wait_timeout: 65                  # 无可用 Key 时等待超时（秒）
  max_retries: 3                    # 失败重试次数

server:
  port: 8000                        # 服务监听端口

logging:
  level: "INFO"
  rotation: "50 MB"
  retention: "7 days"
```

### 启动服务

```bash
python main.py
```

启动成功后：
- **API 地址：** `http://localhost:8000/v1`
- **监控面板：** `http://localhost:8000/dashboard`
- **健康检查：** `http://localhost:8000/health`

### 使用示例

```bash
# Chat Completions（完全兼容 OpenAI 格式）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'

# 获取模型列表
curl http://localhost:8000/v1/models

# 健康检查
curl http://localhost:8000/health
```

也可直接将 `http://localhost:8000/v1` 作为 **API Base URL** 配置到以下客户端：

- [ChatBox](https://github.com/Bin-Huang/chatbox)
- [LobeChat](https://github.com/lobehub/lobe-chat)
- [NextChat](https://github.com/Yidadaa/ChatGPT-Next-Web)
- 任何支持自定义 OpenAI API 地址的工具

---

## ⚙️ 负载均衡策略说明

| 策略 | 名称 | 适用场景 |
|------|------|----------|
| `most_remaining` | **剩余配额优先**（默认） | 优先使用 RPM 余量最多的 Key，最大化整体吞吐量 |
| `round_robin` | **轮询** | 均匀分配请求到各 Key，简单公平 |
| `least_used` | **最少使用** | 优先使用总请求数最少的 Key，适合冷启动场景 |

---

## 📊 监控面板功能

访问 `http://localhost:8000/dashboard` 可查看：

- **📈 请求趋势折线图** — 请求量 / Token 用量 / 平均延迟（支持 1h / 6h / 24h 切换）
- **🥧 模型调用饼图** — 按调用量和 Token 双维度展示模型分布
- **📊 Key Token 柱状图** — 各 Key 的 Prompt / Completion Token 对比
- **📋 请求记录表** — 最近 80 条完整调用记录
- **🪙 Token KPI 卡片** — 累计总量、近 1h 量、分类统计
- **🔑 Key 池状态** — 每个 Key 的 RPM 使用率、健康状态、Token 统计

---

## 🔧 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 高性能异步 HTTP 服务 |
| API 客户端 | OpenAI Python SDK | 与 NVIDIA NIM 通信 |
| 配置管理 | PyYAML | YAML 配置文件解析 |
| 日志 | Loguru | 结构化日志输出 |
| HTTP 客户端 | httpx | 底层异步 HTTP 请求 |
| 数据库 | SQLAlchemy + aiosqlite | SQLite 异步持久化 |
| 前端图表 | ECharts | Dashboard 数据可视化 |

---

## 📋 API 接口清单

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/chat/completions` | 对话补全（兼容 OpenAI） |
| GET | `/v1/models` | 获取可用模型列表 |
| GET | `/health` | 服务健康检查 |
| GET | `/dashboard` | Web 监控面板 |
| GET | `/api/stats/overview` | 统计概览（KPI） |
| GET | `/api/stats/trend` | 时序趋势数据 |
| GET | `/api/stats/models` | 模型分布统计 |
| GET | `/api/stats/keys` | Key 详情统计 |
| GET | `/api/stats/records` | 最近请求记录 |

---

## 🛠️ 开发与测试

```bash
# 运行测试
python -m pytest tests/

# 或直接运行
python test_client.py
```

---

## 📄 许可证

MIT License

---

## ⚠️ 免责声明

本项目为个人学习与研究目的开发，仅供本地使用。使用者需自行承担使用 NVIDIA API 所产生的一切费用与责任。本项目与 NVIDIA 公司无任何关联。
