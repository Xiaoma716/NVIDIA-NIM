"""
模型管理模块
负责：
  1. 启动时从 NVIDIA API 自动拉取完整模型列表
  2. 本地持久化每个模型的启用/禁用状态（models_state.json）
  3. 提供启用/禁用/查询接口供 Dashboard 和 Router 调用
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx
from loguru import logger

from core.config import cfg


# 持久化文件路径
STATE_FILE = Path("models_state.json")


class ModelInfo:
    """单个模型的信息与状态"""

    def __init__(
        self,
        model_id: str,
        owned_by: str = "nvidia",
        enabled: bool = True,
        created: int = 0,
        extra: Optional[Dict] = None,
    ):
        self.model_id = model_id
        self.owned_by = owned_by
        self.enabled = enabled
        self.created = created or int(time.time())
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.model_id,
            "owned_by": self.owned_by,
            "enabled": self.enabled,
            "created": self.created,
            "object": "model",
        }

    def to_state_dict(self) -> Dict[str, Any]:
        """用于持久化到JSON文件"""
        return {
            "model_id": self.model_id,
            "owned_by": self.owned_by,
            "enabled": self.enabled,
            "created": self.created,
        }


class ModelManager:
    """
    模型状态管理器
    - 启动时从 NVIDIA 拉取模型列表
    - 本地 JSON 持久化 enabled/disabled 状态
    - 线程安全的启用/禁用操作
    """

    def __init__(
        self,
        base_url: str,
        api_keys: List[str],
        fallback_list: List[str],
        auto_fetch: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_keys = api_keys
        self.fallback_list = fallback_list
        self.auto_fetch = auto_fetch

        # 模型字典：model_id -> ModelInfo
        self._models: Dict[str, ModelInfo] = {}
        self._lock = threading.Lock()

        # 记录最后一次从NVIDIA拉取的时间
        self.last_fetch_time: Optional[float] = None
        self.last_fetch_status: str = "未拉取"
        self._http_client: Optional[httpx.AsyncClient] = None

    @property
    def default_model(self) -> str:
        return cfg.default_model

    # ------------------------------------------------------------------
    # 初始化入口
    # ------------------------------------------------------------------

    async def initialize(self):
        """
        异步初始化：
        1. 先加载本地持久化状态
        2. 再尝试从 NVIDIA 拉取最新列表（补充新模型）
        """
        # Step 1: 加载本地已保存的状态
        self._load_state_from_file()

        # Step 2: 自动拉取（如果配置开启）
        if self.auto_fetch:
            success = await self.fetch_from_nvidia()
            if not success:
                logger.warning("从 NVIDIA 拉取模型列表失败，使用兜底预置列表")
                self._load_fallback_list()
        else:
            # 不自动拉取时，至少确保兜底列表存在
            if not self._models:
                self._load_fallback_list()

        logger.info(f"模型管理器初始化完成，共 {len(self._models)} 个模型，"
                    f"已启用 {self.get_enabled_count()} 个")

        default = self.default_model
        if default and default in self._models:
            if not self._models[default].enabled:
                self._models[default].enabled = True
                self._save_state_to_file()
                logger.info(f"已自动启用配置中的默认模型: {default}")

    # ------------------------------------------------------------------
    # 从 NVIDIA 拉取模型列表
    # ------------------------------------------------------------------

    async def fetch_from_nvidia(self) -> bool:
        """
        向 NVIDIA API 发起真实请求，获取最新模型列表
        依次尝试所有 Key，任一成功即返回
        """
        if not self.api_keys:
            logger.error("没有可用的 API Key，无法拉取模型列表")
            return False

        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=15.0)

        url = f"{self.base_url}/models"

        for key_index, api_key in enumerate(self.api_keys):
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            try:
                logger.info(f"正在从 NVIDIA 拉取模型列表 (Key-{key_index + 1}/{len(self.api_keys)}): {url}")
                response = await self._http_client.get(url, headers=headers)

                if response.status_code != 200:
                    logger.warning(f"Key-{key_index + 1} 返回 HTTP {response.status_code}，尝试下一个...")
                    continue

                data = response.json()
                raw_models = data.get("data", [])

                if not raw_models:
                    logger.warning(f"Key-{key_index + 1} 返回空列表，尝试下一个...")
                    continue

                new_count = self._merge_models(raw_models)

                self.last_fetch_time = time.time()
                self.last_fetch_status = f"成功 (共{len(raw_models)}个，新增{new_count}个)"
                logger.info(
                    f"NVIDIA 模型列表拉取成功 ✅ | "
                    f"Key: Key-{key_index + 1} | 总计: {len(raw_models)} 个 | 新增: {new_count} 个"
                )

                self._save_state_to_file()
                return True

            except httpx.TimeoutException:
                logger.warning(f"Key-{key_index + 1} 拉取超时，尝试下一个...")
                continue
            except Exception as e:
                logger.warning(f"Key-{key_index + 1} 拉取异常: {e}，尝试下一个...")
                continue

        self.last_fetch_status = "失败 (所有Key均失败)"
        logger.error("所有 API Key 拉取模型列表均失败")
        return False

    def _merge_models(self, raw_models: List[Dict]) -> int:
        """将拉取到的模型合并到本地字典，返回新增数量"""
        new_count = 0
        with self._lock:
            for item in raw_models:
                model_id = item.get("id", "")
                if not model_id:
                    continue
                if model_id not in self._models:
                    self._models[model_id] = ModelInfo(
                        model_id=model_id,
                        owned_by=item.get("owned_by", "nvidia"),
                        enabled=(model_id == self.default_model),
                        created=item.get("created", int(time.time())),
                    )
                    new_count += 1
                else:
                    self._models[model_id].owned_by = item.get("owned_by", "nvidia")
        return new_count

    # ------------------------------------------------------------------
    # 本地兜底列表
    # ------------------------------------------------------------------

    def _load_fallback_list(self):
        """加载配置中的兜底预置模型列表（不覆盖已有状态）"""
        with self._lock:
            added = 0
            for model_id in self.fallback_list:
                if model_id not in self._models:
                    self._models[model_id] = ModelInfo(
                        model_id=model_id,
                        owned_by=self._guess_owner(model_id),
                        enabled=(model_id == self.default_model),
                    )
                    added += 1
        if added:
            logger.info(f"从兜底列表加载了 {added} 个预置模型（仅启用默认模型: {self.default_model}）")
            self._save_state_to_file()

    # ------------------------------------------------------------------
    # 持久化：读/写 models_state.json
    # ------------------------------------------------------------------

    def _save_state_to_file(self):
        """将当前所有模型状态保存到 JSON 文件"""
        try:
            with self._lock:
                state = {
                    model_id: model.to_state_dict()
                    for model_id, model in self._models.items()
                }
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            logger.debug(f"模型状态已保存到 {STATE_FILE}")
        except Exception as e:
            logger.error(f"保存模型状态失败: {e}")

    def _load_state_from_file(self):
        """从 JSON 文件恢复模型状态（包含用户之前设置的 enabled/disabled）"""
        if not STATE_FILE.exists():
            logger.info("未找到本地模型状态文件，将从头初始化")
            return

        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state: Dict = json.load(f)

            with self._lock:
                for model_id, info in state.items():
                    self._models[model_id] = ModelInfo(
                        model_id=info.get("model_id", model_id),
                        owned_by=info.get("owned_by", "nvidia"),
                        enabled=info.get("enabled", True),
                        created=info.get("created", int(time.time())),
                    )
            logger.info(f"从本地文件恢复了 {len(self._models)} 个模型状态")
        except Exception as e:
            logger.error(f"读取本地模型状态文件失败: {e}")

    # ------------------------------------------------------------------
    # 模型 CRUD 操作
    # ------------------------------------------------------------------

    def enable_model(self, model_id: str) -> bool:
        """启用指定模型，返回操作是否成功"""
        with self._lock:
            if model_id not in self._models:
                return False
            self._models[model_id].enabled = True
        self._save_state_to_file()
        logger.info(f"模型已启用: {model_id}")
        return True

    def disable_model(self, model_id: str) -> bool:
        """禁用指定模型，返回操作是否成功"""
        with self._lock:
            if model_id not in self._models:
                return False
            # 不允许禁用全部模型（至少保留1个）
            enabled_count = sum(1 for m in self._models.values() if m.enabled)
            if enabled_count <= 1:
                logger.warning("至少需要保留1个启用的模型，操作被拒绝")
                return False
            self._models[model_id].enabled = False
        self._save_state_to_file()
        logger.info(f"模型已禁用: {model_id}")
        return True

    def toggle_model(self, model_id: str) -> Optional[bool]:
        """切换模型启用状态，返回切换后的状态（None表示模型不存在）"""
        with self._lock:
            if model_id not in self._models:
                return None
            current = self._models[model_id].enabled
        if current:
            success = self.disable_model(model_id)
            return not current if success else current
        else:
            success = self.enable_model(model_id)
            return not current if success else current

    def enable_all(self):
        """启用所有模型"""
        with self._lock:
            for model in self._models.values():
                model.enabled = True
        self._save_state_to_file()
        logger.info("已启用所有模型")

    def disable_all_except_default(self):
        """禁用除默认模型外的所有模型"""
        default = self.default_model
        with self._lock:
            for model_id, model in self._models.items():
                model.enabled = (model_id == default)
        self._save_state_to_file()

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_enabled_models(self) -> List[ModelInfo]:
        """返回所有已启用的模型列表"""
        with self._lock:
            return [m for m in self._models.values() if m.enabled]

    def get_all_models(self) -> List[ModelInfo]:
        """返回所有模型列表（含禁用）"""
        with self._lock:
            return list(self._models.values())

    def get_enabled_count(self) -> int:
        with self._lock:
            return sum(1 for m in self._models.values() if m.enabled)

    def is_model_enabled(self, model_id: str) -> bool:
        """检查指定模型是否启用"""
        with self._lock:
            model = self._models.get(model_id)
            return model.enabled if model else False

    def get_stats(self) -> Dict[str, Any]:
        """返回模型管理器统计信息"""
        with self._lock:
            total = len(self._models)
            enabled = sum(1 for m in self._models.values() if m.enabled)
        return {
            "total_models": total,
            "enabled_models": enabled,
            "disabled_models": total - enabled,
            "last_fetch_time": self.last_fetch_time,
            "last_fetch_status": self.last_fetch_status,
            "auto_fetch": self.auto_fetch,
        }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _guess_owner(model_id: str) -> str:
        """根据 model_id 前缀猜测所属组织"""
        prefix_map = {
            "meta/": "meta",
            "mistralai/": "mistralai",
            "google/": "google",
            "microsoft/": "microsoft",
            "nvidia/": "nvidia",
            "deepseek-ai/": "deepseek-ai",
            "qwen/": "qwen",
            "01-ai/": "01-ai",
            "baichuan-inc/": "baichuan",
        }
        for prefix, owner in prefix_map.items():
            if model_id.startswith(prefix):
                return owner
        return "nvidia"