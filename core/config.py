"""
集中化配置管理模块
单一入口加载 config.yaml，所有模块通过 import cfg 获取配置值
修改 config.yaml 中任意字段后，全系统自动生效（需重启）
"""

import sys
from pathlib import Path

import yaml


class _Config:
    """全局配置单例，启动时从 config.yaml 加载一次"""

    def __init__(self):
        self._data: dict = {}
        self._loaded = False

    def load(self, path: str = "config.yaml"):
        config_path = Path(path)
        if not config_path.exists():
            print(f"找不到配置文件: {path}")
            sys.exit(1)
        with open(config_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f) or {}
        if not self._data:
            print("config.yaml 内容为空")
            sys.exit(1)
        self._loaded = True
        return self._data

    def get(self, key_path: str, default=None):
        """
        支持点号路径访问嵌套配置
        例: cfg.get("nvidia.default_model") -> "meta/llama-3.1-70b-instruct"
        """
        keys = key_path.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    @property
    def data(self) -> dict:
        return self._data

    @property
    def nvidia(self) -> dict:
        return self._data.get("nvidia", {})

    @property
    def server(self) -> dict:
        return self._data.get("server", {})

    @property
    def balancer(self) -> dict:
        return self._data.get("balancer", {})

    @property
    def models(self) -> dict:
        return self._data.get("models", {})

    @property
    def keys(self) -> list:
        return self._data.get("keys", [])

    @property
    def logging(self) -> dict:
        return self._data.get("logging", {})

    @property
    def default_model(self) -> str:
        return self.nvidia.get("default_model", "meta/llama-3.1-70b-instruct")

    @property
    def base_url(self) -> str:
        return self.nvidia.get("base_url", "https://integrate.api.nvidia.com/v1")

    @property
    def rpm_limit(self) -> int:
        return self.nvidia.get("rpm_limit", 40)

    @property
    def rpm_buffer(self) -> int:
        return self.nvidia.get("rpm_buffer", 5)

    @property
    def anthropic(self) -> dict:
        return self._data.get("anthropic", {})

    @property
    def anthropic_model_mapping(self) -> dict:
        return self.anthropic.get("model_mapping", {})

    @property
    def anthropic_default_model(self) -> str:
        return self.anthropic.get("default_model", "claude-3-sonnet-20240229")


cfg = _Config()
