"""
单元测试：覆盖 KeyPool / LoadBalancer / StatsManager / Config 校验
运行方式: pytest tests/ -v
"""

import time
import pytest
import threading

from core.key_pool import APIKey, KeyPool
from core.balancer import LoadBalancer
from core.stats_manager import StatsManager


# ====================================================================
# APIKey 测试
# ====================================================================

class TestAPIKey:
    def test_key_initially_available(self):
        key = APIKey(key="test-key-123", alias="TestKey")
        assert key.is_available() is True

    def test_key_reaches_rpm_limit(self):
        key = APIKey(key="k", alias="K", rpm_limit=5, rpm_buffer=1)
        for _ in range(3):
            key.record_request()
        assert key.is_available() is True
        key.record_request()
        assert key.is_available() is False

    def test_ban_and_auto_unban(self):
        key = APIKey(key="k", alias="K")
        key.record_rate_limit_error()
        assert key.is_available() is False
        assert key.get_ban_remaining_seconds() > 0

        key._ban_until = time.time() - 1
        assert key.is_available() is True

    def test_disable_and_enable(self):
        key = APIKey(key="k", alias="K")
        assert key.is_available() is True
        key.disable("test reason")
        assert key.is_available() is False
        assert key._is_disabled is True
        key.enable()
        assert key.is_available() is True
        assert key._is_disabled is False

    def test_stats_return_structure(self):
        key = APIKey(key="k", alias="MyKey")
        stats = key.get_stats()
        assert "alias" in stats
        assert "available" in stats
        assert "is_disabled" in stats
        assert "is_banned" in stats
        assert "total_requests" in stats
        assert "error_rate_percent" in stats
        assert stats["alias"] == "MyKey"

    def test_concurrent_record_request(self):
        key = APIKey(key="k", alias="K", rpm_limit=100)
        errors = []

        def worker():
            try:
                for _ in range(50):
                    key.record_request()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert key._total_requests == 200

    def test_old_timestamps_cleaned(self):
        key = APIKey(key="k", alias="K", rpm_limit=5, rpm_buffer=1)
        old_time = time.time() - 120
        with key._lock:
            for _ in range(10):
                key._timestamps.append(old_time)
        assert len(key._timestamps) == 10
        assert key.is_available() is True
        assert len(key._timestamps) == 0


# ====================================================================
# KeyPool 测试
# ====================================================================

class TestKeyPool:
    def _make_pool(self):
        return KeyPool(
            keys_config=[
                {"key": "key-a", "alias": "Key-A"},
                {"key": "key-b", "alias": "Key-B"},
                {"key": "key-c", "alias": "Key-C"},
            ],
            rpm_limit=40,
            rpm_buffer=5,
        )

    def test_pool_initialization(self):
        pool = self._make_pool()
        assert len(pool.keys) == 3
        summary = pool.get_pool_summary()
        assert summary["total_keys"] == 3
        assert summary["available_keys"] == 3

    def test_get_available_keys(self):
        pool = self._make_pool()
        available = pool.get_available_keys()
        assert len(available) == 3

    def test_disable_key_via_pool(self):
        pool = self._make_pool()
        assert pool.disable_key("Key-B", "manual test") is True
        available = pool.get_available_keys()
        assert len(available) == 2
        aliases = [k.alias for k in available]
        assert "Key-B" not in aliases

    def test_enable_key_via_pool(self):
        pool = self._make_pool()
        pool.disable_key("Key-B", "test")
        assert pool.enable_key("Key-B") is True
        assert len(pool.get_available_keys()) == 3

    def test_disable_nonexistent_key(self):
        pool = self._make_pool()
        assert pool.disable_key("NonExistent") is False

    def test_pool_summary_with_disabled(self):
        pool = self._make_pool()
        pool.disable_key("Key-A", "test")
        summary = pool.get_pool_summary()
        assert summary["available_keys"] == 2
        assert summary["unavailable_keys"] == 1


# ====================================================================
# LoadBalancer 测试
# ====================================================================

class TestLoadBalancer:

    def _make_balancer(self, strategy="most_remaining"):
        pool = KeyPool(
            keys_config=[
                {"key": "k1", "alias": "K1"},
                {"key": "k2", "alias": "K2"},
                {"key": "k3", "alias": "K3"},
            ],
            rpm_limit=40,
            rpm_buffer=5,
        )
        return LoadBalancer(key_pool=pool, strategy=strategy)

    def test_most_remaining_strategy(self):
        balancer = self._make_balancer("most_remaining")
        key = balancer.get_key()
        assert key is not None
        assert key.alias in ("K1", "K2", "K3")

    def test_round_robin_strategy(self):
        balancer = self._make_balancer("round_robin")
        selected = []
        for _ in range(6):
            k = balancer.get_key()
            selected.append(k.alias)
        assert len(set(selected)) > 1

    def test_least_used_strategy(self):
        balancer = self._make_balancer("least_used")
        key = balancer.get_key()
        assert key is not None

    def test_least_used_thread_safety(self):
        balancer = self._make_balancer("least_used")
        errors = []

        def worker():
            try:
                for _ in range(100):
                    k = balancer.get_key()
                    if k:
                        k.record_request()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread safety error: {errors}"

    def test_invalid_strategy_raises(self):
        pool = KeyPool(
            keys_config=[{"key": "k", "alias": "K"}],
            rpm_limit=40, rpm_buffer=5,
        )
        with pytest.raises(ValueError, match="不支持的策略"):
            LoadBalancer(key_pool=pool, strategy="invalid_strategy")

    def test_get_key_returns_none_when_all_unavailable(self):
        balancer = self._make_balancer()
        for k in balancer.key_pool.keys:
            k.disable("test")
        assert balancer.get_key() is None

    def test_get_key_or_wait_async_exists(self):
        import asyncio
        balancer = self._make_balancer()

        async def test():
            k = await balancer.get_key_or_wait_async()
            return k is not None

        assert asyncio.run(test())


# ====================================================================
# StatsManager 测试
# ====================================================================

class TestStatsManager:

    def _make_stats(self):
        return StatsManager()

    def test_record_basic(self):
        sm = self._make_stats()
        sm.record(model="model-a", key_alias="K1",
                  prompt_tokens=10, completion_tokens=20,
                  latency_ms=500, success=True)
        overview = sm.get_overview()
        assert overview["total"]["requests"] == 1
        assert overview["total"]["prompt_tokens"] == 10
        assert overview["total"]["completion_tokens"] == 20
        assert overview["total"]["total_tokens"] == 30

    def test_record_stream(self):
        sm = self._make_stats()
        sm.record(model="m", key_alias="K1",
                  prompt_tokens=5, completion_tokens=15,
                  latency_ms=200, success=True, stream=True)
        records = sm.get_recent_records(10)
        assert len(records) == 1
        assert records[0]["stream"] is True

    def test_record_error(self):
        sm = self._make_stats()
        sm.record(model="m", key_alias="K1",
                  prompt_tokens=0, completion_tokens=0,
                  latency_ms=100, success=False)
        overview = sm.get_overview()
        assert overview["total"]["errors"] == 1

    def test_timeline_has_data(self):
        sm = self._make_stats()
        sm.record(model="m", key_alias="K1",
                  prompt_tokens=10, completion_tokens=20, latency_ms=300)
        timeline = sm.get_timeline(minutes=60)
        assert len(timeline) >= 1
        total_requests = sum(slot["requests"] for slot in timeline)
        assert total_requests >= 1

    def test_model_stats(self):
        sm = self._make_stats()
        sm.record(model="llama-70b", key_alias="K1",
                  prompt_tokens=100, completion_tokens=200, latency_ms=1000)
        sm.record(model="mistral-7b", key_alias="K2",
                  prompt_tokens=50, completion_tokens=100, latency_ms=500)
        model_stats = sm.get_model_stats()
        assert len(model_stats) == 2
        models = {s["model"] for s in model_stats}
        assert "llama-70b" in models
        assert "mistral-7b" in models

    def test_key_stats(self):
        sm = self._make_stats()
        sm.record(model="m", key_alias="K1", prompt_tokens=10, completion_tokens=20, latency_ms=100)
        sm.record(model="m", key_alias="K2", prompt_tokens=20, completion_tokens=40, latency_ms=200)
        key_stats = sm.get_key_stats()
        assert len(key_stats) == 2
        aliases = {s["key_alias"] for s in key_stats}
        assert aliases == {"K1", "K2"}

    def test_recent_records_limited(self):
        sm = self._make_stats()
        for i in range(100):
            sm.record(model=f"m-{i}", key_alias="K1",
                      prompt_tokens=i, completion_tokens=i*2, latency_ms=i*10)
        records = sm.get_recent_records(limit=10)
        assert len(records) == 10

    def test_max_records_cap(self):
        sm = self._make_stats()
        for i in range(600):
            sm.record(model="m", key_alias="K1",
                      prompt_tokens=1, completion_tokens=1, latency_ms=1)
        records = sm.get_recent_records(limit=600)
        assert len(records) <= 500

    def test_stale_stats_eviction(self):
        sm = self._make_stats()
        sm.record(model="old-model", key_alias="old-key",
                  prompt_tokens=1, completion_tokens=1, latency_ms=1)
        with sm._lock:
            sm._model_last_seen["old-model"] = time.time() - 86400
            sm._key_last_seen["old-key"] = time.time() - 86400
        sm._last_cleanup_time = time.time() - 3600
        sm._evict_stale_stats(time.time())
        model_stats = sm.get_model_stats()
        key_stats = sm.get_key_stats()
        assert len(model_stats) == 0
        assert len(key_stats) == 0

    def test_uptime_format(self):
        sm = self._make_stats()
        overview = sm.get_overview()
        assert ":" in overview["uptime"]
        assert overview["uptime_seconds"] >= 0

    def test_concurrent_record(self):
        sm = self._make_stats()
        errors = []

        def worker():
            try:
                for i in range(50):
                    sm.record(model="m", key_alias="K1",
                              prompt_tokens=i, completion_tokens=i*2,
                              latency_ms=i*10)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert sm._total["requests"] == 200


# ====================================================================
# Config 校验测试
# ====================================================================

class TestConfigValidation:
    """需要将 main.py 的 validate_config 导入或复制逻辑"""

    @staticmethod
    def _validate(config):
        from main import validate_config
        validate_config(config)

    def test_valid_minimal_config(self):
        config = {
            "keys": [{"key": "nvapi-test", "alias": "K1"}],
            "nvidia": {},
            "balancer": {},
        }
        self._validate(config)

    def test_valid_full_config(self):
        config = {
            "keys": [
                {"key": "nvapi-test-1", "alias": "K1"},
                {"key": "nvapi-test-2", "alias": "K2"},
            ],
            "nvidia": {
                "base_url": "https://integrate.api.nvidia.com/v1",
                "rpm_limit": 40,
                "rpm_buffer": 5,
            },
            "balancer": {
                "strategy": "round_robin",
                "wait_timeout": 65.0,
                "max_retries": 3,
            },
            "server": {"port": 8000},
        }
        self._validate(config)

    def test_missing_keys_raises(self):
        config = {"nvidia": {}, "balancer": {}}
        with pytest.raises(SystemExit):
            self._validate(config)

    def test_empty_key_raises(self):
        config = {
            "keys": [{"key": "", "alias": "K1"}],
            "nvidia": {}, "balancer": {},
        }
        with pytest.raises(SystemExit):
            self._validate(config)

    def test_invalid_strategy_raises(self):
        config = {
            "keys": [{"key": "nvapi-test", "alias": "K1"}],
            "nvidia": {},
            "balancer": {"strategy": "bad_strategy"},
        }
        with pytest.raises(SystemExit):
            self._validate(config)

    def test_invalid_port_raises(self):
        config = {
            "keys": [{"key": "nvapi-test", "alias": "K1"}],
            "nvidia": {}, "balancer": {},
            "server": {"port": 99999},
        }
        with pytest.raises(SystemExit):
            self._validate(config)

    def test_invalid_base_url_raises(self):
        config = {
            "keys": [{"key": "nvapi-test", "alias": "K1"}],
            "nvidia": {"base_url": "not-a-url"},
            "balancer": {},
        }
        with pytest.raises(SystemExit):
            self._validate(config)

    def test_negative_rpm_limit_raises(self):
        config = {
            "keys": [{"key": "nvapi-test", "alias": "K1"}],
            "nvidia": {"rpm_limit": -5},
            "balancer": {},
        }
        with pytest.raises(SystemExit):
            self._validate(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
