import os
import yaml
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field


def load_config(path: str = None) -> Dict[str, Any]:
    """Load YAML config from given path or default location `config.yaml` in repo root."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_universe_config(config: Dict[str, Any], name: str = None) -> Dict[str, Any]:
    if not config:
        return {}
    if name is None:
        return config.get("universe", {})
    universes = config.get("universes", {})
    return universes.get(name, {})

# --- Add dataclasses for Settings and related configs ---
@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "postgres"
    database: str = "market_research"
    min_connections: int = 5
    max_connections: int = 20

@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

@dataclass
class ProviderConfig:
    batch_size: int = 50
    rate_limit: float = 2.0
    retry_attempts: int = 3
    retry_delay: int = 5

@dataclass
class Settings:
    symbols: List[str]
    lookback_days: int = 252
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    features: Dict[str, Dict[str, Union[int, float, str]]] = field(default_factory=dict)
    factor_weights: Optional[Dict[str, float]] = None
    host: str = "0.0.0.0"
    port: int = 8000
    @classmethod
    def from_yaml(cls, path: Union[str, os.PathLike]) -> "Settings":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        symbols = config.get("symbols", [])
        if isinstance(symbols, str):
            with open(symbols, "r") as f:
                symbols = [line.strip() for line in f if line.strip()]
        settings = cls(symbols=symbols)
        db_config = config.get("database", {})
        settings.db = DatabaseConfig(**db_config)
        redis_config = config.get("redis", {})
        settings.redis = RedisConfig(**redis_config)
        provider_config = config.get("provider", {})
        settings.provider = ProviderConfig(**provider_config)
        settings.features = config.get("features", {})
        settings.factor_weights = config.get("factor_weights")
        api_config = config.get("api", {})
        settings.host = api_config.get("host", settings.host)
        settings.port = api_config.get("port", settings.port)
        return settings

__all__ = ["Settings", "DatabaseConfig", "RedisConfig", "ProviderConfig"]
