"""
Loads and validates config/crawl.yaml — the crawler's allowlist, seed URLs,
include/exclude patterns, and politeness settings.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.core.config import get_settings


class CrawlConfig(BaseModel):
    seed_urls: list[str] = Field(default_factory=list)
    allowed_domains: list[str] = Field(default_factory=list)
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    max_depth: int = 3
    max_pages: int = 100
    rate_limit_seconds: float = 2.0
    request_timeout_seconds: int = 30
    max_retries: int = 3
    user_agent: str = "KrishiMitraAI-Bot/1.0"
    respect_robots_txt: bool = True


def load_crawl_config(path: str | None = None) -> CrawlConfig:
    settings = get_settings()
    config_path = Path(path) if path else settings.resolve_path(settings.crawl_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Crawl config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return CrawlConfig.model_validate(raw)


def is_url_allowed(url: str, config: CrawlConfig) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.netloc.split(":")[0].lower()
    if config.allowed_domains and not any(host == d or host.endswith(f".{d}") for d in config.allowed_domains):
        return False
    if config.exclude_patterns and any(p in url for p in config.exclude_patterns):
        return False
    if config.include_patterns and not any(p in url for p in config.include_patterns):
        return False
    return True
