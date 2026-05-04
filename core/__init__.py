"""Core configuration and domain definitions for Smart Teacher."""
from .config import Config
from .domains_config import (
    DEFAULT_DOMAIN,
    DEFAULT_COURSE,
    DOMAINS,
    get_chapters,
    get_courses,
    get_domains,
)

__all__ = [
    "Config",
    "DEFAULT_DOMAIN",
    "DEFAULT_COURSE",
    "DOMAINS",
    "get_chapters",
    "get_courses",
    "get_domains",
]
