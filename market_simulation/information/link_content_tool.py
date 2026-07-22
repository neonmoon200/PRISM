from __future__ import annotations

import re
from functools import lru_cache
from html import unescape
from urllib.request import Request, urlopen

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>[\s\S]*?</\1>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _html_to_text(html: str) -> str:
    cleaned = _SCRIPT_STYLE_RE.sub(" ", html)
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = _SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


@lru_cache(maxsize=4096)
def read_link_report_content(
    url: str,
    *,
    timeout_seconds: float = 4.0,
    max_chars: int = 1800,
) -> str:
    """读取 URL 对应网页正文（失败时返回空字符串）。

    使用进程级 LRU 缓存，避免多个 agent 重复抓取同一链接。
    """

    u = str(url or "").strip()
    if not u.startswith(("http://", "https://")):
        return ""
    try:
        req = Request(
            u,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
        with urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310
            raw = resp.read()
        html = raw.decode("utf-8", errors="ignore")
        text = _html_to_text(html)
        if not text:
            return ""
        return text[:max_chars]
    except Exception:
        return ""

