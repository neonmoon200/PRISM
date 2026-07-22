from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pandas import Timestamp


Audience = Literal["all", "retail", "institution"]


@dataclass(frozen=True)
class ProfessionalNews:
    """Professional information released to some or all market participants."""

    news_id: str
    symbol: str
    publish_time: Timestamp
    headline: str
    topic: str
    direction: float
    strength: float = 1.0
    content: str = ""
    credibility: float = 0.70
    urgency: float = 0.50
    audience: Audience = "all"
    #: 散户对该条资讯的独立可见概率上界（0–1）；None 表示沿用 persona 的
    #: ``professional_news_coverage``。机构侧仍由 ``research_coverage`` 控制（默认可为 100%）。
    retail_sample_rate: float | None = None
    source_url: str = ""
    #: 是否来自 ``news_multisource.csv``（用于散户/机构不同的可见性规则）
    from_multisource: bool = False
    #: 散户侧固定延迟天数（None 表示走 persona 的随机时滞规则）。
    retail_delay_days: int | None = None


@dataclass(frozen=True)
class SocialPost:
    """Retail-authored social message routed through engagement-driven exposure."""

    post_id: str
    symbol: str
    author_agent_id: int
    created_time: Timestamp
    topic: str
    direction: float
    strength: float
    sentiment: float
    visibility: Literal["followers", "broadcast"] = "broadcast"
    credibility: float = 0.45
    source_news_id: str | None = None
    referenced_post_id: str | None = None
    content_label: str = ""
    content: str = ""


@dataclass(frozen=True)
class SocialComment:
    """Retail-authored comment attached to a SocialPost."""

    comment_id: str
    post_id: str
    symbol: str
    author_agent_id: int
    created_time: Timestamp
    direction: float
    strength: float
    sentiment: float
    credibility: float = 0.40
    content_label: str = ""
    content: str = ""


@dataclass(frozen=True)
class ResolvedInformation:
    """Agent-specific information after visibility, delay and trust filters."""

    item_id: str
    symbol: str
    source: Literal["news", "social", "comment"]
    available_time: Timestamp
    topic: str
    direction: float
    strength: float
    credibility: float
    sentiment: float
    summary: str
    content: str = ""
    source_url: str = ""
    author_agent_id: int | None = None
    source_news_id: str | None = None
    parent_post_id: str | None = None
    related_to_self: bool = False
