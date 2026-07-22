from __future__ import annotations

from pandas import Timestamp

from market_simulation.information.types import ProfessionalNews


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def build_daily_research_report_news(
    *,
    report_date: Timestamp,
    agent_id: int,
    symbol: str,
    report_text: str,
    credibility: float = 0.72,
    urgency: float = 0.55,
    direction: float = 0.0,
    strength: float = 0.0,
) -> ProfessionalNews | None:
    """把 LLM 生成的纯文本研报包装成 ProfessionalNews。"""
    text = str(report_text or "").strip()
    if not text:
        return None
    date_tag = str(report_date.normalize().date())

    return ProfessionalNews(
        news_id=f"institution_report:{agent_id}:{symbol}:{date_tag}",
        symbol=symbol,
        publish_time=report_date,
        headline=f"机构研报：{symbol} 日度观点",
        topic="fundamental",
        direction=float(direction),
        strength=_clamp(float(strength)),
        content=text[:2000],
        credibility=_clamp(float(credibility)),
        urgency=_clamp(float(urgency)),
        audience="retail",
        retail_sample_rate=1.0,
        from_multisource=False,
        retail_delay_days=1,
    )
