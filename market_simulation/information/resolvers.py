from __future__ import annotations

import hashlib

from pandas import Timedelta, Timestamp

from market_simulation.information.asymmetry_config import is_info_asymmetry_disabled
from market_simulation.information.types import ProfessionalNews, ResolvedInformation
from market_simulation.personas import InstitutionStrategyProfile, RetailPersonaProfile
from market_simulation.states.professional_news_state import ProfessionalNewsState
from market_simulation.states.social_network_state import SocialNetworkState


def _stable_fraction(*parts: object) -> float:
    joined = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


# 与 xueqiu_loader 中 professional_news_coverage 派生一致：高于此阈值视为「能看研报」
_RETAIL_RESEARCH_COVERAGE_THRESHOLD = 0.55


def _retail_can_consume_research(profile: RetailPersonaProfile) -> bool:
    return float(profile.professional_news_coverage) >= _RETAIL_RESEARCH_COVERAGE_THRESHOLD


def _retail_news_coverage(profile: RetailPersonaProfile, item: ProfessionalNews) -> float:
    """机构研报：能看研报的散户 100% 可见；其余新闻仍按 coverage / retail_sample_rate 抽样。"""
    if str(item.news_id).startswith("institution_report:"):
        return 1.0 if _retail_can_consume_research(profile) else 0.0
    if item.retail_sample_rate is not None:
        return float(item.retail_sample_rate)
    return float(profile.professional_news_coverage)


def _retail_news_available_time(
    item: ProfessionalNews,
    *,
    agent_id: int,
    profile: RetailPersonaProfile,
) -> Timestamp:
    if is_info_asymmetry_disabled():
        return item.publish_time
    if item.retail_delay_days is not None:
        return item.publish_time + Timedelta(days=max(0, int(item.retail_delay_days)))
    lag_bucket = int(_stable_fraction("lag", item.news_id, agent_id) * (profile.max_news_lag_days + 1))
    lag_bucket = min(lag_bucket, profile.max_news_lag_days)
    return item.publish_time + Timedelta(days=lag_bucket)


def _institution_news_available_time(
    item: ProfessionalNews,
    *,
    agent_id: int,
    profile: InstitutionStrategyProfile,
) -> Timestamp:
    if is_info_asymmetry_disabled():
        return item.publish_time
    delay_minutes = max(int((1.0 - profile.reaction_speed) * 30.0), 0)
    jitter = int(_stable_fraction("inst_delay", item.news_id, agent_id) * 5.0)
    return item.publish_time + Timedelta(minutes=delay_minutes + jitter)


def _retail_news_visible(
    item: ProfessionalNews,
    *,
    agent_id: int,
    profile: RetailPersonaProfile,
) -> bool:
    if is_info_asymmetry_disabled():
        return True
    if item.audience == "institution":
        return False
    if item.from_multisource:
        if _stable_fraction("retail_ms_half_gate", agent_id, "news_multisource_v1") >= 0.5:
            return False
        if _stable_fraction("retail_cov", item.news_id, agent_id) > 0.30:
            return False
        return True
    cov = _retail_news_coverage(profile, item)
    if cov <= 0.0:
        return False
    return _stable_fraction("retail_cov", item.news_id, agent_id) <= cov


def _institution_news_visible(
    item: ProfessionalNews,
    *,
    agent_id: int,
    profile: InstitutionStrategyProfile,
) -> bool:
    if is_info_asymmetry_disabled():
        return True
    if item.audience == "retail":
        return False
    if item.from_multisource:
        return True
    return _stable_fraction("inst_cov", item.news_id, agent_id) <= profile.research_coverage


def resolve_retail_information(
    *,
    symbol: str,
    as_of: Timestamp,
    agent_id: int,
    profile: RetailPersonaProfile,
    news_state: ProfessionalNewsState,
    social_state: SocialNetworkState,
    seen_news_ids: set[str],
    seen_post_ids: set[str],
    seen_comment_ids: set[str] | None = None,
) -> list[ResolvedInformation]:
    """汇总当前可见的资讯：专业新闻 + 个人 inbox 中已到达的帖子 + 这些帖子的新评论。

    没有关注图——帖子是否能被看到完全取决于它是否被推送进了该 agent 的 inbox
    （初始 3 人随机 + 互动驱动扩散）。
    """

    resolved: list[ResolvedInformation] = []

    for item in news_state.news_for_symbol(symbol):
        if not _retail_news_visible(item, agent_id=agent_id, profile=profile):
            continue
        available_time = _retail_news_available_time(item, agent_id=agent_id, profile=profile)
        if available_time > as_of or item.news_id in seen_news_ids:
            continue
        skepticism = profile.personality.skepticism()
        credibility = max(0.05, min(1.0, item.credibility * (1.10 - 0.35 * skepticism)))
        resolved.append(
            ResolvedInformation(
                item_id=item.news_id,
                symbol=item.symbol,
                source="news",
                available_time=available_time,
                topic=item.topic,
                direction=0.0,
                strength=0.0,
                credibility=credibility,
                sentiment=0.0,
                summary=item.headline,
                content=(item.content or item.headline or "")[:2000],
                source_url=item.source_url,
                source_news_id=item.news_id,
            )
        )

    delay_seconds = profile.social_delay_seconds * (0.5 + profile.personality.social_susceptibility())
    skepticism = profile.personality.skepticism()
    seen_comment_ids = seen_comment_ids if seen_comment_ids is not None else set()

    for post_id, queued_time in social_state.inbox_for(agent_id):
        if post_id in seen_post_ids:
            already_seen_post = True
        else:
            already_seen_post = False
        post = social_state.get_post(post_id)
        if post is None:
            continue
        post_available = max(post.created_time, queued_time) + Timedelta(seconds=delay_seconds)
        if post_available > as_of:
            continue
        if not already_seen_post:
            author_score = social_state.author_engagement_score(post.author_agent_id, as_of=as_of)
            cred_boost = 1.0 + 0.15 * min(2.0, author_score / 5.0)
            credibility = max(
                0.05,
                min(1.0, post.credibility * cred_boost * (1.10 - 0.30 * skepticism)),
            )
            resolved.append(
                ResolvedInformation(
                    item_id=post.post_id,
                    symbol="SOCIAL",
                    source="social",
                    available_time=post_available,
                    topic=post.topic,
                    direction=0.0,
                    strength=0.0,
                    credibility=credibility,
                    sentiment=0.0,
                    summary=post.content_label or post.topic,
                    content=(post.content or post.content_label or post.topic or "")[:2000],
                    author_agent_id=post.author_agent_id,
                    source_news_id=post.source_news_id,
                )
            )
        for comment in social_state.comments_for(post_id):
            if comment.comment_id in seen_comment_ids:
                continue
            if comment.author_agent_id == agent_id:
                continue
            comment_available = max(post_available, comment.created_time + Timedelta(seconds=delay_seconds))
            if comment_available > as_of:
                continue
            credibility = max(0.05, min(1.0, comment.credibility * (1.10 - 0.30 * skepticism)))
            resolved.append(
                ResolvedInformation(
                    item_id=comment.comment_id,
                    symbol="SOCIAL",
                    source="comment",
                    available_time=comment_available,
                    topic=post.topic,
                    direction=0.0,
                    strength=0.0,
                    credibility=credibility,
                    sentiment=0.0,
                    summary=comment.content_label or post.topic,
                    content=(comment.content or comment.content_label or post.topic or "")[:2000],
                    author_agent_id=comment.author_agent_id,
                    source_news_id=post.source_news_id,
                    parent_post_id=post.post_id,
                    related_to_self=(post.author_agent_id == agent_id),
                )
            )

    resolved.sort(key=lambda item: item.available_time)
    return resolved


def resolve_institution_information(
    *,
    symbol: str,
    as_of: Timestamp,
    agent_id: int,
    profile: InstitutionStrategyProfile,
    news_state: ProfessionalNewsState,
    seen_news_ids: set[str],
) -> list[ResolvedInformation]:
    resolved: list[ResolvedInformation] = []
    for item in news_state.news_for_symbol(symbol):
        if not _institution_news_visible(item, agent_id=agent_id, profile=profile):
            continue
        available_time = _institution_news_available_time(item, agent_id=agent_id, profile=profile)
        if available_time > as_of or item.news_id in seen_news_ids:
            continue
        credibility = max(0.05, min(1.0, item.credibility * (0.90 + 0.20 * profile.reaction_speed)))
        resolved.append(
            ResolvedInformation(
                item_id=item.news_id,
                symbol=item.symbol,
                source="news",
                available_time=available_time,
                topic=item.topic,
                direction=0.0,
                strength=0.0,
                credibility=credibility,
                sentiment=0.0,
                summary=item.headline,
                content=(item.content or item.headline or "")[:2000],
                source_url=item.source_url,
                source_news_id=item.news_id,
            )
        )

    resolved.sort(key=lambda item: item.available_time)
    return resolved
