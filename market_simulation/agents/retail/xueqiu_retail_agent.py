"""基于雪球 BigFive 数据的散户 agent。

行为：
* ``read_news(news_item)``  —— 专门的新闻输入接口，外部可主动推送
* ``read_post(post)``       —— 看帖
* ``like_post(post)``       —— 点赞 / 转发
* ``compose_post(time)``    —— 发帖
* ``make_execution_intent`` —— 交易（含止盈止损）

状态：见 ``XueqiuRetailState``，包含资产、情绪、belief（方向/题材/热点）、
credibility（多源信任）、social_behavior、trading_style、带衰减记忆。

LLM：按 ``llm_call_probability`` 概率调用；若本次触发了 LLM 但调用抛错（稽核/超时等），
则本唤醒**不交易**且不再用规则模型兜底。
"""

from __future__ import annotations

import logging
from random import Random
from typing import Any, Literal

from pandas import Timedelta, Timestamp

from market_simulation.agents.core.base import HeterogeneousAgentBase
from market_simulation.agents.core.execution import ExecutionIntent, round_lot
from market_simulation.agents.core.llm_protocols import XueqiuLLM, get_default_llm
from market_simulation.information import (
    ProfessionalNews,
    ResolvedInformation,
    SocialComment,
    SocialPost,
    resolve_retail_information,
)
from market_simulation.information.link_content_tool import read_link_report_content
from market_simulation.personas.xueqiu_loader import XueqiuPersonaProfile
from market_simulation.personas.xueqiu_state import XueqiuRetailState
from market_simulation.social import compose_retail_comment, compose_retail_post
from market_simulation.states.professional_news_state import ProfessionalNewsState
from market_simulation.states.social_network_state import SocialNetworkState
from market_simulation.utils.session_calendar import SessionCalendar
from mlib.core.transaction import Transaction

_logger = logging.getLogger(__name__)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _safe_float(value: Any, default: float) -> float:
    """把 LLM 返回里的潜在脏值（None / 字符串 / 列表 / NaN）安全转 float。"""

    if value is None:
        return default
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if out != out:  # NaN
        return default
    return out


def _classify_news_source(news: ProfessionalNews) -> str:
    topic = (news.topic or "").lower()
    if "regul" in topic or "policy" in topic or "official" in topic:
        return "official_announcement"
    return "pro_news"


class XueqiuRetailAgent(HeterogeneousAgentBase):
    """雪球散户 agent。"""

    def __init__(
        self,
        *,
        symbol: str,
        session_calendar: SessionCalendar,
        start_time: Timestamp,
        end_time: Timestamp,
        profile: XueqiuPersonaProfile,
        reference_price: int = 100_000,
        seed: int = 0,
        llm: XueqiuLLM | None = None,
        llm_call_probability: float = 0.05,
    ) -> None:
        super().__init__(
            symbol=symbol,
            session_calendar=session_calendar,
            start_time=start_time,
            end_time=end_time,
            init_cash=profile.base.initial_cash,
            initial_position=profile.base.initial_position,
            reference_price=reference_price,
            base_interval_seconds=profile.base.base_interval_seconds,
            seed=seed,
        )
        self.profile = profile
        self.state = XueqiuRetailState.from_profile(profile)
        self.seen_news_ids: set[str] = set()
        self.seen_post_ids: set[str] = set()
        self.seen_comment_ids: set[str] = set()
        self._latest_news_summary: str = ""
        self._latest_news_url: str = ""
        self._latest_post_summary: str = ""
        self._latest_comment_summary: str = ""
        self._last_tick_time: Timestamp | None = None
        self.llm = llm if llm is not None else get_default_llm()
        self.llm_call_probability = max(0.0, min(1.0, llm_call_probability))
        # 供 runner 按 agent 分文件落盘：记录本次唤醒中的 LLM 输出
        self._llm_outputs: list[dict[str, Any]] = []

    # =================== 状态同步与时间推进 ===================
    def sync_internal_resources(self) -> None:
        self._llm_outputs = []
        self.state.sync_resources(self.cash, self.holdings, self.symbol)
        now = self.start_time
        if self._last_tick_time is not None:
            now = self._last_tick_time
        self.state.tick(now)

    # =================== 信息收集（仍走仓库默认 resolver） ===================
    def collect_information(self, time: Timestamp) -> list[ResolvedInformation]:
        self._last_tick_time = time
        news_state = self.symbol_states[self.symbol][ProfessionalNewsState.__name__]
        social_state = self.symbol_states[self.symbol][SocialNetworkState.__name__]
        assert isinstance(news_state, ProfessionalNewsState)
        assert isinstance(social_state, SocialNetworkState)
        return resolve_retail_information(
            symbol=self.symbol,
            as_of=time,
            agent_id=self.agent_id,
            profile=self.profile.base,
            news_state=news_state,
            social_state=social_state,
            seen_news_ids=self.seen_news_ids,
            seen_post_ids=self.seen_post_ids,
            seen_comment_ids=self.seen_comment_ids,
        )

    def consume_information(self, item: ResolvedInformation) -> None:
        if item.source == "news":
            self.read_news(item)
        elif item.source == "social":
            self.read_post(item)
        else:
            self.read_comment(item)

    # =================== 行为：看新闻（独立接口） ===================
    def read_news(
        self,
        news: ResolvedInformation | ProfessionalNews,
        *,
        now: Timestamp | None = None,
    ) -> None:
        """专门的新闻输入接口。外部 runner / 测试可直接调。"""

        if isinstance(news, ProfessionalNews):
            source_kind = _classify_news_source(news)
            time = now or news.publish_time
            news_id = news.news_id
            direction = news.direction
            strength = news.strength
            topic = news.topic
            summary = news.headline
            source_url = news.source_url
        else:
            source_kind = "pro_news"
            if news.topic and ("regul" in news.topic.lower() or "policy" in news.topic.lower()):
                source_kind = "official_announcement"
            time = now or news.available_time
            news_id = news.item_id
            direction = news.direction
            strength = news.strength
            topic = news.topic
            summary = news.summary
            source_url = news.source_url
        if news_id in self.seen_news_ids:
            return
        self.seen_news_ids.add(news_id)
        self.state.ingest(
            time=time,
            source=source_kind,  # type: ignore[arg-type]
            direction=direction,
            strength=strength,
            topic=topic or "",
            summary=summary or "",
        )
        self._latest_news_summary = summary or topic or ""
        self._latest_news_url = source_url or ""

    # =================== 行为：看帖 / 点赞 / 转发 ===================
    def read_post(
        self,
        post: ResolvedInformation | SocialPost,
        *,
        now: Timestamp | None = None,
    ) -> None:
        if isinstance(post, SocialPost):
            time = now or post.created_time
            post_id = post.post_id
            direction = post.direction
            strength = post.strength
            topic = post.topic
            summary = post.content_label or post.topic
            author_cred = 0.50
        else:
            time = now or post.available_time
            post_id = post.item_id
            direction = post.direction
            strength = post.strength
            topic = post.topic
            summary = post.summary
            author_cred = post.credibility
        if post_id in self.seen_post_ids:
            return
        self.seen_post_ids.add(post_id)
        # 选择来源：高 credibility 帖 → opinion_leader，否则 peer / crowd
        if author_cred >= 0.65:
            source = "opinion_leader"
        elif author_cred >= 0.40:
            source = "peer_post"
        else:
            source = "retail_crowd"
        self.state.ingest(
            time=time,
            source=source,  # type: ignore[arg-type]
            direction=direction,
            strength=strength,
            topic=topic or "",
            summary=summary or "",
        )
        self._latest_post_summary = summary or topic or ""
        # 点赞 / 评论 / 转发（向 SocialNetworkState 上报，触发曝光扩散）
        self._maybe_engage_with_post(
            time=time,
            post_id=post_id,
            direction=direction,
            author_cred=author_cred,
        )

    def read_comment(
        self,
        comment: ResolvedInformation | SocialComment,
        *,
        now: Timestamp | None = None,
    ) -> None:
        """读评论：评论内容也会推动 belief / 情绪。"""

        if isinstance(comment, SocialComment):
            time = now or comment.created_time
            comment_id = comment.comment_id
            direction = comment.direction
            strength = comment.strength
            summary = comment.content_label
            author_cred = comment.credibility
            topic = ""
        else:
            time = now or comment.available_time
            comment_id = comment.item_id
            direction = comment.direction
            strength = comment.strength
            summary = comment.summary
            author_cred = comment.credibility
            topic = comment.topic
        if comment_id in self.seen_comment_ids:
            return
        self.seen_comment_ids.add(comment_id)
        if author_cred >= 0.65:
            source = "opinion_leader"
        elif author_cred >= 0.40:
            source = "peer_post"
        else:
            source = "retail_crowd"
        self.state.ingest(
            time=time,
            source=source,  # type: ignore[arg-type]
            direction=direction,
            strength=strength * 0.75,
            topic=topic or "",
            summary=summary or "",
        )
        self.state.consumed_comments += 1
        self._latest_comment_summary = summary or topic or ""

    def like_post(self, post_id: str, *, time: Timestamp | None = None) -> None:
        social_state = self.symbol_states.get(self.symbol, {}).get(SocialNetworkState.__name__)
        if isinstance(social_state, SocialNetworkState):
            social_state.record_like(viewer_id=self.agent_id, post_id=post_id, now=time or self._last_tick_time or self.start_time)
        self.state.liked_posts += 1
        self.seen_post_ids.add(post_id)

    def repost_post(self, post_id: str, *, time: Timestamp | None = None) -> None:
        social_state = self.symbol_states.get(self.symbol, {}).get(SocialNetworkState.__name__)
        if isinstance(social_state, SocialNetworkState):
            social_state.record_repost(viewer_id=self.agent_id, post_id=post_id, now=time or self._last_tick_time or self.start_time)
        self.state.reposted_posts += 1
        self.seen_post_ids.add(post_id)

    def comment_on_post(
        self,
        post_id: str,
        *,
        time: Timestamp,
        text_label: str = "",
    ) -> SocialComment | None:
        social_state = self.symbol_states.get(self.symbol, {}).get(SocialNetworkState.__name__)
        if not isinstance(social_state, SocialNetworkState):
            return None
        post = social_state.get_post(post_id)
        if post is None:
            return None
        comment = compose_retail_comment(
            comment_id=f"comment-{self.agent_id}-{post.post_id}-{self.state.authored_posts}-{len(self.seen_comment_ids)}",
            post=post,
            author_agent_id=self.agent_id,
            created_time=time,
            own_belief=self.state.belief,
            own_credibility=0.55,
            text_label=text_label,
        )
        social_state.record_comment(viewer_id=self.agent_id, post_id=post.post_id, comment=comment, now=time)
        self.state.commented_posts += 1
        return comment

    def _maybe_engage_with_post(
        self,
        *,
        time: Timestamp,
        post_id: str,
        direction: float,
        author_cred: float,
    ) -> None:
        sb = self.profile.social_behavior
        aligned = direction * self.state.belief >= 0
        like_p = _clamp(0.10 + 0.40 * sb.repost_tendency + (0.20 if aligned else 0.0) + 0.20 * author_cred)
        repost_p = _clamp(0.04 + 0.40 * sb.repost_tendency * (1.0 if aligned else 0.3))
        comment_p = _clamp(0.05 + 0.30 * sb.posting_tendency + (0.10 if aligned else 0.05))
        text_label = ""
        if self.llm.maybe(self.rng, self.llm_call_probability * 0.5):
            decision = self.llm.react_to_post(
                persona_brief=self.profile.brief_personality,
                post_text=self._latest_post_summary,
                author_credibility=author_cred,
                own_belief=self.state.belief,
            )
            self._llm_outputs.append(
                {
                    "kind": "react_to_post",
                    "post_id": post_id,
                    "decision": decision,
                }
            )
            if isinstance(decision, dict):
                try:
                    self.state.llm_calls += 1
                    action = str(decision.get("action", "")).lower()
                    shift = _safe_float(decision.get("belief_shift"), 0.0)
                    shift = max(-0.3, min(0.3, shift))
                    self.state.belief = max(-1.0, min(1.0, self.state.belief + shift))
                    txt = str(decision.get("text", "")).strip()
                    if txt:
                        text_label = txt[:120]
                    if action == "like":
                        self.like_post(post_id, time=time)
                        return
                    if action == "repost":
                        self.repost_post(post_id, time=time)
                        return
                    if action == "comment":
                        self.comment_on_post(post_id, time=time, text_label=text_label)
                        return
                    if action == "ignore" or action == "unlike":
                        return
                except Exception as exc:
                    self._llm_outputs.append(
                        {"kind": "react_to_post_consume_error", "error": str(exc)[:400], "decision": decision}
                    )
                    _logger.warning(
                        "[散户 agent_id=%s] 处理 react_to_post LLM 决策异常，回退规则路径: %s",
                        self.agent_id,
                        str(exc)[:400],
                    )
            elif decision is not None:
                _logger.warning(
                    "[散户 agent_id=%s] react_to_post LLM 返回非字典 (type=%s)，回退规则路径",
                    self.agent_id,
                    type(decision).__name__,
                )
        if self.rng.random() < repost_p:
            self.repost_post(post_id, time=time)
            return
        if self.rng.random() < like_p:
            self.like_post(post_id, time=time)
        if self.rng.random() < comment_p:
            self.comment_on_post(post_id, time=time, text_label=text_label)

    # =================== 行为：发帖 ===================
    def maybe_publish_social_post(
        self,
        time: Timestamp,
        new_information: list[ResolvedInformation],
    ) -> None:
        social_state = self.symbol_states[self.symbol][SocialNetworkState.__name__]
        assert isinstance(social_state, SocialNetworkState)
        if new_information:
            strongest = max(new_information, key=lambda it: abs(it.direction) * it.strength * it.credibility)
            sig_strength = abs(strongest.direction) * strongest.strength * strongest.credibility
        else:
            strongest = ResolvedInformation(
                item_id=f"noop-{self.agent_id}-{time.isoformat()}",
                symbol=self.symbol,
                source="news",
                available_time=time,
                topic="",
                direction=0.0,
                strength=0.0,
                credibility=0.0,
                sentiment=self.state.sentiment,
                summary="",
            )
            sig_strength = 0.0
        if not self.state.should_post(sig_strength):
            return

        self._publish_one_post(time=time, trigger=strongest, social_state=social_state)

    def compose_post(
        self,
        time: Timestamp,
        *,
        trigger: ResolvedInformation | None = None,
    ) -> SocialPost | None:
        """主动发帖接口（外部测试 / runner 也可调）。"""

        social_state = self.symbol_states.get(self.symbol, {}).get(SocialNetworkState.__name__)
        if not isinstance(social_state, SocialNetworkState):
            return None
        if trigger is None:
            recent = self.state.memory.recent(time, n=1)
            if not recent:
                return None
            top = recent[0]
            trigger = ResolvedInformation(
                item_id=f"mem-{self.agent_id}-{self.state.authored_posts + 1}",
                symbol=self.symbol,
                source="news" if top.source in ("pro_news", "official_announcement") else "social",
                available_time=time,
                topic="memory",
                direction=top.direction,
                strength=max(0.2, top.strength),
                credibility=0.55,
                sentiment=top.direction,
                summary=top.summary or "",
            )
        return self._publish_one_post(time=time, trigger=trigger, social_state=social_state)

    def _publish_one_post(
        self,
        *,
        time: Timestamp,
        trigger: ResolvedInformation,
        social_state: SocialNetworkState,
    ) -> SocialPost | None:
        author_score = social_state.author_engagement_score(self.agent_id, as_of=time)
        influence = _clamp(0.30 + 0.10 * min(5.0, author_score / 2.0))
        rhetoric_style = self._pick_rhetoric_style(
            author_score=author_score,
            signal_strength=abs(trigger.direction) * max(0.2, trigger.strength),
            topic=trigger.topic,
        )
        text_label: str | None = None
        sentiment_override: float | None = None
        if self.llm.maybe(self.rng, self.llm_call_probability):
            decision = self.llm.compose_post(
                persona_brief=self.profile.brief_personality,
                topic=trigger.topic or "",
                direction=trigger.direction,
                sentiment=self.state.sentiment,
                trigger_summary=trigger.summary or trigger.topic or "",
            )
            self._llm_outputs.append(
                {
                    "kind": "compose_post",
                    "symbol": self.symbol,
                    "trigger_item_id": trigger.item_id,
                    "decision": decision,
                }
            )
            if isinstance(decision, dict):
                try:
                    self.state.llm_calls += 1
                    txt = str(decision.get("text", "")).strip()
                    if txt:
                        text_label = txt[:120]
                    if "sentiment" in decision:
                        try:
                            sentiment_override = max(-1.0, min(1.0, float(decision["sentiment"])))
                        except (TypeError, ValueError):
                            sentiment_override = None
                except Exception as exc:
                    self._llm_outputs.append(
                        {"kind": "compose_post_consume_error", "error": str(exc)[:400], "decision": decision}
                    )
                    _logger.warning(
                        "[散户 agent_id=%s] 处理 compose_post LLM 决策异常，使用规则文本: %s",
                        self.agent_id,
                        str(exc)[:400],
                    )
            elif decision is not None:
                _logger.warning(
                    "[散户 agent_id=%s] compose_post LLM 返回非字典 (type=%s)，使用规则文本",
                    self.agent_id,
                    type(decision).__name__,
                )
        post = compose_retail_post(
            post_id=f"post-{self.agent_id}-{self.state.authored_posts + 1}",
            symbol=self.symbol,
            author_agent_id=self.agent_id,
            info=ResolvedInformation(
                item_id=trigger.item_id,
                symbol=trigger.symbol,
                source=trigger.source,
                available_time=time,
                topic=trigger.topic,
                direction=trigger.direction,
                strength=trigger.strength,
                credibility=trigger.credibility,
                sentiment=trigger.sentiment if sentiment_override is None else sentiment_override,
                summary=text_label or trigger.summary,
                author_agent_id=trigger.author_agent_id,
                source_news_id=trigger.source_news_id,
            ),
            posting_intensity=self.profile.social_behavior.posting_tendency,
            influence=influence,
            rhetoric_style=rhetoric_style,
        )
        social_state.publish_post(post, now=time)
        self.state.authored_posts += 1
        return post

    def _pick_rhetoric_style(
        self,
        *,
        author_score: float,
        signal_strength: float,
        topic: str,
    ) -> str:
        """根据声望与信号强度切换话术，模拟争取互动的表达策略。"""
        emotional = signal_strength >= 0.55
        if author_score < 1.8:
            return "hot_take" if emotional else "discussion"
        if author_score < 4.5:
            if emotional and self.rng.random() < 0.55:
                return "hot_take"
            return "data_driven" if self.rng.random() < 0.5 else "discussion"
        if topic in {"policy", "macro", "regulation"}:
            return "deep_dive"
        return "risk_alert" if emotional else "data_driven"

    # =================== 行为：交易决策 ===================
    def make_execution_intent(self, time: Timestamp) -> ExecutionIntent:
        # 止盈止损优先级最高
        lob = self.get_lob()
        mark = float(lob.mid_price) if lob is not None else float(self.reference_price)
        if self.state.take_profit_triggered(mark) or self.state.stop_loss_triggered(mark):
            volume = round_lot(self.tradable_holdings.get(self.symbol, 0))
            if volume >= 100:
                return ExecutionIntent(
                    mode="directional",
                    direction="S",
                    target_volume=volume,
                    aggressiveness=0.85,
                    rationale="take_profit_or_stop_loss",
                )

        if not self.state.should_trade(now=time):
            return ExecutionIntent(mode="none", rationale="conviction below threshold")

        # LLM 询问（小概率）；调用异常则本唤醒不交易（不与规则模型混用）
        llm_decision = None
        llm_failed = False
        if self.llm.maybe(self.rng, self.llm_call_probability):
            try:
                llm_decision = self.llm.decide_trade(
                    persona_brief=self.profile.brief_personality,
                    belief=self.state.belief,
                    sentiment=self.state.sentiment,
                    cash=self.cash,
                    position=self.holdings.get(self.symbol, 0),
                    recent_return=self.recent_return(time, lookback_seconds=600),
                    latest_news=self._latest_news_summary,
                    latest_post=self._latest_post_summary,
                    latest_news_report=read_link_report_content(self._latest_news_url),
                )
                self._llm_outputs.append(
                    {
                        "kind": "trade_single",
                        "symbol": self.symbol,
                        "decision": llm_decision,
                    }
                )
                if llm_decision is not None:
                    self.state.llm_calls += 1
            except Exception as exc:
                llm_failed = True
                self._llm_outputs.append(
                    {
                        "kind": "trade_single",
                        "symbol": self.symbol,
                        "error": str(exc)[:400],
                    }
                )
                _logger.warning(
                    "[散户 agent_id=%s] LLM 调用失败，本唤醒跳过交易: %s",
                    self.agent_id,
                    str(exc)[:400],
                )
        if llm_failed:
            return ExecutionIntent(mode="none", rationale="llm_call_failed_skip_trade")

        belief = self.state.belief
        if not isinstance(llm_decision, dict):
            if llm_decision is not None:
                _logger.warning(
                    "[散户 agent_id=%s] decide_trade LLM 返回非字典 (type=%s)，回退规则路径",
                    self.agent_id,
                    type(llm_decision).__name__,
                )
            llm_decision = None
        if abs(belief) < 0.06 and llm_decision is None:
            return ExecutionIntent(mode="none", rationale="weak directional belief")

        ts = self.profile.trading_style
        sb = self.profile.social_behavior
        if llm_decision is not None:
            try:
                action = str(llm_decision.get("action", "hold")).lower()
                if action == "hold":
                    return ExecutionIntent(mode="none", rationale="llm_hold")
                direction: Literal["B", "S"] = "B" if action == "buy" else "S"
                size_ratio = _clamp(_safe_float(llm_decision.get("size_ratio"), 0.2))
                aggressiveness = _clamp(_safe_float(llm_decision.get("aggressiveness"), 0.4))
            except Exception as exc:
                self._llm_outputs.append(
                    {"kind": "trade_single_consume_error", "error": str(exc)[:400], "decision": llm_decision}
                )
                _logger.warning(
                    "[散户 agent_id=%s] 处理 decide_trade 决策异常，回退规则路径: %s",
                    self.agent_id,
                    str(exc)[:400],
                )
                llm_decision = None
        if llm_decision is None:
            direction = "B" if belief >= 0 else "S"
            size_ratio = _clamp(0.15 + 0.50 * abs(belief) + 0.30 * ts.impulsiveness)
            aggressiveness = _clamp(0.20 + 0.30 * ts.impulsiveness + 0.25 * self.state.excitement + 0.15 * sb.contagion_sensitivity)

        if direction == "S" and self.tradable_holdings.get(self.symbol, 0) < 100:
            return ExecutionIntent(mode="none", rationale="cannot short without inventory")

        if direction == "B":
            target_cash = self.tradable_cash * size_ratio
            volume = round_lot(int(target_cash / max(mark, 1)))
        else:
            holdings = self.tradable_holdings.get(self.symbol, 0)
            volume = round_lot(int(holdings * size_ratio))
        volume = max(0, min(volume, max(self.profile.base.base_order_size * 5, 100)))
        if volume < 100:
            return ExecutionIntent(mode="none", rationale="size_below_lot")

        price_offset = 100 if abs(self.recent_return(time, lookback_seconds=900)) > 0.01 else 0
        return ExecutionIntent(
            mode="directional",
            direction=direction,
            target_volume=volume,
            aggressiveness=aggressiveness,
            price_offset_ticks=price_offset,
            rationale="xueqiu_belief_driven",
        )

    def interval_multiplier(self) -> float:
        ts = self.profile.trading_style
        return max(0.40, 1.40 - 0.70 * ts.frequency - 0.40 * self.state.conviction)

    # =================== 反馈 ===================
    def on_order_executed(self, time: Timestamp, transaction: Transaction, trans_order_id_to_notify: int):
        del_ids = super().on_order_executed(time, transaction, trans_order_id_to_notify)
        if transaction.type in ("B", "S"):
            volume = int(transaction.volume)
            if transaction.order_matched_volume is not None:
                volume = int(transaction.order_matched_volume.get(trans_order_id_to_notify, volume))
            if volume > 0:
                if trans_order_id_to_notify in transaction.buy_id:
                    side: Literal["B", "S"] = "B"
                elif trans_order_id_to_notify in transaction.sell_id:
                    side = "S"
                else:
                    return del_ids
                self.state.on_fill(side=side, price=float(transaction.price), volume=volume, time=time)
        return del_ids

    def update_after_trade(self, pnl_delta: float) -> None:
        time = self._last_tick_time or self.start_time
        self.state.update_after_trade(pnl_delta, time=time)

    def record_order_submission(self, count: int) -> None:
        self.state.submitted_orders += int(count)

    def snapshot_metrics(self) -> dict[str, float]:
        wealth = self.mark_to_market_wealth()
        social_state = self.symbol_states.get(self.symbol, {}).get(SocialNetworkState.__name__)
        author_score = (
            float(social_state.author_engagement_score(self.agent_id, as_of=self._last_tick_time or self.start_time))
            if isinstance(social_state, SocialNetworkState)
            else 0.0
        )
        return {
            "type": 0.0,
            "user_id": float(int(self.profile.user_id)) if self.profile.user_id.isdigit() else 0.0,
            "belief": float(self.state.belief),
            "sentiment": float(self.state.sentiment),
            "stress": float(self.state.stress),
            "excitement": float(self.state.excitement),
            "conviction": float(self.state.conviction),
            "memory_size": float(len(self.state.memory.items)),
            "consumed_news": float(self.state.consumed_news),
            "consumed_posts": float(self.state.consumed_posts),
            "consumed_comments": float(self.state.consumed_comments),
            "liked_posts": float(self.state.liked_posts),
            "reposted_posts": float(self.state.reposted_posts),
            "commented_posts": float(self.state.commented_posts),
            "authored_posts": float(self.state.authored_posts),
            "submitted_orders": float(self.state.submitted_orders),
            "executed_trades": float(self.state.executed_trades),
            "llm_calls": float(self.state.llm_calls),
            "author_engagement_score": author_score,
            "wealth": float(wealth or self.cash),
            "position": float(self.holdings.get(self.symbol, 0)),
            "cash": float(self.cash),
        }


def build_xueqiu_agents(
    *,
    profiles,
    symbol: str,
    session_calendar: SessionCalendar,
    start_time: Timestamp,
    end_time: Timestamp,
    reference_price: int,
    seed: int = 0,
    llm: XueqiuLLM | None = None,
    llm_call_probability: float = 0.05,
) -> list[XueqiuRetailAgent]:
    """从 profile 列表批量构造 agent。"""
    agents: list[XueqiuRetailAgent] = []
    for idx, profile in enumerate(profiles):
        agents.append(
            XueqiuRetailAgent(
                symbol=symbol,
                session_calendar=session_calendar,
                start_time=start_time,
                end_time=end_time,
                profile=profile,
                reference_price=reference_price,
                seed=seed + 100 + idx,
                llm=llm,
                llm_call_probability=llm_call_probability,
            )
        )
    return agents


# 便于 IDE / 静态分析识别未使用的 import
_UNUSED = (Timedelta,)
