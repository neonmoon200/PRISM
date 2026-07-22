from __future__ import annotations

from pandas import Timestamp

from market_simulation.agents.core.base import HeterogeneousAgentBase
from market_simulation.agents.core.execution import ExecutionIntent, round_lot
from market_simulation.information import ResolvedInformation, resolve_retail_information
from market_simulation.personas import RetailAgentState, RetailPersonaProfile
from market_simulation.social import compose_retail_comment, compose_retail_post
from market_simulation.states.professional_news_state import ProfessionalNewsState
from market_simulation.states.social_network_state import SocialNetworkState
from market_simulation.utils.session_calendar import SessionCalendar


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class RetailAgent(HeterogeneousAgentBase):
    """Retail investor driven by Big Five personality, news and social diffusion."""

    def __init__(
        self,
        *,
        symbol: str,
        session_calendar: SessionCalendar,
        start_time: Timestamp,
        end_time: Timestamp,
        profile: RetailPersonaProfile,
        reference_price: int = 100_000,
        seed: int = 0,
    ) -> None:
        super().__init__(
            symbol=symbol,
            session_calendar=session_calendar,
            start_time=start_time,
            end_time=end_time,
            init_cash=profile.initial_cash,
            initial_position=profile.initial_position,
            reference_price=reference_price,
            base_interval_seconds=profile.base_interval_seconds,
            seed=seed,
        )
        self.profile = profile
        self.state = RetailAgentState.from_profile(profile)
        self.seen_news_ids: set[str] = set()
        self.seen_post_ids: set[str] = set()
        self.seen_comment_ids: set[str] = set()
        self._latest_signal: ResolvedInformation | None = None

    def sync_internal_resources(self) -> None:
        self.state.sync_resources(self.cash, self.holdings)

    def collect_information(self, time: Timestamp) -> list[ResolvedInformation]:
        news_state = self.symbol_states[self.symbol][ProfessionalNewsState.__name__]
        social_state = self.symbol_states[self.symbol][SocialNetworkState.__name__]
        assert isinstance(news_state, ProfessionalNewsState)
        assert isinstance(social_state, SocialNetworkState)
        return resolve_retail_information(
            symbol=self.symbol,
            as_of=time,
            agent_id=self.agent_id,
            profile=self.profile,
            news_state=news_state,
            social_state=social_state,
            seen_news_ids=self.seen_news_ids,
            seen_post_ids=self.seen_post_ids,
            seen_comment_ids=self.seen_comment_ids,
        )

    def consume_information(self, item: ResolvedInformation) -> None:
        self._latest_signal = item
        self.state.ingest_information(
            source=item.source,
            time=item.available_time,
            direction=item.direction,
            strength=item.strength,
            credibility=item.credibility,
        )
        if item.source == "news":
            self.seen_news_ids.add(item.item_id)
        elif item.source == "social":
            self.seen_post_ids.add(item.item_id)
            self._maybe_engage_with_post(item, time=item.available_time)
        else:
            self.seen_comment_ids.add(item.item_id)

    def _maybe_engage_with_post(self, post_info: ResolvedInformation, *, time: Timestamp) -> None:
        social_state = self.symbol_states[self.symbol][SocialNetworkState.__name__]
        if not isinstance(social_state, SocialNetworkState):
            return
        posting_intensity = self.profile.personality.posting_intensity()
        aligned = post_info.direction * self.state.belief >= 0
        like_p = _clamp(0.10 + 0.35 * posting_intensity + (0.20 if aligned else 0.0))
        comment_p = _clamp(0.05 + 0.30 * posting_intensity + (0.10 if aligned else 0.05))
        if self.rng.random() < like_p:
            social_state.record_like(viewer_id=self.agent_id, post_id=post_info.item_id, now=time)
        if self.rng.random() < comment_p:
            post = social_state.get_post(post_info.item_id)
            if post is not None:
                comment = compose_retail_comment(
                    comment_id=f"comment-{self.agent_id}-{post.post_id}-{self.state.authored_posts}",
                    post=post,
                    author_agent_id=self.agent_id,
                    created_time=time,
                    own_belief=self.state.belief,
                    own_credibility=post_info.credibility,
                )
                social_state.record_comment(viewer_id=self.agent_id, post_id=post.post_id, comment=comment, now=time)

    def maybe_publish_social_post(self, time: Timestamp, new_information: list[ResolvedInformation]) -> None:
        social_state = self.symbol_states[self.symbol][SocialNetworkState.__name__]
        assert isinstance(social_state, SocialNetworkState)
        posting_intensity = self.profile.personality.posting_intensity()
        if new_information:
            strongest = max(new_information, key=lambda item: abs(item.direction) * item.strength * item.credibility)
            signal_strength = abs(strongest.direction) * strongest.strength * strongest.credibility
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
            signal_strength = 0.0
        if not self.state.should_post(posting_intensity, signal_strength):
            return

        author_score = social_state.author_engagement_score(self.agent_id, as_of=time)
        influence = _clamp(0.30 + 0.10 * min(5.0, author_score / 2.0))
        post = compose_retail_post(
            post_id=f"post-{self.agent_id}-{self.state.authored_posts + 1}",
            symbol=self.symbol,
            author_agent_id=self.agent_id,
            info=ResolvedInformation(
                item_id=strongest.item_id,
                symbol=strongest.symbol,
                source=strongest.source,
                available_time=time,
                topic=strongest.topic,
                direction=strongest.direction,
                strength=strongest.strength,
                credibility=strongest.credibility,
                sentiment=strongest.sentiment,
                summary=strongest.summary,
                author_agent_id=strongest.author_agent_id,
                source_news_id=strongest.source_news_id,
            ),
            posting_intensity=posting_intensity,
            influence=influence,
        )
        social_state.publish_post(post, now=time)
        self.state.record_post()

    def make_execution_intent(self, time: Timestamp) -> ExecutionIntent:
        risk_tolerance = self.profile.personality.risk_tolerance()
        if not self.state.should_trade(risk_tolerance):
            return ExecutionIntent(mode="none", rationale="low conviction")

        belief = self.state.belief
        if abs(belief) < 0.08:
            return ExecutionIntent(mode="none", rationale="weak directional belief")

        direction = "B" if belief >= 0 else "S"
        if direction == "S" and self.tradable_holdings.get(self.symbol, 0) < 100:
            return ExecutionIntent(mode="none", rationale="retail cannot short without inventory")

        social_heat = self.profile.personality.trading_heat()
        volume = round_lot(int(self.profile.base_order_size * (0.8 + 1.5 * self.state.conviction + social_heat)))
        aggressiveness = _clamp(0.20 + 0.35 * social_heat + 0.25 * self.state.stress + 0.20 * self.state.conviction)
        price_offset_ticks = 100 if abs(self.recent_return(time, lookback_seconds=900)) > 0.01 else 0
        return ExecutionIntent(
            mode="directional",
            direction=direction,
            target_volume=volume,
            aggressiveness=aggressiveness,
            price_offset_ticks=price_offset_ticks,
            rationale="retail belief-driven order",
        )

    def interval_multiplier(self) -> float:
        return max(0.45, 1.35 - 0.60 * self.profile.personality.trading_heat() - 0.35 * self.state.conviction)

    def update_after_trade(self, pnl_delta: float) -> None:
        self.state.update_after_trade(pnl_delta)

    def record_order_submission(self, count: int) -> None:
        self.state.record_order_submission(count)

    def snapshot_metrics(self) -> dict[str, float]:
        wealth = self.mark_to_market_wealth()
        return {
            "type": 0.0,
            "belief": float(self.state.belief),
            "conviction": float(self.state.conviction),
            "stress": float(self.state.stress),
            "consumed_news": float(self.state.consumed_news),
            "consumed_posts": float(self.state.consumed_posts),
            "believed_posts": float(self.state.believed_posts),
            "authored_posts": float(self.state.authored_posts),
            "submitted_orders": float(self.state.submitted_orders),
            "executed_trades": float(self.state.executed_trades),
            "wealth": float(wealth or self.cash),
        }
