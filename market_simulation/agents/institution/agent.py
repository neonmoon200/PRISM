from __future__ import annotations

import hashlib
import logging

from typing import Any

from pandas import Timestamp

from market_simulation.agents.core.base import HeterogeneousAgentBase
from market_simulation.agents.core.execution import ExecutionIntent, round_lot
from market_simulation.agents.core.llm_protocols import InstitutionOpenAILLM
from market_simulation.agents.institution.research_report import build_daily_research_report_news
from market_simulation.information import ResolvedInformation, resolve_institution_information
from market_simulation.information.link_content_tool import read_link_report_content
from market_simulation.personas import InstitutionAgentState, InstitutionStrategyProfile
from market_simulation.states.professional_news_state import ProfessionalNewsState
from market_simulation.utils.session_calendar import SessionCalendar


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


_logger = logging.getLogger(__name__)


class InstitutionTraderAgent(HeterogeneousAgentBase):
    """Professional trader driven by a strategy identity and richer news access."""

    def __init__(
        self,
        *,
        symbol: str,
        session_calendar: SessionCalendar,
        start_time: Timestamp,
        end_time: Timestamp,
        profile: InstitutionStrategyProfile,
        reference_price: int = 100_000,
        seed: int = 0,
        institution_llm: InstitutionOpenAILLM | None = None,
        institution_llm_call_probability: float = 0.0,
    ) -> None:
        super().__init__(
            symbol=symbol,
            session_calendar=session_calendar,
            start_time=start_time,
            end_time=end_time,
            init_cash=profile.initial_cash * 1000,  # yuan → tick (mlib exchange unit)
            initial_position=profile.initial_position,
            reference_price=reference_price,
            base_interval_seconds=profile.base_interval_seconds,
            seed=seed,
        )
        self.wakeup_tier = 1  # 同刻唤醒次于散户
        self.profile = profile
        self.state = InstitutionAgentState.from_profile(profile)
        self.seen_news_ids: set[str] = set()
        self.institution_llm = institution_llm
        self.institution_llm_call_probability = institution_llm_call_probability
        self._institution_llm_calls = 0
        self._latest_news_summary: str = ""
        self._latest_news_url: str = ""
        # 机构不做社会学习：仅消费专业新闻。
        self.social_learning_enabled = False
        self._llm_outputs: list[dict[str, Any]] = []
        self._last_research_report_day: Timestamp | None = None

    def get_next_wakeup_time(self, time: Timestamp) -> Timestamp | None:
        """09:30 开盘；10:30 起；交易时段内整点（11/13/14/15 等）。同刻散户优先。"""
        return self.session_calendar.next_institution_daily_wakeup(time)

    def sync_internal_resources(self) -> None:
        self._llm_outputs = []
        self.state.sync_resources(self.cash, self.holdings)

    def collect_information(self, time: Timestamp) -> list[ResolvedInformation]:
        news_state = self.symbol_states[self.symbol][ProfessionalNewsState.__name__]
        assert isinstance(news_state, ProfessionalNewsState)
        return resolve_institution_information(
            symbol=self.symbol,
            as_of=time,
            agent_id=self.agent_id,
            profile=self.profile,
            news_state=news_state,
            seen_news_ids=self.seen_news_ids,
        )

    def _publish_daily_research_report(
        self,
        *,
        time: Timestamp,
        report_text: str,
    ) -> None:
        report_day = time.normalize()
        if self._last_research_report_day is not None and self._last_research_report_day == report_day:
            return
        news_state = self.symbol_states[self.symbol][ProfessionalNewsState.__name__]
        if not isinstance(news_state, ProfessionalNewsState):
            return
        report = build_daily_research_report_news(
            report_date=time,
            agent_id=int(self.agent_id),
            symbol=self.symbol,
            report_text=report_text,
        )
        if report is None:
            return
        news_state.add_news(report)
        self._last_research_report_day = report_day
        self._llm_outputs.append(
            {
                "kind": "institution_daily_report",
                "news_id": report.news_id,
                "symbol": report.symbol,
                "publish_time": str(report.publish_time),
                "retail_delay_days": report.retail_delay_days,
            }
        )

    def _should_generate_daily_report(self, time: Timestamp) -> bool:
        report_day = time.normalize()
        if self._last_research_report_day is not None and self._last_research_report_day == report_day:
            return False
        nxt = self.session_calendar.next_institution_daily_wakeup(time)
        return nxt is None or nxt.normalize() > report_day

    def consume_information(self, item: ResolvedInformation) -> None:
        self.state.ingest_news(
            time=item.available_time,
            direction=item.direction,
            strength=item.strength,
            credibility=item.credibility,
            topic_weight=self.profile.signal_weight(item.topic),
            reaction_speed=self.profile.reaction_speed,
        )
        self.seen_news_ids.add(item.item_id)
        self._latest_news_summary = item.summary or item.topic or ""
        self._latest_news_url = item.source_url or ""

    def _mark_price(self) -> float:
        """以 tick 为单位的中间价，单位换算为元（除以 1000）便于 LLM 阅读。"""
        lob = self.get_lob()
        mid = self._safe_mid_price(lob) if lob is not None else None
        if mid is None or mid <= 0:
            mid = self.reference_price
        return float(mid) / 1000.0

    def make_execution_intent(self, time: Timestamp) -> ExecutionIntent:
        """机构本次唤醒的交易意图**完全由 LLM 决定**。

        仅做两件 agent 侧的事：
        1. 把上下文（含 strategy 提示词、内部状态、新闻、账户）发给 LLM；
        2. 把 LLM 的 JSON 翻译成 ``ExecutionIntent``——是否交易、买/卖、做市/方向、
           规模、激进度全部读自 LLM。

        agent 不再做 ``should_trade`` / ``|signal|`` / 库存规则等策略层过滤；
        资金/持仓不足之类的物理约束由下游 ``build_orders_from_intent`` 兜底。
        """

        recent_ret = self.recent_return(time, lookback_seconds=1_800)
        strategy = self.profile.strategy
        generate_report = self._should_generate_daily_report(time)

        llm = self.institution_llm
        if llm is None:
            return ExecutionIntent(mode="none", rationale="institution_llm_unavailable")
        if (not generate_report) and (not llm.maybe(self.rng, self.institution_llm_call_probability)):
            # 没有 LLM（或概率门未通过）：本次不交易。
            # 严格保证"由 LLM 决定"语义——没有 LLM 就不执行任何规则化决策。
            return ExecutionIntent(mode="none", rationale="institution_llm_unavailable")

        report_excerpt = read_link_report_content(self._latest_news_url)
        report_text = report_excerpt or ""
        report_meta = {
            "latest_news_url": self._latest_news_url,
            "latest_news_report_loaded": bool(report_text),
            "latest_news_report_len": len(report_text),
            "latest_news_report_sha1": hashlib.sha1(report_text.encode("utf-8")).hexdigest() if report_text else "",
            "latest_news_report_preview": report_text[:200],
        }
        try:
            parsed = llm.decide_trade(
                strategy=strategy,
                symbol=self.symbol,
                profile_name=self.profile.name,
                belief=float(self.state.belief),
                conviction=float(self.state.conviction),
                risk_pressure=float(self.state.risk_pressure),
                recent_return=float(recent_ret),
                cash=float(self.cash) / 1000.0,  # tick → yuan for LLM readability
                position=int(self.holdings.get(self.symbol, 0)),
                mark_price=self._mark_price(),
                base_order_size=int(self.profile.base_order_size),
                latest_news_summary=self._latest_news_summary,
                latest_news_url=self._latest_news_url,
                latest_news_report=report_excerpt,
                generate_report=generate_report,
                simulation_time=str(time),
            )
            self._llm_outputs.append(
                {
                    "kind": "institution_trade",
                    "symbol": self.symbol,
                    "decision": parsed,
                    **report_meta,
                }
            )
        except Exception as exc:
            self._llm_outputs.append(
                {
                    "kind": "institution_trade",
                    "symbol": self.symbol,
                    "error": str(exc)[:400],
                    **report_meta,
                }
            )
            _logger.warning(
                "[机构 agent_id=%s symbol=%s] institution LLM 调用失败，本唤醒不交易: %s",
                self.agent_id,
                self.symbol,
                str(exc)[:400],
            )
            return ExecutionIntent(mode="none", rationale="institution_llm_call_failed")
        if parsed is None:
            # strict 模式下 llm 已经抛出；非 strict 解析失败时不下单。
            return ExecutionIntent(mode="none", rationale="institution_llm_unparsable")

        self._institution_llm_calls += 1
        if generate_report:
            report_text = str(parsed.get("research_report_text", "") or "").strip()
            if report_text:
                self._publish_daily_research_report(time=time, report_text=report_text)

        mode_raw = str(parsed.get("mode", "none")).strip().lower()
        if mode_raw not in ("directional", "liquidity", "none"):
            mode_raw = "none"
        if mode_raw == "none":
            reason = str(parsed.get("reason", "") or "")[:180]
            return ExecutionIntent(mode="none", rationale=f"llm_hold: {reason}".strip())

        try:
            size_ratio = float(parsed.get("size_ratio", 1.0))
        except (TypeError, ValueError):
            size_ratio = 1.0
        size_ratio = max(0.0, min(3.0, size_ratio))

        try:
            aggressiveness = float(parsed.get("aggressiveness", 0.5))
        except (TypeError, ValueError):
            aggressiveness = 0.5
        aggressiveness = _clamp(aggressiveness)

        target_volume = round_lot(int(self.profile.base_order_size * size_ratio))
        if target_volume < 100:
            reason = str(parsed.get("reason", "") or "")[:180]
            return ExecutionIntent(mode="none", rationale=f"llm_size_too_small: {reason}".strip())

        reason = str(parsed.get("reason", "") or "")[:180]

        if mode_raw == "liquidity":
            return ExecutionIntent(
                mode="liquidity",
                target_volume=target_volume,
                aggressiveness=aggressiveness,
                passive_spread_ticks=100,
                rationale=f"llm_liquidity: {reason}".strip(),
            )

        direction_raw = str(parsed.get("direction", "")).strip().upper()
        if direction_raw not in ("B", "S"):
            return ExecutionIntent(mode="none", rationale=f"llm_missing_direction: {reason}".strip())

        return ExecutionIntent(
            mode="directional",
            direction=direction_raw,  # type: ignore[arg-type]
            target_volume=target_volume,
            aggressiveness=aggressiveness,
            price_offset_ticks=0,
            rationale=f"llm_directional: {reason}".strip(),
        )

    def interval_multiplier(self) -> float:
        return max(0.35, 1.10 - 0.45 * self.profile.reaction_speed - 0.30 * self.state.conviction)

    def update_after_trade(self, pnl_delta: float) -> None:
        self.state.update_after_trade(pnl_delta)

    def record_order_submission(self, count: int) -> None:
        self.state.record_order_submission(count)

    def snapshot_metrics(self) -> dict[str, float]:
        wealth = self.mark_to_market_wealth()
        return {
            "type": 1.0,
            "social_learning_enabled": 0.0,
            "belief": float(self.state.belief),
            "conviction": float(self.state.conviction),
            "risk_pressure": float(self.state.risk_pressure),
            "consumed_news": float(self.state.consumed_news),
            "submitted_orders": float(self.state.submitted_orders),
            "executed_trades": float(self.state.executed_trades),
            "wealth": float((wealth if wealth is not None else self.cash) / 1000.0),  # tick → yuan
            "institution_llm_calls": float(self._institution_llm_calls),
        }
