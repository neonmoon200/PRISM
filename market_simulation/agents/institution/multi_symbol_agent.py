"""多标的机构 agent：每家机构负责一小组成分股（默认 10 只池子里均分）。"""

from __future__ import annotations

import logging
from typing import Any

from pandas import Timedelta, Timestamp

from market_simulation.agents.core.base import HeterogeneousAgentBase
from market_simulation.agents.core.execution import ExecutionIntent, build_orders_from_intent, round_lot
from market_simulation.agents.core.llm_protocols import InstitutionOpenAILLM
from market_simulation.agents.institution.agent import _clamp
from market_simulation.agents.institution.research_report import build_daily_research_report_news
from market_simulation.information import ResolvedInformation, resolve_institution_information
from market_simulation.information.link_content_tool import read_link_report_content
from market_simulation.personas import InstitutionAgentState, InstitutionStrategyProfile
from market_simulation.states.professional_news_state import ProfessionalNewsState
from market_simulation.states.trade_info_state import TradeInfoState
from market_simulation.utils.session_calendar import SessionCalendar
from mlib.core.lob_snapshot import LobSnapshot
from mlib.core.state import State


_logger = logging.getLogger(__name__)


class InstitutionMultiSymbolTraderAgent(HeterogeneousAgentBase):
    """机构账户在若干标的间择一交易；共享现金，按 symbol 维护研究状态。"""

    def __init__(
        self,
        *,
        symbols: list[str],
        session_calendar: SessionCalendar,
        start_time: Timestamp,
        end_time: Timestamp,
        profile: InstitutionStrategyProfile,
        reference_prices: dict[str, int],
        seed: int = 0,
        institution_llm: InstitutionOpenAILLM | None = None,
        institution_llm_call_probability: float = 0.0,
    ) -> None:
        if not symbols:
            raise ValueError("InstitutionMultiSymbolTraderAgent 至少需要 1 个 symbol")
        primary = symbols[0]
        super().__init__(
            symbol=primary,
            session_calendar=session_calendar,
            start_time=start_time,
            end_time=end_time,
            init_cash=profile.initial_cash * 1000,
            initial_position=profile.initial_position,
            reference_price=int(reference_prices.get(primary, 100_000)),
            base_interval_seconds=profile.base_interval_seconds,
            seed=seed,
        )
        self.wakeup_tier = 1
        self.symbols: list[str] = list(symbols)
        self.profile = profile
        self.reference_prices: dict[str, int] = {
            s: int(reference_prices.get(s, self.reference_price)) for s in self.symbols
        }
        self.states_by_symbol: dict[str, InstitutionAgentState] = {
            s: InstitutionAgentState.from_profile(profile) for s in self.symbols
        }
        self.seen_news_ids: set[str] = set()
        self.institution_llm = institution_llm
        self.institution_llm_call_probability = institution_llm_call_probability
        self._institution_llm_calls = 0
        self._latest_news_by_symbol: dict[str, dict[str, str]] = {}
        self._trade_symbol: str | None = None
        self.social_learning_enabled = False
        self._llm_outputs: list[dict[str, Any]] = []
        self._last_research_report_day: Timestamp | None = None

    def get_next_wakeup_time(self, time: Timestamp) -> Timestamp | None:
        return self.session_calendar.next_institution_daily_wakeup(time)

    def on_market_open(self, time: Timestamp, symbols: list[str]) -> None:
        super().on_market_open(time, symbols)
        if self.initial_position and self.symbol in self.holdings:
            self.holdings[self.symbol] = self.initial_position
            self.tradable_holdings[self.symbol] = self.initial_position

    def sync_internal_resources(self) -> None:
        self._llm_outputs = []
        for sym, state in self.states_by_symbol.items():
            state.sync_resources(self.cash, self.holdings)

    def collect_information(self, time: Timestamp) -> list[ResolvedInformation]:
        merged: list[ResolvedInformation] = []
        for sym in self.symbols:
            news_state = self.symbol_states[sym][ProfessionalNewsState.__name__]
            assert isinstance(news_state, ProfessionalNewsState)
            merged.extend(
                resolve_institution_information(
                    symbol=sym,
                    as_of=time,
                    agent_id=self.agent_id,
                    profile=self.profile,
                    news_state=news_state,
                    seen_news_ids=self.seen_news_ids,
                )
            )
        merged.sort(key=lambda item: item.available_time)
        return merged

    def _publish_daily_research_report(
        self,
        *,
        time: Timestamp,
        symbol: str,
        report_text: str,
    ) -> None:
        report_day = time.normalize()
        if self._last_research_report_day is not None and self._last_research_report_day == report_day:
            return
        news_state = self.symbol_states[symbol][ProfessionalNewsState.__name__]
        if not isinstance(news_state, ProfessionalNewsState):
            return
        report = build_daily_research_report_news(
            report_date=time,
            agent_id=int(self.agent_id),
            symbol=symbol,
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
        sym = item.symbol if item.symbol in self.states_by_symbol else self.symbol
        state = self.states_by_symbol[sym]
        state.ingest_news(
            time=item.available_time,
            direction=item.direction,
            strength=item.strength,
            credibility=item.credibility,
            topic_weight=self.profile.signal_weight(item.topic),
            reaction_speed=self.profile.reaction_speed,
        )
        self.seen_news_ids.add(item.item_id)
        self._latest_news_by_symbol[sym] = {
            "summary": item.summary or item.topic or "",
            "url": item.source_url or "",
        }

    def _mark_price(self, symbol: str) -> float:
        lob = self._get_lob(symbol)
        mid = self._safe_mid_price(lob) if lob is not None else None
        if mid is None or mid <= 0:
            mid = self.reference_prices.get(symbol, self.reference_price)
        return float(mid) / 1000.0

    def _get_lob(self, symbol: str) -> LobSnapshot | None:
        state = self.symbol_states.get(symbol, {}).get(State.__name__)
        if not isinstance(state, State):
            return None
        return state.lob_snapshot

    def recent_return(self, time: Timestamp, lookback_seconds: int = 600, *, symbol: str) -> float:
        state = self.symbol_states.get(symbol, {}).get(TradeInfoState.__name__)
        if not isinstance(state, TradeInfoState) or not state.trade_infos:
            return 0.0
        cutoff = time - Timedelta(seconds=lookback_seconds)
        current_mid = self._safe_mid_price(state.trade_infos[-1].lob_snapshot)
        reference_mid = self._safe_mid_price(state.trade_infos[0].lob_snapshot)
        if current_mid is None or reference_mid is None:
            return 0.0
        for item in reversed(state.trade_infos):
            if item.order.time <= cutoff:
                mid = self._safe_mid_price(item.lob_snapshot)
                if mid is not None:
                    reference_mid = mid
                break
        if reference_mid <= 0:
            return 0.0
        return (current_mid - reference_mid) / reference_mid

    def make_execution_intent(self, time: Timestamp) -> ExecutionIntent:
        self._trade_symbol = None
        generate_report = self._should_generate_daily_report(time)
        llm = self.institution_llm
        if llm is None:
            return ExecutionIntent(mode="none", rationale="institution_llm_unavailable")
        if (not generate_report) and (not llm.maybe(self.rng, self.institution_llm_call_probability)):
            return ExecutionIntent(mode="none", rationale="institution_llm_unavailable")

        candidates: list[dict[str, Any]] = []
        for sym in self.symbols:
            st = self.states_by_symbol[sym]
            news = self._latest_news_by_symbol.get(sym, {})
            url = news.get("url", "")
            report_excerpt = read_link_report_content(url) if url else ""
            candidates.append(
                {
                    "symbol": sym,
                    "belief": float(st.belief),
                    "conviction": float(st.conviction),
                    "risk_pressure": float(st.risk_pressure),
                    "recent_return_30min": float(self.recent_return(time, lookback_seconds=1_800, symbol=sym)),
                    "position": int(self.holdings.get(sym, 0)),
                    "mark_price": self._mark_price(sym),
                    "latest_news_summary": news.get("summary", ""),
                    "latest_news_url": url,
                    "latest_news_report": report_excerpt,
                }
            )

        try:
            parsed = llm.decide_trade_multi(
                strategy=self.profile.strategy,
                profile_name=self.profile.name,
                cash=float(self.cash) / 1000.0,
                base_order_size=int(self.profile.base_order_size),
                candidates=candidates,
                generate_report=generate_report,
                simulation_time=str(time),
            )
            self._llm_outputs.append(
                {
                    "kind": "institution_trade_multi",
                    "symbols": list(self.symbols),
                    "decision": parsed,
                }
            )
        except Exception as exc:
            self._llm_outputs.append(
                {"kind": "institution_trade_multi", "symbols": list(self.symbols), "error": str(exc)[:400]}
            )
            _logger.warning(
                "[多标的机构 agent_id=%s] LLM 调用失败: %s",
                self.agent_id,
                str(exc)[:400],
            )
            return ExecutionIntent(mode="none", rationale="institution_llm_call_failed")

        if parsed is None:
            return ExecutionIntent(mode="none", rationale="institution_llm_unparsable")

        self._institution_llm_calls += 1
        sym = str(parsed.get("symbol", "")).strip()
        report_symbol = sym if sym in self.states_by_symbol else self.symbol
        if generate_report:
            report_text = str(parsed.get("research_report_text", "") or "").strip()
            if report_text:
                self._publish_daily_research_report(time=time, symbol=report_symbol, report_text=report_text)
        if sym not in self.states_by_symbol:
            return ExecutionIntent(mode="none", rationale="institution_llm_bad_symbol")
        self._trade_symbol = sym

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

    def build_orders(self, *, time: Timestamp, intent: ExecutionIntent) -> list[BaseOrder]:
        sym = self._trade_symbol or self.symbol
        lob = self._get_lob(sym)
        if lob is None:
            return self._build_bootstrap_orders(time=time, intent=intent, symbol=sym)
        return build_orders_from_intent(agent=self, time=time, symbol=sym, lob=lob, intent=intent)

    def _build_bootstrap_orders(
        self, *, time: Timestamp, intent: ExecutionIntent, symbol: str
    ) -> list[BaseOrder]:
        if intent.mode == "none" or intent.target_volume <= 0:
            return []
        volume = max(0, intent.target_volume // 100 * 100)
        if volume < 100:
            return []
        ref = self.reference_prices.get(symbol, self.reference_price)
        if intent.mode == "liquidity":
            orders: list[BaseOrder] = []
            buy_volume = min(volume, max(0, int(self.tradable_cash / max(ref, 100)) // 100 * 100))
            sell_volume = min(volume, self.tradable_holdings.get(symbol, 0) // 100 * 100)
            if buy_volume >= 100:
                orders.extend(
                    self.construct_valid_orders(
                        time=time,
                        symbol=symbol,
                        type="B",
                        price=max(100, ref - 100),
                        volume=buy_volume,
                    )
                )
            if sell_volume >= 100:
                orders.extend(
                    self.construct_valid_orders(
                        time=time,
                        symbol=symbol,
                        type="S",
                        price=max(100, ref + 100),
                        volume=sell_volume,
                    )
                )
            return orders
        if intent.mode == "directional" and intent.direction in ("B", "S"):
            if intent.direction == "B":
                buy_volume = min(volume, max(0, int(self.tradable_cash / max(ref, 100)) // 100 * 100))
                if buy_volume < 100:
                    return []
                return self.construct_valid_orders(
                    time=time, symbol=symbol, type="B", price=max(100, ref), volume=buy_volume
                )
            sell_volume = min(volume, self.tradable_holdings.get(symbol, 0) // 100 * 100)
            if sell_volume < 100:
                return []
            return self.construct_valid_orders(
                time=time, symbol=symbol, type="S", price=max(100, ref), volume=sell_volume
            )
        return []

    def interval_multiplier(self) -> float:
        st = self.states_by_symbol[self.symbol]
        return max(0.35, 1.10 - 0.45 * self.profile.reaction_speed - 0.30 * st.conviction)

    def update_after_trade(self, pnl_delta: float) -> None:
        sym = self._trade_symbol or self.symbol
        self.states_by_symbol[sym].update_after_trade(pnl_delta)

    def record_order_submission(self, count: int) -> None:
        sym = self._trade_symbol or self.symbol
        self.states_by_symbol[sym].record_order_submission(count)

    def snapshot_metrics(self) -> dict[str, float]:
        wealth = self.mark_to_market_wealth()
        st = self.states_by_symbol[self.symbol]
        return {
            "type": 1.0,
            "social_learning_enabled": 0.0,
            "belief": float(st.belief),
            "conviction": float(st.conviction),
            "risk_pressure": float(st.risk_pressure),
            "consumed_news": float(sum(s.consumed_news for s in self.states_by_symbol.values())),
            "submitted_orders": float(sum(s.submitted_orders for s in self.states_by_symbol.values())),
            "executed_trades": float(sum(s.executed_trades for s in self.states_by_symbol.values())),
            "wealth": float((wealth if wealth is not None else self.cash) / 1000.0),
            "institution_llm_calls": float(self._institution_llm_calls),
            "num_symbols": float(len(self.symbols)),
        }
