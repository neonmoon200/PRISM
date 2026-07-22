from __future__ import annotations

from abc import abstractmethod
from random import Random

from pandas import Timedelta, Timestamp

from market_simulation.agents.core.execution import ExecutionIntent, build_orders_from_intent
from market_simulation.information import ResolvedInformation
from market_simulation.states.trade_info_state import TradeInfoState
from market_simulation.utils.session_calendar import SessionCalendar
from mlib.core.action import Action
from mlib.core.base_agent import BaseAgent
from mlib.core.base_order import BaseOrder
from mlib.core.lob_snapshot import LobSnapshot
from mlib.core.observation import Observation
from mlib.core.state import State
from mlib.core.transaction import Transaction


class HeterogeneousAgentBase(BaseAgent):
    """Shared execution loop for retail and institution agents."""

    def __init__(
        self,
        *,
        symbol: str,
        session_calendar: SessionCalendar,
        start_time: Timestamp,
        end_time: Timestamp,
        init_cash: float,
        initial_position: int = 0,
        reference_price: int = 100_000,
        base_interval_seconds: float = 30.0,
        seed: int = 0,
    ) -> None:
        super().__init__(init_cash=init_cash, communication_delay=0, computation_delay=0)
        self.wakeup_tier = 0  # 散户/异质 agent 默认同刻优先于机构
        self.symbol = symbol
        self.session_calendar = session_calendar
        self.start_time = start_time
        self.end_time = end_time
        self.initial_position = initial_position
        self.reference_price = reference_price
        self.base_interval_seconds = base_interval_seconds
        self.rng = Random(seed)
        self._last_wealth: float | None = None
        self._last_information_batch: list[ResolvedInformation] = []
        self._last_intent: ExecutionIntent = ExecutionIntent(mode="none", rationale="init")

    def on_market_open(self, time: Timestamp, symbols: list[str]) -> None:
        super().on_market_open(time, symbols)
        self.holdings[self.symbol] = self.initial_position
        self.tradable_holdings[self.symbol] = self.initial_position

    def get_action(self, observation: Observation) -> Action:
        assert self.agent_id == observation.agent.agent_id
        time = observation.time

        if time < self.start_time:
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=self.start_time)
        if time > self.end_time:
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=None)

        self.sync_internal_resources()
        new_information = self.collect_information(time)
        self._last_information_batch = list(new_information)
        for item in new_information:
            self.consume_information(item)

        if not observation.is_market_open_wakup:
            self.maybe_publish_social_post(time, new_information)

        next_wakeup = self.get_next_wakeup_time(time)
        if observation.is_market_open_wakup:
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=next_wakeup)

        if not self.session_calendar.is_open(time):
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=next_wakeup)

        intent = self.make_execution_intent(time)
        self._last_intent = intent
        orders = self.build_orders(time=time, intent=intent)
        self.record_order_submission(len(orders))
        return Action(agent_id=self.agent_id, orders=orders, time=time, next_wakeup_time=next_wakeup)

    def get_next_wakeup_time(self, time: Timestamp) -> Timestamp | None:
        return self.session_calendar.next_trading_agent_wakeup(time)

    def get_market_state(self) -> State | None:
        state = self.symbol_states.get(self.symbol, {}).get(State.__name__)
        if not isinstance(state, State):
            return None
        return state

    def get_lob(self) -> LobSnapshot | None:
        state = self.get_market_state()
        if state is None:
            return None
        return state.lob_snapshot

    @staticmethod
    def _safe_mid_price(lob: LobSnapshot | None) -> int | None:
        if lob is None:
            return None
        try:
            return int(lob.mid_price)
        except ValueError:
            # Empty/corrupted snapshot: use last price if available, otherwise caller fallback.
            if getattr(lob, "last_price", 0) and lob.last_price > 0:
                return int(lob.last_price)
            return None

    def recent_return(self, time: Timestamp, lookback_seconds: int = 600) -> float:
        state = self.symbol_states.get(self.symbol, {}).get(TradeInfoState.__name__)
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

    def build_orders(self, *, time: Timestamp, intent: ExecutionIntent) -> list[BaseOrder]:
        lob = self.get_lob()
        if lob is None:
            return self._build_bootstrap_orders(time=time, intent=intent)
        return build_orders_from_intent(agent=self, time=time, symbol=self.symbol, lob=lob, intent=intent)

    def _build_bootstrap_orders(self, *, time: Timestamp, intent: ExecutionIntent) -> list[BaseOrder]:
        if intent.mode == "none" or intent.target_volume <= 0:
            return []
        volume = max(0, intent.target_volume // 100 * 100)
        if volume < 100:
            return []
        if intent.mode == "liquidity":
            orders: list[BaseOrder] = []
            buy_volume = min(volume, max(0, int(self.tradable_cash / max(self.reference_price, 100)) // 100 * 100))
            sell_volume = min(volume, self.tradable_holdings.get(self.symbol, 0) // 100 * 100)
            if buy_volume >= 100:
                orders.extend(
                    self.construct_valid_orders(
                        time=time,
                        symbol=self.symbol,
                        type="B",
                        price=max(100, self.reference_price - 100),
                        volume=buy_volume,
                    )
                )
            if sell_volume >= 100:
                orders.extend(
                    self.construct_valid_orders(
                        time=time,
                        symbol=self.symbol,
                        type="S",
                        price=max(100, self.reference_price + 100),
                        volume=sell_volume,
                    )
                )
            return orders

        if intent.direction == "B":
            buy_volume = min(volume, max(0, int(self.tradable_cash / max(self.reference_price, 100)) // 100 * 100))
            if buy_volume < 100:
                return []
            return self.construct_valid_orders(
                time=time,
                symbol=self.symbol,
                type="B",
                price=max(100, self.reference_price + intent.price_offset_ticks),
                volume=buy_volume,
            )

        sell_volume = min(volume, self.tradable_holdings.get(self.symbol, 0) // 100 * 100)
        if sell_volume < 100:
            return []
        return self.construct_valid_orders(
            time=time,
            symbol=self.symbol,
            type="S",
            price=max(100, self.reference_price - intent.price_offset_ticks),
            volume=sell_volume,
        )

    def mark_to_market_wealth(self) -> float | None:
        state = self.get_market_state()
        if state is None or state.lob_snapshot is None:
            return None
        mid = self._safe_mid_price(state.lob_snapshot)
        if mid is None:
            return None
        return self.cash + self.holdings.get(self.symbol, 0) * mid

    def on_order_executed(self, time: Timestamp, transaction: Transaction, trans_order_id_to_notify: int) -> list[tuple[str, int]]:
        del_order_ids = super().on_order_executed(time, transaction, trans_order_id_to_notify)
        wealth = self.mark_to_market_wealth()
        if wealth is not None and self._last_wealth is not None:
            self.update_after_trade(wealth - self._last_wealth)
        if wealth is not None:
            self._last_wealth = wealth
        return del_order_ids

    @abstractmethod
    def sync_internal_resources(self) -> None:
        """Push account information into agent-specific state."""

    @abstractmethod
    def collect_information(self, time: Timestamp) -> list[ResolvedInformation]:
        """Resolve visible information for the current wakeup."""

    @abstractmethod
    def consume_information(self, item: ResolvedInformation) -> None:
        """Update internal belief state from one resolved information item."""

    def maybe_publish_social_post(self, time: Timestamp, new_information: list[ResolvedInformation]) -> None:
        del time, new_information

    @abstractmethod
    def make_execution_intent(self, time: Timestamp) -> ExecutionIntent:
        """Turn internal state into a normalized execution intent."""

    @abstractmethod
    def interval_multiplier(self) -> float:
        """Scale the base wakeup interval according to the agent state."""

    @abstractmethod
    def update_after_trade(self, pnl_delta: float) -> None:
        """Write realized trading feedback back to the internal state."""

    @abstractmethod
    def record_order_submission(self, count: int) -> None:
        """Track how many orders were produced."""

    @abstractmethod
    def snapshot_metrics(self) -> dict[str, float]:
        """Export end-of-run metrics for reporting."""
