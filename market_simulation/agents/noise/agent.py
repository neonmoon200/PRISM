from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import pandas as pd

from market_simulation.real_events.event_schedule import EventSchedule
from market_simulation.states.trade_info_state import TradeInfoState
from market_simulation.utils.session_calendar import SessionCalendar
from mlib.core.action import Action
from mlib.core.base_agent import BaseAgent

if TYPE_CHECKING:
    from mlib.core.lob_snapshot import LobSnapshot
    from mlib.core.observation import Observation


class NoiseAgent(BaseAgent):
    """Noise agent, which generates random orders based on predefined distributions."""

    def __init__(
        self,
        symbol: str,
        init_price: int,
        interval_seconds: float,
        session_calendar: SessionCalendar,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        seed: int,
        init_cash: int = 1_000,
        event_schedule: EventSchedule | None = None,
    ) -> None:
        super().__init__(
            init_cash=int(init_cash),
            communication_delay=0,
            computation_delay=0,
        )
        self.wakeup_tier = 2  # 同刻唤醒次于散户与机构
        self.symbol = symbol
        self.init_price = init_price
        self.start_time = start_time
        self.end_time = end_time
        self.interval_seconds = interval_seconds
        self.session_calendar = session_calendar
        # probabilities for order type, price and volume
        self.type_probs: dict[str, float] = {"B": 0.4, "S": 0.4, "C": 0.2}
        self.price_level_probs: dict[int, float] = {
            0: 0.12,
            100: 0.14,
            -100: 0.14,
            300: 0.10,
            -300: 0.10,
            500: 0.08,
            -500: 0.08,
            800: 0.06,
            -800: 0.06,
            1200: 0.04,
            -1200: 0.04,
            2000: 0.02,
            -2000: 0.02,
        }
        # 噪声单保持小额（最大 500 股/手单位），避免偶发 1000/2000 对簿造成过大冲击
        self.volume_probs: dict[int, float] = {100: 0.55, 200: 0.30, 500: 0.15}
        self.rnd = random.Random(seed)
        self.event_schedule = event_schedule

    def _effective_type_probs(self, time: pd.Timestamp) -> dict[str, float]:
        """Optional event_schedule columns: sell_tilt, buy_tilt, cancel_tilt (multiply base probs, renormalize)."""
        t = dict(self.type_probs)
        if self.event_schedule is None:
            return t
        p = self.event_schedule.params_at(time)
        st = p.get("sell_tilt", 1.0)
        bt = p.get("buy_tilt", 1.0)
        ct = p.get("cancel_tilt", 1.0)
        t["S"] *= st
        t["B"] *= bt
        t["C"] *= ct
        s = sum(t.values())
        if s <= 0:
            return dict(self.type_probs)
        return {k: v / s for k, v in t.items()}

    def _effective_volume(self, time: pd.Timestamp, base_volume: int) -> int:
        """Optional noise_volume_mult from schedule."""
        v = base_volume
        if self.event_schedule is not None:
            m = self.event_schedule.mult(time, "noise_volume_mult", 1.0)
            v = max(100, int(v * m))
            if v % 100 != 0:
                v = (v // 100) * 100
                v = max(100, v)
        return v

    def get_action(self, observation: Observation) -> Action:
        """Generate a random action based on the observation."""
        assert self.agent_id == observation.agent.agent_id, f"Agent ID mismatch: {self.agent_id} != {observation.agent.agent_id}"
        time = observation.time
        if time < self.start_time:
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=self.start_time)
        if time > self.end_time:
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=None)
        if not self.session_calendar.is_open(time):
            return Action(
                agent_id=self.agent_id,
                orders=[],
                time=time,
                next_wakeup_time=self.session_calendar.next_trading_time(time),
            )
        lob = self.get_lob()
        base_price = lob.last_price if lob and lob.last_price > 0 else self.init_price
        # generate a random order by sampling from the predefined distributions
        raw_vol = self._sample(self.volume_probs)
        vol = self._effective_volume(time, int(raw_vol))
        orders = self.construct_valid_orders(
            time=time,
            symbol=self.symbol,
            type=self._sample(self._effective_type_probs(time)),
            price=base_price + self._sample(self.price_level_probs),
            volume=vol,
            random=self.rnd,
        )
        noise_mult = 1.0
        if self.event_schedule is not None:
            noise_mult = self.event_schedule.mult(time, "noise_interval_mult", 1.0)
        next_delay_seconds = self.interval_seconds * noise_mult * self.rnd.uniform(0.5, 1.5)
        next_wakeup_time = self.session_calendar.advance(time, next_delay_seconds)
        action = Action(
            agent_id=self.agent_id,
            orders=orders,  # type: ignore
            time=time,
            next_wakeup_time=next_wakeup_time,
        )
        return action

    def get_lob(self) -> LobSnapshot | None:
        """Get the latest LOB snapshot from the TradeInfoState."""
        state = self.symbol_states[self.symbol][TradeInfoState.__name__]
        assert isinstance(state, TradeInfoState)
        if not state.trade_infos:
            return None
        lob = state.trade_infos[-1].lob_snapshot
        return lob

    def _sample(self, probs: dict[Any, float]) -> Any:
        """Sample from a discrete distribution."""
        return self.rnd.choices(population=list(probs.keys()), weights=list(probs.values()), k=1)[0]
