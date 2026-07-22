from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pandas import Timestamp

from mlib.core.base_agent import BaseAgent
from mlib.core.base_order import BaseOrder
from mlib.core.lob_snapshot import LobSnapshot


def round_lot(volume: int) -> int:
    if volume <= 0:
        return 0
    if volume % 100 == 0:
        return volume
    return (volume // 100 + 1) * 100


def best_prices(lob: LobSnapshot) -> tuple[int, int]:
    ask1 = lob.ask_prices[0] if lob.ask_prices else lob.last_price
    bid1 = lob.bid_prices[0] if lob.bid_prices else lob.last_price
    return ask1, bid1


@dataclass(frozen=True)
class ExecutionIntent:
    """A normalized execution request shared by retail and institution agents."""

    mode: Literal["none", "directional", "liquidity"]
    direction: Literal["B", "S", "N"] = "N"
    target_volume: int = 0
    aggressiveness: float = 0.0
    price_offset_ticks: int = 0
    passive_spread_ticks: int = 300
    rationale: str = ""


def build_orders_from_intent(
    *,
    agent: BaseAgent,
    time: Timestamp,
    symbol: str,
    lob: LobSnapshot,
    intent: ExecutionIntent,
) -> list[BaseOrder]:
    if intent.mode == "none" or intent.target_volume <= 0:
        return []

    ask1, bid1 = best_prices(lob)
    try:
        mid = int(lob.mid_price)
    except ValueError:
        mid = int(lob.last_price) if lob.last_price > 0 else max(100, (ask1 + bid1) // 2)
    volume = round_lot(intent.target_volume)
    if volume <= 0:
        return []

    if intent.mode == "liquidity":
        orders: list[BaseOrder] = []
        buy_volume = min(volume, round_lot(int(agent.tradable_cash / max(bid1, 100))))
        if buy_volume >= 100:
            orders.extend(
                agent.construct_valid_orders(
                    time=time,
                    symbol=symbol,
                    type="B",
                    price=max(100, bid1 - intent.passive_spread_ticks),
                    volume=buy_volume,
                )
            )
        sell_volume = min(volume, round_lot(agent.tradable_holdings.get(symbol, 0)))
        if sell_volume >= 100:
            orders.extend(
                agent.construct_valid_orders(
                    time=time,
                    symbol=symbol,
                    type="S",
                    price=max(100, ask1 + intent.passive_spread_ticks),
                    volume=sell_volume,
                )
            )
        return orders

    if intent.direction == "B":
        affordable = round_lot(int(agent.tradable_cash / max(ask1, 100)))
        volume = min(volume, affordable)
        if volume < 100:
            return []
        passive_offset = max(100, int(intent.passive_spread_ticks // 2))
        if intent.aggressiveness >= 0.70:
            price = ask1 + intent.price_offset_ticks
        elif intent.aggressiveness >= 0.35:
            price = mid + intent.price_offset_ticks
        else:
            price = bid1 - passive_offset + intent.price_offset_ticks
        return agent.construct_valid_orders(time=time, symbol=symbol, type="B", price=max(100, price), volume=volume)

    sellable = round_lot(agent.tradable_holdings.get(symbol, 0))
    volume = min(volume, sellable)
    if volume < 100:
        return []
    passive_offset = max(100, int(intent.passive_spread_ticks // 2))
    if intent.aggressiveness >= 0.70:
        price = bid1 - intent.price_offset_ticks
    elif intent.aggressiveness >= 0.35:
        price = mid - intent.price_offset_ticks
    else:
        price = ask1 + passive_offset - intent.price_offset_ticks
    return agent.construct_valid_orders(time=time, symbol=symbol, type="S", price=max(100, price), volume=volume)
