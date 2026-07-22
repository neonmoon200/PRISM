"""多标的（10 行业指数）雪球散户 agent。

设计要点：
* **每个 agent 同时关注 10 个 symbol**，持仓、belief、记忆都按 symbol 拆分。
* 内部状态：一份共享的人格（profile），N 份 ``XueqiuRetailState``（每个 symbol 一份），
  共享 cash。``self.holdings[symbol]`` 仍由 ``BaseAgent`` 的 multi-symbol 字段维护。
* 唤醒节拍：每个交易日 **2 次**（09:30 开盘 / 13:00 下午），见 ``SessionCalendar.next_retail_daily_wakeup``。
* 每次唤醒：
    1. 拉取新讯息（新闻/帖子/与自己相关的互动评论），统一送入认知 LLM；
    2. 认知输出仅更新状态快照，并按 ``useful`` 决定是否写入某个 symbol 的 memory；
    3. 10:30 / 13:00：过统一门槛后触发一次联合 LLM（发帖 + 交易）。
* 新机制下不维护独立 ``social_memory``，也不在认知阶段执行点赞/转发/评论动作。
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, time
from random import Random
from typing import Any, Literal

import pandas as pd
from pandas import Timedelta, Timestamp

from market_simulation.agents.core.base import HeterogeneousAgentBase
from market_simulation.agents.core.execution import (
    ExecutionIntent,
    build_orders_from_intent,
    round_lot,
)
from market_simulation.agents.core.llm_protocols import XueqiuLLM, get_default_llm
from market_simulation.information import (
    ProfessionalNews,
    ResolvedInformation,
    SocialPost,
    resolve_retail_information,
)
from market_simulation.personas.xueqiu_loader import XueqiuPersonaProfile
from market_simulation.personas.xueqiu_state import (
    MemoryItem,
    XueqiuRetailState,
    memory_settings,
    trim_memory_summary,
)
from market_simulation.social import compose_retail_comment, compose_retail_post
from market_simulation.states.professional_news_state import ProfessionalNewsState
from market_simulation.states.social_network_state import SocialNetworkState
from market_simulation.utils.session_calendar import SessionCalendar
from mlib.core.action import Action
from mlib.core.base_order import BaseOrder
from mlib.core.lob_snapshot import LobSnapshot
from mlib.core.observation import Observation
from mlib.core.state import State
from mlib.core.transaction import Transaction


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


_logger = logging.getLogger(__name__)
_DECISION_TEXT_MAX_LEN = 500
_DECISION_COMMENT_MAX_LEN = 40
_DECISION_TOPIC_MAX_LEN = 16
_DECISION_REASON_MAX_LEN = 32
_FOCUS_SYMBOLS_TOP_K = 3
_JOINT_SOCIAL_ITEMS_TOP_K = 3
_COGNITION_MEMORY_PER_SYMBOL = 3
_COGNITION_PREFILTER_MAX_ITEMS = 24
_JOINT_OLD_MEMORY_MAX = 3
_JOINT_NEW_MEMORY_MAX = 4


@dataclass(frozen=True)
class V3AgentConfig:
    enabled: bool = False
    enable_multi_symbol_take_profit_stop_loss: bool = False
    enable_post_holiday_profit_taking_window: bool = False
    holiday_profit_taking_window_days: int = 3
    holiday_gap_min_days: int = 4
    holiday_sentiment_carryover: float = 0.0
    prompt_include_unrealized_pnl: bool = False
    enable_dynamic_trade_throttle: bool = False


class XueqiuMultiSymbolRetailAgent(HeterogeneousAgentBase):
    """关注一篮子 symbol 的散户 agent。"""

    def __init__(
        self,
        *,
        symbols: list[str],
        session_calendar: SessionCalendar,
        start_time: Timestamp,
        end_time: Timestamp,
        profile: XueqiuPersonaProfile,
        reference_prices: dict[str, int],
        seed: int = 0,
        llm: XueqiuLLM | None = None,
        llm_call_probability: float = 0.05,
        v3_agent_config: V3AgentConfig | None = None,
        surge_retail_wake_start: date | None = None,
        surge_retail_wake_end: date | None = None,
        disable_trade_cooldown: bool = False,
    ) -> None:
        if not symbols:
            raise ValueError("XueqiuMultiSymbolRetailAgent 至少需要 1 个 symbol")
        # 父类 sym 字段还是要的：很多兜底逻辑（如 _build_bootstrap_orders）会用到 self.symbol。
        # 我们用 symbols[0] 占位，真实多标的逻辑全部走 self.symbols。
        primary_symbol = symbols[0]
        primary_ref_price = int(reference_prices.get(primary_symbol, 100_000))
        super().__init__(
            symbol=primary_symbol,
            session_calendar=session_calendar,
            start_time=start_time,
            end_time=end_time,
            init_cash=profile.base.initial_cash * 1000,  # yuan → tick (mlib exchange unit)
            initial_position=profile.base.initial_position,
            reference_price=primary_ref_price,
            base_interval_seconds=profile.base.base_interval_seconds,
            seed=seed,
        )
        self.symbols: list[str] = list(symbols)
        self.profile = profile
        self.reference_prices: dict[str, int] = {
            s: int(reference_prices.get(s, primary_ref_price)) for s in self.symbols
        }
        self.states_by_symbol: dict[str, XueqiuRetailState] = {
            s: XueqiuRetailState.from_profile(profile) for s in self.symbols
        }
        self.seen_news_ids_by_symbol: dict[str, set[str]] = {s: set() for s in self.symbols}
        self.seen_post_ids_by_symbol: dict[str, set[str]] = {s: set() for s in self.symbols}
        self.seen_comment_ids_by_symbol: dict[str, set[str]] = {s: set() for s in self.symbols}
        self._latest_news_by_symbol: dict[str, str] = {s: "" for s in self.symbols}
        self._latest_news_url_by_symbol: dict[str, str] = {s: "" for s in self.symbols}
        self._latest_post_by_symbol: dict[str, str] = {s: "" for s in self.symbols}
        self._latest_comment_by_symbol: dict[str, str] = {s: "" for s in self.symbols}
        self._last_tick_time: Timestamp | None = None
        self.llm = llm if llm is not None else get_default_llm()
        self.llm_call_probability = max(0.0, min(1.0, llm_call_probability))
        self.v3_agent_config = v3_agent_config if v3_agent_config is not None else V3AgentConfig()
        self.surge_retail_wake_start = surge_retail_wake_start
        self.surge_retail_wake_end = surge_retail_wake_end
        self.disable_trade_cooldown = bool(disable_trade_cooldown)
        # 供 runner 按 agent 分文件落盘：记录本次唤醒中的 LLM 输入/输出摘要
        self._llm_outputs: list[dict[str, Any]] = []
        self._last_trade_submit_time: Timestamp | None = None
        self._daily_trade_submit_count: dict[Timestamp, int] = {}
        self._daily_post_count: dict[Timestamp, int] = {}
        self._last_post_time: Timestamp | None = None
        # 本次唤醒中 LLM 对每条新讯息的认知分析（item_id -> dict）
        self._info_analysis_by_id: dict[str, dict[str, Any]] = {}
        self._pre_cognition_memory_by_symbol: dict[str, list[MemoryItem]] = {s: [] for s in self.symbols}
        self._new_cognition_memory_by_symbol: dict[str, list[MemoryItem]] = {s: [] for s in self.symbols}
        self._holiday_carryover_applied_days: set[Any] = set()

    # ============== mlib 生命周期 ==============
    def on_market_open(self, time: Timestamp, symbols: list[str]) -> None:
        # BaseAgent._init_holdings 会把所有 symbols 的 holdings/tradable_holdings 置 0
        super().on_market_open(time, symbols)
        # 给 primary symbol 一个 initial position（和单标的版本对齐；也可全部置 0）
        if self.initial_position and self.symbol in self.holdings:
            self.holdings[self.symbol] = self.initial_position
            self.tradable_holdings[self.symbol] = self.initial_position

    # ============== 状态推进 ==============
    def sync_internal_resources(self) -> None:
        self._llm_outputs = []
        for sym, state in self.states_by_symbol.items():
            state.sync_resources(self.cash, self.holdings, sym)
            now = self._last_tick_time or self.start_time
            state.tick(now)

    # ============== 信息：news + posts，按 symbol 拆 ==============
    def collect_information(self, time: Timestamp) -> list[ResolvedInformation]:
        self._last_tick_time = time
        news_state = self.symbol_states[self.symbol][ProfessionalNewsState.__name__]
        social_state = self.symbol_states[self.symbol][SocialNetworkState.__name__]
        assert isinstance(news_state, ProfessionalNewsState)
        assert isinstance(social_state, SocialNetworkState)
        merged: list[ResolvedInformation] = []
        for sym in self.symbols:
            merged.extend(
                resolve_retail_information(
                    symbol=sym,
                    as_of=time,
                    agent_id=self.agent_id,
                    profile=self.profile.base,
                    news_state=news_state,
                    social_state=social_state,
                    seen_news_ids=self.seen_news_ids_by_symbol[sym],
                    seen_post_ids=self.seen_post_ids_by_symbol[sym],
                    seen_comment_ids=self.seen_comment_ids_by_symbol[sym],
                )
            )
        merged.sort(key=lambda x: x.available_time)
        # 去重：帖子去 symbol 后可能在按 symbol 拉取时重复出现。
        deduped: list[ResolvedInformation] = []
        seen_ids: set[str] = set()
        for it in merged:
            if it.item_id in seen_ids:
                continue
            seen_ids.add(it.item_id)
            deduped.append(it)
        return deduped

    @staticmethod
    def _classify_news_source_kind(topic: str | None) -> str:
        t = (topic or "").lower()
        if "regul" in t or "policy" in t or "official" in t:
            return "official_announcement"
        return "pro_news"

    @staticmethod
    def _classify_social_source_kind(author_credibility: float) -> str:
        if author_credibility >= 0.65:
            return "opinion_leader"
        if author_credibility >= 0.40:
            return "peer_post"
        return "retail_crowd"

    def _info_kind(self, item: ResolvedInformation) -> Literal["news", "social", "comment"]:
        if item.source == "news":
            return "news"
        if item.source == "social":
            return "social"
        return "comment"

    def _resolve_source_kind(self, item: ResolvedInformation) -> str:
        if item.source == "news":
            return self._classify_news_source_kind(item.topic)
        return self._classify_social_source_kind(float(item.credibility))

    @staticmethod
    def _source_group(item: ResolvedInformation) -> Literal["news", "social"]:
        return "news" if item.source == "news" else "social"

    def _source_group_trust(self, source_group: Literal["news", "social"]) -> float:
        ref_state = self.states_by_symbol.get(self.symbol)
        if ref_state is None:
            return 0.5
        cred = ref_state.credibility
        if source_group == "news":
            return max(0.05, float(cred.get("pro_news", 0.5) + cred.get("official_announcement", 0.5)) / 2.0)
        return max(
            0.05,
            float(
                cred.get("opinion_leader", 0.5)
                + cred.get("peer_post", 0.5)
                + cred.get("retail_crowd", 0.5)
            )
            / 3.0,
        )

    def _select_cognition_candidates(self, items: list[ResolvedInformation]) -> list[ResolvedInformation]:
        """先按 news/social 信任源配额筛选，再按优先级取样，控制认知 LLM 输入规模。"""
        eligible: list[ResolvedInformation] = []
        for it in items:
            if it.source == "comment" and not bool(getattr(it, "related_to_self", False)):
                continue
            eligible.append(it)
        if not eligible:
            return []

        pool: dict[str, list[ResolvedInformation]] = {"news": [], "social": []}
        for it in eligible:
            pool[self._source_group(it)].append(it)

        total_budget = max(2, min(_COGNITION_PREFILTER_MAX_ITEMS, len(eligible)))
        trust_news = self._source_group_trust("news")
        trust_social = self._source_group_trust("social")
        denom = max(1e-6, trust_news + trust_social)
        news_quota = int(round(total_budget * trust_news / denom))
        social_quota = total_budget - news_quota
        if pool["news"] and news_quota <= 0:
            news_quota = 1
            social_quota = max(0, total_budget - news_quota)
        if pool["social"] and social_quota <= 0:
            social_quota = 1
            news_quota = max(0, total_budget - social_quota)

        def _rank(it: ResolvedInformation) -> tuple[int, float, float]:
            related = 1 if bool(getattr(it, "related_to_self", False)) else 0
            ts = float(getattr(it.available_time, "value", 0.0))
            cred = float(it.credibility)
            return related, ts, cred

        selected: list[ResolvedInformation] = []
        for group, quota in (("news", news_quota), ("social", social_quota)):
            ranked = sorted(pool[group], key=_rank, reverse=True)
            selected.extend(ranked[: max(0, quota)])

        if len(selected) < total_budget:
            selected_ids = {it.item_id for it in selected}
            refill = sorted(
                (it for it in eligible if it.item_id not in selected_ids),
                key=_rank,
                reverse=True,
            )
            selected.extend(refill[: max(0, total_budget - len(selected))])

        selected.sort(key=lambda x: x.available_time)
        return selected

    def _build_self_cognition_snapshot(
        self,
        time: Timestamp,
        *,
        focus_symbols: set[str] | None = None,
    ) -> dict[str, Any]:
        """认知 LLM 用的「自身状态」快照：默认仅保留关注标的，降低上下文体积。"""
        mem_cfg = memory_settings()
        symbols = [
            sym for sym in self.symbols
            if (focus_symbols is None or sym in focus_symbols)
        ]
        if not symbols:
            symbols = list(self.symbols[:1])
        return {
            "wakeup_time": str(time),
            "cash_yuan": float(self.cash) / 1000.0,
            "risk_style": self._risk_style(),
            "memory_config": {
                "capacity_per_symbol": int(mem_cfg["capacity"]),
                "llm_recent_n": int(mem_cfg["llm_recent_n"]),
                "summary_max_tokens": int(mem_cfg["summary_max_tokens"]),
                "half_life_hours": float(mem_cfg["half_life_seconds"]) / 3600.0,
                "llm_score_threshold": float(mem_cfg["llm_score_threshold"]),
                "llm_random_old_count": int(mem_cfg["llm_random_old_count"]),
            },
            "symbols": {
                sym: {
                    "position": int(self.holdings.get(sym, 0)),
                    "tradable_position": int(self.tradable_holdings.get(sym, 0)),
                    "mark_price": float(self._mark_price(sym)) / 1000.0,
                    "state": self._state_snapshot(sym=sym, now=time),
                    "memory_recent": self._memory_snapshot_for_llm(
                        state=self.states_by_symbol[sym], now=time
                    )[:_COGNITION_MEMORY_PER_SYMBOL],
                    "memory_store_count": len(self.states_by_symbol[sym].memory.items),
                }
                for sym in symbols
            },
            "source_credibility": {
                k: round(float(v), 3)
                for k, v in self.states_by_symbol[self.symbol].credibility.items()
            },
            "skepticism": float(self.profile.social_behavior.skepticism),
            "contagion_sensitivity": float(self.profile.social_behavior.contagion_sensitivity),
        }

    def _capture_pre_cognition_memory(self) -> None:
        self._pre_cognition_memory_by_symbol = {
            sym: list(state.memory.items)
            for sym, state in self.states_by_symbol.items()
        }
        self._new_cognition_memory_by_symbol = {sym: [] for sym in self.symbols}

    def _refresh_new_cognition_memory(self) -> None:
        updated: dict[str, list[MemoryItem]] = {}
        for sym, state in self.states_by_symbol.items():
            before = self._pre_cognition_memory_by_symbol.get(sym, [])
            before_ids = {id(it) for it in before}
            updated[sym] = [it for it in state.memory.items if id(it) not in before_ids]
        self._new_cognition_memory_by_symbol = updated

    def _joint_memory_snapshot_for_symbol(
        self,
        *,
        sym: str,
        now: Timestamp,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        old_items = self._pre_cognition_memory_by_symbol.get(sym, [])
        if not old_items:
            old_items = list(self.states_by_symbol[sym].memory.items)
        new_items = self._new_cognition_memory_by_symbol.get(sym, [])

        def _score(it: MemoryItem, half_life_seconds: float) -> float:
            return it.weight(now, half_life_seconds) * (0.3 + abs(it.direction) * max(0.0, it.strength))

        half_life_seconds = float(self.states_by_symbol[sym].memory.half_life_seconds)
        old_ranked = sorted(old_items, key=lambda it: _score(it, half_life_seconds), reverse=True)
        old_pick = old_ranked[:_JOINT_OLD_MEMORY_MAX]
        new_pick = sorted(new_items, key=lambda it: it.time, reverse=True)[:_JOINT_NEW_MEMORY_MAX]

        token_budget = int(memory_settings()["summary_max_tokens"])
        def _pack(items: list[MemoryItem], *, tag: str) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for it in items:
                out.append(
                    {
                        "time": str(it.time),
                        "source": it.source,
                        "direction": float(it.direction),
                        "strength": float(it.strength),
                        "summary": trim_memory_summary(it.summary or "", token_budget=token_budget),
                        "phase": tag,
                    }
                )
            return out

        old_snapshot = _pack(old_pick, tag="old")
        new_snapshot = _pack(new_pick, tag="new")
        merged_snapshot = old_snapshot + new_snapshot
        return old_snapshot, new_snapshot, merged_snapshot

    def _focus_symbols(self, *, now: Timestamp, k: int = 5) -> list[str]:
        """按关注度选 top-k 标的（用于交易/发帖 prompt 缩减）。"""
        scored: list[tuple[str, float]] = []
        for sym in self.symbols:
            st = self.states_by_symbol[sym]
            score = (
                abs(float(st.belief)) * 0.45
                + float(st.conviction) * 0.35
                + float(st.memory.aggregated_strength(now)) * 0.20
            )
            scored.append((sym, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in scored[: max(1, min(k, len(scored)))]]

    def _llm_batch_analyze_information(
        self,
        time: Timestamp,
        new_information: list[ResolvedInformation],
    ) -> dict[str, dict[str, Any]]:
        """对一批新讯息做一次 LLM 认知分析，返回 ``{item_id: analysis_dict}``。

        * LLM 不可用 / 调用失败：返回 ``{}``，下游 ``consume_information`` 走规则兜底；
        * LLM 返回非预期结构：尽力解析，已成功的部分继续生效，缺失的回退；
        * **仅当 ``new_information`` 非空** 才调用 LLM；无新消息则返回 ``{}`` 且下游走规则 ingest。
        """

        if not new_information:
            return {}
        if not self.llm.enabled:
            return {}

        chosen = self._select_cognition_candidates(new_information)

        items_payload: list[dict[str, Any]] = []
        for it in chosen:
            sym = it.symbol if it.symbol in self.states_by_symbol else "SOCIAL"
            source_group = self._source_group(it)
            items_payload.append(
                {
                    "item_id": it.item_id,
                    "symbol": sym,
                    "source": source_group,
                    "source_kind": self._resolve_source_kind(it),
                    "topic": (it.topic or "")[:40],
                    "summary": (it.summary or "")[:240],
                    "content": (it.content or it.summary or it.topic or "")[:1800],
                    "source_credibility": round(float(it.credibility), 3),
                    "related_to_self": bool(getattr(it, "related_to_self", False)),
                    "has_link": bool(getattr(it, "source_url", "") or ""),
                    "available_time": str(it.available_time),
                }
            )
        if not items_payload:
            return {}

        focus_symbols = set(self._focus_symbols(now=time, k=_FOCUS_SYMBOLS_TOP_K))
        persona_state_snapshot = self._build_self_cognition_snapshot(
            time,
            focus_symbols=focus_symbols,
        )

        try:
            decision = self.llm.analyze_information_batch(
                persona_brief=self.profile.brief_personality,
                persona_state_snapshot=persona_state_snapshot,
                items=items_payload,
                extra_context={
                    "batch_new_items_count": len(new_information),
                    "batch_selected_items_count": len(items_payload),
                    "source_trust_news": round(float(self._source_group_trust("news")), 3),
                    "source_trust_social": round(float(self._source_group_trust("social")), 3),
                    "wakeup_time": str(time),
                },
            )
        except Exception as exc:
            self._llm_outputs.append({"kind": "info_analysis_batch", "error": str(exc)[:400]})
            _logger.warning(
                "[多标的散户 agent_id=%s] info_analysis_batch LLM 调用失败，本轮回退规则消费: %s",
                self.agent_id,
                str(exc)[:400],
            )
            return {}

        self._llm_outputs.append(
            {
                "kind": "info_analysis_batch",
                "batch_size": len(items_payload),
                "decision_keys": (
                    list(decision.keys()) if isinstance(decision, dict) else None
                ),
            }
        )

        if not isinstance(decision, dict):
            if decision is not None:
                _logger.warning(
                    "[多标的散户 agent_id=%s] analyze_information_batch 返回非字典 (type=%s)",
                    self.agent_id,
                    type(decision).__name__,
                )
            return {}

        items_out = decision.get("items")
        if not isinstance(items_out, list):
            _logger.warning(
                "[多标的散户 agent_id=%s] analyze_information_batch 返回缺少 items list",
                self.agent_id,
            )
            return {}

        analysis_by_id: dict[str, dict[str, Any]] = {}
        valid_ids = {p["item_id"] for p in items_payload}
        for raw in items_out:
            if not isinstance(raw, dict):
                continue
            iid = str(raw.get("item_id", "")).strip()
            if not iid or iid not in valid_ids:
                continue
            analysis_by_id[iid] = raw
        # 把单条 LLM 决策也落到 _llm_outputs 便于排查
        self._llm_outputs.append(
            {
                "kind": "info_analysis_batch_result",
                "matched": len(analysis_by_id),
                "unmatched_in_response": [
                    str(r.get("item_id", "")) for r in items_out
                    if isinstance(r, dict) and str(r.get("item_id", "")).strip() not in valid_ids
                ][:10],
                "missing_from_response": sorted(valid_ids - set(analysis_by_id.keys()))[:10],
                "cognition_dropped_items": sorted(valid_ids - set(analysis_by_id.keys()))[:10],
            }
        )
        return analysis_by_id

    def consume_information(self, item: ResolvedInformation) -> None:
        sym = item.symbol if item.symbol in self.states_by_symbol else self.symbol
        info_kind = self._info_kind(item)
        source_kind = self._resolve_source_kind(item)
        analysis = self._info_analysis_by_id.get(item.item_id)

        if item.source == "news":
            self.seen_news_ids_by_symbol[sym].add(item.item_id)
            self._latest_news_by_symbol[sym] = item.summary or item.topic or ""
            self._latest_news_url_by_symbol[sym] = item.source_url or ""
        elif item.source == "social":
            self.seen_post_ids_by_symbol[sym].add(item.item_id)
            self._latest_post_by_symbol[sym] = item.summary or item.topic or ""
        else:
            self.seen_comment_ids_by_symbol[sym].add(item.item_id)
            self._latest_comment_by_symbol[sym] = item.summary or item.topic or ""

        if not isinstance(analysis, dict):
            # 新机制：只有认知 LLM 判断为有用的信息才更新状态/写入 memory。
            return
        target_sym = str(analysis.get("target_symbol", sym)).strip()
        if target_sym not in self.states_by_symbol:
            target_sym = sym
        target_state = self.states_by_symbol[target_sym]
        useful = bool(analysis.get("useful", False))
        try:
            target_state.apply_llm_analysis(
                time=item.available_time,
                source=source_kind,  # type: ignore[arg-type]
                info_kind=info_kind,
                belief_shift=_safe_float(analysis.get("belief_shift"), 0.0),
                sentiment_shift=_safe_float(analysis.get("sentiment_shift"), 0.0),
                stress_shift=_safe_float(analysis.get("stress_shift"), 0.0),
                excitement_shift=_safe_float(analysis.get("excitement_shift"), 0.0),
                conviction_shift=_safe_float(analysis.get("conviction_shift"), 0.0),
                memory_direction=_safe_float(analysis.get("memory_direction"), 0.0),
                memory_strength=_safe_float(analysis.get("memory_strength"), 0.0),
                topic=str(analysis.get("topic", item.topic or ""))[:40],
                summary=str(analysis.get("summary", item.summary or ""))[:120],
                credibility_feedback=_safe_float(analysis.get("credibility_feedback"), 0.0),
                store_memory=useful,
            )
            target_state.llm_calls += 1
        except Exception as exc:
            self._llm_outputs.append(
                {"kind": "info_analysis_apply_error", "item_id": item.item_id, "error": str(exc)[:400]}
            )
            _logger.warning(
                "[多标的散户 agent_id=%s] 应用 info_analysis 失败，丢弃本条信息: %s",
                self.agent_id,
                str(exc)[:400],
            )

    def _fade_conviction_without_new_bullish_news(self, new_information: list[ResolvedInformation]) -> None:
        """无新增利好时的温和 conviction 退潮，避免长时间单边亢奋。"""
        bullish_by_symbol: dict[str, float] = {sym: 0.0 for sym in self.symbols}
        bearish_by_symbol: dict[str, float] = {sym: 0.0 for sym in self.symbols}
        for item in new_information:
            if item.source != "news":
                continue
            ana = self._info_analysis_by_id.get(item.item_id)
            if not isinstance(ana, dict):
                continue
            target_sym = str(ana.get("target_symbol", item.symbol)).strip()
            if target_sym not in self.states_by_symbol:
                target_sym = item.symbol if item.symbol in self.states_by_symbol else self.symbol
            if target_sym not in self.states_by_symbol:
                continue
            belief_shift = _safe_float(ana.get("belief_shift"), 0.0)
            mem_direction = _safe_float(ana.get("memory_direction"), 0.0)
            mem_strength = _safe_float(ana.get("memory_strength"), 0.0)
            useful = bool(ana.get("useful", False))
            weight = (0.50 * abs(belief_shift) + 0.50 * abs(mem_direction) * max(0.0, min(1.0, mem_strength)))
            if useful:
                weight *= 1.12
            signal = 0.60 * belief_shift + 0.40 * mem_direction
            if signal > 0.0:
                bullish_by_symbol[target_sym] += max(0.0, signal) * max(0.2, weight)
            elif signal < 0.0:
                bearish_by_symbol[target_sym] += abs(signal) * max(0.2, weight)

        for sym, state in self.states_by_symbol.items():
            state.fade_conviction_without_bullish(
                bullish_news_score=float(bullish_by_symbol.get(sym, 0.0)),
                bearish_news_score=float(bearish_by_symbol.get(sym, 0.0)),
            )

    # ============== 行为：发帖（综合思考，不绑定单标的；symbol 字段仅占位路由） ==============
    def maybe_publish_social_post(
        self,
        time: Timestamp,
        new_information: list[ResolvedInformation],
    ) -> None:
        social_state = self.symbol_states[self.symbol][SocialNetworkState.__name__]
        assert isinstance(social_state, SocialNetworkState)
        # 发帖门槛信号：优先使用「认知 LLM 对社交内容的 sentiment 标签」。
        # 若本轮没有社交标签，则 signal_strength=0（通常会被 gate 拦截）。
        social_sentiment_labels: list[float] = []
        for it in new_information:
            if it.source not in {"social", "comment"}:
                continue
            ana = self._info_analysis_by_id.get(it.item_id)
            if not isinstance(ana, dict):
                continue
            tag = _safe_float(ana.get("sentiment_shift"), 0.0)
            if abs(tag) > 1e-6:
                social_sentiment_labels.append(max(-1.0, min(1.0, tag)))
        sentiment_signal = (
            sum(social_sentiment_labels) / len(social_sentiment_labels)
            if social_sentiment_labels
            else 0.0
        )
        signal_strength = (
            max(abs(v) for v in social_sentiment_labels)
            if social_sentiment_labels
            else 0.0
        )
        gate_prob = self._post_gate_probability(
            time=time,
            has_new_information=bool(new_information),
            signal_strength=float(signal_strength),
        )
        social_gate_reason, state_activation, affect_intensity = self._social_llm_gate_reason(
            has_new_information=bool(new_information),
            signal_strength=float(signal_strength),
        )
        if social_gate_reason is not None:
            self._llm_outputs.append(
                {
                    "kind": "post_state_gate_block_before_llm",
                    "reason": social_gate_reason,
                    "state_activation": round(float(state_activation), 4),
                    "affect_intensity": round(float(affect_intensity), 4),
                    "has_new_information": bool(new_information),
                    "signal_strength": round(float(signal_strength), 4),
                }
            )
            return
        gate_prob = _clamp(
            gate_prob
            * (0.40 + 0.60 * max(float(signal_strength), float(state_activation), float(affect_intensity)))
            * max(0.05, float(self.llm_call_probability)),
            0.01,
            0.90,
        )
        # 发帖门控前置：不满足门控则直接跳过，不触发 LLM，避免无效 token 消耗。
        gate_draw = self.rng.random()
        if gate_draw > gate_prob:
            self._llm_outputs.append(
                {
                    "kind": "post_gate_block_before_llm",
                    "post_gate_probability": round(float(gate_prob), 4),
                    "gate_draw": round(float(gate_draw), 4),
                    "today_posts": self._daily_post_submit_count(time),
                    "has_new_information": bool(new_information),
                }
            )
            return

        focus_symbols = self._focus_symbols(now=time, k=5)
        # 发帖 prompt：社交 memory + 关注度最高 5 个标的的 memory。
        self_context: dict[str, Any] = {
            "wakeup_time": str(time),
            "cash": float(self.cash) / 1000.0,
            "risk_style": self._risk_style(),
            "daily_post_limit": self._daily_post_limit(),
            "today_posts": self._daily_post_submit_count(time),
            "focus_symbols": focus_symbols,
            "positions": {
                sym: {
                    "position": int(self.holdings.get(sym, 0)),
                    "tradable_position": int(self.tradable_holdings.get(sym, 0)),
                    "mark_price": float(self._mark_price(sym)) / 1000.0,
                    "state": self._state_snapshot(sym=sym, now=time),
                    "memory": (
                        self._memory_snapshot_for_llm(state=self.states_by_symbol[sym], now=time)
                        if sym in focus_symbols
                        else []
                    ),
                }
                for sym in self.symbols
            },
        }
        try:
            decision = self.llm.decide_post_free(
                persona_brief=self._social_persona_brief(),
                open_context=self_context,
                post_context={
                    "today_posts": self._daily_post_submit_count(time),
                    "daily_post_limit": self._daily_post_limit(),
                    "has_new_information": bool(new_information),
                    "signal_strength": round(float(signal_strength), 4),
                    "post_gate_probability": round(float(gate_prob), 4),
                    "gate_draw": round(float(gate_draw), 4),
                },
            )
        except Exception as exc:
            self._llm_outputs.append({"kind": "post_free", "error": str(exc)[:400]})
            _logger.warning("[多标的散户 agent_id=%s] decide_post_free 失败，跳过发帖: %s", self.agent_id, str(exc)[:400])
            return
        self._llm_outputs.append({"kind": "post_free", "decision": decision})
        if not isinstance(decision, dict):
            if decision is not None:
                _logger.warning(
                    "[多标的散户 agent_id=%s] 发帖 LLM 返回非字典 (type=%s)，跳过发帖",
                    self.agent_id,
                    type(decision).__name__,
                )
            return
        try:
            action = str(decision.get("action", "skip")).strip().lower()
            if action != "post":
                return
            text = str(decision.get("text", "")).strip()[:120]
            if not text:
                return
            topic = str(decision.get("topic", "")).strip()[:40] or "综合"
            pri_state = self.states_by_symbol[self.symbol]
            pri_state.llm_calls += 1
            self._publish_freeform_post(
                time=time,
                text=text,
                sentiment_signal=float(sentiment_signal),
                topic=topic,
                social_state=social_state,
                state=pri_state,
            )
        except Exception as exc:
            self._llm_outputs.append(
                {"kind": "post_free_consume_error", "error": str(exc)[:400], "decision": decision}
            )
            _logger.warning(
                "[多标的散户 agent_id=%s] 处理 decide_post_free 结果异常，跳过本轮发帖: %s",
                self.agent_id,
                str(exc)[:400],
            )

    def _publish_freeform_post(
        self,
        *,
        time: Timestamp,
        text: str,
        sentiment_signal: float,
        topic: str,
        social_state: SocialNetworkState,
        state: XueqiuRetailState,
    ) -> SocialPost | None:
        """综合发帖：正文与题材由 LLM 决定；帖子不绑定具体标的。"""
        anchor = "SOCIAL"
        author_score = social_state.author_engagement_score(self.agent_id, as_of=time)
        influence = _clamp(0.30 + 0.10 * min(5.0, author_score / 2.0))
        memory_dir = self._portfolio_memory_direction(time=time)
        memory_strength = self._portfolio_memory_strength(time=time)
        rhetoric_style = self._pick_rhetoric_style(
            author_score=author_score,
            signal_strength=max(0.2, memory_strength),
            topic=topic or "综合",
        )
        direction = max(-1.0, min(1.0, memory_dir))
        trigger = ResolvedInformation(
            item_id=f"free-{self.agent_id}-{time.isoformat()}",
            symbol=anchor,
            source="news",
            available_time=time,
            topic=topic or "综合",
            direction=direction,
            strength=max(0.15, min(1.0, memory_strength or 0.35)),
            credibility=0.45,
            sentiment=max(-1.0, min(1.0, float(sentiment_signal))),
            summary=text.strip()[:120],
            content=text.strip()[:2000],
        )
        post = compose_retail_post(
            post_id=f"post-{self.agent_id}-open-{state.authored_posts + 1}",
            symbol=anchor,
            author_agent_id=self.agent_id,
            info=trigger,
            posting_intensity=self.profile.social_behavior.posting_tendency,
            influence=influence,
            rhetoric_style=rhetoric_style,
        )
        social_state.publish_post(post, now=time)
        state.authored_posts += 1
        self._register_post_submit(time=time)
        return post

    def _publish_one_post(
        self,
        *,
        time: Timestamp,
        symbol: str,
        trigger: ResolvedInformation,
        social_state: SocialNetworkState,
        state: XueqiuRetailState,
        text_override: str = "",
        sentiment_override: float | None = None,
    ) -> SocialPost | None:
        author_score = social_state.author_engagement_score(self.agent_id, as_of=time)
        influence = _clamp(0.30 + 0.10 * min(5.0, author_score / 2.0))
        rhetoric_style = self._pick_rhetoric_style(
            author_score=author_score,
            signal_strength=abs(trigger.direction) * max(0.2, trigger.strength),
            topic=trigger.topic,
        )
        _ = sentiment_override
        post_text = (text_override or "").strip()[:120]
        post = compose_retail_post(
            post_id=f"post-{self.agent_id}-{symbol}-{state.authored_posts + 1}",
            symbol="SOCIAL",
            author_agent_id=self.agent_id,
            info=ResolvedInformation(
                item_id=trigger.item_id,
                symbol="SOCIAL",
                source=trigger.source,
                available_time=time,
                topic=trigger.topic,
                direction=trigger.direction,
                strength=trigger.strength,
                credibility=trigger.credibility,
                sentiment=0.0,
                summary=post_text or trigger.summary,
                author_agent_id=trigger.author_agent_id,
                source_news_id=trigger.source_news_id,
            ),
            posting_intensity=self.profile.social_behavior.posting_tendency,
            influence=influence,
            rhetoric_style=rhetoric_style,
        )
        social_state.publish_post(post, now=time)
        state.authored_posts += 1
        self._register_post_submit(time=time)
        return post

    def _pick_rhetoric_style(
        self,
        *,
        author_score: float,
        signal_strength: float,
        topic: str,
    ) -> str:
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

    # ============== 行为：交易决策（多标的，每次唤醒选一只） ==============
    def make_execution_intent(self, time: Timestamp) -> ExecutionIntent:
        # 多标的版本走自定义 get_action 流程，不会被父类调用；保留接口避免 abstract 报错。
        return ExecutionIntent(mode="none", rationale="multi_symbol_uses_custom_get_action")

    def get_action(self, observation: Observation) -> Action:
        assert self.agent_id == observation.agent.agent_id
        time = observation.time
        wakeup_slot = self._retail_wakeup_slot(time)

        if time < self.start_time:
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=self.start_time)
        if time > self.end_time:
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=None)

        self.sync_internal_resources()
        self._apply_holiday_sentiment_carryover(time=time)
        new_information = self.collect_information(time)
        self._last_information_batch = list(new_information)
        self._capture_pre_cognition_memory()
        # 新机制：每次唤醒都先做一次认知分析；是否写入 memory 由 LLM 在 item 级别决定。
        if new_information:
            self._info_analysis_by_id = self._llm_batch_analyze_information(time, new_information)
        else:
            self._info_analysis_by_id = {}
        for item in new_information:
            self.consume_information(item)
        self._fade_conviction_without_new_bullish_news(new_information)
        self._refresh_new_cognition_memory()

        next_wakeup = self.get_next_wakeup_time(time)
        if observation.is_market_open_wakup:
            self._last_intent = ExecutionIntent(mode="none", rationale="market_open_wakeup")
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=next_wakeup)
        if not self.session_calendar.is_open(time):
            self._last_intent = ExecutionIntent(mode="none", rationale="market_closed")
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=next_wakeup)
        if wakeup_slot not in {1, 2, 3, 4}:
            self._last_intent = ExecutionIntent(mode="none", rationale="joint_llm_only_scheduled_wakeups")
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=next_wakeup)
        forced_exit = self._pick_forced_risk_exit_intent(time=time)
        if forced_exit is not None:
            chosen_sym, chosen_intent = forced_exit
            self._last_intent = chosen_intent
            orders = self._build_orders_for_symbol(time=time, symbol=chosen_sym, intent=chosen_intent)
            self.record_order_submission(len(orders))
            if orders:
                self._register_trade_submit(time=time)
            return Action(agent_id=self.agent_id, orders=orders, time=time, next_wakeup_time=next_wakeup)
        if not new_information:
            self._last_intent = ExecutionIntent(mode="none", rationale="no_new_information_skip_joint_llm")
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=next_wakeup)
        sentiment_signal, signal_strength = self._social_signal_from_information(new_information)
        skip_reason = self._joint_llm_gate_reason(
            time=time,
            wakeup_slot=wakeup_slot,
            has_new_information=bool(new_information),
            signal_strength=signal_strength,
        )
        if skip_reason is not None:
            self._last_intent = ExecutionIntent(mode="none", rationale=skip_reason)
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=next_wakeup)

        try:
            chosen_sym, chosen_intent = self._llm_joint_pick_symbol_and_intent_and_maybe_post(
                time=time,
                wakeup_slot=wakeup_slot,
                new_information=new_information,
                sentiment_signal=sentiment_signal,
                signal_strength=signal_strength,
            )
        except Exception as exc:
            self._llm_outputs.append({"kind": "post_trade_joint", "error": str(exc)[:400]})
            _logger.warning("[多标的散户 agent_id=%s] 联合 LLM 调用失败，本唤醒跳过社交与交易: %s", self.agent_id, str(exc)[:400])
            self._last_intent = ExecutionIntent(mode="none", rationale="joint_llm_call_failed")
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=next_wakeup)

        if chosen_intent is None or chosen_sym is None or chosen_intent.mode == "none":
            self._last_intent = chosen_intent or ExecutionIntent(mode="none", rationale="no_intent")
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=next_wakeup)

        self._last_intent = chosen_intent
        orders = self._build_orders_for_symbol(time=time, symbol=chosen_sym, intent=chosen_intent)
        self.record_order_submission(len(orders))
        if orders:
            self._register_trade_submit(time=time)
        return Action(agent_id=self.agent_id, orders=orders, time=time, next_wakeup_time=next_wakeup)

    def get_next_wakeup_time(self, time: Timestamp) -> Timestamp | None:
        """散户唤醒：默认 09:30 / 13:00；疯涨窗口内追加 11:00 / 14:00。"""
        for cal_day in self.session_calendar.trading_dates:
            for anchor in self._retail_wake_anchors_for_day(cal_day):
                if anchor > time:
                    return anchor
        return None

    def _in_surge_retail_wake_window(self, cal_day: date) -> bool:
        if self.surge_retail_wake_start is None or self.surge_retail_wake_end is None:
            return False
        if cal_day not in self.session_calendar.trading_dates:
            return False
        return self.surge_retail_wake_start <= cal_day <= self.surge_retail_wake_end

    def _retail_wake_anchors_for_day(self, cal_day: date) -> list[Timestamp]:
        if cal_day not in self.session_calendar.trading_dates:
            return []
        clock_points = [
            time(9, 30),
            time(13, 0),
        ]
        if self._in_surge_retail_wake_window(cal_day):
            clock_points = [time(9, 30), time(11, 0), time(13, 0), time(14, 0)]
        anchors: list[Timestamp] = []
        for clock in clock_points:
            anchor = Timestamp.combine(cal_day, clock)
            if self.session_calendar.is_open(anchor):
                anchors.append(anchor)
        return sorted(set(anchors))

    def _retail_wakeup_slot(self, time: Timestamp) -> int:
        """返回当日第几次散户唤醒：1=09:30，2=11:00/13:00，3=13:00/14:00，4=14:00（疯涨期）。"""
        cal_day = time.normalize().date()
        anchors = self._retail_wake_anchors_for_day(cal_day)
        for idx, anchor in enumerate(anchors, start=1):
            if int(time.hour) == int(anchor.hour) and int(time.minute) == int(anchor.minute):
                return idx
        return 0

    def _trading_day_index(self, day: Any) -> int | None:
        target = pd.Timestamp(day).normalize().date()
        trading_dates = self.session_calendar.trading_dates
        try:
            return trading_dates.index(target)
        except ValueError:
            return None

    def _holiday_gap_days(self, *, time: Timestamp) -> int:
        idx = self._trading_day_index(time)
        if idx is None or idx <= 0:
            return 0
        trading_dates = self.session_calendar.trading_dates
        prev_day = trading_dates[idx - 1]
        curr_day = trading_dates[idx]
        return max(0, (curr_day - prev_day).days)

    def _is_post_holiday_window(self, *, time: Timestamp) -> bool:
        cfg = self.v3_agent_config
        if not cfg.enable_post_holiday_profit_taking_window:
            return False
        idx = self._trading_day_index(time)
        if idx is None or idx <= 0:
            return False
        trading_dates = self.session_calendar.trading_dates
        max_window = max(1, int(cfg.holiday_profit_taking_window_days))
        max_back = min(max_window, idx)
        for offset in range(1, max_back + 1):
            pivot = idx - offset + 1
            if pivot <= 0:
                continue
            gap_days = (trading_dates[pivot] - trading_dates[pivot - 1]).days
            if gap_days >= max(2, int(cfg.holiday_gap_min_days)):
                return True
        return False

    def _apply_holiday_sentiment_carryover(self, *, time: Timestamp) -> None:
        cfg = self.v3_agent_config
        carry = max(0.0, float(cfg.holiday_sentiment_carryover))
        if carry <= 1e-6:
            return
        day = time.normalize()
        if day in self._holiday_carryover_applied_days:
            return
        if self._holiday_gap_days(time=time) < max(2, int(cfg.holiday_gap_min_days)):
            return
        for state in self.states_by_symbol.values():
            state.conviction = _clamp(float(state.conviction) + carry * float(state.sentiment) * 0.25)
            state.excitement = _clamp(float(state.excitement) + carry * max(0.0, float(state.sentiment)) * 0.15)
        self._holiday_carryover_applied_days.add(day)

    def _unrealized_pnl_ratio(self, *, sym: str) -> float:
        state = self.states_by_symbol[sym]
        if state.position <= 0 or state.avg_cost <= 0:
            return 0.0
        mark = float(max(self._mark_price(sym), 100))
        return (mark - float(state.avg_cost)) / float(state.avg_cost)

    def _pick_forced_risk_exit_intent(self, *, time: Timestamp) -> tuple[str, ExecutionIntent] | None:
        cfg = self.v3_agent_config
        if not cfg.enabled:
            return None
        best: tuple[float, str, ExecutionIntent] | None = None
        in_holiday_window = self._is_post_holiday_window(time=time)
        for sym in self.symbols:
            holdings = int(self.tradable_holdings.get(sym, 0))
            if holdings < 100:
                continue
            state = self.states_by_symbol[sym]
            mark = float(max(self._mark_price(sym), 100))
            pnl_ratio = self._unrealized_pnl_ratio(sym=sym)
            tp_or_sl = False
            rationale = ""
            if cfg.enable_multi_symbol_take_profit_stop_loss:
                if state.take_profit_triggered(mark):
                    tp_or_sl = True
                    rationale = "multi_symbol_take_profit"
                elif state.stop_loss_triggered(mark):
                    tp_or_sl = True
                    rationale = "multi_symbol_stop_loss"
            if (not tp_or_sl) and in_holiday_window and pnl_ratio >= 0.01:
                volume = round_lot(int(holdings * 0.6))
                if volume >= 100:
                    intent = ExecutionIntent(
                        mode="directional",
                        direction="S",
                        target_volume=volume,
                        aggressiveness=0.70,
                        rationale="post_holiday_profit_taking",
                    )
                    score = abs(pnl_ratio) + 0.02
                    if best is None or score > best[0]:
                        best = (score, sym, intent)
                continue
            if not tp_or_sl:
                continue
            volume = round_lot(holdings)
            if volume < 100:
                continue
            intent = ExecutionIntent(
                mode="directional",
                direction="S",
                target_volume=volume,
                aggressiveness=0.85,
                rationale=rationale or "take_profit_or_stop_loss",
            )
            score = abs(pnl_ratio)
            if best is None or score > best[0]:
                best = (score, sym, intent)
        if best is None:
            return None
        return best[1], best[2]

    def _llm_pick_symbol_and_intent(
        self,
        *,
        time: Timestamp,
    ) -> tuple[str | None, ExecutionIntent | None]:
        focus_symbols = set(self._focus_symbols(now=time, k=_FOCUS_SYMBOLS_TOP_K))
        candidates = self._build_trade_candidates(
            time=time,
            focus_symbols=focus_symbols,
            only_focus_symbols=False,
        )
        decision = self.llm.decide_trade_multi(
            persona_brief=self._trade_persona_brief(),
            cash=self.cash,
            candidates=candidates,
            social_memory=[],
        )
        self._llm_outputs.append(
            {
                "kind": "trade_multi",
                "candidate_symbols": [c["symbol"] for c in candidates],
                "decision": decision,
            }
        )
        return self._consume_trade_multi_decision(decision=decision, time=time)

    def _llm_joint_pick_symbol_and_intent_and_maybe_post(
        self,
        *,
        time: Timestamp,
        wakeup_slot: int,
        new_information: list[ResolvedInformation],
        sentiment_signal: float,
        signal_strength: float,
    ) -> tuple[str | None, ExecutionIntent | None]:
        focus_symbols = set(self._focus_symbols(now=time, k=_FOCUS_SYMBOLS_TOP_K))
        candidates = self._build_trade_candidates(
            time=time,
            focus_symbols=focus_symbols,
            only_focus_symbols=True,
        )
        social_items = self._select_joint_social_items(new_information=new_information)
        open_context = self._build_post_open_context(time=time, focus_symbols=focus_symbols)
        post_context = {
            "wakeup_slot": int(wakeup_slot),
            "today_posts": self._daily_post_submit_count(time),
            "daily_post_limit": self._daily_post_limit(),
            "has_new_information": bool(social_items),
            "signal_strength": round(float(signal_strength), 4),
            "state_activation": round(float(self._state_activation_score()), 4),
            "affect_intensity": round(float(self._affect_intensity_score()), 4),
            "new_social_items": social_items,
            "new_news_items": [],
        }
        decision = self.llm.decide_post_trade_joint(
            trade_persona_brief=self._trade_persona_brief(),
            social_persona_brief=self._social_persona_brief(),
            cash=self.cash,
            candidates=candidates,
            social_memory=[],
            open_context=open_context,
            post_context=post_context,
        )
        self._llm_outputs.append(
            {
                "kind": "post_trade_joint",
                "candidate_symbols": [c["symbol"] for c in candidates],
                "decision": decision,
                "wakeup_slot": int(wakeup_slot),
            }
        )
        if not isinstance(decision, dict):
            if decision is not None:
                _logger.warning(
                    "[多标的散户 agent_id=%s] 联合 LLM 返回非字典 (type=%s)，跳过本轮",
                    self.agent_id,
                    type(decision).__name__,
                )
            return None, None
        post_decision = decision.get("post_decision")
        self._consume_post_free_decision(
            time=time,
            decision=post_decision,
            sentiment_signal=sentiment_signal,
        )
        trade_decision = decision.get("trade_decision")
        return self._consume_trade_multi_decision(decision=trade_decision, time=time)

    def _build_trade_candidates(
        self,
        *,
        time: Timestamp,
        focus_symbols: set[str],
        only_focus_symbols: bool = False,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        symbol_list = list(self.symbols)
        if only_focus_symbols:
            symbol_list = [sym for sym in symbol_list if sym in focus_symbols]
        for sym in symbol_list:
            state = self.states_by_symbol[sym]
            exposure = self._symbol_exposure_ratio(symbol=sym)
            _old_mem, _new_mem, merged_mem = self._joint_memory_snapshot_for_symbol(sym=sym, now=time)
            candidates.append(
                {
                    "symbol": sym,
                    "state": self._trade_state_snapshot(sym=sym, now=time),
                    "position": int(self.holdings.get(sym, 0)),
                    "tradable_position": int(self.tradable_holdings.get(sym, 0)),
                    "mark_price": float(self._mark_price(sym)) / 1000.0,
                    "unrealized_pnl_ratio": float(self._unrealized_pnl_ratio(sym=sym)),
                    "current_exposure_ratio": float(exposure),
                    "suggested_max_exposure_ratio": float(self._max_symbol_exposure_ratio()),
                    "memory": (merged_mem if sym in focus_symbols else []),
                }
            )
        return candidates

    def _consume_trade_multi_decision(
        self,
        *,
        decision: Any,
        time: Timestamp,
    ) -> tuple[str | None, ExecutionIntent | None]:
        if not isinstance(decision, dict):
            if decision is not None:
                _logger.warning(
                    "[多标的散户 agent_id=%s] 交易 LLM 返回非字典 (type=%s)，跳过本轮",
                    self.agent_id,
                    type(decision).__name__,
                )
            return None, None
        base_reason = self._trade_gate_reason(time=time)
        if base_reason is not None:
            return None, ExecutionIntent(mode="none", rationale=base_reason)
        try:
            sym = str(decision.get("symbol", "")).strip()
            if sym not in self.states_by_symbol:
                return None, None
            action = str(decision.get("action", "hold")).lower()
            reason = str(decision.get("reason", "")).strip().replace("\n", " ")
            if len(reason) > _DECISION_REASON_MAX_LEN:
                reason = reason[:_DECISION_REASON_MAX_LEN]
            if action == "hold":
                return sym, ExecutionIntent(mode="none", rationale=(reason or "llm_hold"))
            direction: Literal["B", "S"] = "B" if action == "buy" else "S"
            if direction == "S" and self.tradable_holdings.get(sym, 0) < 100:
                return sym, ExecutionIntent(mode="none", rationale="cannot_short_no_inventory")
            size_ratio = _clamp(_safe_float(decision.get("size_ratio"), 0.20))
            aggressiveness = _clamp(_safe_float(decision.get("aggressiveness"), 0.40))
            state = self.states_by_symbol[sym]
            state.llm_calls += 1
            size_ratio = self._style_constrained_size_ratio(symbol=sym, direction=direction, size_ratio=size_ratio)
            if size_ratio <= 0.0:
                return sym, ExecutionIntent(mode="none", rationale="style_risk_gate")
            aggressiveness = min(aggressiveness, self._max_aggressiveness())
            volume = self._size_to_volume(symbol=sym, direction=direction, size_ratio=size_ratio)
            if volume < 100:
                return sym, ExecutionIntent(mode="none", rationale="size_below_lot")
            return sym, ExecutionIntent(
                mode="directional",
                direction=direction,
                target_volume=volume,
                aggressiveness=aggressiveness,
                rationale=(reason or "llm_multi_pick"),
            )
        except Exception as exc:
            self._llm_outputs.append(
                {"kind": "trade_multi_consume_error", "error": str(exc)[:400], "decision": decision}
            )
            _logger.warning(
                "[多标的散户 agent_id=%s] 处理交易 LLM 决策时异常，跳过本轮: %s",
                self.agent_id,
                str(exc)[:400],
            )
            return None, None

    def _build_post_open_context(
        self,
        *,
        time: Timestamp,
        focus_symbols: set[str],
    ) -> dict[str, Any]:
        ordered_focus_symbols = [sym for sym in self.symbols if sym in focus_symbols]
        positions_payload: dict[str, dict[str, Any]] = {}
        for sym in ordered_focus_symbols:
            old_mem, new_mem, merged_mem = self._joint_memory_snapshot_for_symbol(sym=sym, now=time)
            positions_payload[sym] = {
                **self._compact_position_snapshot(sym=sym, now=time),
                "state": self._trade_state_snapshot(sym=sym, now=time),
                "memory_old": old_mem,
                "memory_new": new_mem,
                "memory": merged_mem,
            }
        return {
            "wakeup_time": str(time),
            "cash": float(self.cash) / 1000.0,
            "risk_style": self._risk_style(),
            "daily_post_limit": self._daily_post_limit(),
            "today_posts": self._daily_post_submit_count(time),
            "focus_symbols": ordered_focus_symbols,
            "positions": positions_payload,
        }

    def _select_joint_social_items(
        self,
        *,
        new_information: list[ResolvedInformation],
    ) -> list[dict[str, Any]]:
        """联合 LLM 仅透传少量社交新消息，不透传新闻。"""
        social_state = self.symbol_states[self.symbol][SocialNetworkState.__name__]
        relation_state = social_state if isinstance(social_state, SocialNetworkState) else None
        scored: list[tuple[float, ResolvedInformation]] = []
        for item in new_information:
            if item.source not in {"social", "comment"}:
                continue
            src_author = int(item.author_agent_id) if item.author_agent_id is not None else None
            relation_score = (
                float(relation_state.relation_proximity(self.agent_id, src_author))
                if relation_state is not None and src_author is not None
                else 0.0
            )
            score = (
                0.75 * float(item.credibility)
                + 0.25 * relation_score
            )
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        out: list[dict[str, Any]] = []
        for _, item in scored[:_JOINT_SOCIAL_ITEMS_TOP_K]:
            target_post_id = item.item_id if item.source == "social" else (item.parent_post_id or "")
            src_author = int(item.author_agent_id) if item.author_agent_id is not None else None
            relation_score = (
                float(relation_state.relation_proximity(self.agent_id, src_author))
                if relation_state is not None and src_author is not None
                else 0.0
            )
            out.append(
                {
                    "item_id": item.item_id,
                    "target_post_id": target_post_id,
                    "symbol": item.symbol,
                    "source": item.source,
                    "topic": (item.topic or "")[:40],
                    "summary": (item.summary or "")[:120],
                    "content": (item.content or item.summary or "")[:400],
                    "credibility": round(float(item.credibility), 3),
                    "related_to_self": round(float(relation_score), 4),
                }
            )
        return out

    def _consume_post_free_decision(
        self,
        *,
        time: Timestamp,
        decision: Any,
        sentiment_signal: float,
    ) -> None:
        if not isinstance(decision, dict):
            if decision is not None:
                _logger.warning(
                    "[多标的散户 agent_id=%s] 发帖 LLM 返回非字典 (type=%s)，跳过发帖",
                    self.agent_id,
                    type(decision).__name__,
                )
            return
        social_state = self.symbol_states[self.symbol][SocialNetworkState.__name__]
        if not isinstance(social_state, SocialNetworkState):
            return
        try:
            pri_state = self.states_by_symbol[self.symbol]
            pri_state.llm_calls += 1
            self._apply_joint_social_interactions(
                time=time,
                decision=decision,
                social_state=social_state,
            )
            action = str(decision.get("action", "skip")).strip().lower()
            if action == "post":
                if self._daily_post_submit_count(time) >= self._daily_post_limit():
                    return
                if self._last_post_time is not None and (time - self._last_post_time) < Timedelta(hours=2):
                    return
                text = str(decision.get("text", "")).strip()[:_DECISION_TEXT_MAX_LEN]
                if not text:
                    return
                topic = str(decision.get("topic", "")).strip()[:_DECISION_TOPIC_MAX_LEN] or "综合"
                reason = str(decision.get("reason", "")).strip().replace("\n", " ")
                if len(reason) > _DECISION_REASON_MAX_LEN:
                    reason = reason[:_DECISION_REASON_MAX_LEN]
                if reason:
                    self._llm_outputs.append(
                        {
                            "kind": "post_joint_reason",
                            "reason": reason,
                        }
                    )
                self._publish_freeform_post(
                    time=time,
                    text=text,
                    sentiment_signal=float(sentiment_signal),
                    topic=topic,
                    social_state=social_state,
                    state=pri_state,
                )
        except Exception as exc:
            self._llm_outputs.append(
                {"kind": "post_free_consume_error", "error": str(exc)[:400], "decision": decision}
            )
            _logger.warning(
                "[多标的散户 agent_id=%s] 处理发帖联合决策异常，跳过本轮发帖: %s",
                self.agent_id,
                str(exc)[:400],
            )

    def _apply_joint_social_interactions(
        self,
        *,
        time: Timestamp,
        decision: dict[str, Any],
        social_state: SocialNetworkState,
    ) -> None:
        raw_items = decision.get("interaction_actions")
        if not isinstance(raw_items, list):
            return
        interactions = [it for it in raw_items if isinstance(it, dict)][:3]
        for it in interactions:
            post_id = str(it.get("target_post_id", "") or it.get("item_id", "")).strip()
            if not post_id:
                continue
            post = social_state.get_post(post_id)
            if post is None or int(post.author_agent_id) == int(self.agent_id):
                continue
            action = str(it.get("action", "ignore")).strip().lower()
            sym = post.symbol if post.symbol in self.states_by_symbol else self.symbol
            state = self.states_by_symbol[sym]
            if action == "like":
                social_state.record_like(viewer_id=int(self.agent_id), post_id=post_id, now=time)
                state.liked_posts += 1
                continue
            if action == "repost":
                social_state.record_repost(viewer_id=int(self.agent_id), post_id=post_id, now=time)
                state.reposted_posts += 1
                continue
            if action == "comment":
                text = str(it.get("text", "")).strip()[:_DECISION_COMMENT_MAX_LEN]
                if not text:
                    continue
                comment_id = f"cmt_{int(self.agent_id)}_{int(time.value)}_{self.rng.randint(1000, 9999)}"
                comment = compose_retail_comment(
                    comment_id=comment_id,
                    post=post,
                    author_agent_id=int(self.agent_id),
                    created_time=time,
                    own_belief=float(state.belief),
                    own_credibility=float(post.credibility),
                    text_label=text,
                )
                social_state.record_comment(
                    viewer_id=int(self.agent_id),
                    post_id=post_id,
                    comment=comment,
                    now=time,
                )
                state.commented_posts += 1

    def _state_snapshot(self, *, sym: str, now: Timestamp) -> dict[str, Any]:
        state = self.states_by_symbol[sym]
        return {
            "belief": float(state.belief),
            "sentiment": float(state.sentiment),
            "stress": float(state.stress),
            "excitement": float(state.excitement),
            "conviction": float(state.conviction),
            "hotspot_belief": float(state.hotspot_belief),
            "belief_topics": dict(sorted(state.belief_topics.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]),
            "memory_direction": float(state.memory.aggregated_direction(now)),
            "memory_strength": float(state.memory.aggregated_strength(now)),
        }

    def _trade_state_snapshot(self, *, sym: str, now: Timestamp) -> dict[str, Any]:
        state = self.states_by_symbol[sym]
        return {
            "belief": float(state.belief),
            "sentiment": float(state.sentiment),
            "conviction": float(state.conviction),
            "stress": float(state.stress),
            "memory_direction": float(state.memory.aggregated_direction(now)),
            "memory_strength": float(state.memory.aggregated_strength(now)),
        }

    def _compact_position_snapshot(self, *, sym: str, now: Timestamp) -> dict[str, Any]:
        _ = now
        return {
            "position": int(self.holdings.get(sym, 0)),
            "mark_price": float(self._mark_price(sym)) / 1000.0,
            "unrealized_pnl_ratio": float(self._unrealized_pnl_ratio(sym=sym)),
            "belief": float(self.states_by_symbol[sym].belief),
        }

    def _memory_snapshot(self, *, state: XueqiuRetailState, now: Timestamp) -> list[dict[str, Any]]:
        return self._memory_snapshot_from_items(
            items=state.memory.items,
            half_life_seconds=float(state.memory.half_life_seconds),
            now=now,
        )

    def _social_memory_snapshot(self, *, now: Timestamp) -> list[dict[str, Any]]:
        _ = now
        return []

    def _memory_snapshot_for_llm(
        self,
        *,
        state: XueqiuRetailState,
        now: Timestamp,
    ) -> list[dict[str, Any]]:
        """交易/社交 LLM 专用：最重要4条 + 随机1条（最多5条）。"""
        return self._memory_snapshot_top4_plus_random1(
            items=state.memory.items,
            half_life_seconds=float(state.memory.half_life_seconds),
            now=now,
        )

    def _social_memory_snapshot_for_llm(self, *, now: Timestamp) -> list[dict[str, Any]]:
        """新机制下不再维护独立 social_memory。"""
        _ = now
        return []

    def _memory_snapshot_top4_plus_random1(
        self,
        *,
        items: list[MemoryItem],
        half_life_seconds: float,
        now: Timestamp,
    ) -> list[dict[str, Any]]:
        token_budget = int(memory_settings()["summary_max_tokens"])
        scored: list[tuple[MemoryItem, float]] = []
        for it in items:
            score = it.weight(now, half_life_seconds) * (0.3 + abs(it.direction) * max(0.0, it.strength))
            scored.append((it, float(score)))
        if not scored:
            return []
        scored.sort(key=lambda pair: pair[1], reverse=True)

        selected: list[MemoryItem] = [it for it, _s in scored[:4]]
        pool = [it for it, _s in scored[4:]]
        if pool:
            selected.extend(self.rng.sample(pool, k=1))

        out: list[dict[str, Any]] = []
        for it in selected[:5]:
            summary = trim_memory_summary(it.summary or "", token_budget=token_budget)
            out.append(
                {
                    "time": str(it.time),
                    "source": it.source,
                    "direction": float(it.direction),
                    "strength": float(it.strength),
                    "summary": summary,
                }
            )
        return out

    def _memory_snapshot_from_items(
        self,
        *,
        items: list[MemoryItem],
        half_life_seconds: float,
        now: Timestamp,
    ) -> list[dict[str, Any]]:
        mem_cfg = memory_settings()
        recent_n = int(mem_cfg["llm_recent_n"])
        threshold = float(mem_cfg["llm_score_threshold"])
        random_old_n = int(mem_cfg["llm_random_old_count"])
        token_budget = int(mem_cfg["summary_max_tokens"])

        # 近期显著记忆优先（按衰减后得分）；再随机补一条旧记忆，保留少量长期背景。
        scored: list[tuple[Any, float]] = []
        for it in items:
            score = it.weight(now, half_life_seconds) * (0.3 + abs(it.direction) * max(0.0, it.strength))
            scored.append((it, float(score)))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        hot_limit = max(0, recent_n - max(0, random_old_n))
        selected: list[Any] = [it for it, s in scored if s >= threshold][:hot_limit]
        pool = [it for it, _s in scored if it not in selected]
        if pool and random_old_n > 0:
            pick_n = min(random_old_n, max(0, recent_n - len(selected)), len(pool))
            selected.extend(self.rng.sample(pool, k=pick_n))
        elif len(selected) < max(0, recent_n):
            # 若没有可随机补充的 pool，再用高分项补齐预算。
            for it, _s in scored:
                if it in selected:
                    continue
                selected.append(it)
                if len(selected) >= max(0, recent_n):
                    break
        if not selected and scored:
            selected = [scored[0][0]]

        out: list[dict[str, Any]] = []
        for it in selected:
            summary = trim_memory_summary(it.summary or "", token_budget=token_budget)
            out.append(
                {
                    "time": str(it.time),
                    "source": it.source,
                    "direction": float(it.direction),
                    "strength": float(it.strength),
                    "summary": summary,
                }
            )
        return out

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    # ============== 订单构造 ==============
    def _size_to_volume(
        self,
        *,
        symbol: str,
        direction: Literal["B", "S"],
        size_ratio: float,
    ) -> int:
        mark = max(self._mark_price(symbol), 1)
        if direction == "B":
            target_cash = self.tradable_cash * size_ratio
            volume = round_lot(int(target_cash / max(mark, 100)))
        else:
            holdings = self.tradable_holdings.get(symbol, 0)
            volume = round_lot(int(holdings * size_ratio))
        cap = max(self.profile.base.base_order_size * 5, 100)
        return max(0, min(volume, cap))

    def _build_orders_for_symbol(
        self,
        *,
        time: Timestamp,
        symbol: str,
        intent: ExecutionIntent,
    ) -> list[BaseOrder]:
        if intent.mode == "none" or intent.target_volume <= 0:
            return []
        lob = self._lob_for_symbol(symbol)
        if lob is None:
            return self._build_bootstrap_orders_for_symbol(time=time, symbol=symbol, intent=intent)
        return build_orders_from_intent(agent=self, time=time, symbol=symbol, lob=lob, intent=intent)

    def _build_bootstrap_orders_for_symbol(
        self,
        *,
        time: Timestamp,
        symbol: str,
        intent: ExecutionIntent,
    ) -> list[BaseOrder]:
        ref_price = self.reference_prices.get(symbol, 100_000)
        volume = max(0, intent.target_volume // 100 * 100)
        if volume < 100:
            return []
        if intent.direction == "B":
            buy_volume = min(
                volume,
                max(0, int(self.tradable_cash / max(ref_price, 100)) // 100 * 100),
            )
            if buy_volume < 100:
                return []
            return self.construct_valid_orders(
                time=time, symbol=symbol, type="B",
                price=max(100, ref_price + intent.price_offset_ticks),
                volume=buy_volume,
            )
        sell_volume = min(volume, self.tradable_holdings.get(symbol, 0) // 100 * 100)
        if sell_volume < 100:
            return []
        return self.construct_valid_orders(
            time=time, symbol=symbol, type="S",
            price=max(100, ref_price - intent.price_offset_ticks),
            volume=sell_volume,
        )

    # ============== 工具 ==============
    def _lob_for_symbol(self, symbol: str) -> LobSnapshot | None:
        state_dict = self.symbol_states.get(symbol)
        if not state_dict:
            return None
        st = state_dict.get(State.__name__)
        if not isinstance(st, State):
            return None
        return st.lob_snapshot

    def _mark_price(self, symbol: str) -> int:
        lob = self._lob_for_symbol(symbol)
        if lob is not None:
            return int(lob.mid_price)
        return self.reference_prices.get(symbol, 100_000)

    def interval_multiplier(self) -> float:
        ts = self.profile.trading_style
        avg_conv = sum(s.conviction for s in self.states_by_symbol.values()) / max(1, len(self.states_by_symbol))
        return max(0.40, 1.40 - 0.70 * ts.frequency - 0.40 * avg_conv)

    def _risk_score(self) -> float:
        ts = self.profile.trading_style
        return _clamp(0.55 * ts.frequency + 0.45 * ts.impulsiveness)

    def _state_activation_score(self) -> float:
        """聚合 belief/sentiment/conviction，衡量是否值得触发高成本 LLM 行为。"""
        if not self.states_by_symbol:
            return 0.0
        max_belief = max(abs(float(s.belief)) for s in self.states_by_symbol.values())
        max_sentiment = max(abs(float(s.sentiment)) for s in self.states_by_symbol.values())
        max_conviction = max(float(s.conviction) for s in self.states_by_symbol.values())
        return _clamp(0.45 * max_belief + 0.35 * max_sentiment + 0.20 * max_conviction)

    def _portfolio_memory_direction(self, *, time: Timestamp) -> float:
        if not self.states_by_symbol:
            return 0.0
        vals = [float(s.memory.aggregated_direction(time)) for s in self.states_by_symbol.values()]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    def _portfolio_memory_strength(self, *, time: Timestamp) -> float:
        if not self.states_by_symbol:
            return 0.0
        return max((float(s.memory.aggregated_strength(time)) for s in self.states_by_symbol.values()), default=0.0)

    def _affect_intensity_score(self) -> float:
        """情绪强度：高兴奋/高压力/高情绪偏移时更可能触发社交与交易。"""
        if not self.states_by_symbol:
            return 0.0
        vals: list[float] = []
        for s in self.states_by_symbol.values():
            vals.append(
                _clamp(
                    0.45 * abs(float(s.sentiment))
                    + 0.35 * abs(float(s.excitement))
                    + 0.20 * abs(float(s.stress))
                )
            )
        return max(vals) if vals else 0.0

    def _risk_style(self) -> Literal["conservative", "balanced", "aggressive"]:
        score = self._risk_score()
        if score < 0.34:
            return "conservative"
        if score < 0.66:
            return "balanced"
        return "aggressive"

    def _daily_trade_limit(self) -> int:
        style = self._risk_style()
        if style == "conservative":
            return 1
        if style == "balanced":
            return 2
        return 3

    def _trade_cooldown_hours(self) -> float:
        style = self._risk_style()
        if style == "conservative":
            return 6.0
        if style == "balanced":
            return 3.0
        return 1.0

    def _max_symbol_exposure_ratio(self) -> float:
        style = self._risk_style()
        if style == "conservative":
            return 0.18
        if style == "balanced":
            return 0.30
        return 0.55

    def _max_buy_size_ratio(self) -> float:
        style = self._risk_style()
        if style == "conservative":
            return 0.08
        if style == "balanced":
            return 0.18
        return 0.35

    def _max_aggressiveness(self) -> float:
        style = self._risk_style()
        if style == "conservative":
            return 0.40
        if style == "balanced":
            return 0.60
        return 0.85

    def _daily_submit_count(self, time: Timestamp) -> int:
        day = time.normalize()
        return int(self._daily_trade_submit_count.get(day, 0))

    def _daily_post_submit_count(self, time: Timestamp) -> int:
        day = time.normalize()
        return int(self._daily_post_count.get(day, 0))

    def _register_trade_submit(self, *, time: Timestamp) -> None:
        day = time.normalize()
        self._daily_trade_submit_count[day] = self._daily_trade_submit_count.get(day, 0) + 1
        self._last_trade_submit_time = time

    def _register_post_submit(self, *, time: Timestamp) -> None:
        day = time.normalize()
        self._daily_post_count[day] = self._daily_post_count.get(day, 0) + 1
        self._last_post_time = time

    def _daily_post_limit(self) -> int:
        openness = float(self.profile.base.personality.openness)
        if openness >= 0.75:
            return 3
        if openness >= 0.45:
            return 2
        return 1

    def _post_gate_probability(self, *, time: Timestamp, has_new_information: bool, signal_strength: float) -> float:
        p = self.profile.base.personality
        sb = self.profile.social_behavior
        prob = (
            0.04
            + 0.26 * float(sb.posting_tendency)
            + 0.10 * float(p.openness)     # 开放性越高，允许发帖概率略高
            + 0.06 * float(p.extraversion)
            - 0.08 * float(p.conscientiousness)
        )
        if not has_new_information:
            prob *= 0.35
        elif signal_strength < 0.10:
            prob *= 0.70
        if self._daily_post_submit_count(time) >= self._daily_post_limit():
            prob *= 0.25
        if self._last_post_time is not None and (time - self._last_post_time) < Timedelta(hours=2):
            prob *= 0.60
        return _clamp(prob, 0.01, 0.85)

    def _trade_gate_reason(self, *, time: Timestamp) -> str | None:
        # 先由人格状态门控交易意愿：无明显 conviction 时不调用 LLM。
        if not any(state.should_trade(now=time) for state in self.states_by_symbol.values()):
            return "state_gate_low_conviction"
        if self.v3_agent_config.enable_dynamic_trade_throttle:
            max_intraday_ret = 0.0
            for sym in self.symbols:
                ref = max(100, int(self.reference_prices.get(sym, 100_000)))
                mark = max(100, int(self._mark_price(sym)))
                max_intraday_ret = max(max_intraday_ret, abs((float(mark) - float(ref)) / float(ref)))
            if max_intraday_ret >= 0.095:
                if self._daily_submit_count(time) >= max(1, self._daily_trade_limit() - 1):
                    return "dynamic_throttle_daily_limit"
                if self._last_trade_submit_time is not None:
                    cooldown = Timedelta(hours=self._trade_cooldown_hours() * 1.5)
                    if time - self._last_trade_submit_time < cooldown:
                        return "dynamic_throttle_cooldown"
        if self._daily_submit_count(time) >= self._daily_trade_limit():
            return "daily_trade_limit_reached"
        if not self.disable_trade_cooldown and self._last_trade_submit_time is not None:
            cooldown = Timedelta(hours=self._trade_cooldown_hours())
            if time - self._last_trade_submit_time < cooldown:
                return "trade_cooldown_active"
        return None

    def _social_signal_from_information(
        self,
        new_information: list[ResolvedInformation],
    ) -> tuple[float, float]:
        social_sentiment_labels: list[float] = []
        for it in new_information:
            if it.source not in {"social", "comment"}:
                continue
            if not bool(getattr(it, "related_to_self", False)):
                continue
            ana = self._info_analysis_by_id.get(it.item_id)
            if not isinstance(ana, dict):
                continue
            tag = _safe_float(ana.get("sentiment_shift"), 0.0)
            if abs(tag) > 1e-6:
                social_sentiment_labels.append(max(-1.0, min(1.0, tag)))
        sentiment_signal = (
            sum(social_sentiment_labels) / len(social_sentiment_labels)
            if social_sentiment_labels
            else 0.0
        )
        signal_strength = (
            max(abs(v) for v in social_sentiment_labels)
            if social_sentiment_labels
            else 0.0
        )
        return float(sentiment_signal), float(signal_strength)

    def _joint_llm_gate_reason(
        self,
        *,
        time: Timestamp,
        wakeup_slot: int,
        has_new_information: bool,
        signal_strength: float,
    ) -> str | None:
        if not has_new_information:
            return "joint_gate_requires_new_information"
        state_activation = float(self._state_activation_score())
        affect_intensity = float(self._affect_intensity_score())
        symbol_memory_strength = max(
            (float(s.memory.aggregated_strength(time)) for s in self.states_by_symbol.values()),
            default=0.0,
        )
        memory_activation = _clamp(symbol_memory_strength)
        base_activation = max(float(signal_strength), state_activation, affect_intensity, memory_activation)
        self._llm_outputs.append(
            {
                "kind": "joint_llm_rule_gate",
                "wakeup_slot": int(wakeup_slot),
                "has_new_information": True,
                "signal_strength": round(float(signal_strength), 4),
                "state_activation": round(float(state_activation), 4),
                "affect_intensity": round(float(affect_intensity), 4),
                "memory_activation": round(float(memory_activation), 4),
                "base_activation": round(float(base_activation), 4),
            }
        )
        if base_activation < 0.10:
            return "joint_llm_rule_gate_low_activation"
        return None

    def _social_llm_gate_reason(
        self,
        *,
        has_new_information: bool,
        signal_strength: float,
    ) -> tuple[str | None, float, float]:
        state_activation = float(self._state_activation_score())
        affect_intensity = float(self._affect_intensity_score())
        if (not has_new_information) and signal_strength < 0.08 and state_activation < 0.16:
            return "social_gate_low_signal_no_new_information", state_activation, affect_intensity
        if signal_strength < 0.06 and state_activation < 0.12 and affect_intensity < 0.12:
            return "social_gate_low_emotion_activation", state_activation, affect_intensity
        return None, state_activation, affect_intensity

    def _trade_llm_gate_reason(self, *, time: Timestamp) -> str | None:
        base_reason = self._trade_gate_reason(time=time)
        if base_reason is not None:
            return base_reason
        state_activation = float(self._state_activation_score())
        affect_intensity = float(self._affect_intensity_score())
        if state_activation < 0.16 and affect_intensity < 0.14:
            return "trade_gate_low_emotion_activation"
        prob = _clamp(
            max(0.05, float(self.llm_call_probability))
            * (0.45 + 0.55 * max(state_activation, affect_intensity)),
            0.05,
            1.0,
        )
        draw = self.rng.random()
        self._llm_outputs.append(
            {
                "kind": "trade_llm_probability_gate",
                "probability": round(float(prob), 4),
                "draw": round(float(draw), 4),
                "state_activation": round(float(state_activation), 4),
                "affect_intensity": round(float(affect_intensity), 4),
            }
        )
        if draw > prob:
            return "trade_llm_probability_gate_blocked"
        return None

    def _symbol_exposure_ratio(self, *, symbol: str) -> float:
        mark = float(self._mark_price(symbol))
        pos = float(self.holdings.get(symbol, 0))
        value = max(0.0, mark * pos)
        wealth = max(1.0, float(self.cash) + sum(float(self._mark_price(s)) * float(self.holdings.get(s, 0)) for s in self.symbols))
        return value / wealth

    def _style_constrained_size_ratio(
        self,
        *,
        symbol: str,
        direction: Literal["B", "S"],
        size_ratio: float,
    ) -> float:
        ratio = _clamp(size_ratio)
        if direction == "S":
            return ratio
        exposure = self._symbol_exposure_ratio(symbol=symbol)
        if exposure >= self._max_symbol_exposure_ratio():
            return 0.0
        return min(ratio, self._max_buy_size_ratio())

    def _trade_persona_brief(self) -> str:
        style = self._risk_style()
        extra_prompt = ""
        if self.v3_agent_config.prompt_include_unrealized_pnl:
            extra_prompt = " 候选中的 unrealized_pnl_ratio 表示浮盈/浮亏，请纳入方向与仓位决策。"
        return (
            f"{self.profile.brief_personality}\n"
            f"[交易风格约束] style={style}; "
            f"daily_trade_limit={self._daily_trade_limit()}; "
            f"cooldown_hours={self._trade_cooldown_hours():.1f}; "
            f"max_symbol_exposure={self._max_symbol_exposure_ratio():.2f}; "
            f"max_buy_size_ratio={self._max_buy_size_ratio():.2f}. "
            "保守型优先小额分散持仓，激进型才允许更高集中度。"
            f"{extra_prompt}"
        )

    def _social_persona_brief(self) -> str:
        p = self.profile.base.personality
        return (
            f"{self.profile.brief_personality}\n"
            "[社交发帖约束] "
            f"daily_post_limit={self._daily_post_limit()}; "
            f"openness={float(p.openness):.2f}; "
            f"posting_tendency={float(self.profile.social_behavior.posting_tendency):.2f}. "
            "若今日已发帖或缺乏新讯息，优先 skip。"
        )

    # ============== 反馈 ==============
    def on_order_executed(self, time: Timestamp, transaction: Transaction, trans_order_id_to_notify: int):
        del_ids = super().on_order_executed(time, transaction, trans_order_id_to_notify)
        if transaction.type in ("B", "S"):
            sym = transaction.symbol
            volume = int(transaction.volume)
            if transaction.order_matched_volume is not None:
                volume = int(transaction.order_matched_volume.get(trans_order_id_to_notify, volume))
            if volume > 0 and sym in self.states_by_symbol:
                if trans_order_id_to_notify in transaction.buy_id:
                    side: Literal["B", "S"] = "B"
                elif trans_order_id_to_notify in transaction.sell_id:
                    side = "S"
                else:
                    return del_ids
                self.states_by_symbol[sym].on_fill(
                    side=side, price=float(transaction.price), volume=volume, time=time
                )
        return del_ids

    def update_after_trade(self, pnl_delta: float) -> None:
        time = self._last_tick_time or self.start_time
        # 把 pnl 平均归到所有 symbol（简化），主要为了让 conviction 更新
        for state in self.states_by_symbol.values():
            state.update_after_trade(pnl_delta / max(1, len(self.states_by_symbol)), time=time)

    def record_order_submission(self, count: int) -> None:
        # 计入 primary symbol，方便外部聚合
        self.states_by_symbol[self.symbol].submitted_orders += int(count)

    def mark_to_market_wealth(self) -> float | None:
        """Returns total wealth in yuan (tick units divided by 1000)."""
        wealth_tick = float(self.cash)
        any_lob = False
        for sym in self.symbols:
            lob = self._lob_for_symbol(sym)
            if lob is None:
                continue
            any_lob = True
            wealth_tick += self.holdings.get(sym, 0) * float(lob.mid_price)
        if not any_lob:
            return None
        return wealth_tick / 1000.0

    def snapshot_metrics(self) -> dict[str, float]:
        wealth = self.mark_to_market_wealth()
        agg_belief = sum(s.belief for s in self.states_by_symbol.values()) / max(1, len(self.symbols))
        agg_sent = sum(s.sentiment for s in self.states_by_symbol.values()) / max(1, len(self.symbols))
        agg_conv = sum(s.conviction for s in self.states_by_symbol.values()) / max(1, len(self.symbols))
        consumed_news = sum(s.consumed_news for s in self.states_by_symbol.values())
        consumed_posts = sum(s.consumed_posts for s in self.states_by_symbol.values())
        consumed_comments = sum(s.consumed_comments for s in self.states_by_symbol.values())
        liked = sum(s.liked_posts for s in self.states_by_symbol.values())
        reposted = sum(s.reposted_posts for s in self.states_by_symbol.values())
        commented = sum(s.commented_posts for s in self.states_by_symbol.values())
        authored = sum(s.authored_posts for s in self.states_by_symbol.values())
        submitted = sum(s.submitted_orders for s in self.states_by_symbol.values())
        executed = sum(s.executed_trades for s in self.states_by_symbol.values())
        llm_calls = sum(s.llm_calls for s in self.states_by_symbol.values())
        social_state = self.symbol_states.get(self.symbol, {}).get(SocialNetworkState.__name__)
        author_score = (
            float(social_state.author_engagement_score(self.agent_id, as_of=self._last_tick_time or self.start_time))
            if isinstance(social_state, SocialNetworkState)
            else 0.0
        )
        out: dict[str, float] = {
            "type": 0.0,
            "user_id": float(int(self.profile.user_id)) if self.profile.user_id.isdigit() else 0.0,
            "belief_mean": float(agg_belief),
            "sentiment_mean": float(agg_sent),
            "conviction_mean": float(agg_conv),
            "consumed_news": float(consumed_news),
            "consumed_posts": float(consumed_posts),
            "consumed_comments": float(consumed_comments),
            "liked_posts": float(liked),
            "reposted_posts": float(reposted),
            "commented_posts": float(commented),
            "authored_posts": float(authored),
            "submitted_orders": float(submitted),
            "executed_trades": float(executed),
            "llm_calls": float(llm_calls),
            "author_engagement_score": author_score,
            "wealth": float(wealth if wealth is not None else self.cash / 1000.0),
            "cash": float(self.cash / 1000.0),  # tick → yuan for output
        }
        for sym in self.symbols:
            out[f"position_{sym}"] = float(self.holdings.get(sym, 0))
            out[f"belief_{sym}"] = float(self.states_by_symbol[sym].belief)
        return out


def build_xueqiu_multi_symbol_agents(
    *,
    profiles,
    symbols: list[str],
    session_calendar: SessionCalendar,
    start_time: Timestamp,
    end_time: Timestamp,
    reference_prices: dict[str, int],
    seed: int = 0,
    llm: XueqiuLLM | None = None,
    llm_call_probability: float = 0.05,
    v3_agent_config: V3AgentConfig | None = None,
    surge_retail_wake_start: date | None = None,
    surge_retail_wake_end: date | None = None,
    disable_trade_cooldown: bool = False,
) -> list[XueqiuMultiSymbolRetailAgent]:
    agents: list[XueqiuMultiSymbolRetailAgent] = []
    for idx, profile in enumerate(profiles):
        agents.append(
            XueqiuMultiSymbolRetailAgent(
                symbols=symbols,
                session_calendar=session_calendar,
                start_time=start_time,
                end_time=end_time,
                profile=profile,
                reference_prices=reference_prices,
                seed=seed + 100 + idx,
                llm=llm,
                llm_call_probability=llm_call_probability,
                v3_agent_config=v3_agent_config,
                surge_retail_wake_start=surge_retail_wake_start,
                surge_retail_wake_end=surge_retail_wake_end,
                disable_trade_cooldown=disable_trade_cooldown,
            )
        )
    return agents


_UNUSED = (Timedelta, deepcopy)
