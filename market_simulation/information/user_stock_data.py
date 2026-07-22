"""从仓库根目录 ``stock_data/`` 加载用户准备的行情、成分与多源新闻，生成模拟用数据。

- ``policy_news_xwlb.csv``：政策/宏观类，对所有标的广播，散户与机构均 100% 接入（``retail_sample_rate=1``）。
- ``news_multisource.csv``：个股资讯；公告/财报类等视为「专业」，散户约 30% 可见；其余通稿类视为「公开」，散户 100% 可见。
- ``professional_financial_ths.csv``、``professional_fund_flow_em.csv``：专业研究/资金口径，机构全量、散户 30%。
- ``professional_info_sina_cninfo.csv``：用于聚合行业指数的日收盘价序列（写入/更新 ``stock_data/stock_data.csv``）并从同一文件抽取「成分股日行情摘要」作为公开资讯（散户与机构均 100%）。

列名与导出格式与原先指数行情管线兼容。

参见 ``build_news_from_user_stock_data_dir`` 与 ``ensure_user_artifact_csvs``。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas import Timedelta, Timestamp

from market_simulation.information.news_text_scoring import (
    _classify_topic,
    _credibility,
    _direction_strength,
    _short_headline,
    _urgency,
)
from market_simulation.information.types import ProfessionalNews
from market_simulation.utils.session_calendar import SessionCalendar


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USER_STOCK_DATA_DIR = _REPO_ROOT / "stock_data"


_PROFESSIONAL_TITLE_KWS = (
    "财报",
    "年报",
    "季报",
    "半年报",
    "三季报",
    "一季报",
    "业绩",
    "净利",
    "营收",
    "招股",
    "定增",
    "重组",
    "审计",
    "问询",
    "披露",
)


def _stable_news_id(*parts: object) -> str:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]
    return f"ud-{h}"


def _calendar_sessions_by_date(calendar: SessionCalendar) -> dict[Timestamp, list]:
    """按交易日聚合所有 ``SessionWindow``（A 股下每日通常 2 段）。"""

    by_day: dict[Timestamp, list] = {}
    for s in calendar.sessions:
        by_day.setdefault(s.open_time.normalize(), []).append(s)
    return by_day


# 旧名称保留兼容（返回每日第一段，供仅需识别交易日的调用方）。
def _calendar_session_by_date(calendar: SessionCalendar) -> dict[Timestamp, object]:
    return {d: ss[0] for d, ss in _calendar_sessions_by_date(calendar).items()}


def align_timestamp_to_session(ts: Timestamp, calendar: SessionCalendar) -> Timestamp | None:
    """将真实发布时间映射到模拟交易日窗口内。

    支持每日多段 session（例如 A 股的上午/下午）：
    - 若已在某段内 → 原样返回；
    - 若早于当日首段 → 首段开盘后 +2 分钟；
    - 若处于段间休市（如 11:30~13:00 的午休）→ 下一段开盘后 +2 分钟；
    - 若晚于当日末段 → 末段收盘前 -1 分钟；
    - 若当天根本不是交易日 → ``None``。
    """

    d = ts.normalize()
    by_day = _calendar_sessions_by_date(calendar)
    sessions_today = by_day.get(d)
    if not sessions_today:
        return None
    sessions_today = sorted(sessions_today, key=lambda s: s.open_time)
    for sess in sessions_today:
        if sess.contains(ts):
            return ts
        if ts < sess.open_time:
            return sess.open_time + Timedelta(minutes=2)
    return sessions_today[-1].close_time - Timedelta(minutes=1)


def ensure_user_artifact_csvs(
    data_dir: str | Path | None = None,
    *,
    symbols_filter: Iterable[str] | None = None,
) -> tuple[Path, Path]:
    """从 ``professional_info_sina_cninfo.csv`` / ``news_multisource.csv`` 生成
    ``stock_data.csv`` 与 ``stock_profile.csv``（列需含 ``ts_code`` / ``date`` / ``close`` 等）。
    """

    root = Path(data_dir) if data_dir else DEFAULT_USER_STOCK_DATA_DIR
    info_path = root / "professional_info_sina_cninfo.csv"
    multi_path = root / "news_multisource.csv"
    out_prices = root / "stock_data.csv"
    out_profile = root / "stock_profile.csv"

    if not info_path.is_file():
        raise FileNotFoundError(f"缺少行情成分文件: {info_path}")

    df = pd.read_csv(info_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["component_weight"] = pd.to_numeric(df["component_weight"], errors="coerce").fillna(1.0)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    sym_set = {str(s).strip() for s in symbols_filter} if symbols_filter is not None else None
    group_sym_col = "stock_id"
    if sym_set is not None:
        idx_mask = df["stock_id"].isin(sym_set)
        comp_key: str | None = None
        comp_mask = None
        for col in ("component_symbol_full", "component_symbol", "ts_code"):
            if col not in df.columns:
                continue
            m = df[col].astype(str).str.strip().isin(sym_set)
            if m.any():
                comp_key = col
                comp_mask = m
                break
        if idx_mask.any():
            df = df[idx_mask]
        elif comp_key is not None and comp_mask is not None:
            df = df[comp_mask]
            group_sym_col = comp_key
        else:
            df = df[idx_mask]

    bar_rows: list[dict[str, object]] = []
    for (sid, td), g in df.groupby([group_sym_col, "trade_date"], sort=True):
        w = g["component_weight"].astype(float)
        c = g["close"].astype(float)
        den = float(w.sum()) or 1.0
        wc = float((c * w).sum() / den)
        first = g.iloc[0]
        ind = str(first.get("industry", "") or "")
        sym_out = str(sid).strip()
        display_name = str(first.get("component_name", "") or first.get("name", "") or "")
        bar_rows.append(
            {
                "stock_id": sym_out,
                "trade_date": td,
                "close": wc,
                "Industry": ind,
                "category": ind,
                "name": display_name,
            }
        )
    bar = pd.DataFrame(bar_rows)
    bar = bar.sort_values(["stock_id", "trade_date"])
    bar["pre_close"] = bar.groupby("stock_id")["close"].shift(1)
    bar["change"] = bar["close"] - bar["pre_close"]
    bar["pct_chg"] = (bar["change"] / bar["pre_close"] * 100.0).where(bar["pre_close"].notna() & (bar["pre_close"] != 0))
    bar = bar.rename(columns={"stock_id": "ts_code", "trade_date": "date"})
    # 行情 CSV 期望列
    for col in (
        "amount",
        "elg_amount_net",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "dv_ttm",
        "ma_hfq_5",
        "ma_hfq_10",
        "ma_hfq_30",
        "ma_amount_5",
        "ma_amount_10",
        "ma_amount_30",
    ):
        if col not in bar.columns:
            bar[col] = pd.NA
    bar = bar[
        [
            "ts_code",
            "Industry",
            "category",
            "date",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "amount",
            "elg_amount_net",
            "pe_ttm",
            "pb",
            "ps_ttm",
            "dv_ttm",
            "ma_hfq_5",
            "ma_hfq_10",
            "ma_hfq_30",
            "ma_amount_5",
            "ma_amount_10",
            "ma_amount_30",
            "name",
        ]
    ]
    bar.to_csv(out_prices, index=False)

    if multi_path.is_file():
        m = pd.read_csv(multi_path)
        prof = m.groupby("stock_id", as_index=False).first()[["stock_id", "weight", "name", "industry", "description"]]
        if sym_set is not None:
            prof = prof[prof["stock_id"].astype(str).str.strip().isin(sym_set)]
        prof.to_csv(out_profile, index=False)
    else:
        if group_sym_col == "stock_id":
            meta = df.groupby("stock_id", as_index=False).first()[
                ["stock_id", "weight", "name", "industry", "description"]
            ]
        else:
            gmf = df.groupby(group_sym_col, as_index=False).first()
            display = (
                gmf["component_name"].fillna("").astype(str)
                if "component_name" in gmf.columns
                else (
                    gmf["name"].fillna("").astype(str)
                    if "name" in gmf.columns
                    else gmf[group_sym_col].astype(str).str.strip()
                )
            )
            meta = pd.DataFrame(
                {
                    "stock_id": gmf[group_sym_col].astype(str).str.strip(),
                    "weight": gmf["weight"] if "weight" in gmf.columns else pd.NA,
                    "name": display,
                    "industry": gmf["industry"] if "industry" in gmf.columns else pd.NA,
                    "description": gmf["description"] if "description" in gmf.columns else pd.NA,
                }
            )
        if sym_set is not None:
            meta = meta[meta["stock_id"].astype(str).str.strip().isin(sym_set)]
        meta.to_csv(out_profile, index=False)

    return out_prices, out_profile


def build_news_from_user_stock_data_dir(
    *,
    symbols: list[str],
    calendar: SessionCalendar,
    data_dir: str | Path | None = None,
    seed: int = 0,
) -> list[ProfessionalNews]:
    """读取 ``stock_data/*.csv`` 构建 ``ProfessionalNews`` 列表（时间落在 ``calendar`` 范围内）。"""

    _ = seed
    root = Path(data_dir) if data_dir else DEFAULT_USER_STOCK_DATA_DIR
    t0 = calendar.first_open - Timedelta(days=1)
    t1 = calendar.final_close + Timedelta(days=1)
    sym_set = set(symbols)
    items: list[ProfessionalNews] = []

    policy_path = root / "policy_news_xwlb.csv"
    if policy_path.is_file():
        pdf = pd.read_csv(policy_path)
        pdf["date"] = pd.to_datetime(pdf["date"].astype(str), format="%Y%m%d", errors="coerce")
        pdf = pdf[(pdf["date"] >= t0.normalize()) & (pdf["date"] <= t1.normalize())]
        for idx, row in pdf.iterrows():
            base_ts = Timestamp.combine(row["date"].date(), pd.Timestamp("10:00").time())
            text = f"{row.get('title', '')} {row.get('content', '')}"
            topic = _classify_topic(text)
            _direction, _strength = _direction_strength(text)
            cred = max(_credibility(text), 0.82)
            urg = _urgency(str(row.get("title", "")))
            headline = _short_headline(str(row.get("title", "")))
            for sym in symbols:
                nid = _stable_news_id("policy", idx, sym, row.get("title", ""))
                aligned = align_timestamp_to_session(base_ts, calendar)
                if aligned is None:
                    continue
                items.append(
                    ProfessionalNews(
                        news_id=nid,
                        symbol=sym,
                        publish_time=aligned,
                        headline=headline,
                        topic=topic,
                        direction=0.0,
                        strength=0.0,
                        content=text,
                        credibility=cred,
                        urgency=urg,
                        audience="all",
                        retail_sample_rate=1.0,
                    )
                )

    multi_path = root / "news_multisource.csv"
    if multi_path.is_file():
        mdf = pd.read_csv(multi_path)
        mdf["publish_time"] = pd.to_datetime(mdf["publish_time"], errors="coerce")
        mdf = mdf[(mdf["publish_time"] >= t0) & (mdf["publish_time"] <= t1)]
        mdf = mdf[mdf["stock_id"].isin(sym_set)]
        for idx, row in mdf.iterrows():
            sym = str(row["stock_id"])
            raw_t = row["publish_time"]
            if pd.isna(raw_t):
                continue
            if not isinstance(raw_t, Timestamp):
                raw_t = Timestamp(raw_t)
            title = str(row.get("title", "") or "")
            content = str(row.get("content", "") or "")
            text = f"{title} {content}"
            source_url = str(row.get("url", "") or "").strip()
            topic = _classify_topic(text)
            _direction, _strength = _direction_strength(text)
            cred = _credibility(text)
            urg = _urgency(title)
            headline = _short_headline(title or content)
            aligned = align_timestamp_to_session(raw_t, calendar)
            if aligned is None:
                continue
            nid = _stable_news_id("multi", idx, sym, title, raw_t)
            items.append(
                ProfessionalNews(
                    news_id=nid,
                    symbol=sym,
                    publish_time=aligned,
                    headline=headline,
                    topic=topic,
                    direction=0.0,
                    strength=0.0,
                    content=text,
                    credibility=cred,
                    urgency=urg,
                    audience="all",
                    retail_sample_rate=None,
                    source_url=source_url,
                    from_multisource=True,
                )
            )

    fin_path = root / "professional_financial_ths.csv"
    if fin_path.is_file():
        fdf = pd.read_csv(fin_path)
        fdf = fdf[fdf["stock_id"].isin(sym_set)]
        period_col: str | None = "报告期" if "报告期" in fdf.columns else None
        if period_col is not None:
            for idx, row in fdf.iterrows():
                sym = str(row["stock_id"])
                period = str(row.get(period_col, "") or "")
                if not any(y in period for y in ("2025", "2026", "2024")):
                    continue
                try:
                    base_ts = pd.to_datetime(period)
                except Exception:
                    continue
                if isinstance(base_ts, pd.Series):
                    continue
                base_ts = Timestamp.combine(base_ts.date(), pd.Timestamp("15:00").time())
                if base_ts < t0 or base_ts > t1:
                    continue
                parts = [
                    f"{k}={row[k]}"
                    for k in fdf.columns
                    if k in row.index and str(row[k]) not in ("", "nan", "False")
                ][:12]
                text = "; ".join(parts)
                headline = _short_headline(f"{sym} 财务摘要 {row.get(period_col, '')}".strip())
                topic = "earnings"
                _direction, _strength = _direction_strength(text)
                aligned = align_timestamp_to_session(base_ts, calendar)
                if aligned is None:
                    continue
                nid = _stable_news_id("fin", idx, sym, text[:80])
                items.append(
                    ProfessionalNews(
                        news_id=nid,
                        symbol=sym,
                        publish_time=aligned,
                        headline=headline,
                        topic=topic,
                        direction=0.0,
                        strength=0.0,
                        content=text,
                        credibility=0.78,
                        urgency=0.45,
                        audience="all",
                        retail_sample_rate=0.30,
                    )
                )

    flow_path = root / "professional_fund_flow_em.csv"
    if flow_path.is_file():
        fl = pd.read_csv(flow_path)
        date_col = "日期" if "日期" in fl.columns else None
        if date_col is None:
            pass
        else:
            fl["__d"] = pd.to_datetime(fl[date_col], errors="coerce")
            fl = fl[(fl["__d"] >= t0.normalize()) & (fl["__d"] <= t1.normalize())]
            fl = fl[fl["stock_id"].isin(sym_set)]
            for idx, row in fl.iterrows():
                sym = str(row["stock_id"])
                raw_d = row["__d"]
                if pd.isna(raw_d):
                    continue
                ts = Timestamp.combine(raw_d.date(), pd.Timestamp("14:30").time())
                zj = row.get("主力净流入-净额", "")
                headline = _short_headline(f"{sym} 资金流向 主力净流入 {zj}")
                text = headline
                topic = _classify_topic(text)
                _direction, _strength = _direction_strength(text)
                aligned = align_timestamp_to_session(ts, calendar)
                if aligned is None:
                    continue
                nid = _stable_news_id("flow", idx, sym, raw_d)
                items.append(
                    ProfessionalNews(
                        news_id=nid,
                        symbol=sym,
                        publish_time=aligned,
                        headline=headline,
                        topic=topic,
                        direction=0.0,
                        strength=0.0,
                        content=text,
                        credibility=0.70,
                        urgency=0.40,
                        audience="all",
                        retail_sample_rate=0.30,
                    )
                )

    info_path = root / "professional_info_sina_cninfo.csv"
    if info_path.is_file():
        idf = pd.read_csv(info_path)
        idf["trade_date"] = pd.to_datetime(idf["trade_date"], errors="coerce")
        idf = idf[(idf["trade_date"] >= t0.normalize()) & (idf["trade_date"] <= t1.normalize())]
        idf = idf[idf["stock_id"].isin(sym_set)]
        for idx, row in idf.iterrows():
            sym = str(row["stock_id"])
            d = row["trade_date"]
            if pd.isna(d):
                continue
            comp = str(row.get("component_name", ""))
            close = row.get("close", "")
            headline = _short_headline(f"{comp} 收盘{close} ({d.date()})")
            text = f"{headline} {row.get('company_industry', '')}"
            topic = _classify_topic(text)
            _direction, _strength = _direction_strength(text)
            ts = Timestamp.combine(d.date(), pd.Timestamp("15:00").time())
            aligned = align_timestamp_to_session(ts, calendar)
            if aligned is None:
                continue
            nid = _stable_news_id("ohlc", idx, sym, comp, d)
            items.append(
                ProfessionalNews(
                    news_id=nid,
                    symbol=sym,
                    publish_time=aligned,
                    headline=headline,
                    topic=topic,
                    direction=0.0,
                    strength=0.0,
                    content=text,
                    credibility=0.72,
                    urgency=0.35,
                    audience="all",
                    retail_sample_rate=1.0,
                )
            )

    items.sort(key=lambda x: x.publish_time)
    return items


__all__ = [
    "DEFAULT_USER_STOCK_DATA_DIR",
    "align_timestamp_to_session",
    "build_news_from_user_stock_data_dir",
    "ensure_user_artifact_csvs",
]
