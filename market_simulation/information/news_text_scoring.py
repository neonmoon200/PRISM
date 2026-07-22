"""新闻文本的朴素主题/方向/可信度打分（无外部模型，供 CSV 新闻管线复用）。"""

from __future__ import annotations

import hashlib
import re


_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "policy": ("政策", "监管", "证监会", "国资委", "央行", "财政部", "改革", "发改委", "公告"),
    "macro": ("通胀", "降息", "加息", "GDP", "宏观", "汇率", "PMI", "美联储", "贸易"),
    "earnings": ("业绩", "财报", "营收", "利润", "净利润", "增速", "分红"),
    "tech": ("AI", "人工智能", "芯片", "半导体", "新能源", "光伏", "锂电", "储能", "云计算"),
    "consumption": ("消费", "白酒", "医药", "食品", "零售", "汽车", "家电"),
    "finance": ("银行", "保险", "券商", "金融", "信贷", "理财", "基金"),
    "energy": ("石油", "煤炭", "天然气", "电力", "能源", "矿业", "有色"),
    "sentiment": ("热点", "炒作", "题材", "概念", "情绪", "异动"),
}

_BULLISH = ("上涨", "增长", "突破", "利好", "看好", "扩张", "增持", "回购", "中标", "创新高", "加仓")
_BEARISH = ("下跌", "下滑", "亏损", "利空", "减持", "退市", "破产", "处罚", "下调", "违约", "风险")


def _classify_topic(text: str) -> str:
    counts: dict[str, int] = {}
    for topic, kws in _TOPIC_KEYWORDS.items():
        c = sum(1 for kw in kws if kw in text)
        if c:
            counts[topic] = c
    if not counts:
        return "macro"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _direction_strength(text: str) -> tuple[float, float]:
    bull = sum(1 for kw in _BULLISH if kw in text)
    bear = sum(1 for kw in _BEARISH if kw in text)
    if bull == 0 and bear == 0:
        h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        sign = 1.0 if (h % 2) == 0 else -1.0
        return 0.10 * sign, 0.35
    delta = bull - bear
    direction = max(-1.0, min(1.0, delta / 3.0))
    strength = max(0.30, min(1.0, 0.30 + 0.20 * (bull + bear)))
    return direction, strength


def _credibility(text: str) -> float:
    if any(k in text for k in ("公告", "证监会", "央行", "国资委", "财政部")):
        return 0.85
    if any(k in text for k in ("路透", "新华社", "人民日报", "彭博", "新华网")):
        return 0.80
    return 0.65


def _urgency(text: str) -> float:
    if any(k in text for k in ("【", "突发", "紧急", "重要")):
        return 0.80
    return 0.50


def _short_headline(text: str, limit: int = 200) -> str:
    t = re.sub(r"\s+", " ", str(text).strip())
    return t[:limit] if len(t) > limit else t


__all__ = [
    "_BULLISH",
    "_BEARISH",
    "_TOPIC_KEYWORDS",
    "_classify_topic",
    "_credibility",
    "_direction_strength",
    "_short_headline",
    "_urgency",
]
