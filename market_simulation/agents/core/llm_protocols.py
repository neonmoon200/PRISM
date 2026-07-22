from __future__ import annotations

from random import Random
from typing import Any, Protocol


class XueqiuLLMProtocol(Protocol):
    """Public interface expected by retail agents.

    This protocol intentionally contains no API client, endpoint, key-loading
    logic, or model vendor details. Users can inject their own implementation
    from outside this repository.
    """

    enabled: bool

    def maybe(self, rng: Random, probability: float) -> bool: ...

    def react_to_post(self, **kwargs: Any) -> dict[str, Any] | None: ...

    def compose_post(self, **kwargs: Any) -> dict[str, Any] | None: ...

    def decide_trade(self, **kwargs: Any) -> dict[str, Any] | None: ...

    def analyze_information_batch(self, **kwargs: Any) -> dict[str, Any] | None: ...

    def decide_post_free(self, **kwargs: Any) -> dict[str, Any] | None: ...

    def decide_trade_multi(self, **kwargs: Any) -> dict[str, Any] | None: ...

    def decide_post_trade_joint(self, **kwargs: Any) -> dict[str, Any] | None: ...


class InstitutionLLMProtocol(Protocol):
    """Public interface expected by institution agents."""

    def maybe(self, rng: Random, probability: float) -> bool: ...

    def decide_trade(self, **kwargs: Any) -> dict[str, Any] | None: ...

    def decide_trade_multi(self, **kwargs: Any) -> dict[str, Any] | None: ...


class NullRetailLLM:
    """No-op retail LLM used by the public algorithm release by default."""

    enabled = False

    def maybe(self, rng: Random, probability: float) -> bool:
        return False

    def react_to_post(self, **kwargs: Any) -> dict[str, Any] | None:
        return None

    def compose_post(self, **kwargs: Any) -> dict[str, Any] | None:
        return None

    def decide_trade(self, **kwargs: Any) -> dict[str, Any] | None:
        return None

    def analyze_information_batch(self, **kwargs: Any) -> dict[str, Any] | None:
        return None

    def decide_post_free(self, **kwargs: Any) -> dict[str, Any] | None:
        return None

    def decide_trade_multi(self, **kwargs: Any) -> dict[str, Any] | None:
        return {"action": "hold", "reason": "llm_not_configured"}

    def decide_post_trade_joint(self, **kwargs: Any) -> dict[str, Any] | None:
        return {"trade": {"action": "hold", "reason": "llm_not_configured"}, "post": {"action": "skip"}}


def get_default_llm() -> XueqiuLLMProtocol:
    return NullRetailLLM()


XueqiuLLM = XueqiuLLMProtocol
InstitutionOpenAILLM = InstitutionLLMProtocol


__all__ = [
    "InstitutionLLMProtocol",
    "InstitutionOpenAILLM",
    "NullRetailLLM",
    "XueqiuLLM",
    "XueqiuLLMProtocol",
    "get_default_llm",
]
