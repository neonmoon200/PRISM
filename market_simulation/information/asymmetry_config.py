"""信息差消融开关：默认关闭（保留完整信息差机制）。"""

from __future__ import annotations

import os

_info_asym_disabled: bool = False


def is_info_asymmetry_disabled() -> bool:
    return _info_asym_disabled


def set_info_asymmetry_disabled(disabled: bool) -> None:
    global _info_asym_disabled
    _info_asym_disabled = bool(disabled)


def configure_info_asymmetry_from_env() -> None:
    """读取 ``XUEQIU_ABLATION_NO_INFO_ASYM``（1/true/yes/on 时启用消融）。"""
    env = os.environ.get("XUEQIU_ABLATION_NO_INFO_ASYM", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        set_info_asymmetry_disabled(True)


# 模块导入时即读取环境变量，旧版 runner 无需 --ablation-no-info-asym 也能生效。
configure_info_asymmetry_from_env()
