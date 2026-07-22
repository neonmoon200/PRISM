"""市场模拟中的各类 agent。

目录约定：
* ``config/`` — 静态配置（如雪球 BigFive JSON 与默认路径）；
* ``core/`` — 异质 agent 共享基类、订单意图与 LOB 执行辅助；
* ``retail/``、``institution/``、``noise/``、``trading/`` — 各自主体实现。

对外常用符号从此包根导入即可。
"""

from market_simulation.agents.core import (
    ExecutionIntent,
    HeterogeneousAgentBase,
    best_prices,
    build_orders_from_intent,
    round_lot,
)
from market_simulation.agents.institution import InstitutionTraderAgent
from market_simulation.agents.noise import NoiseAgent
from market_simulation.agents.retail import (
    RetailAgent,
    XueqiuMultiSymbolRetailAgent,
    XueqiuRetailAgent,
    build_xueqiu_agents,
    build_xueqiu_multi_symbol_agents,
)
__all__ = [
    "ExecutionIntent",
    "HeterogeneousAgentBase",
    "InstitutionTraderAgent",
    "NoiseAgent",
    "RetailAgent",
    "XueqiuMultiSymbolRetailAgent",
    "XueqiuRetailAgent",
    "best_prices",
    "build_orders_from_intent",
    "build_xueqiu_agents",
    "build_xueqiu_multi_symbol_agents",
    "round_lot",
]
