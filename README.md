# Market Simulation Algorithms

This repository contains the public algorithm-only subset of the market simulation project.

Included scope:

- `market_simulation/agents`: heterogeneous retail, institution, noise-agent, and execution-intent logic.
- `market_simulation/information`: information-resolution, asymmetry, scoring, and data-loader algorithms.

LLM dependency note:

The original private project uses LLM clients for selected agent decisions. In this public release, those concrete clients are replaced by protocol definitions in `market_simulation/agents/core/llm_protocols.py`. The default retail LLM implementation is a no-op. To reproduce LLM-augmented behavior, inject an external object that implements the protocol methods; keep any API client and credentials outside this repository.

Security note:

Before pushing updates, run a secret scan and inspect staged files:

```bash
git status --short
git diff --cached --name-only
rg -n "api[-_ ]?key|secret|token|password|sk-[A-Za-z0-9]|OPENAI|XUEQIU_.*KEY" .
```
