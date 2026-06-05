# agent-eval

A reusable evaluation + runtime-critique framework for LangGraph agents.

Two modes over one metric core:

- **Offline** — holistic, statistically-gated evaluation for CI/CD (blocks bad releases).
- **Runtime** — an in-flight critic inside nodes/tool-calls that decides **retry / fallback / fail / escalate**.

Covers 4 agent types (Orchestration, RAG, T2S, Plain) × 4 categories (search, response, latency, scenario),
evaluated at any LangGraph level (graph / subgraph / node / tool). Config-first (YAML) with a Python plugin
escape hatch; bring-your-own dataset.

See `docs/research/2026-06-05-agent-eval-sota-research.md` for the SOTA basis, and the implementation plan for
the architecture and phasing.

## Status

Early development — P0 (core spine). Target Python 3.12.
