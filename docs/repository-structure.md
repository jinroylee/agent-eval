# Repository structure

```
agent-eval/
├── README.md                     # overview + quickstart
├── pyproject.toml                # deps + extras (langgraph, t2s, bertscore)
├── docs/                         # this documentation
├── examples/                     # three runnable, self-contained examples
│   ├── README.md                 #   the predict → evaluate workflow
│   ├── judges.py                 #   real Claude judge factories (single + PoLL panel)
│   ├── plain/                    #   common metrics: agent.py, gold.jsonl, plain.yaml, README
│   ├── rag/                      #   + RAG metrics: agent.py, corpus, gold.jsonl, rag.yaml, README
│   └── t2s/                      #   + T2S metrics: agent.py, build_db.py, t2s.yaml, README
├── src/agent_eval/
│   ├── core/                     # the metric core
│   │   ├── contracts.py          #   EvalContext, MetricResult, Aggregation, CostClass, MetaKey
│   │   ├── metric.py             #   Metric protocol + BaseMetric
│   │   ├── registry.py           #   MetricRegistry, BuildContext, default_registry()
│   │   ├── suite.py              #   Suite, SuiteResult
│   │   ├── gate.py               #   GatePolicy, MetricAggregate, decide_gate (direction-aware)
│   │   └── errors.py
│   ├── metrics/                  # the catalog, by agent type
│   │   ├── common.py             #   llm_judge, bertscore, p95_latency, token_usage (+ JudgeMetric base)
│   │   ├── rag.py                #   recall/precision/ndcg_at_k, faithfulness, consistency
│   │   └── t2s.py                #   soft_f1, component_match, ast_valid, t2s_faithfulness/consistency
│   ├── judges/                   # LLM-as-a-judge
│   │   ├── backend.py            #   JudgeRequest/Verdict, LLMJudge, FunctionJudge, lexical stub
│   │   └── panel.py              #   poll_pointwise (PoLL)
│   ├── execution/                # T2S execution grounding
│   │   ├── harness.py · sandbox.py · compare.py   # run SQL read-only; result-set policy
│   ├── stats/                    # the statistics the runner uses
│   │   ├── intervals.py          #   Wilson + bootstrap percentile CI
│   │   └── clustered.py          #   cluster-robust SE
│   ├── datasets/                 # bring-your-own dataset
│   │   ├── base.py               #   field_map → canonical; load_dataset / load_records
│   │   ├── jsonl.py · tabular.py
│   ├── harness/
│   │   └── predict.py            # run a LangGraph agent → fill a predictions dataset
│   ├── config/
│   │   ├── schema.py             #   pydantic config model
│   │   └── loader.py             #   load + build suites + resolve judge + import_attr
│   ├── offline/
│   │   ├── runner.py             #   evaluate(): aggregate + gate
│   │   └── report.py             #   JSON / JUnit / text
│   ├── cli/main.py               # `agent-eval predict | evaluate | version`
│   └── runtime/                  # FOUNDATION: in-flight critic over the same metrics (future mode)
│       ├── critic.py             #   Critic, CriticPolicy, Decision, critic_loop
│       ├── loop_guard.py · fallback.py
└── tests/                        # unit + integration (pytest)
```

## "Where does X live?"

| I want to… | Look in |
|---|---|
| see what each metric reads | `docs/metrics.md`, then `src/agent_eval/metrics/{common,rag,t2s}.py` |
| add a metric | `core/metric.py` (BaseMetric), register in the relevant `metrics/*.py` |
| change how scores aggregate or gate | `offline/runner.py`, `core/gate.py` |
| run my agent to make predictions | `harness/predict.py`, `cli/main.py` (`predict`) |
| wire a real LLM judge | `judges/backend.py`, `examples/judges.py` |
| add a dataset format | `datasets/base.py` + a new `datasets/*.py` |
| understand the config | `config/schema.py`, `docs/configuration.md` |
| build on the (future) runtime critic | `runtime/critic.py`, `docs/runtime-critic.md` |

`docs/research/` holds the original SOTA research report that informed the method choices — a
point-in-time record; the live metric set is whatever [metrics.md](metrics.md) lists.
`docs/final_metric_list.md` is the agreed target metric set this framework implements.
