# 저장소 구조

```
agent-eval/
├── README.md                     # 개요 + 빠른 시작
├── pyproject.toml                # 의존성 + extras(langgraph, t2s, bertscore)
├── docs/                         # 이 문서
├── examples/                     # 실행 가능한 독립형 예제 3개
│   ├── README.md                 #   predict → evaluate 워크플로
│   ├── judges.py                 #   실제 Claude judge 팩토리(단일 + PoLL 심판 패널)
│   ├── plain/                    #   공통 메트릭: agent.py, gold.jsonl, plain.yaml, README
│   ├── rag/                      #   + RAG 메트릭: agent.py, corpus, gold.jsonl, rag.yaml, README
│   └── t2s/                      #   + T2S 메트릭: agent.py, build_db.py, t2s.yaml, README
├── src/agent_eval/
│   ├── core/                     # 메트릭 코어(metric core)
│   │   ├── contracts.py          #   EvalContext, MetricResult, Aggregation, CostClass, MetaKey
│   │   ├── metric.py             #   Metric 프로토콜 + BaseMetric
│   │   ├── registry.py           #   MetricRegistry, BuildContext, default_registry()
│   │   ├── suite.py              #   Suite, SuiteResult
│   │   ├── gate.py               #   GatePolicy, MetricAggregate, decide_gate (방향 인식)
│   │   └── errors.py
│   ├── metrics/                  # 에이전트 유형별 카탈로그
│   │   ├── common.py             #   llm_judge, bertscore, p95_latency, token_usage (+ JudgeMetric 베이스)
│   │   ├── rag.py                #   recall/precision/ndcg_at_k, faithfulness, consistency
│   │   └── t2s.py                #   soft_f1, component_match, ast_valid, t2s_faithfulness/consistency
│   ├── judges/                   # LLM-as-a-judge
│   │   ├── backend.py            #   JudgeRequest/Verdict, LLMJudge, FunctionJudge, 어휘 기반 스텁
│   │   └── panel.py              #   poll_pointwise (PoLL)
│   ├── execution/                # T2S 실행 기반 검증
│   │   ├── harness.py · sandbox.py · compare.py   # SQL을 읽기 전용으로 실행; 결과 집합 비교 정책
│   ├── stats/                    # 러너가 사용하는 통계
│   │   ├── intervals.py          #   Wilson + 부트스트랩 백분위 CI
│   │   └── clustered.py          #   군집 강건(cluster-robust) SE
│   ├── datasets/                 # 직접 준비하는 데이터셋
│   │   ├── base.py               #   field_map → 표준형(canonical); load_dataset / load_records
│   │   ├── jsonl.py · tabular.py
│   ├── harness/
│   │   └── predict.py            # LangGraph 에이전트 실행 → 예측 데이터셋 채우기
│   ├── config/
│   │   ├── schema.py             #   pydantic 설정 모델
│   │   └── loader.py             #   로드 + 스위트(Suite) 빌드 + judge 해석 + import_attr
│   ├── offline/
│   │   ├── runner.py             #   evaluate(): 집계 + 게이트
│   │   └── report.py             #   JSON / JUnit / text
│   ├── cli/main.py               # `agent-eval predict | evaluate | version`
│   └── runtime/                  # 토대(FOUNDATION): 동일 메트릭 기반 실시간 크리틱(in-flight critic) (향후 모드)
│       ├── critic.py             #   Critic, CriticPolicy, Decision, critic_loop
│       ├── loop_guard.py · fallback.py
└── tests/                        # 단위 + 통합 (pytest)
```

## "X는 어디에 있는가?"

| 하고 싶은 일 | 찾아볼 곳 |
|---|---|
| 각 메트릭이 무엇을 읽는지 확인 | `docs/metrics.md`, 그다음 `src/agent_eval/metrics/{common,rag,t2s}.py` |
| 메트릭 추가 | `core/metric.py` (BaseMetric), 해당 `metrics/*.py`에 등록 |
| 점수 집계 또는 게이트 방식 변경 | `offline/runner.py`, `core/gate.py` |
| 에이전트를 실행해 예측 생성 | `harness/predict.py`, `cli/main.py` (`predict`) |
| 실제 LLM judge 연결 | `judges/backend.py`, `examples/judges.py` |
| 데이터셋 형식 추가 | `datasets/base.py` + 새 `datasets/*.py` |
| 설정 이해 | `config/schema.py`, `docs/configuration.md` |
| (향후) 런타임 크리틱 위에 구축 | `runtime/critic.py`, `docs/runtime-critic.md` |

`docs/research/`에는 방법 선택의 근거가 된 원본 SOTA 연구 리포트가 담겨 있다 — 특정 시점의
기록이며, 현재 유효한 메트릭 세트는 [metrics.md](metrics.md)에 나열된 것을 따른다.
`docs/final_metric_list.md`는 이 프레임워크가 구현하는, 합의된 목표 메트릭 세트다.
