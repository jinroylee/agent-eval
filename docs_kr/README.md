# agent-eval — 문서

오프라인으로 동작하는 **LangGraph 에이전트 평가 프레임워크**다. 레이블이 달린 데이터셋 위에서 에이전트를 실행하고,
엄선된 메트릭 집합으로 점수를 매긴 뒤, 통계적으로 방어 가능한 임계값(threshold)을 기준으로 **CI를 게이팅한다**.

```
 gold dataset ──► agent-eval predict ──► predictions ──► agent-eval evaluate ──► gated verdict
```

세 가지 에이전트 유형, 하나의 메트릭 코어(metric core):

- **Common** — `llm_judge`, `bertscore`, `p95_latency`, `token_usage`
- **RAG** — `recall_at_k`, `precision_at_k`, `ndcg_at_k`, `faithfulness`, `consistency`
- **T2S** — `soft_f1`, `component_match`, `ast_valid`, `t2s_faithfulness`, `t2s_consistency`

## 설치

```bash
uv sync --extra langgraph --extra t2s      # run agents + text-to-SQL metrics
```

## 빠른 시작

```bash
uv run agent-eval predict  -c examples/rag/rag.yaml
uv run agent-eval evaluate -c examples/rag/rag.yaml
```

## 다음으로 읽을 문서

| 문서 | 다루는 내용 |
|---|---|
| [concepts.md](concepts.md) | 전체 개념 모델(mental model): 메트릭 코어, `EvalContext`, 정답(GT) 대 정답 불필요(non-GT), 집계(aggregation), 게이트(gate). **먼저 읽을 것.** |
| [metrics.md](metrics.md) | 메트릭 카탈로그(Common / RAG / T2S)와 **각 메트릭이 요구하는 그래프 상태(graph-state) 필드**. |
| [langgraph-integration.md](langgraph-integration.md) | 직접 만든 LangGraph 에이전트 평가하기: 상태 계약(state contract), `state_map`, `agent-eval predict`. |
| [offline-evaluation.md](offline-evaluation.md) | 데이터셋, 게이트 실행, 통계, 리포팅, CI. |
| [configuration.md](configuration.md) | 전체 YAML 레퍼런스. |
| [judges.md](judges.md) | LLM 심판(judge): 루브릭(rubric), 심판 패널(PoLL), 그리고 실제 Claude judge 연결하기. |
| [customizing.md](customizing.md) | 메트릭, 데이터셋 어댑터, judge 백엔드, 또는 플러그인 추가하기. |
| [repository-structure.md](repository-structure.md) | 실제 디렉터리 레이아웃과 "X는 어디에 있는가?"를 알려주는 지도. |
| [runtime-critic.md](runtime-critic.md) | 동일한 메트릭을 재사용하는 향후 실행 중(in-flight) 비평(critique) 모드를 위한 **기반**(오프라인 게이트의 일부는 아님). |
| [final_metric_list.md](final_metric_list.md) | 이 프레임워크가 구현하는, 합의된 목표 메트릭 집합. |

함께 제공되는 [`examples/`](../examples/)가 가장 빠르게 시작하는 방법이다 — 에이전트 유형마다 실행 가능한 예제가 하나씩 들어 있다.
`research/`에는 방법 선택에 근거가 된 원본 SOTA 연구 리포트가 들어 있다(특정 시점의 기록이며, 현재 유효한
메트릭 집합은 [metrics.md](metrics.md)에 나열된 것을 따른다).
