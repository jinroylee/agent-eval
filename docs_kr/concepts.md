# 개념과 아키텍처

이 문서를 가장 먼저 읽어야 한다 — 다른 모든 문서가 여기 담긴 개념들 위에 세워진다.

## 핵심 아이디어 하나: 단일 메트릭 코어, 두 개의 CLI 단계

```
          config (YAML)                      METRIC CORE
                │                 Metric protocol · Registry · Suite · Gate
   EvalContext(input, output,          each Metric: EvalContext → MetricResult(score 0..1, …)
     expected, retrieved_context,                   │
     metadata)                                      │
        │                                           ▼
        ├── agent-eval predict ──► fills predictions by running your LangGraph agent
        └── agent-eval evaluate ─► scores predictions → aggregate w/ CIs → GATE verdict (CI exit code)
```

**메트릭**(metric)은 하나의 **작업 단위**(unit of work), 즉 `EvalContext` 하나를 채점하여 정규화된
`MetricResult`를 반환한다. 오프라인 **러너**(runner)는 그 결과들을 데이터셋 전체에 대해 적절한 통계량으로
집계한 뒤 **게이트**(gate)를 적용한다 → 통과/실패 판정. 이것이 프레임워크의 전부다.

## 핵심 데이터 타입

`from agent_eval.core.contracts import ...` — 작고 불변인(frozen) 타입 네 개.

### `EvalContext` — 채점할 항목 하나

```python
@dataclass(frozen=True)
class EvalContext:
    input: Any                      # the user message / question
    output: Any = None              # the final response to the user  ← common metrics score this
    expected: Any = None            # gold final response (GT metrics only)
    retrieved_context: Sequence[str] = ()   # RAG: retrieved chunk texts
    metadata: Mapping = {}          # agent-specific fields: retrieved_ids, sql, db_ref, latency_ms, …
```

메트릭을 에이전트 타입 전반에서 균일하게 유지해 주는 **불변식**(invariant): `output`은 *언제나 사용자에게
보여지는 최종 응답*이므로, 공통 메트릭(judge, BERTScore)은 모든 에이전트 타입에 대해 이를 채점한다.
에이전트별 산출물은 문서화된 키(`MetaKey.*`) 아래 `metadata`에 담긴다 — RAG 에이전트의 순위가 매겨진 id는
`metadata['retrieved_ids']`에, T2S 에이전트의 SQL은 `metadata['sql']`에 들어간다. 각 메트릭이 정확히 어떤
필드를 읽는지는 [metrics.md](metrics.md)를 참고한다.

### `MetricResult` — 정규화된 점수

```python
@dataclass(frozen=True)
class MetricResult:
    metric: str
    score: float                    # normalized 0..1 (raw magnitude for latency/tokens)
    passed: bool | None = None      # set for binary metrics; None for graded
    confidence: float | None = None # judge/panel agreement
    cost: Cost = Cost()             # tokens / usd / latency_ms (auto-captures latency)
    detail: Mapping = {}            # sub-scores, reasons, parse errors
    error: str | None = None        # set ⇒ the metric could NOT RUN (≠ a low score)
```

`error`와 낮은 `score`를 구분하는 것이 중요하다: 필수 필드가 누락되거나 예외가 발생하면 `error`가 설정되고(그
항목은 분모에서 제외되어 별도로 집계된다), 실제로 결과가 나빴을 뿐이라면 단지 점수가 낮게 나온다. 메트릭은
실행을 절대 중단시키지 않는다 — 예외는 `error`로 포착된다.

### `Metric` — 모든 것이 구현하는 단 하나의 계약

```python
class Metric(Protocol):
    name: str
    requires: frozenset[str]        # EvalContext fields it needs (validated before running)
    cost_class: CostClass           # FREE | CHEAP | EXPENSIVE  (EXPENSIVE ⇒ calls an LLM)
    aggregation: Aggregation        # RATE | MEAN | P95  (how per-item scores roll up)
    higher_is_better: bool          # gate direction (False for latency/tokens)
    unit_interval: bool             # are scores bounded to [0, 1]?
    def score(self, ctx) -> MetricResult: ...
```

이것을 직접 손으로 구현하는 경우는 거의 없다 — `BaseMetric`을 상속하고 `_compute`를 작성하면 된다.
[customizing.md](customizing.md)를 참고한다.

### `Suite`와 `GatePolicy` — 무엇을 실행하고, 어떻게 게이트하는가

```python
@dataclass
class Suite:
    agent_type: str                 # plain | rag | t2s (informational)
    category: str                   # e.g. retrieval | response | correctness
    metrics: Sequence[Metric]
    gate: GatePolicy

@dataclass
class GatePolicy:
    thresholds: Mapping[str, float] # metric_name → threshold (direction per metric.higher_is_better)
    require_pass: Sequence[str]     # hard ship-blockers (must run AND pass)
```

## GT 대 non-GT (핵심 메트릭 축)

각 메트릭은 **정답 기반**(ground-truth, 기준 답/레이블이 필요)이거나 **정답 불필요**(reference-free)이다:

- **GT** — `bertscore`, `recall_at_k`, `precision_at_k`, `ndcg_at_k`, `soft_f1`, `component_match`.
  이들은 `expected` 또는 `metadata`의 정답 필드(`relevant_ids`, `gold_sql`)를 읽는다.
- **non-GT** — `faithfulness`(충실성), `consistency`(일관성), `t2s_faithfulness`, `t2s_consistency`, `ast_valid`,
  `p95_latency`, `token_usage`. 레이블이 필요 없다 — 검색된 근거나 실행된 쿼리 결과에 비추어 output을 판정하거나,
  비용/지연을 측정한다.
- `llm_judge`는 **양쪽 모두** 동작한다: 기준 답이 있으면(`expected`가 설정된 경우) 그에 비추어 채점하고, 없으면
  내재적 품질을 채점한다.

설계 원칙: **객관적 작업은 judge가 아니라 근거 기반 검사로 게이트한다** — T2S 정확성은 쿼리 *실행*(`soft_f1`)으로
게이트하고, judge(LLM 심판)는 주관적 품질과 근거성(groundedness)에 한정된다. 답을 생성해 낸 모델이 자기 답의
정확성을 채점하도록 놔둬서는 절대 안 된다.

## 점수 집계 방식

모든 메트릭은 `aggregation`을 선언하며, 러너는 그에 따라 항목별 점수를 집계한다:

| 집계 | 사용 메트릭 | 통계량 | 구간 |
|---|---|---|---|
| `RATE` | 이진 메트릭 (`ast_valid`) | 통과율 | **Wilson** (소표본에서 커버리지가 부족하지 않음) |
| `MEAN` | 등급형 메트릭 (`recall_at_k`, `soft_f1`, `faithfulness`, judge) | 평균 | **군집 강건**(cluster-robust) SE |
| `P95` | `p95_latency` | 95 백분위수 | **부트스트랩**(bootstrap) (지연은 두꺼운 꼬리(heavy-tailed) 분포) |

비독립(non-iid) 항목(같은 지문을 공유하는 RAG 질문 등)의 경우 `metadata['cluster_id']`를 설정하여, 군집 강건
구간이 인위적으로 좁은 오차 막대를 보고하지 않도록 한다.
[offline-evaluation.md](offline-evaluation.md)를 참고한다.

## 구성 요소가 패키지에 매핑되는 방식

```
core/         contracts · metric (BaseMetric) · registry (+ BuildContext) · suite · gate · errors
metrics/      common · rag · t2s          # the three metric families
judges/       backend (LLMJudge, FunctionJudge) · panel (PoLL)
execution/    harness · sandbox · compare(result-set policy)   # T2S
stats/        intervals (Wilson, bootstrap) · clustered (cluster-robust SE)
datasets/     base (field_map → canonical) · jsonl · tabular
harness/      predict — run a LangGraph agent to fill predictions
config/       schema (pydantic) · loader
offline/      runner (evaluate) · report (JSON/JUnit/text)
cli/          main (predict · evaluate · version)
runtime/      critic · loop_guard · fallback   # foundation for a future in-flight critique mode
```

오프라인 게이트를 구동하는 바로 그 `Metric` 객체들은, 단일 스텝을 채점하고 accept/retry/fallback을 결정하는
**실행 중 크리틱**(in-flight critic)을 구동할 수도 있다 — 이는 미래 런타임 모드를 위한 기반이며, 오프라인
경로와는 분리되어 유지된다. [runtime-critic.md](runtime-critic.md)를 참고한다.
