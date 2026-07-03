# 오프라인 평가(CI/CD 게이트)

오프라인 모드는 다음 질문에 답한다: *이 에이전트를 배포해도 될 만큼 좋은가?* — 감이 아니라 통계로.

## 흐름

1. 레이블이 달린 예제로 구성된 **골드 데이터셋**을 준비한다(참조 / 관련 id / 골드 쿼리).
2. **`agent-eval predict`**가 입력에 대해 에이전트를 실행하고 **예측** 데이터셋을 기록한다
   (이미 예측이 있다면 이 단계는 생략한다 — [langgraph-integration.md](langgraph-integration.md) 참고).
3. 메트릭 **스위트(Suite)**가 각 예측에 점수를 매긴다.
4. 러너가 소표본에서도 정확한 신뢰구간(CI)으로 메트릭별로 **집계**한다.
5. **게이트**가 이를 통과/실패 판정으로 바꾼다; 실패 시 CLI는 0이 아닌 종료 코드를 반환한다.

## 실행하기

```bash
uv run agent-eval evaluate --config eval.yaml
# --category <name>   run one suite only (default: all)
# --dataset <name>    override which configured dataset to score
```

```
=== rag/retrieval  dataset=predictions  n=5 ===
metric                     value                95% CI       thr  gate
recall_at_k                0.900        [0.704, 1.000]      0.70  PASS
precision_at_k             0.633        [0.417, 0.850]         -  info
VERDICT: PASS
```

## Python에서 사용하기

```python
from agent_eval.core.contracts import EvalContext
from agent_eval.core.gate import GatePolicy
from agent_eval.core.suite import Suite
from agent_eval.metrics.rag import RecallAtK
from agent_eval.offline.runner import evaluate

dataset = [
    EvalContext(input="q1", metadata={"retrieved_ids": ["d1", "d2"], "relevant_ids": ["d1"]}),
    EvalContext(input="q2", metadata={"retrieved_ids": ["x"], "relevant_ids": ["d3"]}),
]
suite = Suite("rag", "retrieval", [RecallAtK(k=10)],
              GatePolicy(thresholds={"recall_at_k": 0.4}, require_pass=["recall_at_k"]))
result = evaluate(suite, dataset)

print(result.verdict.passed)
for agg in result.aggregates:
    print(agg.metric, agg.value, (agg.ci_low, agg.ci_high), f"n={agg.n}", f"errors={agg.n_errors}")
```

`evaluate(suite, dataset, alpha=0.05)`는 `SuiteResult(agent_type, category, n_items, aggregates, verdict)`를 반환한다. 여기서 각 `MetricAggregate`는 `value`, `ci_low/ci_high`, `n`, `n_errors`, `aggregation`, `higher_is_better`를 담고, `GateVerdict`는 `passed`, `metric_passed`, `reasons`를 담는다.

## 집계가 동작하는 방식(그리고 이유)

각 메트릭은 `aggregation`을 선언하며, 러너는 각각에 대해 소표본에서도 정확한 통계를 사용한다:

- **`RATE`** (`ast_valid` 같은 이진 메트릭) → **Wilson** 구간을 적용한 통과율. 정규근사/CLT 구간과 달리 작은 *n*에서 커버리지가 부족해지지 않는다.
- **`MEAN`** (`recall_at_k`, `soft_f1`, `faithfulness`, 심사 점수 같은 등급형 메트릭) → **군집 강건** 표준오차(cluster-robust SE)를 적용한 평균. 항목이 서로 독립이 아니라면(non-iid) — 같은 지문을 공유하는 RAG 질문, 멀티턴 시나리오처럼 — `metadata["cluster_id"]`를 설정해 오차 막대가 인위적으로 좁아지지 않게 한다.
- **`P95`** (`p95_latency`) → **부트스트랩(bootstrap)** 구간을 적용한 95번째 백분위수(레이턴시는 두꺼운 꼬리(heavy-tailed) 분포라 정규근사는 틀린 값을 내놓는다).

**오류**(실행할 수 없었던 메트릭 — 필드 누락, 예외, 잘못된 골드)는 `n_errors`에 집계되고 `n`에서는 제외되며, 게이트 사유에 드러난다. 진짜로 나쁜 결과는 그저 점수가 낮게 나올 뿐이며, 이 둘은 결코 혼동되지 않는다.

## 게이트

임계값이 설정된 모든 메트릭은 **각자의 방향으로** 그 임계값을 충족해야 한다: `recall_at_k: 0.7`은 ≥ 0.7을, `p95_latency: 2000`은 ≤ 2000을 의미한다. `require_pass` 메트릭은 반드시 실행되어 통과해야 하는 하드 차단 요인(hard blocker)이다. 임계값이 없는 메트릭은 참고용이다.

## 리포트(JSON / JUnit / text)

```python
from agent_eval.offline import report
report.to_json(result)     # durable machine artifact
report.to_junit([result])  # JUnit XML for CI dashboards; failing thresholds become <failure>
report.to_text(result)     # the human table the CLI prints
```

## CI 연동

```yaml
# .github/workflows/eval.yml (sketch)
- run: uv sync --extra langgraph --extra t2s
- run: uv run agent-eval predict  -c eval/rag.yaml
- run: uv run agent-eval evaluate -c eval/rag.yaml   # non-zero exit fails the job
```

릴리스 간 동일 조건(apples-to-apples) 비교를 위해 **고정된(frozen)** 골든 슬라이스를 버전 관리에 유지한다; 현실성을 위해 프로덕션 실패 사례에서 뽑은 롤링 보충분을 추가한다. 모든 데이터셋에 버전을 매기고, 그것이 게이팅한 에이전트 릴리스와 연결한다.
