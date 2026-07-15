# 런타임 크리틱(runtime critique) (토대)

> **상태: 향후 모드를 위한 토대(foundation)이며, 오프라인 게이트(offline gate)의 일부가 아니다.** 오프라인
> `predict`/`evaluate` 경로는 이 중 어떤 것도 임포트하지 않는다. 현재 메트릭 계약(metric contract)에 맞춰
> 연결되고 테스트로 커버된 상태로 여기에 보관되어 있어, 나중에 런타임 크리틱 모드를 이 위에 구축할 수 있다.

오프라인 러너는 *데이터셋* 전체를 채점하여 릴리스를 게이팅한다. **크리틱(critic)**은 그 실시간(in-flight)
대응물이다. **동일한 `Metric` 객체**로 *단일 스텝*을 채점하고 그 점수를 하나의 결정으로 바꾸어, LangGraph
노드가 잘못된 결과를 내보내기 전에 스스로 교정할 수 있게 한다. 메트릭 코어는 하나, 엔트리포인트는 둘.

## 설계 원칙 (향후 구축을 위해 보존)

- **저비용 우선 에스컬레이션(cheap-first escalation).** 메트릭은 `cost_class` 순서(FREE → CHEAP → EXPENSIVE)로
  실행되며, 근거 기반 검증(grounded check)이 실패하는 순간 크리틱은 즉시 중단한다(short-circuit) — 따라서 무료
  파싱/실행 검증이 이미 문제를 잡아냈다면 LLM judge에 비용을 지불할 일이 결코 없다.
- **생성자/검증자 분리(generator/verifier separation).** 크리틱은 *외부* 검증자(`ast_valid`, 검색(retrieval),
  독립적인 judge)만 사용한다. 답을 생성한 모델에게 자신의 정답 여부를 스스로 채점하도록 요청하는 일은 결코 없다.
- **하드 실패 대 소프트 실패(hard vs soft failure).** 실패한 이진/근거 기반 검증(`aggregation == RATE`, 예:
  `ast_valid`)은 객관적 오류다 → **재시도(retry)**. 신뢰도가 낮은 채점형 검증(`consistency`, `faithfulness`)
  → 검토를 위한 **에스컬레이션(escalate)**. 모두 통과 → **수용(accept)**.

## 구성 요소 (`agent_eval.runtime`)

| | |
|---|---|
| `Critic(metrics, policy)` | `assess(ctx) -> Assessment`; `decide(ctx, attempt) -> (Decision, Assessment)` |
| `CriticPolicy(tau, max_retries)` | 메트릭별 수용 임계값(acceptance threshold) + 재시도 예산(retry budget) |
| `Decision` | `ACCEPT · RETRY · FALLBACK · ESCALATE · ABSTAIN` |
| `critic_loop(generate, build_ctx, critic, fallback)` | 횟수가 제한된 생성 → 검증 → 크리틱 반영 재시도 루프 |
| `LoopGuard` | 진동/무진전 재시도를 중단한다 |
| `default_fallbacks()` | 명명된 전략(`abstain`, `escalate`) |

## 스케치

크리틱은 이미 사용 중인 바로 그 메트릭들로 구성된다 — 예를 들어 `ast_valid`(근거 기반, 하드)로 게이팅되고
`t2s_faithfulness` 신뢰도 검증(채점형, 소프트)을 갖춘 T2S 스텝처럼:

```python
from agent_eval.core.contracts import EvalContext, MetaKey
from agent_eval.metrics.t2s import AstValid, T2SFaithfulness
from agent_eval.judges.backend import FunctionJudge, lexical_overlap_judge
from agent_eval.runtime.critic import Critic, CriticPolicy, critic_loop
from agent_eval.runtime.fallback import default_fallbacks

critic = Critic(
    metrics=[AstValid(), T2SFaithfulness(FunctionJudge(lexical_overlap_judge))],
    policy=CriticPolicy(tau={"t2s_faithfulness": 0.6}, max_retries=2),
)

def generate(attempt, critique):
    # call your model; inject `critique` (the grounded feedback) on retries
    ...

def build_ctx(sql_and_answer):
    return EvalContext(input=question, output=sql_and_answer.answer,
                       metadata={MetaKey.SQL: sql_and_answer.sql, MetaKey.DB_REF: db})

result = critic_loop(generate, build_ctx, critic, fallback=default_fallbacks().get("abstain"))
result.decision   # ACCEPT / ESCALATE / ABSTAIN / ...
```

## 의도적으로 나중으로 미뤄둔 것들

- `agent-eval` CLI 명령어도, 설정 섹션도 없다 (오프라인 흐름은 단순한 표면(surface)을 그대로 유지한다).
- `tau` 임계값은 수작업으로 설정된다. 계획된 설계에서는 오프라인 연구로부터 이를 보정하여, 실시간 크리틱과
  CI 게이트가 운영 지점(operating point)을 공유하도록 한다.
- 아직 LangGraph 노드 래퍼는 없다 — `critic_loop`은 프레임워크에 구애받지 않으며, 이를 노드에 연결하는
  작업(재시도 시 크리틱을 주입하고 `Decision`에 따라 라우팅)이 다음 단계다.
