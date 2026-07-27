# 메트릭 카탈로그

모든 메트릭을 에이전트 타입별로 정리하고, 각 메트릭이 **요구하는 `EvalContext` 필드**를 함께 표시한다.
`metadata['x']` 형태로 표기된 필드는 `EvalContext.metadata`의 키다(`agent_eval.core.contracts.MetaKey`의 상수).
예측 하니스가 설정의 `state_map`을 통해 그래프 상태에서 이 값들을 채운다([langgraph-integration.md](langgraph-integration.md)
참고).

모든 패밀리가 등록된 레지스트리를 만든다:

```python
from agent_eval.core.registry import default_registry
registry = default_registry()   # 14 metric types (T2S included if `sqlglot` is installed)
```

범례: **GT** = 정답(ground truth) 필요 · **agg** = 점수 집계 방식(rate/mean/p95) · ↓ = 낮을수록 좋음.

---

## 1. Common — 모든 에이전트 타입에 적용

이 메트릭들은 **최종 응답**(`output`)을 채점하므로, 일반(plain), RAG, T2S 에이전트 모두에서 동일하게 동작한다.

| 메트릭 | GT? | 읽는 값 | agg | 설명 |
|---|---|---|---|---|
| `llm_judge` | 선택 | `input`, `output` (+ `expected`) | mean | 전반적 품질에 대한 LLM 심판(judge) |
| `bertscore` | 예 | `output`, `expected` | mean | 정답(gold) 답변과의 의미적 유사도 |
| `p95_latency` | 아니오 | `metadata['latency_ms']` | p95 ↓ | 종단 간 테일(tail) 지연 시간(ms) |
| `token_usage` | 아니오 | `metadata['tokens']` | mean ↓ | 쿼리당 평균 소비 토큰 수 |

### `llm_judge`

응답을 **완전성(Completeness), 명확성(Clarity), 유용성(Usefulness), 관련성(Relevance), 친근함(Friendliness)**
기준으로 채점한다. 기본값으로 각 기준을 **개별적으로** 판정한다 — 기준마다 집중된 judge 호출 1회(항목당
5회; PoLL 패널이면 그만큼 배수가 된다). 메트릭 점수는 기준별 점수의 평균이고(각각 1–5 점수를 `[0, 1]`로
정규화), 기준별 점수는 항목별로 `detail['criteria']`에 담기며, `evaluate`가 이를 기준별 `breakdown`으로
집계해 리포트에 표시한다. `expected`가 설정되면 해당 참조 기준으로 채점하고, 없으면 내재적 품질을
채점한다. judge 백엔드가 필요하다 — [judges.md](judges.md) 참고. 설정된 judge가 없으면
결정론적(deterministic) 오프라인 스텁으로 폴백한다. 파라미터: `criteria` — 기준별 판정을 위한 리스트(각
항목은 이름 또는 `{name, description}`), 또는 **일반 문자열 루브릭**(종합 점수 1회 호출);
`scale`(기본값 `(1, 5)`); `system_prompt`(judge 페르소나 재정의 — [judges.md](judges.md)).
**심판 패널(PoLL)**(기준마다 여러 judge의 평균)을 지원한다 — `judge.factory`를
리스트를 반환하는 함수로 설정한다.

### `bertscore`

`output`과 `expected` 사이의 BERTScore F1(`rescale_with_baseline`로 해석 가능한 범위로 조정). `[bertscore]`
extra가 필요하다. 표현이 다르지만 올바른 답변, 즉 정확 일치(exact match)로는 놓칠 답변을 잡아낸다. 파라미터:
`lang`, `model_type`, `rescale_with_baseline`, `scorer`(테스트용 주입).

### `p95_latency` / `token_usage`

성능 메트릭이며 **낮을수록 좋음**이므로, 게이트 임계값이 상한(upper bound)으로 동작한다
(`p95_latency: 2000.0`은 ≤ 2000 ms를 의미). `latency_ms`는 하니스가 자동으로 채우고,
`tokens`는 그래프 상태가 토큰 수를 노출할 때 채워진다(`state_map`으로 매핑).

---

## 2. RAG — 검색 품질 + 답변 근거성

검색 메트릭은 **id** 기반으로 동작한다. `metadata['retrieved_ids']`는 검색기가 반환한 랭킹된 리스트(가장
관련성 높은 것부터)이고, `metadata['relevant_ids']`는 정답(gold) 집합이다. `k`는 평가마다 고정된다(설정의
`defaults.k` 또는 메트릭 파라미터).

| 메트릭 | GT? | 읽는 값 | agg | 설명 |
|---|---|---|---|---|
| `recall_at_k` | 예 | `metadata['retrieved_ids']`, `metadata['relevant_ids']` | mean | top-k 안에 든 정답(gold) id의 비율(**검색 게이트**) |
| `precision_at_k` | 예 | 동일 | mean | top-k 중 관련 있는 것의 비율 |
| `ndcg_at_k` | 예 | 동일 (+ 등급별 이득에는 `metadata['relevance']`) | mean | 순위 가중 관련도 |
| `faithfulness` | 아니오 | `retrieved_context`, `output` | mean | 모든 주장이 검색된 청크로 뒷받침되는가? |
| `consistency` | 아니오 | `metadata['repeated_outputs']` (+ `input`) | mean | 같은 질문을 반복 실행했을 때 같은 답을 주는가? |

`recall_at_k`는 다운스트림의 모든 것에 상한을 씌우므로 보통 하드 게이트로 쓴다. `faithfulness`(모든
주장이 검색된 청크로 *뒷받침되어야* 함)는 근거에 대해 judge로 채점한다. `consistency`는 **반복 실행에
걸쳐** judge로 채점한다: 같은 질문을 N번 실행하고(`prediction.n_runs`) 답변의 모든 쌍(pair)이 서로
일치하는지 판정한다 — 점수는 쌍별 평균이며, 질문당 judge 호출은 N(N−1)/2번이다. 두 메트릭 모두
`system_prompt` 파라미터를 받는다([judges.md](judges.md) 참고). `ndcg_at_k`는 기본적으로 이진(binary)
관련도를 쓴다. 등급별 관련도를 원하면 `metadata['relevance']`(`{id: gain}`)를 전달한다.

---

## 3. T2S — text-to-SQL 정확성 + 근거성

`[t2s]` extra(`sqlglot`)가 필요하다. 생성된 SQL은 `metadata['sql']`에, 정답(gold) SQL은
`metadata['gold_sql']`에, 실행 대상 데이터베이스는 `metadata['db_ref']`에 담긴다. 최종 자연어 답변은
`output`이다.

| 메트릭 | GT? | 읽는 값 | agg | 설명 |
|---|---|---|---|---|
| `soft_f1` | 예 | `metadata['sql']`, `metadata['gold_sql']`, `metadata['db_ref']` | mean | 실행된 결과 집합(result set)의 셀-F1(**정확성 게이트**) |
| `component_match` | 예 | `metadata['sql']`, `metadata['gold_sql']` | mean | AST 겹침(테이블 + 프로젝션) — 진단용 |
| `ast_valid` | 아니오 | `metadata['sql']` | rate | SQL이 파싱되는가? |
| `t2s_faithfulness` | 아니오 | `metadata['sql']`, `metadata['db_ref']`, `output` | mean | 자연어(NL) 답변이 쿼리 결과를 충실하게 보고하는가? |
| `t2s_consistency` | 아니오 | `metadata['repeated_sql']` (+ `input`) | mean | 같은 질문을 반복 실행했을 때 동등한 SQL을 생성하는가? |

**객관적 정확성은 실행으로 게이팅하며, 결코 judge로 게이팅하지 않는다.** `soft_f1`은 예측 SQL과 정답(gold)
SQL을 모두 실행하고 결과 집합을 비교한다(셀-백 F1로 부분 점수 부여). judge는 자연어 `output`이 쿼리가
*실제로 반환한* 것을 충실하게 보고하는지에만 국한된다. 결과 집합 비교는 명시적이고 설정 가능하다 —
`defaults.result_set_policy`가 행 순서, 중복, NULL, 부동소수점 허용 오차(float tolerance)를 제어한다(실행
메트릭에서 문서화된 조용한 실패(silent-failure)의 원인). 실행 메트릭의 파라미터: `dialect`, `result_policy`,
`timeout_s`. judge 메트릭(`t2s_faithfulness`, `t2s_consistency`)은 `system_prompt` 파라미터도
받는다([judges.md](judges.md) 참고).

> T2S 예제([`examples/t2s`](../examples/t2s/))는 역할 분담을 보여주도록 구성되어 있다:
> 패러프레이즈된 쿼리는 `soft_f1`을 1.0으로 유지하면서 `component_match`는 떨어뜨리고, `WHERE`가 누락되면
> `ast_valid`는 1.0으로 유지되지만 `soft_f1`은 하락한다. 정확성과 근거성은 별개의 문제다.

---

## "consistency" 명명에 관한 참고

최종 메트릭 세트는 *consistency*를 RAG와 T2S 양쪽에 모두 올려둔다. 둘 다 **같은 질문을 반복 실행했을
때의 자기일관성(self-consistency)**을 측정하며(반복 실행은 `prediction.n_runs`로 채운다), 비교 대상만
다르다 — 최종 답변(`metadata['repeated_outputs']`) 대 생성된 SQL(`metadata['repeated_sql']`). 그래서
레지스트리는 이들을 별개의 타입 이름으로 노출한다: `consistency`(RAG)와 `t2s_consistency`(T2S).
`faithfulness` / `t2s_faithfulness`는 근거 기반(grounding) 의미를 그대로 유지한다.
