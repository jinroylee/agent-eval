# 설정 레퍼런스(YAML)

하나의 파일에 에이전트를 어떻게 실행할지, 데이터가 어디에 있는지, 어떤 judge를 사용할지, 무엇을 채점하고 게이트할지를 기술한다. 이 파일은 pydantic 스키마(`agent_eval.config.schema`)로 검증된다. 환경 변수는 치환된다(`${VAR}` / `$VAR`).

```yaml
version: 1
agent_type: rag                  # plain | rag | t2s — 참고용(아무것도 좌우하지 않음)

defaults:                        # 공유 메트릭 기본값
  k: 5                           # recall/precision/ndcg의 고정 검색 깊이
  dialect: sqlite                # T2S SQL 다이얼렉트(dialect)
  system_prompt: null            # 모든 judge 메트릭의 judge 페르소나(docs/judges.md 참고)
  result_set_policy:             # T2S 결과 집합(result set) 비교 의미론
    row_order: ignore            #   ignore | strict
    duplicates: keep             #   keep | dedup
    nulls: distinct              #   distinct | coalesce
    float_tolerance: 1.0e-6

judge:                           # 선택 사항 — 생략하면 결정론적 오프라인 스텁을 사용한다
  factory: my_pkg.judges:claude_judge   # JudgeBackend를 반환하는 "module:attr" 또는 "file.py:attr"
                                        # (또는 이들의 리스트 = PoLL 패널)

datasets:
  gold:
    adapter: jsonl               # jsonl | tabular
    path: data/gold.jsonl
    format: null                 # tabular 전용: csv | tsv | excel | parquet (null이면 추론)
    field_map:                   # 타깃 필드 <- 소스 컬럼
      input: question
      expected: reference
      relevant_ids: relevant
    include_unmapped: true       # 매핑되지 않은 소스 컬럼을 메타데이터에 유지
  predictions:
    adapter: jsonl
    path: data/predictions.jsonl

prediction:                      # `agent-eval predict`가 예측을 채우는 방식 (선택 사항)
  graph: my_pkg.agent:graph      # 컴파일된 LangGraph, 또는 이를 반환하는 무인자 팩토리
  source: gold                   # 에이전트를 실행할 입력 데이터셋
  target: predictions            # 예측을 기록할 데이터셋 (반드시 jsonl)
  input_key: input               # 질문이 전달되는 그래프 상태 키
  n_runs: 1                      # >1이면 같은 입력으로 그래프를 N번 실행해 반복 생성 결과를 저장한다
                                 #     (repeated_outputs / repeated_sql — 자기일관성 메트릭용)
  state_map:                     # 표준 필드 <- 그래프 최종 상태 키
    output: output
    retrieved_ids: retrieved_ids
    retrieved_context: retrieved_context

suites:                          # 하나 이상의 명명된 평가
  retrieval:
    dataset: predictions         # 채점할 데이터셋 (기본값은 첫 번째 데이터셋)
    metrics:                     # 단순 타입 문자열, 또는 {type, name, params}
      - recall_at_k
      - {type: precision_at_k, name: precision_at_k, params: {k: 10}}
    gate:
      thresholds: {recall_at_k: 0.7}   # 메트릭 → 임계값 (방향은 metric.higher_is_better에 따름)
      require_pass: [recall_at_k]      # 하드 배포 차단 요인 (반드시 실행되고 통과해야 함)
```

## 섹션

| 섹션 | 용도 |
|---|---|
| `agent_type` | 라벨(`plain` / `rag` / `t2s`). 참고용 — 메트릭 집합은 스위트(Suite)에 나열된 대로 결정된다. |
| `defaults` | 메트릭 팩토리에 주입되는 공유 파라미터: `k`, `dialect`, `result_set_policy`, `system_prompt`(모든 judge 메트릭의 judge 페르소나). 메트릭 자체의 `params`가 이를 재정의한다. |
| `judge` | judge 기반 메트릭에 사용할 LLM judge 백엔드를 결정한다. [judges.md](judges.md) 참고. 생략 시 ⇒ 결정론적 스텁. |
| `datasets` | 명명된 데이터셋. `field_map`은 소스 컬럼을 표준 필드에 투영한다(target ← source). |
| `prediction` | `agent-eval predict`를 LangGraph에 연결한다. [langgraph-integration.md](langgraph-integration.md) 참고. |
| `suites` | 메트릭과 게이트로 구성된 명명된 그룹. `agent-eval evaluate`는 이들 전체를 실행한다(또는 `--category <name>`로 특정 스위트만). |

## 메트릭 스펙

스위트 내의 메트릭은 단순 타입 문자열(`recall_at_k`)이거나 딕셔너리다:

```yaml
- {type: recall_at_k, name: recall_at_10, params: {k: 10}}
- type: llm_judge
  params:
    system_prompt: "당신은 꼼꼼한 한국어 평가자입니다."   # judge 페르소나 재정의
    criteria:                              # 기준별 판정(기준마다 judge 호출 1회);
      - Accuracy                           #   일반 문자열 루브릭이면 종합 점수 1회 호출
      - {name: Tone, description: "정중하고 전문적인 어조"}
```

`type`은 메트릭을 선택한다(타입 + 파라미터는 [metrics.md](metrics.md) 참고). `name`은 리포팅 이름이다(기본값은
`type`). `params`는 생성자 인자다. judge 기반 메트릭은 설정된 judge를 자동으로 전달받으므로 `params`에
넘기지 않는다.

## 게이트

`thresholds`는 메트릭 이름을 숫자에 매핑한다. 비교 방향은 메트릭을 따른다.
`recall_at_k: 0.7`은 **≥ 0.7**을 의미하고, `p95_latency: 2000`은 **≤ 2000**을 의미한다(낮을수록 좋은
메트릭). `require_pass`는 반드시 실행되고 *또한* 통과해야 하는 메트릭을 나열한다 — 모든 항목에서 에러가
났거나 스위트에 없는 메트릭은 게이트를 실패시킨다. 임계값이 없는 메트릭은 `info`로 리포팅된다.

## CLI

```bash
agent-eval predict  -c eval.yaml                 # 에이전트 실행 → 예측
agent-eval evaluate -c eval.yaml                 # 모든 스위트
agent-eval evaluate -c eval.yaml --category retrieval   # 스위트 하나
agent-eval evaluate -c eval.yaml --dataset other_predictions   # 데이터셋 재정의
```

`evaluate`의 종료 코드: **0** 모든 게이트 통과, **1** 게이트 실패, **2** 설정/사용법 오류.
