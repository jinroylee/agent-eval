# LangGraph 에이전트 평가하기

이 문서는 프레임워크를 *여러분의* 컴파일된 LangGraph에 연결해 게이팅된 판정을 얻어내는 과정을 다루는 실전
가이드다. 프레임워크가 에이전트에게서 필요로 하는 것은 작은 **상태 계약(state contract)** 하나뿐이다. 즉, 그래프의
최종 상태에서 어느 필드가 메트릭이 채점할 대상을 담는지다.

```
 your gold dataset ──► agent-eval predict ──► predictions.jsonl ──► agent-eval evaluate ──► verdict
                       (runs your graph)                            (scores + gate)
```

하니스가 그래프를 구동할 수 있게 해 주는 extra를 설치한다:

```bash
uv sync --extra langgraph        # add --extra t2s for text-to-SQL metrics
```

## 1. 상태 계약

하니스는 입력마다 컴파일된 그래프를 한 번씩 호출하고 **최종 상태 dict**를 읽는다. 설정의 `state_map`
(`canonical field: your state key`)을 통해 어떤 상태 키가 메트릭이 읽는 표준 필드(canonical field)에
매핑되는지 알려준다. 표준 필드와 이를 읽는 메트릭은 다음과 같다:

| 표준 필드 | 매핑 위치 | 읽는 메트릭 |
|---|---|---|
| `output` | `EvalContext.output` | `llm_judge`, `bertscore`, `faithfulness`, `consistency`, `t2s_*` |
| `retrieved_context` | `EvalContext.retrieved_context` | `faithfulness`, `consistency` |
| `retrieved_ids` | `metadata['retrieved_ids']` | `recall_at_k`, `precision_at_k`, `ndcg_at_k` |
| `sql` | `metadata['sql']` | `soft_f1`, `component_match`, `ast_valid`, `t2s_*` |
| `tokens` | `metadata['tokens']` | `token_usage` |

`latency_ms`는 자동으로 채워진다(하니스가 각 실행 시간을 측정한다). 표준 필드 집합에 없는 것은 모두 자기 이름
그대로 `metadata`에 들어간다. 전체 필드 레퍼런스는 [metrics.md](metrics.md)에 있다.

메트릭이 필요로 하는 것은 무엇이든 그래프 상태의 키로 노출한다. 예를 들어 RAG 에이전트는 검색과 근거성을 채점할
수 있도록, 검색한 랭킹된 id와 청크 텍스트를 상태에 넣어야 한다:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class RagState(TypedDict, total=False):
    input: str                    # the question (the harness sets this)
    retrieved_ids: list[str]      # ranked doc ids   → recall/precision/ndcg
    retrieved_context: list[str]  # chunk texts      → faithfulness/consistency
    output: str                   # the final answer → llm_judge
    tokens: int

# ... add nodes ...
graph = builder.compile()         # the config points at this object
```

## 2. 골드 데이터셋

입력과 보유한 정답(GT)을 담은 파일(JSONL 또는 CSV/Excel/Parquet)이다. GT 메트릭만 레이블이 필요하며, 정답
불필요(reference-free) 메트릭은 레이블이 필요 없다.

```jsonl
{"question": "What is the capital of France?", "relevant": ["d1","d2"], "reference": "Paris is the capital of France."}
```

## 3. 설정

```yaml
version: 1
agent_type: rag
defaults: {k: 3}

datasets:
  gold:                                    # your labels
    adapter: jsonl
    path: data/gold.jsonl
    field_map: {input: question, expected: reference, relevant_ids: relevant}
  predictions:                             # written by `predict`, read by `evaluate`
    adapter: jsonl
    path: data/predictions.jsonl

prediction:
  graph: my_pkg.agent:graph                # "module:attr" or "path/to/agent.py:graph"
  source: gold                             # dataset of inputs to run the agent on
  target: predictions                      # dataset to write predictions into
  input_key: input                         # graph state key the question is passed under
  state_map:                               # canonical field <- your graph state key
    retrieved_ids: retrieved_ids
    retrieved_context: retrieved_context
    output: output

suites:
  retrieval:
    dataset: predictions
    metrics: [recall_at_k, precision_at_k, ndcg_at_k]
    gate: {thresholds: {recall_at_k: 0.7}, require_pass: [recall_at_k]}
  response:
    dataset: predictions
    metrics: [faithfulness, consistency, llm_judge]
    gate: {thresholds: {faithfulness: 0.6}, require_pass: [faithfulness]}
```

`field_map`(target ← source)은 골드 컬럼을 표준 필드에 매핑하고, `state_map`(target ← graph key)은
에이전트의 출력에 대해 동일하게 매핑한다. 둘은 하나의 어휘를 공유한다. 바로 최상위 필드(`input`, `output`,
`expected`, `retrieved_context`)와 메타데이터 키(그 외 전부)다. 전체 레퍼런스: [configuration.md](configuration.md).

## 4. 실행

```bash
uv run agent-eval predict  -c eval.yaml      # runs your graph → predictions.jsonl
uv run agent-eval evaluate -c eval.yaml      # scores + gates (exit non-zero on failure)
```

`predict`는 각 골드 레코드를 에이전트가 생성한 예측과 병합해 바로 채점할 수 있는 예측 파일을 쓴다. 그다음
`evaluate`가 스위트를 실행하고 게이팅된 리포트를 출력한다.

## 그래프 리졸빙

`prediction.graph`는 `module:attr`(임포트 가능한 패키지) 또는 `path/to/file.py:attr`(스크립트 — 형제
모듈을 임포트할 수 있도록 해당 디렉터리가 `sys.path`에 추가된다)이다. 이 속성은 다음 중 하나여야 한다:

- **컴파일된 그래프**(`.invoke`를 가진 것이면 무엇이든), 또는
- 컴파일된 그래프를 반환하는 **인자 없는 팩토리**(zero-arg factory).

`run_once`는 `graph.invoke({input_key: question})`를 호출하고 반환된 상태 dict를 읽는다. 그래프가 단일
키보다 더 풍부한 입력 구조를 필요로 한다면, 입력을 적절히 변환하는 작은 팩토리로 감싼다.

## Python에서 구동하기

CLI가 하는 모든 것은 라이브러리로도 사용할 수 있다:

```python
from agent_eval.config.loader import load_config, build_suite, build_dataset_spec
from agent_eval.core.registry import default_registry
from agent_eval.datasets.base import load_dataset
from agent_eval.harness.predict import predict
from agent_eval.offline.runner import evaluate

cfg = load_config("eval.yaml")
predict(cfg)                                                   # fill predictions
suite = build_suite(cfg, "retrieval", default_registry())
result = evaluate(suite, load_dataset(build_dataset_spec(cfg, "predictions")))
print(result.verdict.passed)
```

## LangGraph를 쓰지 않는다면?

하니스는 편의 기능일 뿐이다. 이미 (어떤 에이전트든, 어떤 프레임워크에서든) 예측을 가지고 있다면 `predict`를
완전히 건너뛴다. 표준 필드를 담은 예측 파일을 작성하고 `evaluate`를 직접 실행하면 된다. `output`은 그저 "최종
응답"일 뿐이고, `metadata`가 나머지를 담는다.
