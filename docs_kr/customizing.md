# 커스터마이징 및 확장

이 프레임워크는 의도적으로 작게 유지된다. 가장 필요할 만한 기능을 추가하는 방법을 소개한다.

## 메트릭 추가

`BaseMetric`를 서브클래스로 만들고 `_compute`를 구현한다. 무엇을 읽는지(`requires`), 어떻게 집계하는지,
게이트 방향은 무엇인지를 선언한다. 예외와 누락된 필드는 프레임워크가 알아서 `MetricResult(error=...)`로
변환해 주므로, 정상 경로(happy path)만 처리하면 된다.

```python
from agent_eval.core.contracts import Aggregation, CostClass, EvalContext, MetricResult
from agent_eval.core.metric import BaseMetric

class AnswerLength(BaseMetric):
    name = "answer_length"
    requires = frozenset({"output"})
    cost_class = CostClass.FREE
    aggregation = Aggregation.MEAN
    higher_is_better = False        # shorter is better, say
    unit_interval = False           # raw word count, not a [0, 1] score

    def _compute(self, ctx: EvalContext) -> MetricResult:
        return MetricResult(self.name, float(len(str(ctx.output).split())))
```

직접 사용하거나(`AnswerLength().score(ctx)`), 설정(config)에서 이름으로 지정할 수 있도록 등록한다:

```python
registry.register("answer_length", lambda params, ctx: AnswerLength(**params))
```

팩토리는 `(params, ctx)`를 받으며, 여기서 `ctx`는 해석된 judge(`ctx.judge`, `ctx.panel`)와
기본값(defaults)(`ctx.defaults["k"]` 등)을 담고 있는 `BuildContext`다. 기준 답 기반 메트릭은 `ctx.expected`를
읽고, 메타데이터 기반 메트릭은 `ctx.metadata['your_key']`를 읽으며 값이 없으면 `ValueError`를 발생시킨다(이는
`error`가 되어 해당 항목을 제외한다).

## 플러그인으로 메트릭 추가 (포크 없이)

직접 만든 패키지의 엔트리 포인트(entry point)를 통해 팩토리를 노출하면, `default_registry()`가
`load_plugins()`를 통해 이를 가져온다:

```toml
# pyproject.toml of your package
[project.entry-points."agent_eval.metrics"]
answer_length = "my_pkg.metrics:make_answer_length"
```

## judge 백엔드 추가

`JudgeBackend` 프로토콜(메서드 하나)을 구현하거나, 아무 모델로든 `LLMJudge`를 재사용한다. 설정의
`judge.factory`가 이를 반환하는(심판 패널(PoLL)의 경우 리스트를 반환하는) 함수를 가리키도록 한다.
[judges.md](judges.md)를 참고한다.

```python
from agent_eval.judges.backend import JudgeRequest, JudgeVerdict

class MyJudge:
    def evaluate(self, request: JudgeRequest) -> JudgeVerdict:
        ...  # call your model, return a normalized score
```

## 데이터셋 어댑터 추가

어댑터는 `field_map`을 통해 소스 레코드를 `EvalContext`로 정규화한다. 새로운 소스를 지원하려면
`datasets/jsonl.py`를 따라 한다: 행마다 `record_to_context(raw, field_map, include_unmapped)`를 yield하고,
예측 하네스가 입력을 읽을 수 있도록 `read_*_records` 헬퍼(원시 dict를 반환)를 노출한다. 그런 다음
`datasets/base.py`의 `load_dataset` / `load_records`에 분기를 추가한다.

일회성 인메모리 데이터라면 어댑터가 전혀 필요 없다. `EvalContext` 객체를 직접 만들어 그 리스트를 `evaluate`에
넘기면 된다([offline-evaluation.md](offline-evaluation.md) 참고).

## 실행 백엔드 추가 (T2S)

T2S 메트릭은 `readonly_connection`을 통해 `ExecutionHarness`와 통신한다. SQLite 백엔드는
`execution/sandbox.py`에 있으며, Postgres나 다른 백엔드는 동일한 `run(sql, db_ref) -> ExecResult` 인터페이스
뒤에 끼워 넣으면 되고, 결과 집합 비교 정책(`execution/compare.py`)은 변경 없이 그대로 재사용된다.

## 초점 유지

이 프레임워크는 의도적으로 [metrics.md](metrics.md)에 있는 메트릭만 정확히 제공한다. 확장할 때는 이미
존재하는 메트릭을 다시 만들기보다 실제 공백을 메우는 메트릭을 추가하는 편이 좋다. 그리고 객관적 과제는
judge가 아니라 근거 기반 검증으로 게이트한다.
