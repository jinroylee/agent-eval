# LLM-as-a-judge(LLM 심판)

다섯 가지 메트릭이 LLM judge를 사용한다: `llm_judge`(전반적 품질), `faithfulness` / `t2s_faithfulness`
(근거성), 그리고 `consistency` / `t2s_consistency`(같은 질문의 반복 실행 간 자기일관성 — 쌍별 판정).
이들은 하나의
작은 백엔드 인터페이스를 공유하므로, 모델을 한 번만 연결하면 모든 judge 메트릭이 이를 사용한다.

> **judge는 주관적 품질과 근거성에만 국한하고, 객관적 정확성에는 절대 사용하지 않는다.** T2S 쿼리
> 정확성은 *실행*(`soft_f1`)으로 판정하며, judge는 그것을 만들어 낸 답을 결코 채점하지 않는다. 이는
> 의도적인 설계 규칙이지 실수가 아니다.

## 인터페이스

메트릭은 구조화된 `JudgeRequest`를 만들고, 백엔드는 `JudgeVerdict`를 반환한다:

```python
@dataclass(frozen=True)
class JudgeRequest:
    instruction: str                 # the rubric — what to grade and how
    question: str = ""               # the user input, if relevant
    response: str = ""               # the response under test (graded)
    reference: str = ""              # gold answer, if grading against ground truth
    context: Sequence[str] = ()      # evidence (retrieved chunks, executed rows)
    scale: tuple[int, int] = (1, 5)  # integer score range to ask for
    system_prompt: str = ""          # persona/framing preamble; "" → the backend's default

@dataclass(frozen=True)
class JudgeVerdict:
    score: float                     # normalized to [0, 1]
    confidence: float | None = None
    reason: str = ""
```

세 가지 백엔드가 기본 제공된다(`agent_eval.judges.backend`):

- **`LLMJudge(complete, name)`** — 프로바이더 독립적이다. `complete(prompt) -> text` 콜러블을 넘기기만
  하면 되므로, 이 패키지가 어떤 SDK에도 의존하지 않고 어떤 모델이든 사용할 수 있다. 요청을
  `SCORE: <int>` + `REASON: …`를 요구하는 프롬프트로 렌더링한 뒤, 점수를 파싱하고 정규화한다.
- **`FunctionJudge(fn)`** — `fn(request) -> float | JudgeVerdict`를 감싸며, 결정론적/오프라인 채점과
  테스트에 사용한다.
- **`lexical_overlap_judge`** — 예제가 **API 키 없이** 실행되도록 미리 만들어 둔 결정론적 스텁(토큰
  중첩)이다. 실제 judge가 아니라 대체물이며, 점수는 조악하다.

judge가 설정되지 않으면, judge 메트릭은 어휘 스텁으로 폴백한다.

## 시스템 프롬프트

모든 judge 프롬프트는 페르소나/프레이밍 서문으로 시작한다 — 기본값은
`"You are a strict, impartial evaluator."`이다. judge가 설정되는 곳이라면 어디서든 교체할 수 있다
(더 구체적인 쪽이 이긴다):

1. **메트릭 단위** — judge 기반 메트릭의 `params: {system_prompt: ...}`, 또는 설정의
   `defaults.system_prompt`로 모든 judge 메트릭에 한 번에 지정.
2. **백엔드 단위** — judge 팩토리를 작성할 때 `LLMJudge(complete, system_prompt=...)`.
3. 내장 기본값.

이 커스터마이징은 서문만 교체한다: 메트릭의 루브릭(`instruction`)과 `SCORE:`/`REASON:` 형식 계약은
항상 그 뒤에 붙으므로, 커스텀 시스템 프롬프트가 점수 파싱을 깨뜨릴 수는 없다. `FunctionJudge`와
오프라인 스텁은 `request.system_prompt`를 보긴 하지만 무시한다. API 서버에서는 요청의
`system_prompt` 파라미터를 쓰거나, 서버 전역 기본값으로 `AGENT_EVAL_JUDGE_SYSTEM_PROMPT`를
설정한다([api-server.md](api-server.md) 참고).

## 실제 Claude judge 연결하기

설정의 `judge.factory`가 백엔드를 반환하는 함수를 가리키도록 지정한다:

```yaml
judge:
  factory: my_pkg.judges:claude_judge
```

```python
# my_pkg/judges.py
from agent_eval.judges.backend import LLMJudge

def claude_judge(model: str = "claude-opus-4-8") -> LLMJudge:
    import anthropic
    client = anthropic.Anthropic()              # reads ANTHROPIC_API_KEY

    def complete(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            output_config={"effort": "low"},     # grading one response is a small, scoped task
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    return LLMJudge(complete, name="claude")
```

실행 가능한 버전(단일 judge + PoLL 패널)은 [`../examples/judges.py`](../examples/judges.py)에 포함되어 있다.

```bash
uv pip install anthropic && export ANTHROPIC_API_KEY=...
```

## 심판 패널(PoLL)

작고 다양한 심판 패널은 단일 대형 judge만큼 사람의 평가와 잘 상관되고, 비용이 더 적게 들며, 단독
judge가 스스로 알아차리지 못하는 자기 선호 편향을 상쇄한다. 팩토리에서 백엔드의 **리스트**를 반환하면
프레임워크가 그 점수들을 평균 내고, 패널 합의도(`1 − spread`)를 verdict의 `confidence`로 보고한다:

```python
def claude_panel() -> list[LLMJudge]:
    return [
        LLMJudge(_claude_complete("claude-opus-4-8"),  name="opus"),
        LLMJudge(_claude_complete("claude-sonnet-4-6"), name="sonnet"),
    ]
```

```yaml
judge:
  factory: my_pkg.judges:claude_panel
```

첫 번째 백엔드가 주 백엔드이며, 나머지가 패널을 구성한다. 어떤 항목의 합의도가 낮으면 사람이
검토하도록 표시할 뿐, 게이트를 자동으로 바꾸지는 않는다.

## 루브릭(rubric)

각 judge 메트릭은 고정된, 용도에 맞게 만들어진 루브릭을 전송한다(`llm_judge`의 루브릭은 `criteria`
파라미터로 재정의할 수 있다):

| 메트릭 | 채점 대상 |
|---|---|
| `llm_judge` | 완전성, 명확성, 유용성, 관련성, 친절함 — 기본값으로 각 기준을 **개별적으로** 판정한다(기준별 점수는 `detail['criteria']`에 담기고, 전체 점수는 평균; `criteria`를 일반 문자열로 주면 종합 점수 하나). 기준 답이 주어지면 그에 대비하여 채점한다. |
| `faithfulness` | 답변의 모든 주장이 검색된 청크에 의해 **뒷받침되는지** |
| `consistency` | 같은 질문을 반복 실행했을 때 **같은 답**을 주는지(쌍별) |
| `t2s_faithfulness` | NL 답변이 **실행된 쿼리 결과**를 충실하게 보고하는지 |
| `t2s_consistency` | 같은 질문을 반복 실행했을 때 **동등한 SQL**을 생성하는지(쌍별) |
