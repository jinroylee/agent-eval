# LLM-as-a-judge

Four metrics use an LLM judge: `llm_judge` (overall quality), `faithfulness` / `consistency` (RAG
groundedness), and `t2s_faithfulness` / `t2s_consistency` (T2S groundedness). They share one small
backend interface, so you wire a model once and every judge metric uses it.

> **Confine judges to subjective quality and groundedness — never objective correctness.** T2S query
> correctness is gated on *execution* (`soft_f1`), and a judge never grades the answer that produced
> it. This is a deliberate design rule, not an oversight.

## The interface

A metric builds a structured `JudgeRequest`; a backend returns a `JudgeVerdict`:

```python
@dataclass(frozen=True)
class JudgeRequest:
    instruction: str                 # the rubric — what to grade and how
    question: str = ""               # the user input, if relevant
    response: str = ""               # the response under test (graded)
    reference: str = ""              # gold answer, if grading against ground truth
    context: Sequence[str] = ()      # evidence (retrieved chunks, executed rows)
    scale: tuple[int, int] = (1, 5)  # integer score range to ask for

@dataclass(frozen=True)
class JudgeVerdict:
    score: float                     # normalized to [0, 1]
    confidence: float | None = None
    reason: str = ""
```

Three backends ship (`agent_eval.judges.backend`):

- **`LLMJudge(complete, name)`** — provider-agnostic. You pass a `complete(prompt) -> text` callable,
  so any model works without this package depending on any SDK. It renders the request into a prompt
  asking for `SCORE: <int>` + `REASON: …`, then parses and normalizes the score.
- **`FunctionJudge(fn)`** — wraps `fn(request) -> float | JudgeVerdict`, for deterministic/offline
  judging and tests.
- **`lexical_overlap_judge`** — a ready-made deterministic stub (token overlap) so the examples run
  with **no API key**. It's a stand-in, not a real judge — scores are crude.

When no judge is configured, the judge metrics fall back to the lexical stub.

## Wiring a real Claude judge

Point your config's `judge.factory` at a function returning a backend:

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

A runnable version (single judge + a PoLL panel) ships at [`../examples/judges.py`](../examples/judges.py).

```bash
uv pip install anthropic && export ANTHROPIC_API_KEY=...
```

## Panel of judges (PoLL)

A small, diverse panel of judges correlates with human ratings as well as a single large judge,
costs less, and cancels the self-preference bias a lone judge can't see in itself. Return a **list**
of backends from your factory and the framework averages their scores, reporting panel agreement
(`1 − spread`) as the verdict's `confidence`:

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

The first backend is the primary; the rest form the panel. Low agreement on an item flags it for a
human to look at — it doesn't change the gate automatically.

## The rubrics

Each judge metric sends a fixed, purpose-built rubric (override `llm_judge`'s via its `criteria`
param):

| metric | grades |
|---|---|
| `llm_judge` | Completeness, Clarity, Usefulness, Relevance, Friendliness (vs. the reference if one is given) |
| `faithfulness` | every claim in the answer is **supported by** the retrieved chunks |
| `consistency` | the answer does **not contradict** the retrieved chunks |
| `t2s_faithfulness` | the NL answer faithfully reports the **executed query result** |
| `t2s_consistency` | the NL answer does not contradict the executed query result |
