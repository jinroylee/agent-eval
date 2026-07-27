# LLM-as-a-judge

Five metrics use an LLM judge: `llm_judge` (overall quality), `faithfulness` / `t2s_faithfulness`
(groundedness), and `consistency` / `t2s_consistency` (self-consistency across repeated runs of the
same query, judged pairwise). They share one small
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
    system_prompt: str = ""          # persona/framing preamble; "" → the backend's default

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

## The system prompt

Every judge prompt opens with a persona/framing preamble — by default
`"You are a strict, impartial evaluator."`. You can replace it wherever a judge is configured
(most specific wins):

1. **Per metric** — `params: {system_prompt: ...}` on any judge-based metric, or
   `defaults.system_prompt` in the config to set it for all judge metrics at once.
2. **Per backend** — `LLMJudge(complete, system_prompt=...)` when writing a judge factory.
3. The built-in default.

The customization replaces the preamble only: the metric's rubric (`instruction`) and the
`SCORE:`/`REASON:` format contract are always appended after it, so a custom system prompt can
never break score parsing. `FunctionJudge` and the offline stub see `request.system_prompt` but
ignore it. On the API server, pass the `system_prompt` request param, or set
`AGENT_EVAL_JUDGE_SYSTEM_PROMPT` as the server-wide default (see [api-server.md](api-server.md)).

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
| `llm_judge` | Completeness, Clarity, Usefulness, Relevance, Friendliness — each criterion judged **separately** by default (per-criterion scores in `detail['criteria']`, overall = mean; a plain-string `criteria` gives one holistic score). Vs. the reference if one is given. |
| `faithfulness` | every claim in the answer is **supported by** the retrieved chunks |
| `consistency` | repeated runs of the same query give **the same answer** (pairwise) |
| `t2s_faithfulness` | the NL answer faithfully reports the **executed query result** |
| `t2s_consistency` | repeated runs of the same query generate **equivalent SQL** (pairwise) |
