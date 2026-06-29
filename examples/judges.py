"""Real LLM-judge factories for the examples — wire a Claude judge into any config.

Point a config's ``judge.factory`` at one of these (e.g. ``examples/judges.py:claude_judge``).
Without a ``judge`` section the framework uses a deterministic token-overlap stub, so the examples
run with no API key; these factories turn the judge metrics (llm_judge, faithfulness, consistency,
t2s_faithfulness, t2s_consistency) into real LLM evaluations.

Requires the Anthropic SDK and a key:  pip install anthropic ; export ANTHROPIC_API_KEY=...
"""

from __future__ import annotations

from agent_eval.judges.backend import LLMJudge


def _claude_complete(model: str):
    """Return a ``complete(prompt) -> text`` callable backed by Claude."""
    import anthropic  # lazy: only needed when a real judge is configured

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    def complete(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            output_config={"effort": "low"},  # grading one response is a small, scoped task
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    return complete


def claude_judge(model: str = "claude-opus-4-8") -> LLMJudge:
    """A single Claude judge."""
    return LLMJudge(_claude_complete(model), name="claude")


def claude_panel() -> list[LLMJudge]:
    """A diverse-family PoLL panel (Opus + Sonnet) — averages scores, reports panel agreement.

    A diverse panel of judges correlates with humans as well as one large judge while cancelling
    any single model's self-preference bias. The loader treats a returned list as the panel.
    """
    return [
        LLMJudge(_claude_complete("claude-opus-4-8"), name="opus"),
        LLMJudge(_claude_complete("claude-sonnet-4-6"), name="sonnet"),
    ]
