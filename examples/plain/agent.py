"""A toy *plain* LangGraph agent (single-shot Q&A) — the agent under evaluation.

Real agents call an LLM here; this one is deterministic so the example runs anywhere with no API
key. What matters for evaluation is the **state contract**: the fields the graph exposes, which the
config's ``state_map`` points the metrics at.

Required state fields (what the common metrics read, via ``state_map``):
    output  -> PlainState['answer']   # the final response (judge + bertscore score this)
    tokens  -> PlainState['tokens']   # optional: tokens spent (token_usage)
(``latency_ms`` is filled automatically by the prediction harness.)
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class PlainState(TypedDict, total=False):
    input: str  # the user's question (the harness sets this)
    answer: str  # the final response
    tokens: int  # tokens spent on the turn


# A tiny "knowledge base" standing in for an LLM.
_ANSWERS = {
    "How do I reset my password?": (
        "Go to Settings > Security and click 'Reset password'. You'll get an email with a link "
        "that lets you choose a new password."
    ),
    "What are your support hours?": "Our support team is available 24/7 by chat and email.",
    "Where can I download invoices?": (
        "Open Billing > Invoices, then click the download icon next to any invoice to get a PDF."
    ),
    "How do I cancel my subscription?": (
        "You can cancel anytime under Billing > Subscription by clicking 'Cancel plan'. "
        "Your plan stays active until the end of the current billing period."
    ),
    "Do you have a mobile app?": "Yes — our mobile app is available for both iOS and Android.",
}
_FALLBACK = "I'm not sure about that; let me connect you with a human agent."


def answer_node(state: PlainState) -> dict:
    response = _ANSWERS.get(state["input"], _FALLBACK)
    return {"answer": response, "tokens": len(response.split())}


def build_graph():
    graph = StateGraph(PlainState)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return graph.compile()


graph = build_graph()  # the config points at this: "examples/plain/agent.py:graph"
