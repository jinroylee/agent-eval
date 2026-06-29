"""A toy *RAG* LangGraph agent (retrieve → answer) — the agent under evaluation.

Deterministic (keyword-overlap retrieval, no LLM) so the example runs anywhere. The point is the
**state contract**: a RAG agent must expose, for evaluation, both *what it retrieved* and *what it
answered*.

Required state fields (pointed at by the config's ``state_map``):
    retrieved_ids      -> ranked retrieved doc ids   (recall@k / precision@k / ndcg@k)
    retrieved_context  -> retrieved chunk *texts*     (faithfulness / consistency)
    output             -> the final answer            (llm_judge)
    tokens             -> tokens spent                 (token_usage)

``k`` is fixed for the evaluation (``defaults.k`` in the config) and the retriever uses the same
``TOP_K`` — keep them in sync so you measure the operating point you ship.
"""

from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

TOP_K = 3

# A tiny corpus: doc id -> passage text. A real agent retrieves from a vector store.
CORPUS = {
    "d1": "The Eiffel Tower is located in Paris, the capital of France.",
    "d2": "Paris is the most populous city in France and sits on the Seine river.",
    "d3": "William Shakespeare wrote the play Hamlet around the year 1600.",
    "d4": "Hamlet is one of Shakespeare's best known tragedies.",
    "d5": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "d6": "The freezing point of water is 0 degrees Celsius.",
    "d7": "Apollo 11 landed the first humans on the Moon on July 20, 1969.",
    "d8": "Neil Armstrong was the first person to walk on the Moon.",
}


class RagState(TypedDict, total=False):
    input: str  # the question (set by the harness)
    retrieved_ids: list[str]  # ranked doc ids
    retrieved_context: list[str]  # ranked doc texts
    output: str  # the final answer
    tokens: int


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve_node(state: RagState) -> dict:
    q = _tokens(state["input"])
    ranked = sorted(CORPUS, key=lambda d: len(q & _tokens(CORPUS[d])), reverse=True)
    top = ranked[:TOP_K]
    return {"retrieved_ids": top, "retrieved_context": [CORPUS[d] for d in top]}


def answer_node(state: RagState) -> dict:
    # Ground the answer in the top passage (a real agent would synthesize with an LLM).
    context = state.get("retrieved_context") or []
    answer = context[0] if context else "I couldn't find anything relevant."
    return {"output": answer, "tokens": sum(len(c.split()) for c in context) + len(answer.split())}


def build_graph():
    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


graph = build_graph()  # "examples/rag/agent.py:graph"
