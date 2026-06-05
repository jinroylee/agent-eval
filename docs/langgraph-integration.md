# Evaluating & critiquing a LangGraph agent

This is the end-to-end guide: how to point agent-eval at a real LangGraph agent, both **offline**
(score it before deploy) and **at runtime** (critique it in-flight), at **whatever graph level** you
choose.

> Requires the instrumentation extra: `uv sync --extra agentevals` (pulls in `langgraph`,
> `langchain-core`). The T2S example below also wants `--extra t2s`.

## Two ways to attach

| | **Observational** (`attach: observe`) | **Active** (`attach: active`) |
|---|---|---|
| What it does | Runs the graph **unmodified**, captures per-level I/O, scores **post-hoc** | **Injects a critic node** that routes the graph (retry/fallback) in-flight |
| Touches your graph? | No | Yes (you add a node + edge) |
| Use for | Offline evaluation; production observability | The runtime critic (retry/fallback/escalate) |
| Mechanism | `astream_events(version="v2")` capture | `make_critic_node` → `Command(goto=…)` |

## A tiny example agent

A two-node text-to-SQL graph: `generate_sql` turns a question into SQL, `run_query` executes it.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    question: str
    sql: str
    db: str
    answer: str

def generate_sql(state: State):
    sql = my_text_to_sql(state["question"], hint=state.get("_critique"))  # your model
    return {"sql": sql}

def run_query(state: State):
    return {"answer": execute(state["sql"], state["db"])}  # your DB call

g = StateGraph(State)
g.add_node("generate_sql", generate_sql)
g.add_node("run_query", run_query)
g.add_edge(START, "generate_sql")
g.add_edge("generate_sql", "run_query")
g.add_edge("run_query", END)
app = g.compile()
```

## 1. Topology introspection (validate your targets up front)

Before any run, confirm your eval targets name real nodes — fail fast with the list of valid ones:

```python
from agent_eval.instrumentation.topology import introspect, validate_selector
from agent_eval.core.contracts import EvalTarget, Level

idx = introspect(app)
idx.nodes        # {'generate_sql', 'run_query'}   (__start__/__end__ excluded)
idx.edges        # [('__start__','generate_sql'), ('generate_sql','run_query'), ...]

validate_selector(idx, EvalTarget(level=Level.NODE, selector="generate_sql"))   # ok
validate_selector(idx, EvalTarget(level=Level.NODE, selector="typo"))           # raises SelectorNotFound
```

## 2. Observe — capture I/O at any level (non-intrusive)

`observe` runs the graph and returns one `EvalContext` per requested target. The same `EvalContext`
shape is produced at every level — only which fields are filled differs:

| Level | `input` | `output` |
|---|---|---|
| `GRAPH` (`selector="*"`) | the graph input | the final state |
| `NODE` | that node's input state | that node's output state |
| `TOOL` | the call args | the tool result |
| `SUBGRAPH` | boundary input | boundary state |

```python
import asyncio
from agent_eval.instrumentation.observe import observe

targets = [
    EvalTarget(level=Level.NODE,  selector="generate_sql"),  # capture the produced SQL
    EvalTarget(level=Level.GRAPH, selector="*"),             # capture the final answer
]
contexts = asyncio.run(observe(app, {"question": "How many employees?", "db": "co.sqlite"}, targets))

by_level = {c.metadata["level"]: c for c in contexts}
generated_sql = by_level["node"].output      # {'sql': 'SELECT COUNT(*) ...'}
final_state   = by_level["graph"].output     # {'question': ..., 'sql': ..., 'answer': ...}
```

## 3. Offline evaluation of the captured agent

Run the agent over your golden dataset, capture the level you care about, score it, gate it. Here we
score the **generated SQL** with **Execution Accuracy** (a deterministic, execution-grounded check —
no LLM judging correctness):

```python
from agent_eval.core.contracts import EvalContext
from agent_eval.core.gate import GatePolicy
from agent_eval.core.suite import Suite
from agent_eval.metrics.ast_query_ import ExecutionAccuracy
from agent_eval.offline.runner import evaluate

async def captured_sql(question, db):
    [node_ctx] = await observe(app, {"question": question, "db": db},
                               [EvalTarget(level=Level.NODE, selector="generate_sql")])
    return node_ctx.output["sql"]

# gold = [{"question": ..., "gold_sql": ..., "db": "co.sqlite"}, ...]
dataset = [
    EvalContext(input=row["question"], output=asyncio.run(captured_sql(row["question"], row["db"])),
                expected=row["gold_sql"], metadata={"db_ref": row["db"]})
    for row in gold
]
suite = Suite("t2s", "search", [ExecutionAccuracy()],
              GatePolicy(thresholds={"execution_accuracy": 0.8}, require_pass=["execution_accuracy"]))
print(evaluate(suite, dataset).verdict.passed)
```

In practice you'd capture once and run a whole suite of metrics. For a *non-LangGraph* agent, skip
`observe` entirely and build `EvalContext`s directly from your agent's outputs — the offline runner
doesn't care where the outputs came from.

## 4. Active — inject a runtime critic that retries

Now wire the critic **onto the graph** so a bad query is caught and regenerated before it executes.
`make_critic_node` builds a node that runs the critic and routes via `Command`:

```python
from langgraph.graph import StateGraph, START, END
from agent_eval.instrumentation.active_node import make_critic_node
from agent_eval.agents.t2s.critic import build_t2s_critic
from agent_eval.config.loader import load_config
from agent_eval.core.contracts import EvalContext

cfg = load_config("t2s.yaml")
critic = build_t2s_critic(cfg)   # tier-1: AST validity → schema linking → dry-run execution

critic_node = make_critic_node(
    critic,
    build_ctx=lambda s: EvalContext(input=s["question"], output=s["sql"], metadata={"db_ref": s["db"]}),
    on_accept="run_query",     # valid SQL → execute
    on_retry="generate_sql",   # invalid → regenerate (with critique injected into state)
    on_fallback="give_up",     # exhausted retries → safe fallback
)

g = StateGraph(State)
g.add_node("generate_sql", generate_sql)   # reads state["_critique"] on retry to repair
g.add_node("critic", critic_node)
g.add_node("run_query", run_query)
g.add_node("give_up", lambda s: {"answer": "(could not produce a valid query)"})
g.add_edge(START, "generate_sql")
g.add_edge("generate_sql", "critic")       # after generating, critique before executing
g.add_edge("run_query", END)
g.add_edge("give_up", END)
app = g.compile()                          # critic_node's Command(goto=…) handles its own routing
```

What happens at runtime:

1. `generate_sql` produces SQL → control goes to `critic`.
2. The critic runs the deterministic tier (parse → schema-link → dry-run). 
   - **valid** → `Command(goto="run_query")` (continue).
   - **invalid**, retries remain → `Command(goto="generate_sql", update={"_critic_attempt": n+1, "_critique": "<grounded error>"})`. Your `generate_sql` reads `state["_critique"]` to fix the query.
   - **invalid**, retries exhausted (or no progress) → `Command(goto="give_up")`.

The execution harness *is* the verifier here — no LLM grades correctness. The grounded error (e.g.
*"no such table: emp"*) is injected so the next attempt can actually fix it.

> Tool-call critique works the same way: build the critic with `build_orchestration_critic(cfg)`
> (BFCL-style tool-arg validity), put the critic node after your tool-calling node, and have
> `build_ctx` pull the planned tool call into `EvalContext.trajectory`. A dedicated `active_tool`
> ToolNode-wrapper is on the roadmap; `make_critic_node` covers the pattern today.

## 5. Choosing the level (declaratively)

Your config declares the target per suite; the loader turns it into an `EvalTarget`:

```yaml
suites:
  search:   {target: {level: tool,  selector: run_query,    attach: observe}, ...}
  response: {target: {level: node,   selector: explain,      attach: observe}, ...}
  scenario: {target: {level: graph,  selector: "*",          attach: observe}, ...}
```

- **graph / subgraph** → end-to-end or flow-boundary behavior (scenario tests, latency).
- **node** → a specific step's output quality (e.g. the generated SQL, the drafted answer).
- **tool** → a single tool call's args and result (validate before it runs; check what it returned).

Use **observe** to score any level non-intrusively; use **active** to actually intervene at a node or
tool. Both paths converge on the same `EvalContext` and the same metrics/critic — so a metric you
trust offline is the exact check you run in-flight.

## Recap

- `introspect` / `validate_selector` — discover the graph, fail fast on bad selectors.
- `observe` — non-intrusive capture at graph/subgraph/node/tool → `EvalContext`s → offline `evaluate`.
- `make_critic_node` — inject a `Critic` that routes the graph via `Command` (retry/fallback).
- The **same** `Critic` and metrics power both, with thresholds calibrated offline.
