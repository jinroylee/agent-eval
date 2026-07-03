"""T2S metrics: result-set soft_f1 (pre-computed), AST component_match/ast_valid, judge groundedness.

`soft_f1` and the `t2s_*` judge metrics compare/inspect PRE-COMPUTED result sets supplied in the
data — the gold rows come from the dataset (`metadata['gold_execution_result']`) and the predicted
rows from the agent's own state (`metadata['execution_result']`). No database is touched at eval time.
`component_match` / `ast_valid` are AST-only and read the SQL text.
"""

from agent_eval.core.contracts import EvalContext, MetaKey
from agent_eval.judges.backend import FunctionJudge, lexical_overlap_judge
from agent_eval.metrics.t2s import AstValid, ComponentMatch, SoftF1, T2SConsistency, T2SFaithfulness


def _ctx(output="", **meta):
    return EvalContext(input="q", output=output, metadata=meta)


# --- soft_f1: compares the predicted vs gold result sets, no DB -----------------------------------
def test_soft_f1_perfect_match():
    rows = [["Alice"], ["Carol"]]
    r = SoftF1().score(_ctx(**{MetaKey.EXECUTION_RESULT: rows, MetaKey.GOLD_EXECUTION_RESULT: rows}))
    assert r.error is None and r.score == 1.0


def test_soft_f1_partial_credit_on_superset():
    gold = [["Alice"], ["Carol"]]
    pred = [["Alice"], ["Bob"], ["Carol"]]  # a dropped filter -> extra rows -> partial, not zero
    r = SoftF1().score(_ctx(**{MetaKey.EXECUTION_RESULT: pred, MetaKey.GOLD_EXECUTION_RESULT: gold}))
    assert 0.0 < r.score < 1.0


def test_soft_f1_ignores_row_order_by_default():
    gold = [["Alice"], ["Bob"]]
    pred = [["Bob"], ["Alice"]]
    r = SoftF1().score(_ctx(**{MetaKey.EXECUTION_RESULT: pred, MetaKey.GOLD_EXECUTION_RESULT: gold}))
    assert r.score == 1.0


def test_soft_f1_disjoint_results_score_zero():
    r = SoftF1().score(_ctx(**{MetaKey.EXECUTION_RESULT: [["X"]], MetaKey.GOLD_EXECUTION_RESULT: [["Y"]]}))
    assert r.error is None and r.score == 0.0


def test_soft_f1_errors_when_a_result_set_is_missing():
    # A missing result set means the metric CANNOT run (error), not a low score.
    assert SoftF1().score(_ctx(**{MetaKey.EXECUTION_RESULT: [["A"]]})).error  # gold missing
    assert SoftF1().score(_ctx(**{MetaKey.GOLD_EXECUTION_RESULT: [["A"]]})).error  # predicted missing
    assert SoftF1().score(_ctx()).error


# --- component_match / ast_valid: AST-only, read SQL text (unchanged, no DB) ----------------------
def test_component_match_dips_on_alias():
    r = ComponentMatch().score(
        _ctx(**{MetaKey.SQL: "SELECT AVG(salary) AS a FROM emp", MetaKey.GOLD_SQL: "SELECT AVG(salary) FROM emp"})
    )
    assert 0.0 < r.score < 1.0  # same table, different projection text


def test_ast_valid():
    assert AstValid().score(_ctx(**{MetaKey.SQL: "SELECT 1"})).passed is True
    assert AstValid().score(_ctx(**{MetaKey.SQL: "SELECT FROM WHERE"})).passed is False


def test_metrics_error_without_their_inputs():
    assert AstValid().score(_ctx()).error  # no sql
    assert SoftF1().score(_ctx()).error  # no result sets


# --- judge groundedness: judge the NL answer against the predicted result set ---------------------
def test_t2s_faithfulness_judges_answer_against_result():
    judge = FunctionJudge(lexical_overlap_judge)
    ctx = _ctx(output="Alice and Carol earn over 100000", **{MetaKey.EXECUTION_RESULT: [["Alice"], ["Carol"]]})
    r = T2SFaithfulness(judge).score(ctx)
    assert r.error is None and r.score > 0.0


def test_t2s_consistency_judges_answer_against_result():
    judge = FunctionJudge(lexical_overlap_judge)
    ctx = _ctx(output="Alice and Carol earn over 100000", **{MetaKey.EXECUTION_RESULT: [["Alice"], ["Carol"]]})
    r = T2SConsistency(judge).score(ctx)
    assert r.error is None and r.score > 0.0


def test_t2s_judge_errors_without_result():
    judge = FunctionJudge(lexical_overlap_judge)
    # no execution_result -> nothing to judge the answer against -> error
    assert T2SFaithfulness(judge).score(_ctx(output="something")).error
