"""T2S metrics: result-set soft_f1 (pre-computed), AST component_match/ast_valid, judge groundedness.

`soft_f1` and the `t2s_*` judge metrics compare/inspect PRE-COMPUTED result sets supplied in the
data — the gold rows come from the dataset (`metadata['gold_execution_result']`) and the predicted
rows from the agent's own state (`metadata['execution_result']`). No database is touched at eval time.
`component_match` / `ast_valid` are AST-only and read the SQL text.
"""

from agent_eval.core.contracts import EvalContext, MetaKey
from agent_eval.judges.backend import FunctionJudge, lexical_overlap_judge
from agent_eval.metrics.t2s import (
    AstValid,
    ComponentMatch,
    SoftF1,
    T2SConsistency,
    T2SFaithfulness,
    _digest_lines,
)


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


# --- result-set digest: a bounded statistical summary fed to the judge (not the raw rows) ----------
def test_digest_reports_shape_and_numeric_aggregates():
    lines = _digest_lines([[1, 10.0], [2, 20.0], [3, 30.0]])
    text = "\n".join(lines)
    assert "3 row(s), 2 column(s)" in text
    assert "min=1" in text and "max=3" in text  # column 1 spans 1..3
    assert "sum=60" in text and "mean=20" in text  # column 2 aggregates over ALL rows


def test_digest_lists_distinct_text_values_and_sample():
    text = "\n".join(_digest_lines([["Alice"], ["Bob"], ["Alice"]]))
    assert "text" in text
    assert "distinct=2" in text
    assert "Alice" in text and "Bob" in text


def test_digest_scalar_and_empty():
    assert "single value" in " ".join(_digest_lines([[42]]))
    assert _digest_lines([]) == ["(empty result set)"]


def test_digest_reports_nulls():
    text = "\n".join(_digest_lines([[None], [5], [7]]))
    assert "non-null=2" in text and "nulls=1" in text


def test_digest_is_bounded_for_huge_result_sets():
    lines = _digest_lines([[i] for i in range(100_000)])
    assert len(lines) < 30  # not one line per row
    assert sum(len(ln) for ln in lines) < 2000  # compact regardless of size
    text = "\n".join(lines)
    assert "100000 row(s)" in text
    assert "max=99999" in text  # aggregate still computed over the WHOLE set


def test_digest_omits_value_listing_when_distinct_exceeds_cap():
    text = "\n".join(_digest_lines([[i] for i in range(100)], max_distinct=20))
    assert "values=[" not in text  # 100 distinct > cap -> list omitted, aggregates kept
    assert "sum=" in text


def test_digest_caps_columns():
    text = "\n".join(_digest_lines([list(range(50))], max_columns=20))
    assert "more columns" in text


def test_t2s_faithfulness_feeds_bounded_digest_to_judge():
    captured = {}

    def spy(request):
        captured["context"] = request.context
        return 1.0

    rows = [[i] for i in range(10_000)]
    ctx = _ctx(output="The query returned 10000 rows.", **{MetaKey.EXECUTION_RESULT: rows})
    r = T2SFaithfulness(FunctionJudge(spy)).score(ctx)
    assert r.error is None
    joined = "\n".join(captured["context"])
    assert "10000 row(s)" in joined
    assert len(joined) < 2000  # the 10k rows were summarized, not dumped into the judge
