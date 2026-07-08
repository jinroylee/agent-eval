"""T2S metrics: result-set soft_f1 (pre-computed), AST component_match/ast_valid, judge groundedness.

`soft_f1` and the `t2s_*` judge metrics compare/inspect PRE-COMPUTED result sets supplied in the
data — the gold rows come from the dataset (`metadata['gold_execution_result']`) and the predicted
rows from the agent's own state (`metadata['execution_result']`). No database is touched at eval time.
Each result set is canonically an array of `dict[str, Any]` rows (`column -> value`); loosely-typed
encodings are coerced on read. `component_match` / `ast_valid` are AST-only and read the SQL text.
"""

import json

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


# --- soft_f1: compares the predicted vs gold result sets (dict rows), no DB -----------------------
def test_soft_f1_perfect_match():
    rows = [{"name": "Alice"}, {"name": "Carol"}]
    r = SoftF1().score(_ctx(**{MetaKey.EXECUTION_RESULT: rows, MetaKey.GOLD_EXECUTION_RESULT: rows}))
    assert r.error is None and r.score == 1.0


def test_soft_f1_partial_credit_on_superset():
    gold = [{"name": "Alice"}, {"name": "Carol"}]
    pred = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}]  # dropped filter -> extra rows -> partial
    r = SoftF1().score(_ctx(**{MetaKey.EXECUTION_RESULT: pred, MetaKey.GOLD_EXECUTION_RESULT: gold}))
    assert 0.0 < r.score < 1.0


def test_soft_f1_ignores_row_order_by_default():
    gold = [{"name": "Alice"}, {"name": "Bob"}]
    pred = [{"name": "Bob"}, {"name": "Alice"}]
    r = SoftF1().score(_ctx(**{MetaKey.EXECUTION_RESULT: pred, MetaKey.GOLD_EXECUTION_RESULT: gold}))
    assert r.score == 1.0


def test_soft_f1_missing_column_gives_partial_credit():
    # the atomic-fact case: each cell is a (column, value) fact; omitting a whole column from every
    # row costs recall but not precision -> 6 correct facts of 6 predicted, 8 gold -> F1 = 6/7
    gold = [
        {"date": 20251015, "day": "Thu", "alerts": 102, "downtime": 15},
        {"date": 20251015, "day": "Thu", "alerts": 204, "downtime": 10},
    ]
    pred = [
        {"date": 20251015, "day": "Thu", "alerts": 102},   # 'downtime' column omitted
        {"date": 20251015, "day": "Thu", "alerts": 204},
    ]
    r = SoftF1().score(_ctx(**{MetaKey.EXECUTION_RESULT: pred, MetaKey.GOLD_EXECUTION_RESULT: gold}))
    assert abs(r.score - 6 / 7) < 1e-9  # P=1.0, R=0.75 -> 2*1*0.75/1.75 = 0.857142...


def test_soft_f1_facts_are_column_aware():
    # each fact is (column, value): a value only counts under the SAME column, so an aliased/renamed
    # column no longer matches...
    aliased = SoftF1().score(_ctx(**{
        MetaKey.EXECUTION_RESULT: [{"employee": "Alice"}, {"employee": "Carol"}],
        MetaKey.GOLD_EXECUTION_RESULT: [{"name": "Alice"}, {"name": "Carol"}],
    }))
    assert aliased.score == 0.0
    # ...and a value placed in the wrong column is not a true positive (a bare-value bag would give 1.0)
    swapped = SoftF1().score(_ctx(**{
        MetaKey.EXECUTION_RESULT: [{"alerts": 15, "downtime": 102}],
        MetaKey.GOLD_EXECUTION_RESULT: [{"alerts": 102, "downtime": 15}],
    }))
    assert swapped.score == 0.0


def test_soft_f1_disjoint_results_score_zero():
    r = SoftF1().score(_ctx(**{
        MetaKey.EXECUTION_RESULT: [{"v": "X"}], MetaKey.GOLD_EXECUTION_RESULT: [{"v": "Y"}],
    }))
    assert r.error is None and r.score == 0.0


def test_soft_f1_errors_when_a_result_set_is_missing():
    # A missing result set means the metric CANNOT run (error), not a low score.
    assert SoftF1().score(_ctx(**{MetaKey.EXECUTION_RESULT: [{"c": "A"}]})).error  # gold missing
    assert SoftF1().score(_ctx(**{MetaKey.GOLD_EXECUTION_RESULT: [{"c": "A"}]})).error  # predicted missing
    assert SoftF1().score(_ctx()).error


# --- coercion: loosely-typed result sets are cast to the canonical dict rows on read --------------
def test_soft_f1_coerces_json_string_result_set():
    # a store that persisted the rows as a JSON string still evaluates (parsed, not rejected)
    rows = [{"name": "Alice"}, {"name": "Bob"}]
    r = SoftF1().score(_ctx(**{
        MetaKey.EXECUTION_RESULT: json.dumps(rows),   # the whole set as text
        MetaKey.GOLD_EXECUTION_RESULT: rows,
    }))
    assert r.error is None and r.score == 1.0


def test_soft_f1_coerces_positional_and_scalar_rows():
    # positional cells -> {"col1": ...}; a bare scalar row -> {"col1": scalar}; same column -> match
    r = SoftF1().score(_ctx(**{
        MetaKey.EXECUTION_RESULT: [["Alice"], "Bob"],            # a list row and a scalar row
        MetaKey.GOLD_EXECUTION_RESULT: [["Alice"], ["Bob"]],
    }))
    assert r.error is None and r.score == 1.0


def test_soft_f1_coerces_single_unwrapped_row_dict():
    r = SoftF1().score(_ctx(**{
        MetaKey.EXECUTION_RESULT: {"total": 5},                  # one row handed over without the list
        MetaKey.GOLD_EXECUTION_RESULT: [{"total": 5}],
    }))
    assert r.error is None and r.score == 1.0


def test_soft_f1_errors_on_uncastable_result_set():
    # a string that is not JSON cannot be coerced to rows -> error (not a silent 0.0)
    assert SoftF1().score(_ctx(**{
        MetaKey.EXECUTION_RESULT: "not json at all",
        MetaKey.GOLD_EXECUTION_RESULT: [{"c": "A"}],
    })).error


# --- component_match / ast_valid: AST-only, read SQL text (unchanged, no DB) ----------------------
def test_component_match_dips_on_alias():
    pred_sql = "SELECT AVG(salary) AS a FROM emp"
    gold_sql = "SELECT AVG(salary) FROM emp"
    r = ComponentMatch().score(_ctx(**{MetaKey.SQL: pred_sql, MetaKey.GOLD_SQL: gold_sql}))
    assert 0.0 < r.score < 1.0  # same table, different projection text


def test_ast_valid():
    assert AstValid().score(_ctx(**{MetaKey.SQL: "SELECT 1"})).passed is True
    assert AstValid().score(_ctx(**{MetaKey.SQL: "SELECT FROM WHERE"})).passed is False


def test_ast_valid_tolerates_escaped_whitespace_from_raw_data():
    # raw data may carry LITERAL escaped-whitespace (a backslash-n, \\n, \t) that would otherwise make
    # sqlglot choke on the stray backslash; the metric normalizes it and parses the real structure.
    assert AstValid().score(_ctx(**{MetaKey.SQL: "SELECT a\\nFROM t"})).passed is True
    assert AstValid().score(_ctx(**{MetaKey.SQL: "SELECT a\\r\\nFROM t"})).passed is True
    assert AstValid().score(_ctx(**{MetaKey.SQL: "SELECT a\\tFROM t"})).passed is True
    assert AstValid().score(_ctx(**{MetaKey.SQL: "\\nSELECT a FROM t"})).passed is True
    # a genuine syntax error still fails — normalization doesn't paper over real breakage
    assert AstValid().score(_ctx(**{MetaKey.SQL: "SELECT FROM WHERE"})).passed is False


def test_component_match_normalizes_escaped_whitespace():
    # both sides carry escape artifacts; structure still compares (same table + projection -> 1.0)
    r = ComponentMatch().score(_ctx(**{
        MetaKey.SQL: "SELECT name\\nFROM employee",
        MetaKey.GOLD_SQL: "SELECT name FROM employee",
    }))
    assert r.error is None and r.score == 1.0


def test_metrics_error_without_their_inputs():
    assert AstValid().score(_ctx()).error  # no sql
    assert SoftF1().score(_ctx()).error  # no result sets


# --- judge groundedness: judge the NL answer against the predicted result set ---------------------
def test_t2s_faithfulness_judges_answer_against_result():
    judge = FunctionJudge(lexical_overlap_judge)
    ctx = _ctx(output="Alice and Carol earn over 100000",
               **{MetaKey.EXECUTION_RESULT: [{"name": "Alice"}, {"name": "Carol"}]})
    r = T2SFaithfulness(judge).score(ctx)
    assert r.error is None and r.score > 0.0


def test_t2s_consistency_judges_answer_against_result():
    judge = FunctionJudge(lexical_overlap_judge)
    ctx = _ctx(output="Alice and Carol earn over 100000",
               **{MetaKey.EXECUTION_RESULT: [{"name": "Alice"}, {"name": "Carol"}]})
    r = T2SConsistency(judge).score(ctx)
    assert r.error is None and r.score > 0.0


def test_t2s_judge_errors_without_result():
    judge = FunctionJudge(lexical_overlap_judge)
    # no execution_result -> nothing to judge the answer against -> error
    assert T2SFaithfulness(judge).score(_ctx(output="something")).error


# --- result-set digest: a bounded statistical summary fed to the judge (not the raw rows) ----------
def test_digest_reports_shape_and_numeric_aggregates():
    lines = _digest_lines([{"id": 1, "v": 10.0}, {"id": 2, "v": 20.0}, {"id": 3, "v": 30.0}])
    text = "\n".join(lines)
    assert "3 row(s), 2 column(s)" in text
    assert "min=1" in text and "max=3" in text  # column 'id' spans 1..3
    assert "sum=60" in text and "mean=20" in text  # column 'v' aggregates over ALL rows


def test_digest_labels_columns_by_name():
    text = "\n".join(_digest_lines([{"dept": "Eng", "headcount": 3}, {"dept": "HR", "headcount": 2}]))
    assert "'dept'" in text and "'headcount'" in text  # named columns, not "Column 1/2"


def test_digest_lists_distinct_text_values_and_sample():
    text = "\n".join(_digest_lines([{"name": "Alice"}, {"name": "Bob"}, {"name": "Alice"}]))
    assert "text" in text
    assert "distinct=2" in text
    assert "Alice" in text and "Bob" in text


def test_digest_scalar_and_empty():
    assert "single value" in " ".join(_digest_lines([{"total": 42}]))
    assert _digest_lines([]) == ["(empty result set)"]


def test_digest_reports_nulls():
    text = "\n".join(_digest_lines([{"x": None}, {"x": 5}, {"x": 7}]))
    assert "non-null=2" in text and "nulls=1" in text


def test_digest_is_bounded_for_huge_result_sets():
    lines = _digest_lines([{"n": i} for i in range(100_000)])
    assert len(lines) < 30  # not one line per row
    assert sum(len(ln) for ln in lines) < 2000  # compact regardless of size
    text = "\n".join(lines)
    assert "100000 row(s)" in text
    assert "max=99999" in text  # aggregate still computed over the WHOLE set


def test_digest_omits_value_listing_when_distinct_exceeds_cap():
    text = "\n".join(_digest_lines([{"n": i} for i in range(100)], max_distinct=20))
    assert "values=[" not in text  # 100 distinct > cap -> list omitted, aggregates kept
    assert "sum=" in text


def test_digest_caps_columns():
    text = "\n".join(_digest_lines([{f"c{i}": i for i in range(50)}], max_columns=20))
    assert "more columns" in text


def test_t2s_faithfulness_feeds_bounded_digest_to_judge():
    captured = {}

    def spy(request):
        captured["context"] = request.context
        return 1.0

    rows = [{"n": i} for i in range(10_000)]
    ctx = _ctx(output="The query returned 10000 rows.", **{MetaKey.EXECUTION_RESULT: rows})
    r = T2SFaithfulness(FunctionJudge(spy)).score(ctx)
    assert r.error is None
    joined = "\n".join(captured["context"])
    assert "10000 row(s)" in joined
    assert len(joined) < 2000  # the 10k rows were summarized, not dumped into the judge
