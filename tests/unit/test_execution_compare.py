"""Tests for the explicit result-set comparison policy (row/dup/NULL/column-order/float-tol)."""

from agent_eval.execution.compare import ResultSetPolicy, result_sets_equal, soft_f1


def test_row_order_ignored_by_default():
    p = ResultSetPolicy()
    assert result_sets_equal([(1, 2), (3, 4)], [(3, 4), (1, 2)], p)


def test_strict_row_order():
    p = ResultSetPolicy(row_order="strict")
    assert not result_sets_equal([(1,), (2,)], [(2,), (1,)], p)
    assert result_sets_equal([(1,), (2,)], [(1,), (2,)], p)


def test_duplicates_keep_vs_dedup():
    assert not result_sets_equal([(1,), (1,)], [(1,)], ResultSetPolicy(duplicates="keep"))
    assert result_sets_equal([(1,), (1,)], [(1,)], ResultSetPolicy(duplicates="dedup"))


def test_float_tolerance():
    p = ResultSetPolicy(float_tolerance=1e-6)
    assert result_sets_equal([(1.0000001,)], [(1.0,)], p)
    assert not result_sets_equal([(1.1,)], [(1.0,)], p)


def test_int_and_float_compare_equal():
    assert result_sets_equal([(6,)], [(6.0,)], ResultSetPolicy())


def test_nulls_distinct_from_zero():
    p = ResultSetPolicy()
    assert result_sets_equal([(None, 1)], [(None, 1)], p)
    assert not result_sets_equal([(None, 1)], [(0, 1)], p)


def test_column_order_ignore_vs_strict():
    assert result_sets_equal([(1, 2)], [(2, 1)], ResultSetPolicy(column_order="ignore"))
    assert not result_sets_equal([(1, 2)], [(2, 1)], ResultSetPolicy(column_order="strict"))


def test_soft_f1():
    p = ResultSetPolicy()
    assert soft_f1([(1, 2)], [(1, 2)], p) == 1.0
    assert soft_f1([(9,)], [(1,)], p) == 0.0
    assert abs(soft_f1([(1, 2, 3)], [(1, 2, 4)], p) - 2 / 3) < 1e-9  # 2 shared of 3
    assert soft_f1([], [], p) == 1.0  # both empty
