"""The statistics the runner uses: Wilson interval, bootstrap percentile CI, clustered SE."""

import numpy as np
import pytest

from agent_eval.stats.clustered import clustered_se
from agent_eval.stats.intervals import percentile_ci, wilson_interval


def test_wilson_brackets_point_and_stays_in_unit():
    p, lo, hi = wilson_interval(3, 4)
    assert abs(p - 0.75) < 1e-9 and 0.0 <= lo < p < hi <= 1.0


def test_wilson_validates_inputs():
    with pytest.raises(ValueError):
        wilson_interval(5, 4)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_percentile_ci_is_deterministic_and_in_range():
    data = list(range(100))
    q, lo, hi = percentile_ci(data, q=0.95)
    q2, _, _ = percentile_ci(data, q=0.95)
    assert q == q2  # fixed default seed -> reproducible
    assert lo <= q <= hi and 0 <= q <= 99


def test_percentile_ci_empty_raises():
    with pytest.raises(ValueError):
        percentile_ci([], q=0.95)


def test_clustered_se_reduces_to_iid_for_singletons():
    vals = [0.1, 0.5, 0.9, 0.3, 0.7]
    mean, se = clustered_se(vals, list(range(len(vals))))
    iid = float(np.std(vals, ddof=0)) / np.sqrt(len(vals))  # sum_sg2 form == population-ish
    assert abs(mean - np.mean(vals)) < 1e-9 and se > 0 and abs(se - iid) < 0.05


def test_clustered_se_widens_with_correlated_clusters():
    # Perfectly correlated within clusters: cluster 0 all-0, cluster 1 all-1.
    vals = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    _, se_iid = clustered_se(vals, list(range(len(vals))))
    _, se_clustered = clustered_se(vals, [0, 0, 0, 0, 1, 1, 1, 1])  # two clusters
    assert se_clustered > se_iid


def test_clustered_se_single_cluster_is_nan():
    _, se = clustered_se([0.1, 0.2, 0.3], [0, 0, 0])
    assert se != se  # nan
