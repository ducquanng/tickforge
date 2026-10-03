import numpy as np
import pytest

from tickforge.validation import PurgedKFold, WalkForward, assert_no_overlap


@pytest.mark.parametrize("horizon,embargo", [(0, 0), (5, 0), (30, 10)])
def test_purged_kfold_never_overlaps(horizon, embargo):
    n = 1000
    seen = []
    for train, test in PurgedKFold(5, horizon, embargo).split(n):
        assert_no_overlap(train, test, horizon)
        assert np.intersect1d(train, test).size == 0
        after = train[train > test.max()]
        if after.size:
            assert after.min() - test.max() > horizon + embargo
        seen.append(test)
    np.testing.assert_array_equal(np.sort(np.concatenate(seen)), np.arange(n))


def test_unpurged_split_is_caught():
    with pytest.raises(AssertionError):
        assert_no_overlap(np.arange(0, 100), np.arange(100, 200), horizon=5)


def test_walk_forward_trains_strictly_on_the_past():
    folds = list(WalkForward(train_size=100, test_size=20, horizon=5).split(250))
    assert len(folds) == 7
    for train, test in folds:
        assert len(train) == 100 and len(test) == 20
        assert train.max() + 5 < test.min()
        assert_no_overlap(train, test, 5)
    assert folds[1][1][0] - folds[0][1][0] == 20


def test_walk_forward_expanding():
    folds = list(WalkForward(train_size=50, test_size=10, expanding=True).split(100))
    assert all(tr[0] == 0 for tr, _ in folds)
    assert len(folds[-1][0]) > len(folds[0][0])
