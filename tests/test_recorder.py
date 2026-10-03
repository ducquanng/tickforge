import pytest

from tickforge.live.recorder import BookSync, SequenceGap

SNAP = {"lastUpdateId": 100, "bids": [["10.0", "1"], ["9.9", "2"]], "asks": [["10.1", "1"]]}


def ev(first, final, bids=(), asks=()):
    return {"U": first, "u": final, "E": 0, "b": list(bids), "a": list(asks)}


def test_stale_events_are_skipped_and_first_event_must_straddle_snapshot():
    s = BookSync(0.01)
    s.load_snapshot(SNAP)
    assert s.apply(ev(90, 100, bids=[["10.0", "0"]])) is False
    assert s.book.best_bid() == (10.0, 1.0)
    assert s.apply(ev(99, 102, bids=[["10.0", "0"]])) is True
    assert s.book.best_bid()[0] == pytest.approx(9.9)
    assert s.apply(ev(103, 104, asks=[["10.05", "3"]])) is True
    row = s.row(1, 2)
    assert row["ask"] == pytest.approx(10.05) and row["imbalance_1"] == pytest.approx(-0.2)


def test_gap_raises():
    s = BookSync(0.1)
    s.load_snapshot(SNAP)
    s.apply(ev(101, 102))
    with pytest.raises(SequenceGap):
        s.apply(ev(104, 105))


def test_requires_snapshot():
    with pytest.raises(RuntimeError):
        BookSync(0.1).apply(ev(1, 2))
