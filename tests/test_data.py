import datetime as dt
import hashlib
import io
import zipfile

import numpy as np
import pytest

from tickforge.data import binance
from tickforge.data.store import TickStore

FUTURES = b"agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n2,78549.5,0.016,3,7,1788220800016,true\n1,78549.6,0.025,1,2,1788220800003,false\n"
SPOT = b"1,78581.30000000,0.00012000,1,1,1788220800322740,False,True\n2,78581.31000000,0.01700000,2,2,1788220800491646,True,True\n"
DEPTH = b"timestamp,percentage,depth,notional\n2026-09-01 00:00:04,-1.00,2393.254,187143700.0\n2026-09-01 00:00:04,1.00,1895.927,149678600.0\n"


def test_futures_and_spot_formats_normalise_to_one_schema():
    f, s = binance.parse_agg_trades(FUTURES), binance.parse_agg_trades(SPOT)
    assert list(f.columns) == list(s.columns) == ["ts", "price", "qty", "buyer_maker", "agg_id"]
    assert f["ts"].tolist() == [1788220800003, 1788220800016]  # sorted
    assert f["buyer_maker"].tolist() == [False, True]
    assert s["ts"].tolist() == [1788220800322, 1788220800491]  # microseconds -> ms
    assert s["buyer_maker"].tolist() == [False, True]
    assert f.dtypes.equals(s.dtypes)


def test_book_depth_timestamps_are_utc_millis():
    d = binance.parse_book_depth(DEPTH)
    assert d["ts"].tolist() == [1788220804000] * 2
    assert d["pct"].tolist() == [-1.0, 1.0]


def _zip(raw: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", raw)
    return buf.getvalue()


def test_fetch_verifies_checksum(monkeypatch):
    blob = _zip(FUTURES)
    good = (hashlib.sha256(blob).hexdigest() + "  file.zip\n").encode()
    client = binance.BinancePublicData()
    monkeypatch.setattr(client, "_get", lambda url: good if url.endswith(".CHECKSUM") else blob)
    assert len(client.fetch("aggTrades", "BTCUSDT", dt.date(2026, 9, 1))) == 2
    monkeypatch.setattr(client, "_get", lambda url: b"deadbeef  file.zip" if url.endswith(".CHECKSUM") else blob)
    with pytest.raises(binance.ChecksumError):
        client.fetch("aggTrades", "BTCUSDT", dt.date(2026, 9, 1))
    monkeypatch.setattr(client, "_get", lambda url: None)
    assert client.fetch("aggTrades", "BTCUSDT", dt.date(2026, 9, 1)) is None


def test_url_layout():
    u = binance.BinancePublicData("um").url("aggTrades", "BTCUSDT", dt.date(2026, 9, 1))
    assert u == "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-09-01.zip"


def test_store_round_trip_is_idempotent_and_queryable(tmp_path):
    store = TickStore(tmp_path)
    df = binance.parse_agg_trades(FUTURES)

    class Fake:
        calls = 0

        def fetch(self, kind, symbol, day):
            Fake.calls += 1
            return df if day.day == 1 else None

    a, b = dt.date(2026, 9, 1), dt.date(2026, 9, 2)
    assert store.ingest("aggTrades", "BTCUSDT", a, b, client=Fake()) == {"written": 1, "skipped": 0, "missing": 1}
    assert store.ingest("aggTrades", "BTCUSDT", a, b, client=Fake()) == {"written": 0, "skipped": 1, "missing": 1}
    back = store.read("aggTrades", "BTCUSDT", a, b)
    np.testing.assert_array_equal(back["ts"], df["ts"])
    q = store.sql("SELECT symbol, count(*) AS n FROM read_parquet('{root}/um/aggTrades/*/*.parquet', hive_partitioning=true) GROUP BY 1")
    assert q.to_dict("records") == [{"symbol": "BTCUSDT", "n": 2}]
    with pytest.raises(FileNotFoundError):
        store.read("aggTrades", "ETHUSDT", a, b)
