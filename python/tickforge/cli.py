"""Command line entry point: ``tickforge ingest | run | record``."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt

from tickforge.data.store import TickStore

TICK_SIZES = {"BTCUSDT": 0.1, "ETHUSDT": 0.01}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tickforge")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="download Binance public data into the Parquet store")
    ing.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ing.add_argument("--kinds", nargs="+", default=["aggTrades", "bookDepth"])
    ing.add_argument("--start", type=dt.date.fromisoformat, required=True)
    ing.add_argument("--end", type=dt.date.fromisoformat, required=True)
    ing.add_argument("--market", default="um")
    ing.add_argument("--root", default="data")

    run = sub.add_parser("run", help="run the walk-forward experiment and write a report")
    run.add_argument("--symbol", default="BTCUSDT")
    run.add_argument("--cross-symbol", default="ETHUSDT")
    run.add_argument("--start", type=dt.date.fromisoformat, required=True)
    run.add_argument("--end", type=dt.date.fromisoformat, required=True)
    run.add_argument("--train-days", type=int, default=21)
    run.add_argument("--tick-size", type=float, default=None)
    run.add_argument("--root", default="data")
    run.add_argument("--out", default="results")

    rec = sub.add_parser("record", help="record the live spot L2 order book")
    rec.add_argument("--symbol", default="BTCUSDT")
    rec.add_argument("--seconds", type=float, default=60)
    rec.add_argument("--out", default="data/live/book.parquet")

    a = ap.parse_args(argv)
    if a.cmd == "ingest":
        store = TickStore(a.root)
        for sym in a.symbols:
            for kind in a.kinds:
                print(sym, kind, store.ingest(kind, sym, a.start, a.end, a.market))
    elif a.cmd == "run":
        from tickforge.experiment import ExperimentConfig, run_experiment
        from tickforge.report import write_report

        cfg = ExperimentConfig(
            symbol=a.symbol,
            cross_symbol=a.cross_symbol,
            start=a.start,
            end=a.end,
            train_days=a.train_days,
            tick_size=a.tick_size or TICK_SIZES.get(a.symbol, 0.01),
        )
        res = run_experiment(TickStore(a.root), cfg)
        path = write_report(res, a.out)
        print(path.read_text())
    elif a.cmd == "record":
        from tickforge.live.recorder import record

        df = asyncio.run(record(a.symbol, a.seconds, a.out))
        print(f"recorded {len(df)} book updates to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
