//! Rust core of tickforge. The pure-Rust modules carry the logic and tests;
//! the `python` feature adds the PyO3 bindings exposed as `tickforge._core`.

pub mod bars;
pub mod book;
pub mod engine;

#[cfg(feature = "python")]
mod py {
    use crate::{bars, book, engine};
    use numpy::{IntoPyArray, PyReadonlyArray1};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    fn err(e: String) -> PyErr {
        PyValueError::new_err(e)
    }

    /// Aggregate trades into bars. `kind` is one of "time", "tick", "volume",
    /// "dollar". Returns a dict of numpy arrays.
    #[pyfunction]
    fn build_bars<'py>(
        py: Python<'py>,
        ts: PyReadonlyArray1<'py, i64>,
        price: PyReadonlyArray1<'py, f64>,
        qty: PyReadonlyArray1<'py, f64>,
        buyer_maker: PyReadonlyArray1<'py, bool>,
        kind: &str,
        threshold: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let kind = match kind {
            "time" => bars::BarKind::Time(threshold as i64),
            "tick" => bars::BarKind::Tick(threshold as u64),
            "volume" => bars::BarKind::Volume(threshold),
            "dollar" => bars::BarKind::Dollar(threshold),
            other => return Err(err(format!("unknown bar kind '{other}'"))),
        };
        let b = bars::build_bars(
            ts.as_slice()?,
            price.as_slice()?,
            qty.as_slice()?,
            buyer_maker.as_slice()?,
            kind,
        )
        .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("open_ts", b.open_ts.into_pyarray(py))?;
        d.set_item("close_ts", b.close_ts.into_pyarray(py))?;
        d.set_item("open", b.open.into_pyarray(py))?;
        d.set_item("high", b.high.into_pyarray(py))?;
        d.set_item("low", b.low.into_pyarray(py))?;
        d.set_item("close", b.close.into_pyarray(py))?;
        d.set_item("volume", b.volume.into_pyarray(py))?;
        d.set_item("dollar", b.dollar.into_pyarray(py))?;
        d.set_item("buy_volume", b.buy_volume.into_pyarray(py))?;
        d.set_item("n_trades", b.n_trades.into_pyarray(py))?;
        Ok(d)
    }

    /// Replay the trade tape against a stream of target positions.
    #[pyfunction]
    #[pyo3(signature = (trade_ts, trade_price, buyer_maker, order_ts, target, *, latency=0, taker_fee_bps=0.0, maker_fee_bps=0.0, slippage_bps=0.0, tick_size=0.01, initial_cash=0.0, maker=false))]
    #[allow(clippy::too_many_arguments)]
    fn run_backtest<'py>(
        py: Python<'py>,
        trade_ts: PyReadonlyArray1<'py, i64>,
        trade_price: PyReadonlyArray1<'py, f64>,
        buyer_maker: PyReadonlyArray1<'py, bool>,
        order_ts: PyReadonlyArray1<'py, i64>,
        target: PyReadonlyArray1<'py, f64>,
        latency: i64,
        taker_fee_bps: f64,
        maker_fee_bps: f64,
        slippage_bps: f64,
        tick_size: f64,
        initial_cash: f64,
        maker: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let p = engine::Params {
            latency,
            taker_fee_bps,
            maker_fee_bps,
            slippage_bps,
            tick_size,
            initial_cash,
            maker,
        };
        let o = engine::run(
            trade_ts.as_slice()?,
            trade_price.as_slice()?,
            buyer_maker.as_slice()?,
            order_ts.as_slice()?,
            target.as_slice()?,
            p,
        )
        .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("equity", o.equity.into_pyarray(py))?;
        d.set_item("position", o.position.into_pyarray(py))?;
        d.set_item("fill_ts", o.fill_ts.into_pyarray(py))?;
        d.set_item("fill_price", o.fill_price.into_pyarray(py))?;
        d.set_item("fill_qty", o.fill_qty.into_pyarray(py))?;
        d.set_item("fill_fee", o.fill_fee.into_pyarray(py))?;
        d.set_item("final_equity", o.final_equity)?;
        d.set_item("total_fees", o.total_fees)?;
        d.set_item("n_cancelled", o.n_cancelled)?;
        Ok(d)
    }

    fn side(s: &str) -> PyResult<book::Side> {
        match s {
            "bid" | "b" => Ok(book::Side::Bid),
            "ask" | "a" => Ok(book::Side::Ask),
            other => Err(err(format!("unknown side '{other}'"))),
        }
    }

    /// Price-level limit order book.
    #[pyclass]
    struct OrderBook {
        inner: book::OrderBook,
    }

    #[pymethods]
    impl OrderBook {
        #[new]
        fn new(tick_size: f64) -> PyResult<Self> {
            Ok(OrderBook {
                inner: book::OrderBook::new(tick_size).map_err(err)?,
            })
        }
        fn clear(&mut self) {
            self.inner.clear()
        }
        fn update(&mut self, side_: &str, price: f64, qty: f64) -> PyResult<()> {
            self.inner.update(side(side_)?, price, qty);
            Ok(())
        }
        /// Apply many (price, qty) level updates to one side.
        fn update_many(&mut self, side_: &str, levels: Vec<(f64, f64)>) -> PyResult<()> {
            let s = side(side_)?;
            for (p, q) in levels {
                self.inner.update(s, p, q);
            }
            Ok(())
        }
        fn best_bid(&self) -> Option<(f64, f64)> {
            self.inner.best_bid()
        }
        fn best_ask(&self) -> Option<(f64, f64)> {
            self.inner.best_ask()
        }
        fn mid(&self) -> Option<f64> {
            self.inner.mid()
        }
        fn spread(&self) -> Option<f64> {
            self.inner.spread()
        }
        fn microprice(&self) -> Option<f64> {
            self.inner.microprice()
        }
        fn imbalance(&self, levels: usize) -> Option<f64> {
            self.inner.imbalance(levels)
        }
        fn depth(&self, side_: &str, levels: usize) -> PyResult<Vec<(f64, f64)>> {
            Ok(self.inner.depth(side(side_)?, levels))
        }
        fn sweep_vwap(&self, side_: &str, qty: f64) -> PyResult<Option<f64>> {
            Ok(self.inner.sweep_vwap(side(side_)?, qty))
        }
        fn n_levels(&self) -> (usize, usize) {
            self.inner.n_levels()
        }
    }

    #[pymodule]
    fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(build_bars, m)?)?;
        m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
        m.add_class::<OrderBook>()?;
        Ok(())
    }
}
