//! Aggregation of raw trade ticks into bars.
//!
//! Besides OHLCV the bars carry signed order flow (aggressor-side volume),
//! which is the raw material for most microstructure features.
//!
//! Point-in-time contract: `close_ts` is the first timestamp at which the bar
//! is fully known. A bar that is still forming when the input ends is dropped,
//! so appending more ticks never changes a bar that was already emitted.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BarKind {
    /// Fixed wall-clock interval, in the same unit as the input timestamps.
    Time(i64),
    /// Fixed number of trades.
    Tick(u64),
    /// Fixed traded base-asset volume.
    Volume(f64),
    /// Fixed traded quote-asset notional.
    Dollar(f64),
}

#[derive(Debug, Default, Clone)]
pub struct Bars {
    pub open_ts: Vec<i64>,
    pub close_ts: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub dollar: Vec<f64>,
    pub buy_volume: Vec<f64>,
    pub n_trades: Vec<i64>,
}

impl Bars {
    pub fn len(&self) -> usize {
        self.close.len()
    }
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

struct Acc {
    open_ts: i64,
    last_ts: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    dollar: f64,
    buy_volume: f64,
    n: i64,
}

impl Acc {
    fn start(ts: i64, p: f64) -> Self {
        Acc {
            open_ts: ts,
            last_ts: ts,
            open: p,
            high: p,
            low: p,
            close: p,
            volume: 0.0,
            dollar: 0.0,
            buy_volume: 0.0,
            n: 0,
        }
    }
    fn add(&mut self, ts: i64, p: f64, q: f64, buyer_maker: bool) {
        self.last_ts = ts;
        if p > self.high {
            self.high = p;
        }
        if p < self.low {
            self.low = p;
        }
        self.close = p;
        self.volume += q;
        self.dollar += p * q;
        if !buyer_maker {
            // The buyer was the aggressor.
            self.buy_volume += q;
        }
        self.n += 1;
    }
    fn flush(&self, close_ts: i64, out: &mut Bars) {
        out.open_ts.push(self.open_ts);
        out.close_ts.push(close_ts);
        out.open.push(self.open);
        out.high.push(self.high);
        out.low.push(self.low);
        out.close.push(self.close);
        out.volume.push(self.volume);
        out.dollar.push(self.dollar);
        out.buy_volume.push(self.buy_volume);
        out.n_trades.push(self.n);
    }
}

/// Build bars from time-ordered trades. All slices must have equal length.
///
/// Time bars are aligned to multiples of the interval; intervals with no
/// trades produce no bar (callers that need a regular grid reindex afterwards).
pub fn build_bars(
    ts: &[i64],
    price: &[f64],
    qty: &[f64],
    buyer_maker: &[bool],
    kind: BarKind,
) -> Result<Bars, String> {
    let n = ts.len();
    if price.len() != n || qty.len() != n || buyer_maker.len() != n {
        return Err("input arrays must have equal length".into());
    }
    match kind {
        BarKind::Time(i) if i <= 0 => return Err("time interval must be positive".into()),
        BarKind::Tick(0) => return Err("tick threshold must be positive".into()),
        BarKind::Volume(v) | BarKind::Dollar(v) if v.is_nan() || v <= 0.0 => {
            return Err("threshold must be positive".into());
        }
        _ => {}
    }
    let mut out = Bars::default();
    let mut acc: Option<Acc> = None;
    let mut bucket = 0i64;
    for k in 0..n {
        if k > 0 && ts[k] < ts[k - 1] {
            return Err(format!("timestamps not sorted at index {k}"));
        }
        if let BarKind::Time(interval) = kind {
            let b = ts[k].div_euclid(interval);
            if let Some(a) = &acc
                && b != bucket
            {
                a.flush((bucket + 1) * interval, &mut out);
                acc = None;
            }
            bucket = b;
            acc.get_or_insert_with(|| Acc::start(b * interval, price[k]))
                .add(ts[k], price[k], qty[k], buyer_maker[k]);
            continue;
        }
        let a = acc.get_or_insert_with(|| Acc::start(ts[k], price[k]));
        a.add(ts[k], price[k], qty[k], buyer_maker[k]);
        let done = match kind {
            BarKind::Tick(t) => a.n as u64 >= t,
            BarKind::Volume(v) => a.volume >= v,
            BarKind::Dollar(d) => a.dollar >= d,
            BarKind::Time(_) => unreachable!(),
        };
        if done {
            a.flush(a.last_ts, &mut out);
            acc = None;
        }
    }
    // The bar still forming at end-of-input is intentionally dropped.
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<bool>) {
        (
            vec![0, 10, 999, 1000, 1500, 3000, 3001],
            vec![100.0, 101.0, 99.0, 100.0, 102.0, 103.0, 104.0],
            vec![1.0, 2.0, 1.0, 1.0, 1.0, 5.0, 1.0],
            vec![false, true, false, false, true, false, false],
        )
    }

    #[test]
    fn time_bars_ohlcv_and_flow() {
        let (t, p, q, m) = sample();
        let b = build_bars(&t, &p, &q, &m, BarKind::Time(1000)).unwrap();
        // Buckets 0 and 1 are complete; bucket 3 is still forming and dropped.
        assert_eq!(b.len(), 2);
        assert_eq!(b.open_ts, vec![0, 1000]);
        assert_eq!(b.close_ts, vec![1000, 2000]);
        assert_eq!(
            (b.open[0], b.high[0], b.low[0], b.close[0]),
            (100.0, 101.0, 99.0, 99.0)
        );
        assert_eq!(b.volume[0], 4.0);
        assert_eq!(b.buy_volume[0], 2.0);
        assert_eq!(b.n_trades, vec![3, 2]);
        assert!((b.dollar[0] - (100.0 + 202.0 + 99.0)).abs() < 1e-9);
    }

    #[test]
    fn volume_bars_close_on_threshold() {
        let (t, p, q, m) = sample();
        let b = build_bars(&t, &p, &q, &m, BarKind::Volume(3.0)).unwrap();
        assert_eq!(b.volume, vec![3.0, 3.0, 5.0]);
        assert_eq!(b.close_ts, vec![10, 1500, 3000]);
    }

    #[test]
    fn tick_bars() {
        let (t, p, q, m) = sample();
        let b = build_bars(&t, &p, &q, &m, BarKind::Tick(2)).unwrap();
        assert_eq!(b.len(), 3);
        assert_eq!(b.close, vec![101.0, 100.0, 103.0]);
    }

    #[test]
    fn appending_ticks_never_rewrites_emitted_bars() {
        let (t, p, q, m) = sample();
        for kind in [
            BarKind::Time(1000),
            BarKind::Volume(3.0),
            BarKind::Dollar(250.0),
            BarKind::Tick(2),
        ] {
            let full = build_bars(&t, &p, &q, &m, kind).unwrap();
            for cut in 1..t.len() {
                let part = build_bars(&t[..cut], &p[..cut], &q[..cut], &m[..cut], kind).unwrap();
                let k = part.len();
                assert!(k <= full.len());
                assert_eq!(part.close[..], full.close[..k]);
                assert_eq!(part.close_ts[..], full.close_ts[..k]);
                assert_eq!(part.volume[..], full.volume[..k]);
            }
        }
    }

    #[test]
    fn rejects_unsorted_and_bad_params() {
        assert!(
            build_bars(
                &[2, 1],
                &[1.0, 1.0],
                &[1.0, 1.0],
                &[true, true],
                BarKind::Tick(1)
            )
            .is_err()
        );
        assert!(build_bars(&[1], &[1.0], &[1.0], &[true], BarKind::Time(0)).is_err());
        assert!(build_bars(&[1], &[1.0], &[1.0], &[true], BarKind::Volume(0.0)).is_err());
        assert!(build_bars(&[1], &[1.0, 2.0], &[1.0], &[true], BarKind::Tick(1)).is_err());
    }
}
