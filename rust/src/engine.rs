//! Tick-level backtest engine.
//!
//! The engine replays the trade tape and executes a stream of target
//! positions against it. Historical top-of-book quotes are not part of the
//! free dataset, so the quote is reconstructed from the tape: a
//! seller-initiated print marks the bid, a buyer-initiated print marks the ask.
//!
//! Execution assumptions (all deliberately on the pessimistic side):
//! * An order decided at time `t` reaches the market at `t + latency`.
//! * Taker orders cross the reconstructed spread, pay `slippage_bps` on top,
//!   and pay the taker fee.
//! * Maker orders rest at the touch and fill only when a later aggressor
//!   print trades strictly *through* their price, i.e. the order is treated
//!   as last in queue. Unfilled orders are cancelled at the next decision.
//! * Order size is assumed small relative to displayed liquidity; there is no
//!   market-impact model.

#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub latency: i64,
    pub taker_fee_bps: f64,
    pub maker_fee_bps: f64,
    pub slippage_bps: f64,
    pub tick_size: f64,
    pub initial_cash: f64,
    pub maker: bool,
}

#[derive(Debug, Default, Clone)]
pub struct Outcome {
    /// Mark-to-market equity at each order's arrival time, before it acts.
    pub equity: Vec<f64>,
    /// Position held right after each order's arrival.
    pub position: Vec<f64>,
    pub fill_ts: Vec<i64>,
    pub fill_price: Vec<f64>,
    /// Signed: positive for buys.
    pub fill_qty: Vec<f64>,
    pub fill_fee: Vec<f64>,
    pub final_equity: f64,
    pub total_fees: f64,
    pub n_cancelled: u64,
}

struct Working {
    limit: f64,
    qty: f64, // signed
}

struct State {
    p: Params,
    cash: f64,
    pos: f64,
    bid: Option<f64>,
    ask: Option<f64>,
    last: Option<f64>,
    working: Option<Working>,
    out: Outcome,
}

impl State {
    fn fill(&mut self, ts: i64, price: f64, qty: f64, fee_bps: f64) {
        let fee = (price * qty).abs() * fee_bps * 1e-4;
        self.cash -= price * qty + fee;
        self.pos += qty;
        self.out.total_fees += fee;
        self.out.fill_ts.push(ts);
        self.out.fill_price.push(price);
        self.out.fill_qty.push(qty);
        self.out.fill_fee.push(fee);
    }

    fn on_trade(&mut self, ts: i64, price: f64, buyer_maker: bool) {
        if let Some(w) = &self.working {
            let through = if w.qty > 0.0 {
                buyer_maker && price < w.limit
            } else {
                !buyer_maker && price > w.limit
            };
            if through {
                let (limit, qty) = (w.limit, w.qty);
                self.working = None;
                self.fill(ts, limit, qty, self.p.maker_fee_bps);
            }
        }
        if buyer_maker {
            self.bid = Some(price);
            if let Some(a) = self.ask
                && a <= price
            {
                self.ask = Some(price + self.p.tick_size);
            }
        } else {
            self.ask = Some(price);
            if let Some(b) = self.bid
                && b >= price
            {
                self.bid = Some(price - self.p.tick_size);
            }
        }
        self.last = Some(price);
    }

    fn equity(&self) -> f64 {
        let mark = match (self.bid, self.ask) {
            (Some(b), Some(a)) => (a + b) / 2.0,
            _ => self.last.unwrap_or(0.0),
        };
        self.cash + self.pos * mark
    }

    fn on_order(&mut self, ts: i64, target: f64) {
        if self.working.take().is_some() {
            self.out.n_cancelled += 1;
        }
        let delta = target - self.pos;
        let (Some(bid), Some(ask)) = (self.bid, self.ask) else {
            return; // No two-sided market observed yet.
        };
        if delta.abs() < 1e-12 {
            return;
        }
        if self.p.maker {
            let limit = if delta > 0.0 { bid } else { ask };
            self.working = Some(Working { limit, qty: delta });
        } else {
            let slip = self.p.slippage_bps * 1e-4;
            let price = if delta > 0.0 {
                ask * (1.0 + slip)
            } else {
                bid * (1.0 - slip)
            };
            self.fill(ts, price, delta, self.p.taker_fee_bps);
        }
    }
}

/// Run a backtest. `order_ts` are decision times (sorted) and `target` the
/// desired position in base-asset units after each decision.
pub fn run(
    trade_ts: &[i64],
    trade_price: &[f64],
    buyer_maker: &[bool],
    order_ts: &[i64],
    target: &[f64],
    p: Params,
) -> Result<Outcome, String> {
    let n = trade_ts.len();
    if trade_price.len() != n || buyer_maker.len() != n {
        return Err("trade arrays must have equal length".into());
    }
    if order_ts.len() != target.len() {
        return Err("order arrays must have equal length".into());
    }
    if p.latency < 0 {
        return Err("latency must be non-negative".into());
    }
    if order_ts.windows(2).any(|w| w[1] < w[0]) {
        return Err("order timestamps not sorted".into());
    }
    if trade_ts.windows(2).any(|w| w[1] < w[0]) {
        return Err("trade timestamps not sorted".into());
    }
    if target.iter().any(|x| !x.is_finite()) {
        return Err("targets must be finite".into());
    }
    let mut s = State {
        p,
        cash: p.initial_cash,
        pos: 0.0,
        bid: None,
        ask: None,
        last: None,
        working: None,
        out: Outcome::default(),
    };
    s.out.equity.reserve(order_ts.len());
    s.out.position.reserve(order_ts.len());
    let mut j = 0usize;
    for (i, &t) in order_ts.iter().enumerate() {
        let arrive = t + p.latency;
        while j < n && trade_ts[j] <= arrive {
            s.on_trade(trade_ts[j], trade_price[j], buyer_maker[j]);
            j += 1;
        }
        s.out.equity.push(s.equity());
        s.on_order(arrive, target[i]);
        s.out.position.push(s.pos);
    }
    while j < n {
        s.on_trade(trade_ts[j], trade_price[j], buyer_maker[j]);
        j += 1;
    }
    if s.working.take().is_some() {
        s.out.n_cancelled += 1;
    }
    s.out.final_equity = s.equity();
    Ok(s.out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(maker: bool) -> Params {
        Params {
            latency: 0,
            taker_fee_bps: 0.0,
            maker_fee_bps: 0.0,
            slippage_bps: 0.0,
            tick_size: 1.0,
            initial_cash: 1000.0,
            maker,
        }
    }

    // bid 99 / ask 101, then the market moves up to 109 / 111.
    const TS: [i64; 4] = [1, 2, 10, 11];
    const PX: [f64; 4] = [99.0, 101.0, 109.0, 111.0];
    const BM: [bool; 4] = [true, false, true, false];

    #[test]
    fn taker_crosses_spread_both_ways() {
        let o = run(&TS, &PX, &BM, &[5, 20], &[1.0, 0.0], params(false)).unwrap();
        assert_eq!(o.fill_price, vec![101.0, 109.0]);
        assert_eq!(o.fill_qty, vec![1.0, -1.0]);
        assert!((o.final_equity - 1008.0).abs() < 1e-9);
        assert_eq!(o.position, vec![1.0, 0.0]);
        // Marked at mid before each order acts.
        assert!((o.equity[0] - 1000.0).abs() < 1e-9);
        assert!((o.equity[1] - (1000.0 - 101.0 + 110.0)).abs() < 1e-9);
    }

    #[test]
    fn flat_round_trip_loses_spread_and_fees() {
        let ts = [1, 2, 10, 11];
        let px = [99.0, 101.0, 99.0, 101.0];
        let mut p = params(false);
        p.taker_fee_bps = 10.0;
        p.slippage_bps = 100.0;
        let o = run(&ts, &px, &BM, &[5, 20], &[1.0, 0.0], p).unwrap();
        let buy = 101.0 * 1.01;
        let sell = 99.0 * 0.99;
        let fees = (buy + sell) * 1e-3;
        assert!((o.total_fees - fees).abs() < 1e-9);
        assert!((o.final_equity - (1000.0 - buy + sell - fees)).abs() < 1e-9);
    }

    #[test]
    fn latency_delays_the_fill() {
        let mut p = params(false);
        p.latency = 6;
        // Decided at 5, arrives at 11, after the move: pays 111 instead of 101.
        let o = run(&TS, &PX, &BM, &[5], &[1.0], p).unwrap();
        assert_eq!(o.fill_price, vec![111.0]);
        assert_eq!(o.fill_ts, vec![11]);
    }

    #[test]
    fn maker_needs_a_trade_through() {
        // Buy limit rests at the bid (99). A sell print at 99 is not enough.
        let ts = [1, 2, 6, 7, 8];
        let px = [99.0, 101.0, 99.0, 98.0, 100.0];
        let bm = [true, false, true, true, false];
        let o = run(&ts, &px, &bm, &[5], &[2.0], params(true)).unwrap();
        assert_eq!(o.fill_ts, vec![7]);
        assert_eq!(o.fill_price, vec![99.0]);
        assert_eq!(o.fill_qty, vec![2.0]);
        assert_eq!(o.n_cancelled, 0);
    }

    #[test]
    fn maker_order_is_cancelled_when_market_runs_away() {
        let o = run(&TS, &PX, &BM, &[5, 20], &[1.0, 1.0], params(true)).unwrap();
        assert!(o.fill_ts.is_empty());
        // First order cancelled by the second, second cancelled at the end.
        assert_eq!(o.n_cancelled, 2);
        assert!((o.final_equity - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn no_fill_before_two_sided_market() {
        let o = run(&TS, &PX, &BM, &[0, 1], &[1.0, 1.0], params(false)).unwrap();
        assert!(o.fill_ts.is_empty());
    }

    #[test]
    fn reconstructed_quote_never_crosses() {
        // Ask print at 101 then a bid print above it at 103.
        let o = run(
            &[1, 2, 3],
            &[99.0, 101.0, 103.0],
            &[true, false, true],
            &[4],
            &[1.0],
            params(false),
        )
        .unwrap();
        assert_eq!(o.fill_price, vec![104.0]);
    }

    #[test]
    fn input_validation() {
        assert!(run(&[2, 1], &[1.0, 1.0], &[true, true], &[], &[], params(false)).is_err());
        assert!(run(&TS, &PX, &BM, &[5, 4], &[1.0, 0.0], params(false)).is_err());
        assert!(run(&TS, &PX, &BM, &[5], &[f64::NAN], params(false)).is_err());
    }
}
