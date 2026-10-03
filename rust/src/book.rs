//! Price-level (L2) limit order book.
//!
//! Prices are stored as integer ticks in a `BTreeMap`, so levels stay ordered
//! and float keys never have to be compared. Used to replay depth streams
//! captured by the live recorder.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

#[derive(Debug, Clone)]
pub struct OrderBook {
    tick: f64,
    bids: BTreeMap<i64, f64>,
    asks: BTreeMap<i64, f64>,
}

impl OrderBook {
    pub fn new(tick: f64) -> Result<Self, String> {
        if !tick.is_finite() || tick <= 0.0 {
            return Err("tick size must be positive and finite".into());
        }
        Ok(OrderBook {
            tick,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        })
    }

    fn key(&self, price: f64) -> i64 {
        (price / self.tick).round() as i64
    }

    fn px(&self, key: i64) -> f64 {
        key as f64 * self.tick
    }

    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
    }

    /// Set the resting quantity at a price level; zero removes the level.
    pub fn update(&mut self, side: Side, price: f64, qty: f64) {
        let k = self.key(price);
        let levels = match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        if qty <= 0.0 {
            levels.remove(&k);
        } else {
            levels.insert(k, qty);
        }
    }

    pub fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids.iter().next_back().map(|(k, q)| (self.px(*k), *q))
    }

    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks.iter().next().map(|(k, q)| (self.px(*k), *q))
    }

    pub fn mid(&self) -> Option<f64> {
        Some((self.best_bid()?.0 + self.best_ask()?.0) / 2.0)
    }

    pub fn spread(&self) -> Option<f64> {
        Some(self.best_ask()?.0 - self.best_bid()?.0)
    }

    /// Size-weighted mid: leans towards the side with less resting size.
    pub fn microprice(&self) -> Option<f64> {
        let (bp, bq) = self.best_bid()?;
        let (ap, aq) = self.best_ask()?;
        Some((bp * aq + ap * bq) / (bq + aq))
    }

    /// Top `levels` of one side, best first, as (price, qty).
    pub fn depth(&self, side: Side, levels: usize) -> Vec<(f64, f64)> {
        match side {
            Side::Bid => self
                .bids
                .iter()
                .rev()
                .take(levels)
                .map(|(k, q)| (self.px(*k), *q))
                .collect(),
            Side::Ask => self
                .asks
                .iter()
                .take(levels)
                .map(|(k, q)| (self.px(*k), *q))
                .collect(),
        }
    }

    /// (bid_qty - ask_qty) / (bid_qty + ask_qty) over the top `levels`.
    pub fn imbalance(&self, levels: usize) -> Option<f64> {
        let b: f64 = self.bids.values().rev().take(levels).sum();
        let a: f64 = self.asks.values().take(levels).sum();
        if b + a > 0.0 {
            Some((b - a) / (b + a))
        } else {
            None
        }
    }

    /// Average price paid by a market order of `qty` that walks the book.
    /// `side` is the side being consumed (Ask for a buy). `None` if the
    /// visible book cannot absorb the order.
    pub fn sweep_vwap(&self, side: Side, qty: f64) -> Option<f64> {
        if qty.is_nan() || qty <= 0.0 {
            return None;
        }
        let mut left = qty;
        let mut cost = 0.0;
        let mut walk = |k: &i64, q: &f64| {
            let take = left.min(*q);
            cost += take * self.px(*k);
            left -= take;
            left <= 1e-12
        };
        let filled = match side {
            Side::Ask => self.asks.iter().any(|(k, q)| walk(k, q)),
            Side::Bid => self.bids.iter().rev().any(|(k, q)| walk(k, q)),
        };
        if filled { Some(cost / qty) } else { None }
    }

    pub fn n_levels(&self) -> (usize, usize) {
        (self.bids.len(), self.asks.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book() -> OrderBook {
        let mut b = OrderBook::new(0.1).unwrap();
        b.update(Side::Bid, 100.0, 2.0);
        b.update(Side::Bid, 99.9, 5.0);
        b.update(Side::Ask, 100.1, 1.0);
        b.update(Side::Ask, 100.3, 4.0);
        b
    }

    #[test]
    fn top_of_book() {
        let b = book();
        assert_eq!(b.best_bid().unwrap().1, 2.0);
        assert!((b.best_ask().unwrap().0 - 100.1).abs() < 1e-9);
        assert!((b.mid().unwrap() - 100.05).abs() < 1e-9);
        assert!((b.spread().unwrap() - 0.1).abs() < 1e-9);
        // Less size on the ask, so the microprice sits above the mid.
        assert!(b.microprice().unwrap() > b.mid().unwrap());
    }

    #[test]
    fn update_replaces_and_removes() {
        let mut b = book();
        b.update(Side::Bid, 100.0, 7.0);
        assert_eq!(b.best_bid().unwrap().1, 7.0);
        b.update(Side::Bid, 100.0, 0.0);
        assert!((b.best_bid().unwrap().0 - 99.9).abs() < 1e-9);
        assert_eq!(b.n_levels(), (1, 2));
    }

    #[test]
    fn float_noise_maps_to_same_level() {
        let mut b = OrderBook::new(0.1).unwrap();
        b.update(Side::Ask, 0.1 + 0.2, 1.0);
        b.update(Side::Ask, 0.3, 0.0);
        assert!(b.best_ask().is_none());
    }

    #[test]
    fn imbalance_and_depth() {
        let b = book();
        assert!((b.imbalance(1).unwrap() - (2.0 - 1.0) / 3.0).abs() < 1e-12);
        assert!((b.imbalance(5).unwrap() - (7.0 - 5.0) / 12.0).abs() < 1e-12);
        assert_eq!(b.depth(Side::Bid, 5).len(), 2);
        assert!(b.depth(Side::Bid, 5)[0].0 > b.depth(Side::Bid, 5)[1].0);
    }

    #[test]
    fn sweep_walks_levels() {
        let b = book();
        let v = b.sweep_vwap(Side::Ask, 3.0).unwrap();
        assert!((v - (100.1 + 2.0 * 100.3) / 3.0).abs() < 1e-9);
        assert!(b.sweep_vwap(Side::Ask, 5.5).is_none());
        let s = b.sweep_vwap(Side::Bid, 2.0).unwrap();
        assert!((s - 100.0).abs() < 1e-9);
    }
}
