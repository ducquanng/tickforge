"""Forward-looking targets. These are the only functions allowed to look ahead.

A label at row ``t`` with horizon ``h`` depends on bars ``t+1 .. t+h``; the
validation splitters need ``h`` to purge overlapping samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Log return from the close of bar ``t`` to the close of bar ``t+horizon``."""
    logp = np.log(close)
    return logp.shift(-horizon) - logp


def forward_log_rv(close: pd.Series, horizon: int) -> pd.Series:
    """Log realised volatility of one-bar returns over bars ``t+1 .. t+horizon``."""
    r2 = np.log(close).diff() ** 2
    fwd = r2.rolling(horizon).mean().shift(-horizon)
    return 0.5 * np.log(fwd.clip(lower=1e-14))
