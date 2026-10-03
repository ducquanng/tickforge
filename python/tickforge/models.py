"""Models behind one small interface, so baselines and ML models are
evaluated by exactly the same code path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Model: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class Zero:
    """Predicts zero: the random-walk benchmark for returns."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


@dataclass
class Column:
    """Uses one feature as the forecast, optionally transformed and rescaled
    by a single coefficient fitted on the training set."""

    column: str
    log: bool = False
    fit_scale: bool = False
    coef_: float = 1.0
    intercept_: float = 0.0

    def _x(self, X):
        x = X[self.column].to_numpy(dtype=float)
        return np.log(np.clip(x, 1e-7, None)) if self.log else x

    def fit(self, X, y):
        if self.fit_scale:
            x, yv = self._x(X), y.to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(yv)
            self.coef_, self.intercept_ = np.polyfit(x[ok], yv[ok], 1)
        return self

    def predict(self, X):
        return self.intercept_ + self.coef_ * np.nan_to_num(self._x(X))


@dataclass
class HAR:
    """Heterogeneous autoregressive volatility model (Corsi, 2009), in logs:
    regress future log-RV on log-RV over short, medium and long windows."""

    columns: tuple[str, ...] = ("rv_30", "rv_240", "rv_1440")
    model_: LinearRegression = field(default_factory=LinearRegression, repr=False)

    def _x(self, X):
        return np.log(np.clip(X[list(self.columns)].to_numpy(dtype=float), 1e-7, None))

    def fit(self, X, y):
        self.model_.fit(self._x(X), y.to_numpy())
        return self

    def predict(self, X):
        return self.model_.predict(self._x(X))


@dataclass
class RidgeModel:
    alpha: float = 10.0

    def fit(self, X, y):
        self.fill_ = X.median()
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.model_.fit(X.fillna(self.fill_), y)
        return self

    def predict(self, X):
        return self.model_.predict(X.fillna(self.fill_))


@dataclass
class GBM:
    """LightGBM regressor. Hyperparameters are fixed up front and deliberately
    conservative (shallow trees, strong subsampling); they are never tuned on
    test data."""

    params: dict = field(default_factory=dict)
    n_estimators: int = 300
    seed: int = 0

    def fit(self, X, y):
        p = {
            "objective": "regression",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 200,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_lambda": 10.0,
            "verbose": -1,
            "n_jobs": 2,
            "seed": self.seed,
        }
        p.update(self.params)
        self.model_ = lgb.LGBMRegressor(n_estimators=self.n_estimators, **p).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def importance(self) -> pd.Series:
        b = self.model_.booster_
        return pd.Series(b.feature_importance("gain"), index=b.feature_name()).sort_values(ascending=False)
