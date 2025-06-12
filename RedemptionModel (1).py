import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class RedemptionModel:
    def __init__(self, X: pd.DataFrame, target_col: str):
        self.X = X.copy()
        self.target_col = target_col
        self._add_time_features()
        self._add_lag_features(lag=1)
        self._add_rolling_mean(window=7)
        self.X.dropna(inplace=True)  # Drop any rows with missing feature values
        self.X = self.X[self.X[self.target_col] > 0]  # Remove rows with zero target
        self.results = {}

    def _add_time_features(self):
        idx = self.X.index
        self.X["month"] = idx.month
        self.X["quarter"] = idx.quarter
        self.X["dayofweek"] = idx.dayofweek

    def _add_lag_features(self, lag: int = 1):
        self.X[f"lag_{lag}"] = self.X[self.target_col].shift(lag)

    def _add_rolling_mean(self, window: int = 7):
        self.X[f"roll_mean_{window}"] = (
            self.X[self.target_col].rolling(window, min_periods=1).mean()
        )

    def safe_mape(self, truth: pd.Series, preds: pd.Series) -> float:
        """MAPE only where truth is not zero."""
        mask = truth != 0
        return MAPE(truth[mask], preds[mask])

    def run_models(self, n_splits: int = 4, test_size: int = 365,
                   b_boot: int = 50, q_lo: float = 0.10, q_hi: float = 0.90):
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        features = ["dayofweek", "lag_1", "roll_mean_7"]

        for fold, (train_idx, test_idx) in enumerate(tscv.split(self.X)):
            train = self.X.iloc[train_idx]
            test = self.X.iloc[test_idx]

            # --- Base Model ---
            base_pred = self._base_model(train, test)
            self._store(fold, "Base", test[self.target_col], base_pred)
            self._plot(test.index, test[self.target_col], base_pred, label=f"Base – fold {fold}")

            # --- Linear Regression with features ---
            lr = LinearRegression()
            lr.fit(train[features], train[self.target_col])
            preds_lr = lr.predict(test[features])
            self._store(fold, "Linear+Features", test[self.target_col], preds_lr)
            self._plot(test.index, test[self.target_col], preds_lr, label=f"Linear+Features – fold {fold}")

            # --- Random Forest with features ---
            rf = RandomForestRegressor(n_estimators=100, random_state=fold)
            rf.fit(train[features], train[self.target_col])
            preds_rf = rf.predict(test[features])
            self._store(fold, "RF+Features", test[self.target_col], preds_rf)
            self._plot(test.index, test[self.target_col], preds_rf, label=f"RF+Features – fold {fold}")

            # --- QRF Bootstrapped Intervals (for the same features) ---
            qdf = self._bootstrap_rf_model(train, test, features=features,
                                           b_boot=b_boot, q_lo=q_lo, q_hi=q_hi)
            self._store_interval(fold, "QRF", test[self.target_col], qdf)
            self._plot_interval(test.index, test[self.target_col], qdf, label=f"QRF – fold {fold}")

    def _base_model(self, train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        res = sm.tsa.seasonal_decompose(train[self.target_col], period=365)
        res_clip = res.seasonal.clip(lower=0)
        res_clip.index = res_clip.index.dayofyear
        avg_season = res_clip.groupby(res_clip.index).mean().to_dict()
        return pd.Series(index=test.index,
                         data=[avg_season.get(idx.dayofyear, 0)
                               for idx in test.index])

    def _bootstrap_rf_model(self, train: pd.DataFrame, test: pd.DataFrame,
                            features=None, b_boot: int = 50, q_lo: float = 0.10,
                            q_hi: float = 0.90) -> pd.DataFrame:
        if features is None:
            features = [c for c in train.columns if c not in [self.target_col]]

        y_train = train[self.target_col]
        preds = np.zeros((b_boot, len(test)))
        for b in tqdm(range(b_boot), desc="Bootstraps", leave=False):
            boot_idx = resample(train.index, replace=True)
            Xb = train.loc[boot_idx, features]
            yb = y_train.loc[boot_idx]
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                random_state=b,
                n_jobs=-1,
            )
            rf.fit(Xb, yb)
            preds[b] = rf.predict(test[features])
        y_hat = preds.mean(axis=0)
        lo = np.quantile(preds, q_lo, axis=0)
        hi = np.quantile(preds, q_hi, axis=0)
        return pd.DataFrame({
            "pred": y_hat,
            "lo": lo,
            "hi": hi
        }, index=test.index)

    def _store(self, fold, model_name, truth, pred):
        mape = self.safe_mape(truth, pd.Series(pred, index=truth.index))
        self.results.setdefault(model_name, {})[fold] = mape

    def _store_interval(self, fold, model_name, truth, qdf):
        mape = self.safe_mape(truth, qdf["pred"])
        coverage = ((truth >= qdf["lo"]) & (truth <= qdf["hi"])).mean()
        width = (qdf["hi"] - qdf["lo"]).mean()
        self.results.setdefault(model_name, {})[fold] = {
            "MAPE": mape,
            "cov": coverage,
            "w": width,
        }

    def _plot(self, idx, truth, pred, label=""):
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.scatter(idx, truth, s=6, color="grey", label="Observed")
        ax.plot(idx, pred, label=label)
        ax.legend(); plt.show()

    def _plot_interval(self, idx, truth, qdf, label=""):
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.scatter(idx, truth, s=6, color="grey", label="Observed")
        ax.plot(idx, qdf["pred"], color="C2", label=label)
        ax.fill_between(idx, qdf["lo"], qdf["hi"], color="C2", alpha=0.2,
                        label="80% PI")
        ax.legend(); plt.show()

def load_data(file: str) -> pd.DataFrame:
    df = (
        pd.read_csv(
            file,
            dtype={"_id": int, "Redemption Count": int, "Sales Count": int},
            parse_dates=["Timestamp"],
        )
        .sort_values("Timestamp")
        .set_index("Timestamp")
        .resample("1D")
        .sum()
    )
    return df

if __name__ == "__main__":
    data = load_data("Toronto Island Ferry Ticket Counts.csv")
    model = RedemptionModel(data, "Redemption Count")
    model.run_models()
    print(pd.DataFrame(model.results))




