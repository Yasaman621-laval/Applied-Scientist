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

class SalesForecastModel:
    def __init__(self, X: pd.DataFrame, target_col: str):
        self.X = X.copy()
        self.target_col = target_col

        # Feature Engineering
        self.X['dayofweek'] = self.X.index.dayofweek
        self.X['lag_1'] = self.X[self.target_col].shift(1)
        self.X['roll_mean_7'] = self.X[self.target_col].rolling(7, min_periods=1).mean()
        # Drop missing values
        self.X.dropna(inplace=True)
        # Drop rows with zero sales, lag, or rolling mean
        before = self.X.shape[0]
        self.X = self.X[
            (self.X[self.target_col] > 0) &
            (self.X['lag_1'] > 0) &
            (self.X['roll_mean_7'] > 0)
        ]
        after = self.X.shape[0]
        print(f"Rows before filtering: {before}, after filtering: {after}")
        print(self.X[[self.target_col, 'lag_1', 'roll_mean_7', 'dayofweek']].head(10))
        print(f"Number of zeros in target: {(self.X[self.target_col] == 0).sum()}")
        self.results = {}

    def safe_mape(self, truth: pd.Series, preds: pd.Series) -> float:
        mask = truth != 0
        return MAPE(truth[mask], preds[mask])

    def run_models(self, n_splits: int = 4, test_size: int = 365,
                   b_boot: int = 50, q_lo: float = 0.10, q_hi: float = 0.90):
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        features = ["dayofweek", "lag_1", "roll_mean_7"]

        for fold, (train_idx, test_idx) in enumerate(tscv.split(self.X)):
            train = self.X.iloc[train_idx]
            test = self.X.iloc[test_idx]

            print(f"\n=== Fold {fold} ===")
            print(f"Train size: {len(train)}, Test size: {len(test)}")
            print(f"Train date range: {train.index.min()} to {train.index.max()}")
            print(f"Test date range: {test.index.min()} to {test.index.max()}")
            print(f"Train min/max Sales: {train[self.target_col].min()} / {train[self.target_col].max()}")
            print(f"Test min/max Sales: {test[self.target_col].min()} / {test[self.target_col].max()}")
            print(f"Train min/max lag_1: {train['lag_1'].min()} / {train['lag_1'].max()}")
            print(f"Test min/max lag_1: {test['lag_1'].min()} / {test['lag_1'].max()}")
            print(f"Train min/max roll_mean_7: {train['roll_mean_7'].min()} / {train['roll_mean_7'].max()}")
            print(f"Test min/max roll_mean_7: {test['roll_mean_7'].min()} / {test['roll_mean_7'].max()}")

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

            # --- QRF Bootstrapped Intervals ---
            qdf = self._bootstrap_rf_model(train, test, features=features,
                                           b_boot=b_boot, q_lo=q_lo, q_hi=q_hi)
            self._store_interval(fold, "QRF", test[self.target_col], qdf)
            self._plot_interval(test.index, test[self.target_col], qdf, label=f"QRF – fold {fold}")

            print("Sample predictions (actual vs. models):")
            print(pd.DataFrame({
                "actual": test[self.target_col].values[:5],
                "Base": base_pred.values[:5],
                "LR": preds_lr[:5],
                "RF": preds_rf[:5],
                "QRF": qdf['pred'].values[:5],
            }))

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
    model = SalesForecastModel(data, "Sales Count")
    model.run_models()
    print("\nSummary Table:")
    print(pd.DataFrame(model.results))
