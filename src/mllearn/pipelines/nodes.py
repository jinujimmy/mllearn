import pandas as pd
from typing import Dict, Tuple

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rename_columns(df: pd.DataFrame, renaming_map: Dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=renaming_map)


def create_lag_features(df: pd.DataFrame, lag_features: list[str]) -> pd.DataFrame:
    return df.assign(**{f"lag_{lag}": df[lag].shift(1) for lag in lag_features})


def prepare_model_table(df: pd.DataFrame, leakage_or_id: list[str]) -> pd.DataFrame:
    """Parse time, sort, drop leakage columns. Same as notebook Step 1."""
    out = df.copy()
    out["dteday"] = pd.to_datetime(out["dteday"])
    out["datetime"] = out["dteday"] + pd.to_timedelta(out["hr"], unit="h")
    out = out.sort_values("datetime").reset_index(drop=True)
    return out.drop(columns=leakage_or_id)


def time_split(
    df: pd.DataFrame, model_params: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """First n rows before cutoff = train. Same as notebook Step 2 (iloc)."""
    cutoff = pd.Timestamp(model_params["cutoff"])
    target = model_params["target"]
    drop_from_features = set(model_params["drop_from_features"]) | {target}
    timestamps = pd.to_datetime(df["datetime"])
    n_train = int((timestamps < cutoff).sum())
    feature_cols = [c for c in df.columns if c not in drop_from_features]
    X = df[feature_cols]
    y = df[[target]]
    return X.iloc[:n_train], y.iloc[:n_train], X.iloc[n_train:], y.iloc[n_train:]


def train_hist_gb(
    X_train: pd.DataFrame, y_train: pd.DataFrame, random_state: int
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(random_state=random_state)
    model.fit(X_train, y_train.squeeze())
    return model


def evaluate_model(
    model: HistGradientBoostingRegressor,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    y_train: pd.DataFrame,
) -> dict:
    y_true = y_test.squeeze()
    y_tr = y_train.squeeze()
    pred = model.predict(X_test)
    y_mean = float(y_tr.mean())
    pred_mean = pd.Series(y_mean, index=y_true.index)
    return {
        "train_mean_total_users": y_mean,
        "train_mean_mae": float(mean_absolute_error(y_true, pred_mean)),
        "train_mean_rmse": float(mean_squared_error(y_true, pred_mean) ** 0.5),
        "hist_gb_mae": float(mean_absolute_error(y_true, pred)),
        "hist_gb_rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_true)),
    }
