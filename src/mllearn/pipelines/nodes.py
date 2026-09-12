import pandas as pd
from typing import Dict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rename_columns(df: pd.DataFrame, renaming_map: Dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=renaming_map)


def prepare_model_table(df: pd.DataFrame, leakage_or_id: list[str]) -> pd.DataFrame:
    """Parse time, sort, drop leakage columns. Same as notebook Step 1."""
    out = df.copy()
    out["dteday"] = pd.to_datetime(out["dteday"])
    out["datetime"] = out["dteday"] + pd.to_timedelta(out["hr"], unit="h")
    out = out.sort_values("datetime").reset_index(drop=True)
    return out.drop(columns=leakage_or_id)


def time_split(
    df: pd.DataFrame, cutoff: str, target: str = "total_users"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered split. Same as notebook Step 2: rows before cutoff = train."""
    times = pd.to_datetime(df["datetime"])
    n_train = int((times < pd.Timestamp(cutoff)).sum())
    drop_from_x = {target, "datetime", "dteday"}
    feature_cols = [c for c in df.columns if c not in drop_from_x]
    X = df[feature_cols]
    y = df[[target]]
    return (
        X.iloc[:n_train].reset_index(drop=True),
        y.iloc[:n_train].reset_index(drop=True),
        X.iloc[n_train:].reset_index(drop=True),
        y.iloc[n_train:].reset_index(drop=True),
    )


def predict_model(
    X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame
) -> pd.DataFrame:
    """Fit HGB on train, predict on test. Score (MAE/RMSE) is the next process."""
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train.squeeze())
    pred = model.predict(X_test)
    return pd.DataFrame({"predicted_users": pred})


def score_model(y_true: pd.DataFrame, y_hat: pd.DataFrame) -> dict:
    """Test MAE/RMSE. Same numbers as the notebook score() helper."""
    mae = float(mean_absolute_error(y_true.squeeze(), y_hat.squeeze()))
    rmse = float(mean_squared_error(y_true.squeeze(), y_hat.squeeze()) ** 0.5)
    print(f"{'HistGradientBoosting':20s}  MAE={mae:6.1f}  RMSE={rmse:6.1f}")
    return {"mae": round(mae, 1), "rmse": round(rmse, 1)}
