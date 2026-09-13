import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor
from catboost import CatBoostRegressor
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
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    random_state: int,
) -> pd.DataFrame:
    """Fit the model named in YAML, predict on test."""
    if model_name == "hist_gb":
        model = HistGradientBoostingRegressor(random_state=random_state)
    elif model_name == "catboost":
        model = CatBoostRegressor(random_state=random_state, verbose=0)
    elif model_name == "random_forest":
        model = RandomForestRegressor(random_state=random_state)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.fit(X_train, y_train.squeeze())
    pred = model.predict(X_test)
    return pd.DataFrame({"predicted_users": pred})


def score_model(y_true: pd.DataFrame, y_hat: pd.DataFrame, name: str) -> dict:
    """Test MAE/RMSE. Same numbers as the notebook score() helper."""
    mae = float(mean_absolute_error(y_true.squeeze(), y_hat.squeeze()))
    rmse = float(mean_squared_error(y_true.squeeze(), y_hat.squeeze()) ** 0.5)
    print(f"{name:20s}  MAE={mae:6.1f}  RMSE={rmse:6.1f}")
    return {"name": name, "mae": round(mae, 1), "rmse": round(rmse, 1)}


def plot_test_errors(
    model_table: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    predictions: pd.DataFrame,
    cutoff: str,
    name: str,
) -> Figure:
    """Notebook Step 4: first test week actual vs pred, then MAE by hour."""
    times = pd.to_datetime(model_table["datetime"])
    test_times = times[times >= pd.Timestamp(cutoff)].reset_index(drop=True)
    test = pd.DataFrame(
        {
            "datetime": test_times.to_numpy(),
            "hr": X_test["hr"].to_numpy(),
            "actual": y_test.squeeze().to_numpy(),
            "pred": predictions.squeeze().to_numpy(),
        }
    )
    test["abs_err"] = (test["actual"] - test["pred"]).abs()
    week_end = pd.Timestamp(cutoff) + pd.Timedelta(days=7)
    week = test[(test["datetime"] >= cutoff) & (test["datetime"] < week_end)]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(week["datetime"], week["actual"], label="actual")
    axes[0].plot(week["datetime"], week["pred"], label="predicted")
    axes[0].set_title(f"{name}: test week 1–7 Oct 2012 actual vs predicted")
    axes[0].set_ylabel("riders per hour")
    axes[0].legend()

    err_by_hr = test.groupby("hr")["abs_err"].mean()
    axes[1].bar(err_by_hr.index, err_by_hr.values)
    axes[1].set_title(f"{name}: test set mean |error| by hour of day")
    axes[1].set_xlabel("hr")
    axes[1].set_ylabel("MAE (riders)")
    axes[1].set_xticks(range(24))
    mae = float(mean_absolute_error(test["actual"], test["pred"]))
    rmse = float(mean_squared_error(test["actual"], test["pred"]) ** 0.5)
    fig.suptitle(
        f"{name}  |  MAE={mae:.1f}  RMSE={rmse:.1f}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    return fig


def compare_models(
    metrics_hist_gb: dict, metrics_catboost: dict, metrics_random_forest: dict
) -> tuple[dict, Figure]:
    """JSON scores plus one PNG table naming hist_gb, catboost, and random_forest."""
    rows = [metrics_hist_gb, metrics_catboost, metrics_random_forest]
    comparison = {row["name"]: {"mae": row["mae"], "rmse": row["rmse"]} for row in rows}
    names = ", ".join(row["name"] for row in rows)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")
    ax.set_title(
        f"Test MAE / RMSE: {names}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    table = ax.table(
        cellText=[[row["name"], row["mae"], row["rmse"]] for row in rows],
        colLabels=["model", "MAE", "RMSE"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 2)
    fig.tight_layout()
    return comparison, fig

