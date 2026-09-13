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


MODEL_COLORS = {
    "hist_gb": "#1f77b4",
    "catboost": "#ff7f0e",
    "random_forest": "#2ca02c",
}
PRED_LEGEND = {
    "hist_gb": "predicted-HG",
    "catboost": "predicted-CB",
    "random_forest": "predicted-RF",
}
DISPLAY_NAMES = {
    "hist_gb": "HistGradient",
    "catboost": "Catboost",
    "random_forest": "RF",
}


def _test_error_frame(
    model_table: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    predictions: pd.DataFrame,
    cutoff: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test rows with actual/pred, plus the first week after cutoff."""
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
    return test, week


def _draw_week_and_hour(
    ax_week, ax_hour, week: pd.DataFrame, test: pd.DataFrame, heading: str
) -> None:
    ax_week.plot(week["datetime"], week["actual"], label="actual")
    ax_week.plot(week["datetime"], week["pred"], label="predicted")
    ax_week.set_title(f"{heading} Charts")
    ax_week.set_ylabel("riders per hour")
    ax_week.legend()

    err_by_hr = test.groupby("hr")["abs_err"].mean()
    ax_hour.bar(err_by_hr.index, err_by_hr.values)
    ax_hour.set_title("mean |error| by hour of day")
    ax_hour.set_xlabel("hr")
    ax_hour.set_ylabel("MAE (riders)")
    ax_hour.set_xticks(range(24))


def plot_test_errors(
    model_table: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    predictions: pd.DataFrame,
    cutoff: str,
    name: str,
) -> Figure:
    """Notebook Step 4: first test week actual vs pred, then MAE by hour."""
    test, week = _test_error_frame(model_table, X_test, y_test, predictions, cutoff)
    heading = DISPLAY_NAMES.get(name, name)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    _draw_week_and_hour(axes[0], axes[1], week, test, heading)
    mae = float(mean_absolute_error(test["actual"], test["pred"]))
    rmse = float(mean_squared_error(test["actual"], test["pred"]) ** 0.5)
    fig.suptitle(
        f"{heading} Scores  |  MAE={mae:.1f}  RMSE={rmse:.1f}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    return fig


def compare_models(
    model_table: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    cutoff: str,
    predictions_hist_gb: pd.DataFrame,
    predictions_catboost: pd.DataFrame,
    predictions_random_forest: pd.DataFrame,
    metrics_hist_gb: dict,
    metrics_catboost: dict,
    metrics_random_forest: dict,
) -> tuple[dict, Figure]:
    """One PNG: overlay actual + three predictions, then hour errors in the same colors."""
    rows = [metrics_hist_gb, metrics_catboost, metrics_random_forest]
    preds = [
        predictions_hist_gb,
        predictions_catboost,
        predictions_random_forest,
    ]
    comparison = {row["name"]: {"mae": row["mae"], "rmse": row["rmse"]} for row in rows}

    frames = [
        _test_error_frame(model_table, X_test, y_test, pred, cutoff)
        for pred in preds
    ]
    test0 = frames[0][0]
    test_start = pd.to_datetime(test0["datetime"].min()).date()
    test_end = pd.to_datetime(test0["datetime"].max()).date()
    n_days = (pd.Timestamp(test_end) - pd.Timestamp(test_start)).days + 1

    fig = plt.figure(figsize=(12, 9.5), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[3.4, 3.4, 1.1])
    ax_week = fig.add_subplot(gs[0])
    ax_hour = fig.add_subplot(gs[1])
    ax_scores = fig.add_subplot(gs[2])

    daily_actual = test0.groupby(test0["datetime"].dt.normalize())["actual"].mean()
    ax_week.plot(
        daily_actual.index,
        daily_actual.to_numpy(),
        color="#222222",
        linewidth=2,
        label="actual",
    )
    for metrics, (test, _) in zip(rows, frames):
        name = metrics["name"]
        daily_pred = test.groupby(test["datetime"].dt.normalize())["pred"].mean()
        ax_week.plot(
            daily_pred.index,
            daily_pred.to_numpy(),
            color=MODEL_COLORS[name],
            label=PRED_LEGEND[name],
        )
    ax_week.set_ylabel("mean riders / hour")
    ax_week.set_title(
        f"Test daily trend ({test_start} to {test_end}, {n_days} days): "
        "actual vs predicted-HG, predicted-CB, predicted-RF"
    )
    ax_week.legend(loc="upper right", ncol=2)

    hours = list(range(24))
    bar_width = 0.25
    offsets = (-bar_width, 0.0, bar_width)
    for offset, metrics, (test, _) in zip(offsets, rows, frames):
        name = metrics["name"]
        err_by_hr = test.groupby("hr")["abs_err"].mean().reindex(hours)
        ax_hour.bar(
            [h + offset for h in hours],
            err_by_hr.to_numpy(),
            width=bar_width,
            color=MODEL_COLORS[name],
            label=PRED_LEGEND[name],
        )
    ax_hour.set_xticks(hours)
    ax_hour.set_xlabel("hr")
    ax_hour.set_ylabel("MAE (riders)")
    ax_hour.set_title(
        f"MAE by hour of day — full test set ({n_days} days, same colors as the trend)"
    )
    ax_hour.legend(loc="upper right")

    ax_scores.axis("off")
    ax_scores.set_title("Scores", fontweight="bold", loc="left")
    table = ax_scores.table(
        cellText=[
            [PRED_LEGEND[row["name"]], row["mae"], row["rmse"]] for row in rows
        ],
        colLabels=["model", "MAE", "RMSE"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 1.6)
    for i, row in enumerate(rows):
        table[(i + 1, 0)].get_text().set_color(MODEL_COLORS[row["name"]])
        table[(i + 1, 0)].get_text().set_fontweight("bold")

    return comparison, fig

