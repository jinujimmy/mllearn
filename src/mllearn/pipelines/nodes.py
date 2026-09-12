import pandas as pd
from typing import Dict


def rename_columns(df: pd.DataFrame, renaming_map: Dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=renaming_map)


def prepare_model_table(df: pd.DataFrame, leakage_or_id: list[str]) -> pd.DataFrame:
    """Parse time, sort, drop leakage columns. Same as notebook Step 1."""
    out = df.copy()
    out["dteday"] = pd.to_datetime(out["dteday"])
    out["datetime"] = out["dteday"] + pd.to_timedelta(out["hr"], unit="h")
    out = out.sort_values("datetime").reset_index(drop=True)
    return out.drop(columns=leakage_or_id)
