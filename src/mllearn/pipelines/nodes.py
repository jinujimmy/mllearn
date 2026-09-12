import pandas as pd
from typing import Dict

def rename_columns(df:pd.DataFrame,renaming_map: Dict[str,str]) -> pd.DataFrame:
    return df.rename(columns=renaming_map)

def create_lag_features(df:pd.DataFrame,lag_features: list[str]) -> pd.DataFrame:
    return df.assign(**{f"lag_{lag}": df[lag].shift(1) for lag in lag_features}) 

