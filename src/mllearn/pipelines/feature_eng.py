from kedro.pipeline import Pipeline, node, pipeline

from .nodes import rename_columns


def create_feature_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=rename_columns,
                inputs=["raw_data", "params:feature_engineering.renaming_columns"],
                outputs="renamed_data",
                name="rename_columns",
            ),
            # node(
            #     func=create_lag_features,
            #     inputs="renamed_data",
            #     outputs="lag_features",
            #     name="create_lag_features",
            # ),
        ]
    )
