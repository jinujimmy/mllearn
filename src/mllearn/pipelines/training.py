from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_model_table, time_split, predict_model


def create_training_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_model_table,
                inputs=["renamed_data", "params:prepare_data.leakage_or_id"],
                outputs="model_table",
                name="prepare_model_table",
            ),
            node(
                func=time_split,
                inputs=["model_table", "params:time_split.cutoff"],
                outputs=["X_train", "y_train", "X_test", "y_test"],
                name="time_split",
            ),
            node(
                func=predict_model,
                inputs=["X_train", "y_train", "X_test"],
                outputs="predictions",
                name="predict_model",
            ),
        ]
    )
