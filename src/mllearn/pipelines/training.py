from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, prepare_model_table, time_split, train_hist_gb


def create_training_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_model_table,
                inputs=["renamed_data", "params:model.leakage_or_id"],
                outputs="model_table",
                name="prepare_model_table",
            ),
            node(
                func=time_split,
                inputs=["model_table", "params:model"],
                outputs=["X_train", "y_train", "X_test", "y_test"],
                name="time_split",
            ),
            node(
                func=train_hist_gb,
                inputs=["X_train", "y_train", "params:model.random_state"],
                outputs="regressor",
                name="train_hist_gb",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test", "y_train"],
                outputs="model_metrics",
                name="evaluate_model",
            ),
        ]
    )
