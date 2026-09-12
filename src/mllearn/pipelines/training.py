from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    prepare_model_table,
    time_split,
    predict_model,
    score_model,
    plot_test_errors,
)


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
            node(
                func=score_model,
                inputs=["y_test", "predictions"],
                outputs="metrics",
                name="score_model",
            ),
            node(
                func=plot_test_errors,
                inputs=[
                    "model_table",
                    "X_test",
                    "y_test",
                    "predictions",
                    "params:time_split.cutoff",
                ],
                outputs="error_plots",
                name="plot_test_errors",
            ),
        ]
    )
