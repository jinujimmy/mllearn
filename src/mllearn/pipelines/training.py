from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    compare_models,
    plot_test_errors,
    predict_model,
    prepare_model_table,
    score_model,
    time_split,
)

MODEL_NAMES = ("hist_gb", "catboost", "random_forest")


def _model_branch(model_name: str) -> Pipeline:
    """Predict, score, and plot one learner. Dataset names are prefixed with model_name."""

    def _predict(X_train, y_train, X_test, random_state):
        return predict_model(X_train, y_train, X_test, model_name, random_state)

    _predict.__name__ = f"predict_{model_name}"

    def _score(y_true, y_hat):
        return score_model(y_true, y_hat, model_name)

    _score.__name__ = f"score_{model_name}"

    def _plot(model_table, X_test, y_test, predictions, cutoff):
        return plot_test_errors(
            model_table, X_test, y_test, predictions, cutoff, model_name
        )

    _plot.__name__ = f"plot_{model_name}"

    return pipeline(
        [
            node(
                func=_predict,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_test",
                    "params:predict.random_state",
                ],
                outputs=f"{model_name}.predictions",
                name=f"predict_{model_name}",
            ),
            node(
                func=_score,
                inputs=["y_test", f"{model_name}.predictions"],
                outputs=f"{model_name}.metrics",
                name=f"score_{model_name}",
            ),
            node(
                func=_plot,
                inputs=[
                    "model_table",
                    "X_test",
                    "y_test",
                    f"{model_name}.predictions",
                    "params:time_split.cutoff",
                ],
                outputs=f"{model_name}.error_plots",
                name=f"plot_{model_name}",
            ),
        ]
    )


def create_training_pipeline() -> Pipeline:
    model_pipelines = _model_branch(MODEL_NAMES[0])
    for name in MODEL_NAMES[1:]:
        model_pipelines += _model_branch(name)

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
        ]
    ) + model_pipelines + pipeline(
        [
            node(
                func=compare_models,
                inputs=[
                    "hist_gb.metrics",
                    "catboost.metrics",
                    "random_forest.metrics",
                ],
                outputs=["comparison", "comparison_plot"],
                name="compare_models",
            ),
        ]
    )
