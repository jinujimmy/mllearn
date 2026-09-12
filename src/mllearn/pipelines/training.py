from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_model_table


def create_training_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_model_table,
                inputs=["renamed_data", "params:prepare_data.leakage_or_id"],
                outputs="model_table",
                name="prepare_model_table",
            ),
        ]
    )
