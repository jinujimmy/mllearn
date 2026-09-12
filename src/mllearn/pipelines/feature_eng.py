from kedro.pipeline import node, pipeline
from .nodes import rename_columns 

def create_feature_pipeline() -> pipeline:
    return pipeline([
        node(
            func=rename_columns,
            inputs=["raw_data", "params:feature_engineering.renaming_columns"],
            outputs="renamed_data",
            name="rename_columns",
        ),
    ])