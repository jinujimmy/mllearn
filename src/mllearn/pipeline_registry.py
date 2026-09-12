"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

from .pipelines.feature_eng import create_feature_pipeline
from .pipelines.training import create_training_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    feature_engineering_pipeline = create_feature_pipeline()
    training_pipeline = create_training_pipeline()

    return {
        "__default__": feature_engineering_pipeline + training_pipeline,
        "feature_eng": feature_engineering_pipeline,
        "training": training_pipeline,
    }
