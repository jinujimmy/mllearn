"""Project pipelines."""
from __future__ import annotations

from .pipelines.feature_eng import create_feature_pipeline

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.
    
    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    feature_engineering_pipeline = create_feature_pipeline()

    return {
        "__default__": feature_engineering_pipeline,
        "feature_eng": feature_engineering_pipeline,
    }
