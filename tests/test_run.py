"""Kedro pipeline registration tests."""

from pathlib import Path

from kedro.framework.startup import bootstrap_project

from mllearn.pipeline_registry import register_pipelines


class TestPipelines:
    def test_default_pipeline_has_train_and_evaluate_nodes(self):
        bootstrap_project(Path.cwd())
        pipes = register_pipelines()
        names = {node.name for node in pipes["__default__"].nodes}
        assert "rename_columns" in names
        assert "prepare_model_table" in names
        assert "time_split" in names
        assert "train_hist_gb" in names
        assert "evaluate_model" in names
        assert not any(node.name == "create_lag_features" for node in pipes["__default__"].nodes)
