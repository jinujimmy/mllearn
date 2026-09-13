# Later / optional

whatever we say will be done later or optional should be noted in this list.

## Open

- [ ] 2026-09-12 — Re-add lag features (lag 24 and 168 on `total_users`) in `nodes.py` and wire a Kedro node. Removed `create_lag_features` from this prepare-only step.
- [ ] 2026-09-12 — Clarify `training.py` and `pipeline_registry.py`. Why: `training.py` currently only wires prepare, so the names are confusing; revisit file names / comments after more nodes exist.
- [ ] 2026-09-13 — Understand plots: Step 4 notebook (actual vs pred week, MAE by `hr`) and Kedro `plot_test_errors` / `error_plots.png`. Same plots for the lag model later.


## Done
