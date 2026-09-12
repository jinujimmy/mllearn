# Later / optional

whatever we say will be done later or optional should be noted in this list.

## Open

- [ ] 2026-09-12 — Lag features in the notebook: lag 24 and lag 168 on `total_users`, same cutoff `2012-10-01`, compare MAE to ~44.8 especially at hr 7/18/19. Why: commute-hour errors; user asked to learn lags later.
- [ ] 2026-09-12 — Fit CatBoost on the same `X_train`/`X_test` and compare MAE/RMSE to HistGradientBoosting. Why: CatBoost is imported, not trained; optional extra model class.
- [ ] 2026-09-12 — Signed residual plot (`actual - pred` vs `hr`) if MAE-by-hour is not enough. Why: original plan said residual vs hr; we used mean |error| by hour instead.
- [ ] 2026-09-12 — Mean-by-`hr` dummy baseline (in addition to global train mean 185.2). Why: optional scoring idea from the original plan.
- [ ] 2026-09-12 — Feature importance on the HistGradientBoosting model (`hr` should dominate). Why: optional confirmation, not required for the baseline.
- [ ] 2026-09-12 — Open a PR and merge `feature/baseline-regressor` / `feature/kedro-baseline` into `main`. Why: work lives on branches; optional GitHub hygiene.

## Done

- [x] 2026-09-12 — Kedro nodes for the no-lag baseline: prepare → time split → train → evaluate. `create_lag_features` still commented. Metrics match the notebook (hist_gb MAE ~44.8).
