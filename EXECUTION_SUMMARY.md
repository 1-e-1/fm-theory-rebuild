# Phase 1 execution summary

Verified formal configurations completed: **25/50**.

The 50-run denominator is fixed. Tests/pilots/sampling checks and DEFERRED_BY_USER tasks are excluded from that denominator.

- **SEQ_IID: 25/25 completed**; COMPLETED=25.
- **ROBOT_8DOF: 0/25 completed**; BLOCKED=25.

Robot evidence and precise missing requirements are in `ROBOT_RESOURCE_AUDIT.json` / `MISSING_RESULTS.md`. No alternate robot, geometric proxy or expert fallback has been used. Empty robot tables indicate no observations; they do not indicate zero errors or zero collisions.

Only artifact-verified COMPLETED runs contribute numerical metrics. `main_per_seed.csv` retains all 50 identities and states. FM_STANDARD main metrics use the same-checkpoint Heun reference; its Euler metrics remain in explicit `standard_euler.*` columns. Other main models use their original prescribed Euler/Euler-Maruyama history scheme. Solver/source columns and separate sampling costs identify these choices. `summary.csv` labels the available training-seed count out of five; an incomplete seed group is a partial statistic. `paired_per_seed.csv` retains the three research contrasts (complex ODE minus real ODE, complex SDE minus complex ODE, VDC ODE minus complex ODE), plus four model-minus-standard-Heun contrasts. The 95% Student t intervals and seven-comparison Holm family per dataset/metric are described in `STATISTICAL_ANALYSIS_PLAN.json`; particles and projections are not treated as training repeats.

Full formal checkpoint, training-log and floating-sample directory: `D:\work_space\fm_theory_rebuild_v1\outputs\phase1_iid_robot`. Nonformal pilot artifacts are separately retained under `D:\work_space\fm_theory_rebuild_v1\outputs\pilots`. `full_artifact_inventory.csv` provides concrete paths and sizes for all outputs. Review copies of curves, configuration, evaluation, sampling diagnostics and floating samples preserve numeric source arrays.

Training state, checkpoint choice and formal budgets must be taken from the locked protocol and each preserved attempt. Reaching a budget does not demonstrate convergence. Original and validation-best/final weights must remain under the full output directory. Prior failures and retries are retained.

Factual distribution-quality observations and metric means are in RESULT_OBSERVATIONS.md; reaching the training budget does not establish convergence. Exact frozen-protocol reproduction, reaggregation and checkpoint recovery commands are in REPRODUCTION.md.

This package reports experiments and factual progress. SEQ_IID evidence is limited to independent sequence generation and establishes no long-memory claim. No model winner is predetermined. Completion of this phase does not authorize deferred experiments.

