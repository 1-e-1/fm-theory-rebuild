# Factual Phase 1 result observations

Exactly 25 IID runs completed from scratch, each with 12,000 optimizer updates (300,000 formal updates total). All five training seeds are retained for every model. The 25 robot configurations remain BLOCKED. The delivered model source is unchanged.

Primary SEQ_IID metrics below are mean +/- sample SD over the five training seeds. SWD is measured in shared train-standardized complete-vector space; marginal W1 is in raw data units. MMD2 uses the fixed multibandwidth kernel and cross-independent-ensemble generated pairs. Smaller values mean smaller discrepancy for these individual metrics.

| Model | SWD (raw reference) | MMD2 (raw reference) | Marginal W1 |
|---|---:|---:|---:|
| FM_STANDARD | 0.0821379 +/- 0.0145 | 0.000654276 +/- 0.0004507 | 0.232869 +/- 0.002626 |
| FM_REAL_ODE | 0.0845051 +/- 0.006961 | 0.000637807 +/- 0.0002401 | 0.226476 +/- 0.0009559 |
| FM_COMPLEX_SDE | 0.0689574 +/- 0.002855 | 0.000221843 +/- 0.0001194 | 0.231531 +/- 0.002799 |
| FM_COMPLEX_ODE | 0.0889799 +/- 0.008144 | 0.000796306 +/- 0.0002956 | 0.22724 +/- 0.001244 |
| FM_COMPLEX_VDC_ODE | 0.0828288 +/- 0.00737 | 0.00057824 +/- 0.0002474 | 0.225727 +/- 0.001202 |

FM_STANDARD uses the same final checkpoint with Heun for the primary statistical baseline; Euler is retained as a supplement. Other primary models use the prescribed Euler/Euler-Maruyama history scheme. Full precision results, 95% Student t intervals, paired seed differences and seven-comparison Holm corrections are in tables/summary.csv, tables/paired_per_seed.csv and tables/paired_comparisons.csv.

The pooled raw-coordinate histograms in figures/iid_marginal_diagnostics.png remain broad and approximately single-peaked for all five models, while the reference mixture has three peaks. This agrees with the remaining marginal W1 discrepancy (model means approximately 0.226-0.233). Low projected SWD alone does not establish recovery of the full independent mixture distribution. Learning curves and validation values are retained; reaching 12,000 updates does not demonstrate convergence. No extra budget was selected after seeing test results.

The summed recorded formal training wall time is 21268.049 seconds (5.908 hours). This includes each training loop's validation/checkpoint/log overhead and excludes final sampling, evaluation, checkpoint diagnostics and pilots. Detailed per-run/model costs and accounting limits are in tables/compute_costs.csv, source/source_reference/FINAL_COMPUTE_SUMMARY.json and COMPUTE_ACCOUNTING_NOTES.md.

Robot distribution quality, feasibility, collision, endpoint, length and smoothness metrics are unavailable because the actual USD/geometry/FK/metadata/task-expert data were not supplied. Header-only robot tables represent absent observations. No candidate or expert-fallback result was fabricated.

These are experimental observations only. IID evidence establishes no long-memory conclusion. Deferred experiments were not executed.
