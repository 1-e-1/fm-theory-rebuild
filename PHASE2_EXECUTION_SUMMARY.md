# Phase 2 execution summary

Artifact-verified formal runs completed: **75/75**.

The new 75-run denominator contains three selected datasets, five delivered models and five training seeds. Prior Phase 1 results retain their original 25/50 status and are not relabeled or included in this denominator.

- **SEQ_DELAY: 25/25 completed**; COMPLETED=25.
- **SEQ_ACCUM: 25/25 completed**; COMPLETED=25.
- **SEQ_MULTISCALE: 25/25 completed**; COMPLETED=25.

Every task retains the complete L=64, C=2, D=128 C-order flattened state. Observation time s, channel identity and generation time tau are distinct. Each task's DATA_SPEC records exact formulas and engineering choices. Train-only normalization and split membership are shared by all five models/seeds within that task; datasets are summarized separately.

FM_STANDARD primary metrics use same-final-checkpoint Heun samples. Its Euler numbers are retained as standard_euler.* supplementary columns. Other models use their original prescribed history Euler/Euler-Maruyama solver. Solver identity and training/sampling/spatial-computation costs are recorded separately.

The tables preserve all 75 identities. Numerical quality summaries contain only artifact-verified COMPLETED runs and report their available training-seed count out of five. Seven fixed paired comparisons comprise three research contrasts (complex ODE minus real ODE, complex SDE minus complex ODE, VDC ODE minus complex ODE) and four model-minus-standard-Heun contrasts. Mean, sample SD, 95% Student t interval, same-seed raw differences and seven-comparison Holm-adjusted tests are separate fields. Interacting particles, projected directions and metric pairs are not independent training repeats. Secondary diagnostic metrics remain descriptive and do not establish a model winner.

Metric registration and numerical arrays cover the selected tasks' distribution and temporal-dependency evaluation, with their actual channel/observation axes described in METRIC_SPEC. No IID-only interpretation is imposed on these tasks. Validation pilots do not access formal test, do not initialize formal weights and are excluded from the 75-run denominator.

Real-versus-real comparisons reuse a pre-registered fixed reference across model/training-seed runs. Zero training-seed SD for a shared reference reflects this reuse and does not establish a reference with no uncertainty; retained real-reference blocks/arrays document finite-sample variation.

Full formal weights/logs/floating samples: `D:\work_space\fm_theory_rebuild_v1\outputs\phase2_3datasets`. Independent pilot artifacts: `D:\work_space\fm_theory_rebuild_v1\outputs\phase2_pilots`. Full artifact paths and sizes are in full_artifact_inventory.csv. Configurations, actual per-attempt source snapshots, all prior failure/resume records, raw curves, floating samples and metric provenance remain traceable.

A formal checkpoint alone does not demonstrate convergence. Final/best weights, complete configured updates, samples and final evaluation are required for COMPLETED. Missing/failed/blocked and deferred items are retained explicitly. Other datasets, control models, ablations and parameter sweeps remain outside this phase. This is factual experiment reporting, not a paper; completion does not authorize further queues.

