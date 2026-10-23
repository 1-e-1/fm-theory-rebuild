# Phase 2 model and implementation provenance

Delivered theory_fm_models.py is unchanged from the original snapshot: fd97ebc960271869c4906cf4ad8f67252e30ebb39b84f461cc882442a2c76fbe. No mathematical model, loss target, architecture or operator correction was made. All networks remain feedforward Tanh MLPs. The fixed real state-space system generates MULTISCALE data only and is not a learned model/network.

Original experiment/ sources are unchanged. New engineering adapters add three task identities, full L64/C2/D128 data and dedicated manifests. The new diagnostics.py preserves the original same-checkpoint mathematics while adapting D64 tensors and predeclared coordinate pairs to D128 and task provenance. The original, new source, exact diff and 47-test evidence are in phase2_reference and phase2_tests.

Before any pilot/formal execution, execution core hashes, shared data specifications and scope are frozen in phase2_reference/PRE_PILOT_FREEZE.json. Any later evidence-based core bug fix requires a preserved version, diff, retest and source-aware new attempt; formal protocol or model definitions are never silently changed.
