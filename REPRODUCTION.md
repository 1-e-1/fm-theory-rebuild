# Phase 1 reproduction and recovery

The completed original project is `D:\work_space\fm_theory_rebuild_v1`. Use the Python/PyTorch/CUDA environment recorded in `ENVIRONMENT.json`. The review ZIP contains evidence, data, samples and sources; full checkpoints are delivered separately in `outputs/phase1_iid_robot`, with concrete paths in `full_artifact_inventory.csv`. Extracting only the review ZIP does not recreate the complete original workspace layout.

## Verify and reaggregate the delivered workspace

Run in the original project root. These commands do not train:

```powershell
python -m experiment.scope --verify
python -m experiment.verify
python -m experiment.reporting
python -m experiment.plots
python -m experiment.reporting --package
```

The expected final state is 25 `SEQ_IID` configurations `COMPLETED` and 25 `ROBOT_8DOF` configurations `BLOCKED`, retaining denominator 50. `experiment.verify` also checks full checkpoints and frozen execution-source hashes. Use that command instead of `experiment.runner verify`: the latter has a documented NumPy-bool JSON-printing issue in its preserved execution version.

## Resume an interrupted execution of this fixed queue

```powershell
python -m experiment.runner resume
python -m experiment.verify
python -m experiment.reporting --package
```

Resume restores the saved model, optimizer, Python/NumPy/Torch/CUDA RNG states from `checkpoint_resume.pt`. It retains previous logs and attempts. If training is already finished but evaluation is missing, the same final checkpoint is evaluated. Complete configurations are skipped. The delivered original project has no unfinished IID runs, so resume is unnecessary here. Robot training remains blocked until the actual user assets, FK, metadata and task data are verified; there is no robot substitute or automatic unblocking command.

## Reproduce the 25 available runs from scratch with the frozen protocol

The following PowerShell procedure creates a **new** copy inside the workspace and refuses an existing destination. It copies no checkpoints, output directory, completed manifest or completed scope. It reuses the exact delivered frozen data, normalization, metric registration and `LOCKED_PROTOCOL.json`; do not rerun `lock`, since measured pilot costs would rewrite protocol metadata. Original pilot and protocol-selection evidence remain available in the delivered artifacts.

This is a future reproduction command, not an additional execution performed for this delivery. It requires several hours of GPU time and creates its own 25 IID training runs while retaining the 25 robot rows as blocked.

```powershell
$phaseOriginal = 'D:\work_space\fm_theory_rebuild_v1'
$phaseReproduction = Join-Path $phaseOriginal 'reproduction_phase1'
if (Test-Path -LiteralPath $phaseReproduction) {
    throw 'Reproduction destination already exists; choose another fresh directory.'
}
New-Item -ItemType Directory -Path $phaseReproduction | Out-Null
$phaseFiles = @(
    'theory_fm_models.py', 'START_CODEX_PHASE1_50_RUNS.txt',
    'MODEL_GUIDE.md', 'CODEX_RUNBOOK.md', 'MODEL_CONTRACT.json',
    'MODEL_SPEC.json', 'FILES_SHA256.json', 'MODEL_CHANGELOG.md',
    'DATA_SPEC.json', 'LOCKED_PROTOCOL.json', 'ENVIRONMENT.json',
    'STATISTICAL_ANALYSIS_PLAN.json', 'EXPERIMENT_IMPLEMENTATION_LOG.md',
    'COMPUTE_ACCOUNTING_NOTES.md', 'REPRODUCTION.md', 'RESULT_OBSERVATIONS.md'
)
foreach ($phaseFile in $phaseFiles) {
    Copy-Item -LiteralPath (Join-Path $phaseOriginal $phaseFile) -Destination $phaseReproduction
}
foreach ($phaseDirectory in @('experiment', 'tests', 'configs', 'data', 'original_snapshot', 'source_reference')) {
    Copy-Item -LiteralPath (Join-Path $phaseOriginal $phaseDirectory) -Destination (Join-Path $phaseReproduction $phaseDirectory) -Recurse
}
Set-Location -LiteralPath $phaseReproduction
python -m experiment.initialize_scope
if ($LASTEXITCODE -ne 0) { throw 'Scope initialization failed.' }
python -m experiment.runner prepare
if ($LASTEXITCODE -ne 0) { throw 'Frozen data verification failed.' }
python tests/model_correctness.py
if ($LASTEXITCODE -ne 0) { throw 'Model correctness checks failed.' }
python -m experiment.runner run
if ($LASTEXITCODE -ne 0) { throw 'Queue execution failed; inspect preserved progress.' }
python -m experiment.verify
if ($LASTEXITCODE -ne 0) { throw 'Artifact verification failed.' }
python -m experiment.reporting
python -m experiment.plots
python -m experiment.reporting --package
```

All formal runs use the five specified seeds and 12,000 updates under one shared IID protocol. Seeds are advanced in groups of five models. Exact bitwise matching on another GPU/library version is not promised; frozen data, protocol, source, per-seed outputs and cost measurements remain identifiable. The `DATA_SPEC.json` directory string records original provenance; data loading resolves the copied `data/seq_iid` relative to the new source root and verifies byte hashes.

The original `PHASE1_SCOPE.json` must never be reinitialized or edited to reset progress. Deferred datasets, controls and ablations remain outside this phase.
