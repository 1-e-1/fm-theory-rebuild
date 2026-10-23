# Field-memory Flow Models

Five theory-derived flow models, fixed-space field history and two phases of time-series experiments.

Five original model definitions are preserved. Phase 1 records 25/50 completed (25 robot configurations blocked); Phase 2 records 75/75 completed. The repository does not relabel those denominators.

## Setup

Use a dedicated Python environment and install a compatible PyTorch/torchvision pair for your platform, then:

```sh
python -m pip install -r requirements.txt
```

Run from the repository root. See [method scope](METHOD_SCOPE.md), [publication notes](PUBLICATION_NOTES.md) and [model definitions](MODEL_GUIDE.md) for the exact scope. [Phase 1 reproduction](REPRODUCTION.md) and [Phase 2 reproduction](PHASE2_REPRODUCTION.md) distinguish portable checks from commands that require the original full artifacts.

## Distribution limits

This release contains source, selected configuration, original tests and research documentation. It does not bundle datasets, pretrained weights, robot scenes or third-party simulation libraries. Historical success rates and paper figures are not new measurements from this release. Simulator and hardware claims retain their original limitations. Do not launch full training queues just to test an installation.

## License

Original material is licensed under [MIT](LICENSE). Third-party libraries, assets and datasets retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).

## Historical experimental status

Phase 1: SEQ_IID 25/25 completed; ROBOT_8DOF 0/25, blocked by missing verified robot assets and expert data. Total 25/50. Phase 2: SEQ_DELAY, SEQ_ACCUM and SEQ_MULTISCALE each 25/25; total 75/75. These are separate denominators. Aggregate evidence and original reports are included; full checkpoints and raw samples are separate.
