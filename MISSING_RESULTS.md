# Missing and blocked Phase 1 results

All 25 `ROBOT_8DOF` formal triples remain in the 50-run manifest as **BLOCKED**. None has been trained or evaluated. This is an asset verification blockade; it must not be filled with another arm, LASA, planar paths or invented geometry.

Required evidence absent from this project and not supplied as a concrete external asset path:

- Actual 8 DOF right-arm `.usd`/`.usda`/`.usdc` entry file and all referenced geometry/material/texture assets.
- Correct robot link/collision geometry and calibrated FK implementation/data, with provenance.
- Eight-joint ordering, TCP and base frames, angle branch/unwrap rule, sourced joint/velocity/acceleration limits and collision coverage.
- Task definitions (start, target, fixed obstacles), task-based train/validation/test membership, and at least eight valid expert trajectories per condition. The shared 32-coefficient representation fitting error and infeasibility audit have therefore not been performed.
- Concrete, user-authorized asset file paths to inspect; the latest task supplies no external USD or expert-data filename.

Current project file listing was checked and contains no robot asset/data files. `CODEX_RUNBOOK.md` names `D:\work_space\cfm`, `D:\work_space\lunwen2.1`, `D:\conda_envs\env_isaaclab`, and `D:\Isaac_lab` only as historical discovery hints. Each directory exists. Only literal existence was checked; their content was not scanned, and no assertion is made that robot files are absent inside them.

Default executable: `C:\Users\ASUS\AppData\Local\Programs\Python\Python314\python.exe`. `pxr`, `isaaclab`, and `isaacsim` are not importable there. Separate Isaac environments were not inspected; optional Isaac replay alone is not required to block geometry-backed training. Here the actual geometry/FK/data are independently unavailable.

SEQ_IID is independent and continues. `FAILED` formal runs, incomplete progress and artifact verification failures are generated from the manifest by `python -m experiment.reporting --package`. Other research tasks are listed separately in `DEFERRED_BY_USER.json`; they are not failures or missing Phase 1 results.
