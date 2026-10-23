# 第二阶段复现与断点恢复

当前工程根目录为 `D:\work_space\fm_theory_rebuild_v1`。使用 `PHASE2_ENVIRONMENT.json` 记录的现有Python/PyTorch/CUDA环境，不升级全局依赖。新阶段固定75项，三数据任务各25项；第一阶段50项保持原状态，不能改写其分母或标签。

用户已明确确认“按任务书顺序选择3个”：`SEQ_DELAY`、`SEQ_ACCUM`、`SEQ_MULTISCALE`；确认记录于2026-10-09 05:43:02 UTC，原文与来源见 `PHASE2_USER_SELECTION_CONFIRMATION.json`。这与先前默认选择一致，不改变队列、已锁定协议或实际执行历史。

## 查看真实进度与检查

```powershell
python -m phase2_experiment.monitor
python -m phase2_experiment.scope --verify
python -m phase2_experiment.verify
```

以 `PHASE2_MANIFEST.json`、`PHASE2_PROGRESS.md` 和每attempt真实日志为准。`PHASE2_PROCESS_STATE.json`记录前台监督进程实际PID、命令、开始/结束状态及UTF-8控制台日志位置；进程退出不自动等于75项实验完成。

## 恢复已中断的固定正式队列

```powershell
python -u -m phase2_experiment.orchestrate resume
python -m phase2_experiment.verify
```

监督入口先检查第一阶段保护哈希，然后调用新阶段恢复入口。恢复保留attempt和旧日志，加载同attempt的optimizer、Python/NumPy/Torch/CUDA RNG及checkpoint。完整配置被跳过；已有完整训练但缺评价时，对同一final checkpoint补齐评价。缺可恢复checkpoint的旧attempt保留，新建下一个编号从原seed训练。恢复不会更换模型、数据、协议或seed，不启动旧全量入口。

如只需对某个新阶段完整checkpoint重新评价：

```powershell
python -m phase2_experiment.runner evaluate --directory 'outputs/phase2_3datasets/SEQ_DELAY__FM_STANDARD__seed17/attempt_001'
```

评价会核对真实训练预算及来源，读取正式test，按已锁定采样和指标规则执行，不训练、不用test选择checkpoint。第一阶段目录会被此入口拒绝。

## 重新汇总及打包

```powershell
python -m phase2_experiment.reporting
python -m phase2_experiment.plots
python -m phase2_experiment.reporting --package
```

正式全量weights、logs和浮点samples在 `outputs/phase2_3datasets/`；独立pilot在 `outputs/phase2_pilots/`。审查ZIP保留源、数据、数值证据和完整本机artifact清单，权重目录单独交付。只对通过artifact门禁的COMPLETED行计算统计；未完成项不填零、不删除。

## 在新目录按冻结协议从头复现

新目录必须为空，保留实际执行源码、数据、配置及来源保护记录；不复制任何旧权重或completed新阶段manifest。以下准备步骤将必要的第一阶段只读证据一同复制，以满足原版保护哈希；它们仅作来源记录，不参与新训练。

在原工程根目录运行以下Python脚本（PowerShell here-string），目标目录已存在时拒绝覆盖：

```powershell
@'
import json, shutil
from pathlib import Path
origin = Path.cwd().resolve()
fresh = origin / 'reproduction_phase2'
if fresh.exists():
    raise SystemExit('Destination exists; select a fresh reproduction directory.')
fresh.mkdir()
guard = json.loads((origin/'phase2_reference/PHASE1_GUARD.json').read_text(encoding='utf-8'))
for relative in guard['files']:
    source = (origin/relative).resolve()
    if not source.is_relative_to(origin):
        raise SystemExit('Guard source outside original project.')
    target = fresh/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
for name in ['phase2_experiment', 'phase2_tests', 'phase2_reference', 'tests', 'data/phase2_3datasets']:
    shutil.copytree(origin/name, fresh/name, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest*'))
for name in ['PHASE2_AUTHORIZATION.md', 'PHASE2_USER_SELECTION_CONFIRMATION.json', 'PHASE2_STATISTICAL_ANALYSIS_PLAN.json',
             'PHASE2_MODEL_CHANGELOG.md', 'PHASE2_IMPLEMENTATION_LOG.md',
             'PHASE2_ENVIRONMENT.json', 'PHASE2_REPRODUCTION.md']:
    shutil.copy2(origin/name, fresh/name)
# initialize_scope requires an empty new Phase2 manifest; copy the locked
# protocol only after initialization, and never re-select it from test data.
print(fresh)
'@ | python -
Set-Location -LiteralPath 'D:\work_space\fm_theory_rebuild_v1\reproduction_phase2'
@'
from phase2_experiment.scope import initialize_scope, DATASETS
initialize_scope(DATASETS, authority_note='From-scratch reproduction of the delivered fixed three-dataset Phase2 protocol.')
'@ | python -
Copy-Item -LiteralPath '..\PHASE2_LOCKED_PROTOCOL.json' -Destination '.\PHASE2_LOCKED_PROTOCOL.json' -Force
foreach ($phaseDataset in @('SEQ_DELAY','SEQ_ACCUM','SEQ_MULTISCALE')) {
    python -m phase2_experiment.runner prepare --dataset $phaseDataset
    if ($LASTEXITCODE -ne 0) { throw 'Frozen data verification failed.' }
}
python -m pytest phase2_tests -q
if ($LASTEXITCODE -ne 0) { throw 'Phase2 correctness checks failed.' }
python -u -m phase2_experiment.orchestrate run
python -m phase2_experiment.verify
python -m phase2_experiment.reporting
python -m phase2_experiment.plots
python -m phase2_experiment.reporting --package
```

这些是未来从头复现命令，本次交付不执行第二套75项训练。冻结协议包含原独立pilot选择证据；从头复现不重新运行`lock`，不将原checkpoint导入新训练。源、数据、每任务协议和seed保持可追溯；换GPU或库版本时不承诺逐位相同。

若要独立重做pilot以检查新环境，只对明确指定数据执行，例如 `python -m phase2_experiment.runner pilot --dataset SEQ_DELAY --pilot-steps 1000`，并将其作为新非正式attempt保存。它不改变已有正式协议，也不计入75项。

第一阶段不能用第二阶段初始化或恢复命令重置；不得自动训练其余暂缓数据、控制组或消融。


## Frozen next-phase training budget

The 15 independent 1000-update train/validation pilots are complete. Formal common tiers are SEQ_DELAY=12000, SEQ_ACCUM=24000 and SEQ_MULTISCALE=24000, all with N32/H17/Q64 and the same five models and five seeds. See PHASE2_PROTOCOL_SELECTION.json and the immutable PHASE2_LOCKED_PROTOCOL.json. Linear pilot-based projection is 42.562 training hours for all75; autonomous generation, final evaluation, checkpoint checks and packaging add time. This is an estimate, not a promised deadline or a convergence claim.

All formal runs start from seed-initialized weights; pilot weights and Phase1 weights are never initializers. The current source is frozen at11 scientific core files. Run `python -m phase2_experiment.monitor` to inspect actual persisted progress; COMPLETED requires the full budget and genuine final artifacts.

The historical top-level `datasets_pending_lock` field in PHASE2_LOCKED_PROTOCOL.json is an initialization note retained by the original initializer. It is not the runtime lock state. All three dataset-specific payloads are locked and validated by `_protocol` before the first formal launch; authoritative actual choices and hashes are recorded in PHASE2_PROTOCOL_SELECTION.json. The protocol file remains unchanged after formal execution starts.
