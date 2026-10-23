"""Independent, scope-checked Phase-2 trainer; never writes Phase-1 state.

Delivered model and accepted-history sampler are imported unchanged. The three
approved datasets are resolved from PHASE2_SCOPE and PHASE2_MANIFEST, not from
any old all/full queue. No formal protocol is automatically selected.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from theory_fm_models import (MAIN_MODELS, ModelConfig, build_model,
                             optimization_groups, parameter_counts)
from experiment.runner import training_batch
from experiment.sampling import sample

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase2_3datasets"
PILOTS = ROOT / "outputs/phase2_pilots"
PROTOCOL_PATH = ROOT / "PHASE2_LOCKED_PROTOCOL.json"
MODELS = tuple(MAIN_MODELS)
SEEDS = (17, 29, 43, 71, 101)
FORMAL_TIERS = (12000, 24000, 48000)
SOURCE_CORE = ("theory_fm_models.py", "experiment/runner.py", "experiment/sampling.py",
               "experiment/data.py", "experiment/metrics.py", "experiment/diagnostics.py")
SOURCE_PHASE2_CORE = ("phase2_experiment/runner.py", "phase2_experiment/data.py",
                      "phase2_experiment/metrics.py", "phase2_experiment/diagnostics.py",
                      "phase2_experiment/scope.py")
FIELDS = ("step", "elapsed_seconds", "loss", "raw_loss", "val_loss", "grad_norm",
          "grad_norm_after_clip", "clipped", "tau", "orders_json")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_protocol_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False).encode("utf-8")).hexdigest()


def execution_sources():
    # Monitor/reporting/plot code is not a training, label, sampling or evaluation
    # dependency. Its later changes must not invalidate a preserved checkpoint.
    paths = [ROOT/path for path in SOURCE_CORE+SOURCE_PHASE2_CORE]
    initializer=ROOT/"phase2_experiment/__init__.py"
    if initializer.is_file():paths.append(initializer)
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _data_spec(dataset_id):
    from phase2_experiment.data import data_dir
    path = data_dir(dataset_id)/"DATA_SPEC.json"
    spec = read_json(path)
    if spec.get("dataset_id") != dataset_id:
        raise ValueError("Phase-2 data specification identity differs from the requested dataset")
    if spec.get("length") != 64 or spec.get("channels") != 2 or spec.get("state_dim") != 128:
        raise ValueError("The authorized Phase-2 complete sequence representation is L64/C2/D128")
    return path, spec


def config(dataset_id, protocol=None):
    _, spec = _data_spec(dataset_id)
    protocol = protocol or {}
    return ModelConfig(state_dim=spec["state_dim"], hidden_dim=128, hidden_layers=3,
        query_chunk=protocol.get("query_chunk", 64), source_chunk=protocol.get("source_chunk",32))


def load_data(dataset_id, split, device):
    from phase2_experiment.data import load_split
    data=load_split(dataset_id, split, device=device)
    _, spec=_data_spec(dataset_id)
    if data.ndim!=2 or data.shape[1]!=spec["state_dim"] or data.dtype!=torch.float32:
        raise ValueError("Loaded split must contain complete float32 D128 states")
    return data


def verify_data(dataset_id):
    path,spec=_data_spec(dataset_id)
    for record in spec["files"]:
        target=(path.parent/record["filename"]).resolve()
        if not target.is_relative_to(path.parent.resolve()):
            raise ValueError("Data provenance file points outside its dataset directory")
        if not target.is_file() or sha256_file(target)!=record["sha256"]:
            raise ValueError(f"Frozen Phase-2 data changed: {target}")
    return path,spec


def model_correctness_gate():
    report=read_json(ROOT/"tests/model_correctness_report.json")
    digest=sha256_file(ROOT/"theory_fm_models.py")
    if report.get("failed")!=0 or report.get("source_sha256")!=digest:
        raise RuntimeError("The unchanged delivered model must pass its numerical correctness gate")


def _protocol(dataset_id):
    from phase2_experiment.scope import validate_scope
    validate_scope()
    payload=read_json(PROTOCOL_PATH).get(dataset_id)
    if not isinstance(payload,dict) or payload.get("updates") not in FORMAL_TIERS:
        raise ValueError(f"No locked formal protocol for {dataset_id}")
    expected={"Q":64,"E":1,"hidden":128,"hidden_layers":3,"network_lr":2e-4,
              "order_lr":2e-5,"weight_decay":1e-4,"grad_clip":1.0,
              "sigma_min":.05,"kappa":.05,"tau_min":None,"checkpoint":"final"}
    for name,value in expected.items():
        if payload.get(name)!=value:
            raise ValueError(f"Locked protocol changes a shared audited setting: {name}")
    if payload.get("N") not in (32,64) or payload.get("H") not in (17,33):
        raise ValueError("Formal N/H must use the task-book authorized common choices")
    if payload.get("sampling_ensembles")!=16 or payload.get("sampling_particles")!=64 or payload.get("sampling_steps")!=64:
        raise ValueError("The locked common primary sample protocol must be E16/N64/64 steps")
    return payload


def validate(model,val,protocol=None,batches=4):
    protocol=protocol or {"H":17,"N":32,"Q":64}
    generator=torch.Generator(device=val.device).manual_seed(602113)
    losses=[]
    was_training=model.training
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            loss,_,_=training_batch(val,model,H=protocol["H"],N=protocol["N"],Q=protocol["Q"],
                                    generator=generator,validation=True)
            losses.append(float(loss))
    model.train(was_training)
    if not np.isfinite(losses).all():raise FloatingPointError("Nonfinite validation loss")
    return float(np.mean(losses))


def snapshot_sources(directory,hashes):
    target=Path(directory)/"source"
    initial_snapshot=not (target/"source_hashes.json").exists()
    for relative,digest in hashes.items():
        source=ROOT/relative
        if sha256_file(source)!=digest:raise ValueError(f"Execution source changed while taking snapshot: {relative}")
        destination=target/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if sha256_file(destination)!=digest:raise ValueError(f"Preserved source snapshot differs: {relative}")
        else:shutil.copy2(source,destination)
    if initial_snapshot:
        # Save all then-current Phase-2 source once, including reporting files,
        # while only the explicit core whitelist must match when resuming.
        for source in sorted((ROOT/"phase2_experiment").glob("*.py")):
            relative=source.relative_to(ROOT).as_posix()
            destination=target/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():shutil.copy2(source,destination)
        all_hashes={path.relative_to(target).as_posix():sha256_file(path) for path in target.rglob("*.py")}
        write_json(target/"source_hashes.json",all_hashes)
        write_json(target/"core_execution_source_hashes.json",hashes)


def save_checkpoint(path,model,optimizer,step,best,elapsed,cfg,clip_count,run_config):
    payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,
        "best_validation":best,"elapsed_seconds":elapsed,"model_config":asdict(cfg),
        "clip_count":clip_count,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
        "torch_rng":torch.get_rng_state(),
        "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "model_sha256":run_config["source_sha256"],
        "run_identity":{key:run_config[key] for key in ("dataset_id","model_id","seed","kind","run_id","steps")},
        "data_spec_sha256":run_config["data_spec_sha256"],
        "execution_source_hashes":run_config["execution_source_hashes"],
        "dataset_protocol_sha256":run_config.get("dataset_protocol_sha256")}
    path=Path(path);temp=path.with_suffix(".tmp")
    torch.save(payload,temp);os.replace(temp,path)


def assert_run_context(conf,checkpoint,*,formal=True):
    from phase2_experiment.scope import validate_config
    validate_config(conf["dataset_id"],conf["model_id"],conf["seed"])
    if conf.get("phase")!="PHASE2_3DATASETS" or conf.get("kind")!=("formal" if formal else "pilot"):
        raise ValueError("Checkpoint context does not belong to the requested Phase-2 run type")
    if checkpoint.get("model_sha256")!=conf["source_sha256"]:
        raise ValueError("Checkpoint/model provenance differs from configuration")
    if checkpoint.get("run_identity")!={key:conf[key] for key in ("dataset_id","model_id","seed","kind","run_id","steps")}:
        raise ValueError("Checkpoint identity differs from its dataset/model/seed/configuration")
    if json.dumps(checkpoint["model_config"],sort_keys=True)!=json.dumps(conf["model_config"],sort_keys=True):
        raise ValueError("Checkpoint model configuration mismatch")
    if checkpoint.get("data_spec_sha256")!=conf["data_spec_sha256"]:
        raise ValueError("Checkpoint/data provenance mismatch")
    if checkpoint.get("execution_source_hashes")!=conf["execution_source_hashes"]:
        raise ValueError("Checkpoint/execution source provenance mismatch")
    if execution_sources()!=conf["execution_source_hashes"]:
        raise ValueError("Phase-2 execution source set or hashes differ from this preserved run")
    path,_=verify_data(conf["dataset_id"])
    if sha256_file(path)!=conf["data_spec_sha256"]:
        raise ValueError("Frozen dataset specification changed")
    if formal:
        protocol=_protocol(conf["dataset_id"])
        digest=dataset_protocol_hash(protocol)
        if digest!=conf["dataset_protocol_sha256"] or digest!=checkpoint.get("dataset_protocol_sha256"):
            raise ValueError("Frozen per-dataset formal protocol changed")
        for name in ("N","H","Q","E"):
            if conf[name]!=protocol[name]:raise ValueError(f"Run differs from the locked protocol: {name}")
        if conf["steps"]!=protocol["updates"]:
            raise ValueError("Run budget differs from the locked formal protocol")
    if not (0<=checkpoint["step"]<=conf["steps"]):
        raise ValueError("Checkpoint step outside the preserved run budget")


def restore_checkpoint(checkpoint,model,optimizer):
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.set_rng_state(checkpoint["torch_rng"].cpu())
    if checkpoint.get("cuda_rng"):
        if not torch.cuda.is_available():raise ValueError("A CUDA checkpoint cannot continue training without the original CUDA RNG environment")
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])


def preserve_resume_log(log_path,step):
    """Preserve every already-executed row before reconciling with the checkpoint."""
    log_path=Path(log_path)
    if not log_path.is_file():raise ValueError("Resume checkpoint lacks its complete training log")
    shutil.copy2(log_path,log_path.with_name(f"training_before_resume_{time.time_ns()}.csv"))
    with log_path.open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    committed=[row for row in rows if int(row["step"])<=step]
    if [int(row["step"]) for row in committed]!=list(range(1,step+1)):
        raise ValueError("Committed training rows are incomplete or duplicated")
    with log_path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(committed)


def pilot_generation_diagnostics(samples,dataset_id,directory):
    """Evaluate the pilot against validation only; never call formal test metrics."""
    from phase2_experiment.metrics import pilot_validation_diagnostics
    metrics=pilot_validation_diagnostics(samples,dataset_id)
    if not isinstance(metrics,dict) or metrics.get("test_accessed") is not False:
        raise ValueError("Pilot generation diagnostics must explicitly attest no test access")
    numbers=[value for value in metrics.values() if isinstance(value,(int,float)) and not isinstance(value,bool)]
    if not numbers or not np.isfinite(numbers).all():
        raise ValueError("Pilot validation generation metrics must contain finite numerical evidence")
    write_json(Path(directory)/"pilot_validation_diagnostics.json",metrics)
    return metrics


def train_one(dataset_id,model_id,seed,steps=None,*,pilot=False,resume=False):
    from phase2_experiment.scope import validate_config,validate_scope,update_run,run_identifier
    validate_scope()
    validate_config(dataset_id,model_id,seed)
    model_correctness_gate()
    path,spec=verify_data(dataset_id)
    protocol={"N":32,"H":17,"Q":64,"E":1} if pilot else _protocol(dataset_id)
    if pilot:
        steps=200 if steps is None else steps
        if not 1<=steps<=2000:raise ValueError("Pilot budget must be 1..2000 updates")
    else:
        if steps is not None and steps!=protocol["updates"]:raise ValueError("Explicit formal budget differs from locked protocol")
        steps=protocol["updates"]
    run_id=run_identifier(dataset_id,model_id,seed)
    base=(PILOTS if pilot else OUT)/run_id
    attempts=sorted(base.glob("attempt_*"))
    if resume and attempts and (attempts[-1]/"checkpoint_resume.pt").is_file():
        directory=attempts[-1]
    else:
        # A previous attempt without a restorable checkpoint is preserved;
        # a fresh run receives a new attempt number and the same original seed.
        directory=base/f"attempt_{len(attempts)+1:03d}"
    directory.mkdir(parents=True,exist_ok=True)
    cfg=config(dataset_id,protocol)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(seed)
    model=build_model(model_id,cfg).to(device=device,dtype=torch.float32)
    optimizer=torch.optim.AdamW(optimization_groups(model))
    source_hashes=execution_sources()
    run_config={"phase":"PHASE2_3DATASETS","dataset_id":dataset_id,"model_id":model_id,"seed":seed,
        "run_id":run_id,"kind":"pilot" if pilot else "formal","steps":steps,
        "model_config":json.loads(json.dumps(asdict(cfg))),"N":protocol["N"],"H":protocol["H"],
        "Q":protocol["Q"],"E":protocol["E"],"sigma_min":.05,
        "time_sampling":"0.5 Uniform + 0.5 U^4; inverse density weight; no truncation",
        "parameter_counts":parameter_counts(model),"device":str(device),"checkpoint_for_primary_evaluation":"final",
        "EMA":"not used","data_spec_sha256":sha256_file(path),"data_spec_path":path.relative_to(ROOT).as_posix(),
        "source_sha256":source_hashes["theory_fm_models.py"],"execution_source_hashes":source_hashes,
        "environment":{"python":sys.version,"torch":torch.__version__,"CUDA":torch.version.cuda,
                       "device_name":torch.cuda.get_device_name() if device.type=="cuda" else "CPU"},
        "checkpoint_import":"None: initialization from training seed, or same attempt resume only"}
    if not pilot:
        digest=dataset_protocol_hash(protocol)
        run_config.update(dataset_protocol_sha256=digest,locked_protocol_sha256=digest,
                          locked_protocol_file_sha256=sha256_file(PROTOCOL_PATH))
    resume_path=directory/"checkpoint_resume.pt"
    if resume_path.is_file():
        old=read_json(directory/"config.json")
        for key in ("phase","dataset_id","model_id","seed","kind","steps","model_config","N","H","Q","E","device","environment",
                    "data_spec_sha256","source_sha256","execution_source_hashes","dataset_protocol_sha256"):
            if old.get(key)!=run_config.get(key):raise ValueError(f"Resume context differs at {key}")
        # Retain provenance of the original lock-file snapshot; a later lock of
        # another task does not change this task's canonical protocol hash.
        run_config=old
    snapshot_sources(directory,source_hashes)
    write_json(directory/"config.json",run_config)
    train=load_data(dataset_id,"train",device);val=load_data(dataset_id,"validation",device)
    start_step,best,previous_elapsed,clip_count=0,float("inf"),0.0,0
    if resume_path.is_file():
        checkpoint=torch.load(resume_path,map_location=device,weights_only=False)
        assert_run_context(run_config,checkpoint,formal=not pilot)
        restore_checkpoint(checkpoint,model,optimizer)
        start_step,best,previous_elapsed,clip_count=checkpoint["step"],checkpoint["best_validation"],checkpoint["elapsed_seconds"],checkpoint["clip_count"]
        preserve_resume_log(directory/"training.csv",start_step)
    if not pilot:
        update_run(run_id,"RUNNING",attempt=directory.name,attempt_dir=str(directory),output_dir=str(directory),updates=steps)
    started=time.perf_counter()
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats()
    final_val_loss=None
    with (directory/"training.csv").open("a" if start_step else "w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS)
        if not start_step:writer.writeheader()
        for step in range(start_step+1,steps+1):
            optimizer.zero_grad(set_to_none=True)
            loss,tau,raw=training_batch(train,model,H=protocol["H"],N=protocol["N"],Q=protocol["Q"])
            if not torch.isfinite(loss):raise FloatingPointError(f"Nonfinite loss at step {step}; tau={tau}")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(norm):raise FloatingPointError(f"Nonfinite gradient at step {step}")
            clipped=int(norm>1);clip_count+=clipped
            optimizer.step()
            elapsed=previous_elapsed+time.perf_counter()-started
            val_loss=""
            if step%(100 if pilot else 1000)==0 or step==steps:
                final_val_loss=val_loss=validate(model,val,protocol)
                if val_loss<best:
                    best=val_loss
                    save_checkpoint(directory/"checkpoint_best.pt",model,optimizer,step,best,elapsed,cfg,clip_count,run_config)
            writer.writerow({"step":step,"elapsed_seconds":elapsed,"loss":float(loss.detach()),"raw_loss":raw,
                "val_loss":val_loss,"grad_norm":float(norm),"grad_norm_after_clip":float(norm)*min(1.0,1/(float(norm)+1e-6)),
                "clipped":clipped,"tau":tau,"orders_json":json.dumps(model.order_summary()) if step%100==0 or step==1 else ""})
            if step%100==0 or step==steps:
                handle.flush()
                save_checkpoint(resume_path,model,optimizer,step,best,elapsed,cfg,clip_count,run_config)
                print(json.dumps({"phase":"PHASE2_3DATASETS","kind":run_config["kind"],"run":run_id,"step":step,
                    "of":steps,"seconds":round(elapsed,2),"loss":float(loss.detach()),"val_loss":val_loss}),flush=True)
    elapsed=previous_elapsed+time.perf_counter()-started
    save_checkpoint(directory/"checkpoint_final.pt",model,optimizer,steps,best,elapsed,cfg,clip_count,run_config)
    write_json(directory/"training_cost.json",{"training_seconds":elapsed,"updates":steps,
        "peak_training_memory_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else 0,
        "clip_count":clip_count,"parameter_counts":parameter_counts(model),
        "V_evaluation_points":steps*protocol["H"]*protocol["Q"] if model.needs_history else 0,
        "W_network_evaluation_points":steps*protocol["H"]*protocol["Q"]*protocol["N"]*2 if model.needs_history else 0,
        "spatial_reverse_calls":steps*protocol["H"]*math.ceil(protocol["Q"]/cfg.query_chunk)*(2 if model.spec.family=="two_potential" else 1) if model.needs_history else 0})
    if pilot:
        arr,cost=sample(model,ensembles=2,particles=32,steps=32,seed=9011)
        np.save(directory/"pilot_samples.npy",arr)
        write_json(directory/"sampling_diagnostics.json",cost)
        generation=pilot_generation_diagnostics(arr,dataset_id,directory)
        if final_val_loss is None:final_val_loss=validate(model,val,protocol)
        write_json(directory/"pilot_result.json",{"dataset_id":dataset_id,"model_id":model_id,"seed":seed,
            "validation_loss":best,"final_validation_loss":final_val_loss,"generated_mean":float(arr.mean()),
            "generated_variance":float(arr.var()),"validation_mean":float(val.mean()),"validation_variance":float(val.var()),
            "training_seconds":elapsed,"sampling":cost,"validation_generation":generation,
            "test_accessed":False,"kind":"pilot_not_formal"})
    else:
        evaluate_one(directory)
        update_run(run_id,"COMPLETED",attempt_dir=str(directory),output_dir=str(directory),updates=steps)
    return directory


def _attempt_path(directory):
    directory=Path(directory).resolve()
    if not directory.is_relative_to(OUT.resolve()) or directory.name[:8]!="attempt_":
        raise ValueError("Formal evaluation is restricted to preserved Phase-2 attempt directories")
    return directory


def evaluate_one(directory):
    directory=_attempt_path(directory)
    from phase2_experiment.metrics import evaluate_run
    from phase2_experiment.diagnostics import checkpoint_diagnostics
    conf=read_json(directory/"config.json")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint=torch.load(directory/"checkpoint_final.pt",map_location=device,weights_only=False)
    assert_run_context(conf,checkpoint)
    if checkpoint["step"]!=conf["steps"]:raise ValueError("Final evaluation requires all locked formal updates")
    cfg=ModelConfig(**checkpoint["model_config"])
    model=build_model(conf["model_id"],cfg).to(device=device,dtype=torch.float32)
    model.load_state_dict(checkpoint["model"])
    protocol=_protocol(conf["dataset_id"])
    arr,cost=sample(model,ensembles=protocol["sampling_ensembles"],particles=protocol["sampling_particles"],
                    steps=protocol["sampling_steps"],seed=801017)
    np.save(directory/"samples.npy",arr)
    write_json(directory/"evaluation.json",evaluate_run(arr,directory,conf["dataset_id"]))
    costs={"primary":cost}
    if conf["model_id"]=="FM_STANDARD":
        reference,extra=sample(model,ensembles=protocol["sampling_ensembles"],particles=protocol["sampling_particles"],
                               steps=protocol["sampling_steps"],seed=801017,solver="heun")
        np.save(directory/"samples_heun.npy",reference)
        write_json(directory/"evaluation_heun.json",evaluate_run(reference,directory/"heun_metrics",conf["dataset_id"]))
        costs["heun_reference"]=extra
    write_json(directory/"sampling_diagnostics.json",costs)
    diagnostics=checkpoint_diagnostics(model,directory/"checkpoint_checks",load_data(conf["dataset_id"],"validation",device),
                                       dataset_id=conf["dataset_id"])
    write_json(directory/"checkpoint_checks.json",diagnostics)


def complete_artifacts(directory):
    from phase2_experiment.scope import completion_issues
    directory=_attempt_path(directory)
    if not (directory/"config.json").exists():return False
    conf=read_json(directory/"config.json")
    run={key:conf[key] for key in ("run_id","dataset_id","model_id","seed")}
    run.update(attempt_dir=str(directory),output_dir=str(directory))
    if completion_issues(run):return False
    checkpoint=torch.load(directory/"checkpoint_final.pt",map_location="cpu",weights_only=False)
    assert_run_context(conf,checkpoint)
    return checkpoint["step"]==conf["steps"]


def lock_protocol(dataset_id,updates,*,rationale,N=32,H=17):
    """Persist an explicitly chosen shared tier after all five task pilots.

    This function does not choose a tier; its caller must provide the tier and
    validation/cost rationale. Once any formal attempt exists, relocking stops.
    """
    from phase2_experiment.scope import DATASETS,validate_config,load_manifest
    validate_config(dataset_id,MODELS[0],SEEDS[0])
    if updates not in FORMAL_TIERS or N not in (32,64) or H not in (17,33) or not rationale.strip():
        raise ValueError("Provide an authorized shared tier, N/H and a nonempty validation/cost rationale")
    manifest=load_manifest()
    if any(run["status"] not in ("PENDING","BLOCKED") for run in manifest["runs"]) or any(OUT.glob("*/attempt_*")):
        raise ValueError("Formal execution has begun; preserving all locked protocols without relocking")
    pilots=[]
    for model_id in MODELS:
        runs=sorted((PILOTS/f"{dataset_id}__{model_id}__seed17").glob("attempt_*"))
        if not runs or not (runs[-1]/"pilot_result.json").is_file():raise ValueError(f"Complete validation-only pilot missing: {dataset_id}/{model_id}")
        directory=runs[-1]
        result=read_json(directory/"pilot_result.json")
        if result.get("test_accessed") is not False or result.get("dataset_id")!=dataset_id or result.get("model_id")!=model_id:
            raise ValueError("Pilot provenance or validation-only restriction differs")
        generation=result.get("validation_generation")
        if not isinstance(generation,dict) or generation.get("test_accessed") is not False:
            raise ValueError("Pilot lacks validation-only generated-quality/dependency evidence")
        cp=torch.load(directory/"checkpoint_final.pt",map_location="cpu",weights_only=False)
        conf=read_json(directory/"config.json")
        assert_run_context(conf,cp,formal=False)
        if cp["step"]!=conf["steps"] or not np.isfinite(result["validation_loss"]):raise ValueError("Pilot lacks a complete finite validation result")
        pilots.append({"model_id":model_id,"attempt":str(directory),**result})
    payload={"updates":updates,"selection_rationale":rationale,"N":N,"H":H,"Q":64,"E":1,
        "hidden":128,"hidden_layers":3,"state_dim":128,"query_chunk":64,"source_chunk":32,
        "network_lr":2e-4,"order_lr":2e-5,"weight_decay":1e-4,"grad_clip":1.0,"sigma_min":.05,"kappa":.05,
        "tau_min":None,"time_distribution":".5 Uniform + .5 U^4; inverse-density weighting",
        "sampling_ensembles":16,"sampling_particles":64,"sampling_steps":64,"checkpoint":"final",
        "best_validation_checkpoint_retained":True,
        "solver":"first cell_average_PC; then PL node explicit Euler / Euler-Maruyama",
        "standard_reference":"same checkpoint and initial samples; Heun primary statistical baseline; Euler supplement",
        "same_checkpoint_checks":{"steps":[32,64,128],"particles":[32,64,128],"ensembles":2,"extra_training_updates":0},
        "pilots":pilots}
    full=read_json(PROTOCOL_PATH) if PROTOCOL_PATH.exists() else {"phase":"PHASE2_3DATASETS","model_order":list(MODELS),"seed_order":list(SEEDS)}
    existing=full.get(dataset_id)
    placeholder=isinstance(existing,dict) and existing.get("status")=="NOT_LOCKED" and existing.get("updates") is None
    if existing is not None and not placeholder and existing!=payload:
        raise ValueError("A different task protocol is already locked; preserve it")
    full[dataset_id]=payload
    write_json(PROTOCOL_PATH,full)
    return payload


def run_queue():
    from phase2_experiment.scope import DATASETS,load_manifest,update_run,validate_config
    for dataset_id in DATASETS:_protocol(dataset_id)
    # Dataset, seed and model ordering is fixed and checked by the independent
    # Phase-2 scope loader, not inferred from output-directory contents.
    for run in load_manifest()["runs"]:
        dataset_id,model_id,seed=run["dataset_id"],run["model_id"],run["seed"]
        validate_config(dataset_id,model_id,seed)
        if run["status"]=="BLOCKED":continue
        run_id=run["run_id"];dirs=sorted((OUT/run_id).glob("attempt_*"))
        try:
            if dirs and complete_artifacts(dirs[-1]):
                update_run(run_id,"COMPLETED",attempt_dir=str(dirs[-1]),output_dir=str(dirs[-1]))
                continue
            if dirs and (dirs[-1]/"checkpoint_final.pt").exists():
                evaluate_one(dirs[-1])
                update_run(run_id,"COMPLETED",attempt_dir=str(dirs[-1]),output_dir=str(dirs[-1]))
            else:train_one(dataset_id,model_id,seed,resume=bool(dirs))
        except Exception as error:
            failure={"at":datetime.now(timezone.utc).isoformat(),"error":repr(error),"traceback":traceback.format_exc()}
            dirs=sorted((OUT/run_id).glob("attempt_*"))
            if dirs:
                write_json(dirs[-1]/f"failure_{time.time_ns()}.json",failure)
                write_json(dirs[-1]/"failure.json",failure)
            update_run(run_id,"FAILED",reason=str(error))
            print(json.dumps({"failed":run_id,**failure}),flush=True)


def main():
    from phase2_experiment.scope import DATASETS
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("command",choices=("prepare","pilot","lock","run","resume","evaluate","verify"))
    parser.add_argument("--dataset",choices=DATASETS)
    parser.add_argument("--model",choices=MODELS)
    parser.add_argument("--seed",type=int,choices=SEEDS,default=17)
    parser.add_argument("--pilot-steps",type=int,default=200)
    parser.add_argument("--updates",type=int,choices=FORMAL_TIERS)
    parser.add_argument("--rationale")
    parser.add_argument("--N",type=int,choices=(32,64),default=32)
    parser.add_argument("--H",type=int,choices=(17,33),default=17)
    parser.add_argument("--directory")
    args=parser.parse_args();os.chdir(ROOT)
    if args.command in ("prepare","pilot","lock") and not args.dataset:parser.error("This command requires an explicitly scoped --dataset")
    if args.command=="prepare":
        from phase2_experiment.scope import validate_scope
        validate_scope()
        from phase2_experiment.data import prepare_data
        print(json.dumps(prepare_data(args.dataset),indent=2))
    elif args.command=="pilot":
        if not 1<=args.pilot_steps<=2000:parser.error("pilot steps must be 1..2000")
        for model_id in ((args.model,) if args.model else MODELS):
            train_one(args.dataset,model_id,args.seed,args.pilot_steps,pilot=True)
    elif args.command=="lock":
        if args.updates is None or not args.rationale:parser.error("lock requires --updates and --rationale selected from validation/cost evidence")
        print(json.dumps(lock_protocol(args.dataset,args.updates,rationale=args.rationale,N=args.N,H=args.H),indent=2))
    elif args.command in ("run","resume"):run_queue()
    elif args.command=="evaluate":
        if not args.directory:parser.error("evaluate requires --directory")
        evaluate_one(args.directory)
    elif args.command=="verify":
        print(json.dumps({str(directory):complete_artifacts(directory) for directory in OUT.glob("*/attempt_*")},indent=2))


if __name__=="__main__":main()
