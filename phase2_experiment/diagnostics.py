"""Validation fields and same-checkpoint sampling refinement, with zero updates.

This module never changes the formal queue or trains an additional model.
The D=128 field diagnostics have no true divergence/curl labels.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from theory_fm_models import FieldHistory, MAIN_MODELS
from experiment.sampling import sample

DIAGNOSTIC_SEED = 771043
CURL_PAIRS = ((0, 1), (2, 3), (62, 63), (126, 127))


def _write_json(path: Path, obj: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def couple_brownian(finest: torch.Tensor, steps: int) -> torch.Tensor:
    """Return Brownian increments dB, not xi or sigma*dB, by exact grouping."""
    if steps < 1 or finest.shape[0] % steps:
        raise ValueError("Coarse steps must divide the finest Brownian grid")
    stride = finest.shape[0] // steps
    return finest.reshape(steps, stride, *finest.shape[1:]).sum(1)


def _finite_file(path: Path, shape: tuple) -> bool:
    if not path.is_file():
        return False
    try:
        arr = np.load(path, allow_pickle=False)
        return arr.shape == shape and np.issubdtype(arr.dtype, np.floating) and bool(np.isfinite(arr).all())
    except (OSError, ValueError):
        return False


def checkpoint_diagnostics(model, output_dir, val: torch.Tensor, *, dataset_id: str) -> dict:
    """Run/recover 5 unique sampling checks on one unchanged Phase-2 checkpoint.

    Sampling checks: N32 at 32/64/128 steps; N64/N128 at 64 steps, each with
    two independent ensembles. Coarse Brownian increments sum the same 128-step
    Brownian increments. N comparison uses nested subsets of a common N128
    initial population and Brownian sample. The largest/finest samples are finite
    numerical references, not exact solutions; no convergence rate is fitted.

    Field check uses only validation inputs and an independent reference law.
    It records an externally supplied P4 trace estimate and 4 exact curl pairs,
    without accuracy claims or true labels. No gradient update occurs.
    """
    from phase2_experiment.scope import validate_config
    validate_config(dataset_id,model.spec.model_id,17)
    if model.spec.model_id not in MAIN_MODELS or model.cfg.state_dim != 128 or model.cfg.condition_dim:
        raise ValueError("This Phase-2 diagnostic protocol supports only the five D128 unconditional selected sequence models")
    if val.ndim != 2 or val.shape[1] != 128 or len(val) < 32:
        raise ValueError("Expected independent validation sequences flattened as [samples,128]")
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    if val.device != device or val.dtype != dtype:
        raise ValueError("Validation, model and diagnostics must share dtype/device")
    directory = Path(output_dir) / "checkpoint_diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    state_hash = _state_hash(model)
    index_path = directory / "diagnostics.json"
    previous = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    # JSON serializes config tuples as lists; compare its canonical form.
    cfg_json=json.loads(json.dumps(asdict(model.cfg)))
    if previous and (previous.get("model_state_sha256") != state_hash
                     or previous.get("protocol_version") != 1
                     or previous.get("model_config") != cfg_json
                     or previous.get("dataset_id") != dataset_id):
        raise ValueError("Existing auxiliary diagnostics belong to a different checkpoint/protocol; preserve them in their original attempt")
    report = previous or {
        "protocol_version": 1, "status": "RUNNING", "kind": "same_checkpoint_auxiliary_evaluation",
        "formal_configuration_count_added": 0, "training_updates": 0,
        "dataset_id":dataset_id,"model_id": model.spec.model_id, "model_state_sha256": state_hash,
        "model_config": asdict(model.cfg), "seed": DIAGNOSTIC_SEED,
        "data_access": "validation_only_for_field_inputs; autonomous_sampling_no_real_endpoints",
        "ensembles": 2, "checks": {},
        "interpretation": "Finite largest N / finest time grid is not an exact solution. No fitted convergence rate. No true div/curl labels.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    _write_json(index_path, report)
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(DIAGNOSTIC_SEED)
    initial = torch.randn((2,128,128),generator=generator,dtype=dtype).to(device)
    finest = (torch.randn((128,2,128,128),generator=generator,dtype=dtype)
              / (128 ** .5)).to(device)
    # Save the exact shared randomness so checks can be reproduced independently.
    randomness_path = directory / "shared_initial_and_brownian.npz"
    if not randomness_path.exists():
        np.savez(randomness_path, initial=initial.cpu().numpy(),
                 brownian_128=finest.cpu().numpy(), seed=np.asarray(DIAGNOSTIC_SEED))
    try:
        field_path = directory / "validation_field_diagnostics.npz"
        if "field" not in report or not field_path.is_file():
            tic = time.perf_counter()
            if device.type == "cuda":torch.cuda.reset_peak_memory_stats()
            # Distinct draws from the validation set and independent noises.
            g = torch.Generator(device=device).manual_seed(DIAGNOSTIC_SEED+1)
            qi = torch.randint(len(val),(4,),device=device,generator=g)
            ri = torch.randint(len(val),(32,),device=device,generator=g)
            query_y=val[qi].unsqueeze(0);reference_y=val[ri].unsqueeze(0)
            query_eps=torch.randn(query_y.shape,device=device,dtype=dtype,generator=g)
            reference_eps=torch.randn(reference_y.shape,device=device,dtype=dtype,generator=g)
            tau=.5
            queries=tau*query_y+(1-.95*tau)*query_eps
            times=torch.linspace(0,tau,17,device=device,dtype=torch.float64)
            refs=times.to(dtype)[None,:,None,None]*reference_y[:,None]+(1-.95*times).to(dtype)[None,:,None,None]*reference_eps[:,None]
            history=FieldHistory(times,refs) if model.needs_history else None
            probes=(2*torch.randint(2,(4,1,4,128),device=device,generator=g)-1).to(dtype)
            fields=model.differential_fields(queries,tau,history=history,
                curl_pairs=CURL_PAIRS,divergence_probes=probes,create_graph=False)
            arrays={"query_indices":qi.cpu().numpy(),"reference_indices":ri.cpu().numpy(),
                    "queries":queries.cpu().numpy(),"reference_history":refs.cpu().numpy(),
                    "reference_times":times.cpu().numpy(),"drift":fields.drift.cpu().numpy(),
                    "divergence_probe_estimate":fields.divergence.cpu().numpy(),
                    "curl_components":fields.curl.cpu().numpy(),"curl_pairs":np.asarray(CURL_PAIRS),
                    "trace_probes":probes.cpu().numpy(),"J":model.J.detach().cpu().numpy(),
                    "tau":np.asarray(tau),"has_true_labels":np.asarray(False)}
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise FloatingPointError("Nonfinite validation field diagnostic")
            np.savez(field_path,**arrays)
            if device.type=="cuda":torch.cuda.synchronize()
            report["field"]={"raw_npz":field_path.name,"seconds":time.perf_counter()-tic,
                "tau":tau,"query_count":4,"N":32,"H":17,"trace_probes":4,
                "divergence_estimator":fields.divergence_estimator,"curl_pairs":CURL_PAIRS,
                "true_labels_available":False,"accuracy_metrics":"Not applicable",
                "peak_memory_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else 0}
            _write_json(index_path,report)
        for particles,steps in ((32,32),(32,64),(32,128),(64,64),(128,64)):
            key=f"N{particles}_steps{steps}"
            target=directory/f"samples_{key}.npy"
            if key in report["checks"] and _finite_file(target,(2,particles,128)):
                continue
            arr,cost=sample(model,ensembles=2,particles=particles,steps=steps,
                seed=DIAGNOSTIC_SEED,initial=initial[:,:particles],
                brownian=couple_brownian(finest[:,:,:particles],steps))
            if not np.isfinite(arr).all():
                raise FloatingPointError(f"Nonfinite autonomous diagnostic {key}")
            np.save(target,arr)
            report["checks"][key]={"samples":target.name,"status":"COMPLETED", "cost":cost,
                "training_updates":0,"common_initial":"N128 nested subset",
                "Brownian":"sum of shared 128-step dB increments" if model.kappa else "Not applicable (ODE)"}
            _write_json(index_path,report)
        a32=np.load(directory/"samples_N32_steps32.npy",allow_pickle=False).astype(np.float64)
        a64=np.load(directory/"samples_N32_steps64.npy",allow_pickle=False).astype(np.float64)
        a128=np.load(directory/"samples_N32_steps128.npy",allow_pickle=False).astype(np.float64)
        n64=np.load(directory/"samples_N64_steps64.npy",allow_pickle=False).astype(np.float64)
        n128=np.load(directory/"samples_N128_steps64.npy",allow_pickle=False).astype(np.float64)
        def rms(a,b):return float(np.sqrt(np.mean((a-b)**2)))
        report["difference_RMS"]={"steps32_vs128_N32":rms(a32,a128),
            "steps64_vs128_N32":rms(a64,a128),"steps32_vs64_N32":rms(a32,a64),
            "N32_vsN128_steps64_first32":rms(a64,n128[:,:32]),
            "N64_vsN128_steps64_first64":rms(n64,n128[:,:64]),
            "N32_vsN64_steps64_first32":rms(a64,n64[:,:32])}
        if _state_hash(model)!=state_hash:
            raise RuntimeError("Diagnostic evaluation changed model parameters or buffers")
        report["total_sampling_seconds"]=sum(item["cost"]["sampling_seconds"] for item in report["checks"].values())
        report["total_diagnostic_seconds"]=report["total_sampling_seconds"]+report["field"]["seconds"]
        report["status"]="COMPLETED"
        _write_json(index_path,report)
        return report
    except Exception as exc:
        report["status"]="FAILED"
        report.setdefault("failures",[]).append({"exception":repr(exc),"completed_checks":list(report["checks"])})
        _write_json(index_path,report)
        raise
    finally:
        model.train(was_training)
