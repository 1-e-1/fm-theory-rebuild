"""Independent Phase-2 integration inputs; zero pilot/formal optimization updates."""
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from theory_fm_models import MAIN_MODELS, ModelConfig, FieldHistory, build_model
from experiment.sampling import sample
from phase2_experiment import runner
from phase2_experiment import scope
from phase2_experiment.diagnostics import checkpoint_diagnostics, couple_brownian, CURL_PAIRS


@pytest.mark.parametrize("device,dtype",[("cpu",torch.float64),("cuda",torch.float32)])
def test_d128_five_models_forward_backward_and_accepted_sampling(device,dtype):
    if device=="cuda" and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=ModelConfig(state_dim=128,hidden_dim=16,hidden_layers=2,query_chunk=64,source_chunk=32)
    x=torch.randn(1,2,128,generator=torch.Generator().manual_seed(942),dtype=dtype).to(device)
    sources=torch.randn(1,3,4,128,generator=torch.Generator().manual_seed(943),dtype=dtype).to(device)
    times=torch.tensor([0,.2,.5],device=device,dtype=torch.float64)
    history=FieldHistory(times,sources)
    for name in MAIN_MODELS:
        model=build_model(name,cfg).to(device=device,dtype=dtype)
        parameters={key:value.clone() for key,value in model.state_dict().items()}
        output=model(x,.5,history if model.needs_history else None)
        assert output.drift.shape==(1,2,128) and torch.isfinite(output.drift).all()
        output.drift.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        probes=torch.ones(2,1,2,128,device=device,dtype=dtype)
        fields=model.differential_fields(x,.5,history if model.needs_history else None,
            curl_pairs=CURL_PAIRS,divergence_probes=probes)
        assert fields.curl.shape==(1,2,4) and fields.divergence.shape==(1,2)
        assert torch.isfinite(fields.curl).all() and torch.isfinite(fields.divergence).all()
        noise=torch.randn((4,1,4,128),generator=torch.Generator().manual_seed(944),dtype=dtype).to(device)/2
        initial=torch.randn((1,4,128),generator=torch.Generator().manual_seed(945),dtype=dtype).to(device)
        arr,info=sample(model,ensembles=1,particles=4,steps=4,initial=initial,brownian=noise)
        assert arr.shape==(1,4,128) and np.isfinite(arr).all()
        assert info["history_source"]=="generated_accepted_states_only"
        for key,value in model.state_dict().items():torch.testing.assert_close(value,parameters[key],rtol=0,atol=0)


def test_imported_training_batch_keeps_d128_complete_fp_target_and_independent_reference():
    assert runner.training_batch.__module__=="experiment.runner"
    data=torch.randn(100,128,generator=torch.Generator().manual_seed(521))
    seed=509
    replay=torch.Generator().manual_seed(seed)
    qi=torch.randint(100,(64,),generator=replay);ri=torch.randint(100,(32,),generator=replay)
    y=data[qi][None];yr=data[ri][None]
    eps=torch.randn(y.shape,generator=replay);er=torch.randn(yr.shape,generator=replay)
    u=torch.rand((),dtype=torch.float64,generator=replay);coin=torch.rand((),generator=replay)
    tau=u if coin<.5 else u**4
    sigma=1-.95*tau
    target=y-.95*eps-.05*eps/sigma.float()
    class Capture:
        needs_history=True
        kappa=.05
        def __call__(self,x,t,history,create_graph):
            self.x,self.t,self.history=x,t,history
            return SimpleNamespace(drift=torch.zeros_like(x))
    model=Capture()
    loss,time,raw=runner.training_batch(data,model,generator=torch.Generator().manual_seed(seed))
    torch.testing.assert_close(model.x,tau.float()*y+sigma.float()*eps)
    torch.testing.assert_close(model.history.particles[:,0],er)
    torch.testing.assert_close(model.history.particles[:,-1],tau.float()*yr+sigma.float()*er)
    torch.testing.assert_close(loss,target.square().mean()/(.5+.125*tau**(-.75)).float())
    assert model.history.particles.shape==(1,17,32,128)
    assert not torch.equal(model.history.particles[:,-1],model.x[:,:32])
    assert time==float(tau)


def test_d128_diagnostics_five_checks_and_coupled_noise_are_recoverable(tmp_path):
    model=build_model("FM_STANDARD",ModelConfig(state_dim=128,hidden_dim=4,hidden_layers=1))
    val=torch.randn(64,128,generator=torch.Generator().manual_seed(995))
    state={name:value.clone() for name,value in model.state_dict().items()}
    result=checkpoint_diagnostics(model,tmp_path,val,dataset_id="SEQ_DELAY")
    assert result["status"]=="COMPLETED" and result["training_updates"]==0
    assert result["dataset_id"]=="SEQ_DELAY" and len(result["checks"])==5
    assert result["field"]["true_labels_available"] is False
    directory=tmp_path/"checkpoint_diagnostics"
    with np.load(directory/"validation_field_diagnostics.npz") as data:
        assert data["queries"].shape==(1,4,128)
        assert data["reference_history"].shape==(1,17,32,128)
        assert data["trace_probes"].shape==(4,1,4,128)
        np.testing.assert_array_equal(data["curl_pairs"],[[0,1],[2,3],[62,63],[126,127]])
    with np.load(directory/"shared_initial_and_brownian.npz") as data:
        fine=torch.from_numpy(data["brownian_128"])
        assert data["initial"].shape==(2,128,128)
        torch.testing.assert_close(couple_brownian(fine,32).sum(0),fine.sum(0),rtol=1e-6,atol=2e-6)
    original_times={name:check["cost"]["sampling_seconds"] for name,check in result["checks"].items()}
    recovered=checkpoint_diagnostics(model,tmp_path,val,dataset_id="SEQ_DELAY")
    assert original_times=={name:check["cost"]["sampling_seconds"] for name,check in recovered["checks"].items()}
    with pytest.raises(ValueError):checkpoint_diagnostics(model,tmp_path,val,dataset_id="SEQ_ACCUM")
    for name,value in model.state_dict().items():torch.testing.assert_close(value,state[name],rtol=0,atol=0)


def test_source_snapshots_keep_original_and_phase2_same_names_distinct(tmp_path,monkeypatch):
    sources={"experiment/diagnostics.py":b"original diagnostic source",
             "phase2_experiment/diagnostics.py":b"independent D128 source"}
    hashes={}
    for relative,content in sources.items():
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        hashes[relative]=hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    attempt=tmp_path/"outputs/attempt_001"
    runner.snapshot_sources(attempt,hashes)
    for relative,content in sources.items():assert (attempt/"source"/relative).read_bytes()==content
    (attempt/"source/phase2_experiment/diagnostics.py").write_text("changed")
    with pytest.raises(ValueError):runner.snapshot_sources(attempt,hashes)


def test_reporting_changes_do_not_change_training_core_digest(tmp_path,monkeypatch):
    for relative in runner.SOURCE_CORE+runner.SOURCE_PHASE2_CORE:
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(relative)
    monitor=tmp_path/"phase2_experiment/monitor.py";monitor.write_text("first monitor")
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    first=runner.execution_sources()
    monitor.write_text("updated monitoring only")
    (tmp_path/"phase2_experiment/plots.py").write_text("new plot implementation")
    assert first==runner.execution_sources()
    assert "phase2_experiment/monitor.py" not in first
    assert "phase2_experiment/plots.py" not in first
    runner.snapshot_sources(tmp_path/"attempt_001",first)
    assert (tmp_path/"attempt_001/source/phase2_experiment/monitor.py").is_file()
    assert (tmp_path/"attempt_001/source/phase2_experiment/plots.py").is_file()


def test_dataset_protocol_digest_is_stable_and_matches_scope():
    first={"updates":12000,"Q":64,"H":17,"N":32,"E":1}
    assert runner.dataset_protocol_hash(first)==scope.payload_digest(first)
    full={"SEQ_DELAY":first,"SEQ_ACCUM":{"updates":24000}}
    original=runner.dataset_protocol_hash(full["SEQ_DELAY"])
    full["SEQ_MULTISCALE"]={"updates":48000}
    assert original==runner.dataset_protocol_hash(full["SEQ_DELAY"])


def test_resume_log_preserves_all_rows_before_checkpoint_reconciliation(tmp_path):
    path=tmp_path/"training.csv"
    rows=[]
    for step in (1,2,3):
        row={field:"" for field in runner.FIELDS};row["step"]=step;rows.append(row)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=runner.FIELDS);writer.writeheader();writer.writerows(rows)
    runner.preserve_resume_log(path,2)
    with path.open(newline="",encoding="utf-8") as handle:assert [int(r["step"]) for r in csv.DictReader(handle)]==[1,2]
    old=list(tmp_path.glob("training_before_resume_*.csv"));assert len(old)==1
    with old[0].open(newline="",encoding="utf-8") as handle:assert [int(r["step"]) for r in csv.DictReader(handle)]==[1,2,3]


def test_checkpoint_rng_identity_and_context_rejection(tmp_path,monkeypatch):
    model=build_model("FM_STANDARD",ModelConfig(state_dim=128,hidden_dim=4,hidden_layers=1))
    optimizer=torch.optim.AdamW(model.parameters())
    spec=tmp_path/"DATA_SPEC.json";spec.write_text("fixture spec")
    protocol={"updates":12000,"N":32,"H":17,"Q":64,"E":1}
    hashes={"theory_fm_models.py":"fixture_model_hash","phase2_experiment/runner.py":"fixture_runner_hash"}
    identity={"dataset_id":"SEQ_DELAY","model_id":"FM_STANDARD","seed":17,"kind":"formal",
              "run_id":"SEQ_DELAY__FM_STANDARD__seed17","steps":12000}
    conf={**identity,"phase":"PHASE2_3DATASETS","model_config":asdict(model.cfg),
          "source_sha256":hashes["theory_fm_models.py"],"execution_source_hashes":hashes,
          "data_spec_sha256":runner.sha256_file(spec),"dataset_protocol_sha256":runner.dataset_protocol_hash(protocol),
          **{key:protocol[key] for key in ("N","H","Q","E")}}
    random.seed(478);np.random.seed(479);torch.manual_seed(480)
    path=tmp_path/"checkpoint_resume.pt"
    runner.save_checkpoint(path,model,optimizer,100,.8,2.3,model.cfg,2,conf)
    expected=(random.random(),np.random.random(),torch.rand(3))
    checkpoint=torch.load(path,weights_only=False)
    runner.restore_checkpoint(checkpoint,model,optimizer)
    assert random.random()==expected[0] and np.random.random()==expected[1]
    torch.testing.assert_close(torch.rand(3),expected[2],rtol=0,atol=0)
    monkeypatch.setattr(runner,"execution_sources",lambda:hashes)
    monkeypatch.setattr(runner,"verify_data",lambda dataset:(spec,{}))
    monkeypatch.setattr(runner,"_protocol",lambda dataset:protocol)
    runner.assert_run_context(conf,checkpoint)
    tampered=dict(checkpoint);tampered["run_identity"]=dict(checkpoint["run_identity"],seed=29)
    with pytest.raises(ValueError):runner.assert_run_context(conf,tampered)
    monkeypatch.setattr(runner,"_protocol",lambda dataset:dict(protocol,H=33))
    with pytest.raises(ValueError):runner.assert_run_context(conf,checkpoint)


def test_queue_order_is_exact_scope_and_outside_model_rejected_without_training(tmp_path,monkeypatch):
    runs=[{"dataset_id":d,"model_id":m,"seed":s,"run_id":scope.run_identifier(d,m,s),"status":"PENDING"}
          for d,m,s in scope.EXPECTED]
    monkeypatch.setattr(scope,"load_manifest",lambda:{"runs":runs})
    monkeypatch.setattr(runner,"OUT",tmp_path)
    monkeypatch.setattr(runner,"_protocol",lambda dataset:{"updates":12000})
    calls=[]
    monkeypatch.setattr(runner,"train_one",lambda d,m,s,**kwargs:calls.append((d,m,s)))
    runner.run_queue()
    assert calls==list(scope.EXPECTED) and len(calls)==75
    calls.clear();runs[0]["model_id"]="FM_STANDARD_SDE"
    with pytest.raises(ValueError):runner.run_queue()
    assert not calls


def test_no_automatic_protocol_selection_and_no_phase1_artifact_evaluation(tmp_path,monkeypatch):
    with pytest.raises(ValueError):runner.lock_protocol("SEQ_DELAY",500,rationale="short budget")
    with pytest.raises(ValueError):runner.lock_protocol("SEQ_DELAY",12000,rationale="")
    monkeypatch.setattr(runner,"OUT",tmp_path/"phase2")
    with pytest.raises(ValueError):runner.evaluate_one(tmp_path/"phase1/attempt_001")


def test_initialize_placeholders_can_lock_once_but_cannot_relock(tmp_path,monkeypatch):
    monkeypatch.setattr(scope,"ROOT",tmp_path)
    monkeypatch.setattr(scope,"SCOPE_PATH",tmp_path/"PHASE2_SCOPE.json")
    monkeypatch.setattr(scope,"MANIFEST_PATH",tmp_path/"PHASE2_MANIFEST.json")
    monkeypatch.setattr(scope,"PROTOCOL_PATH",tmp_path/"PHASE2_LOCKED_PROTOCOL.json")
    monkeypatch.setattr(runner,"PROTOCOL_PATH",scope.PROTOCOL_PATH)
    monkeypatch.setattr(runner,"OUT",tmp_path/"outputs/phase2_3datasets")
    monkeypatch.setattr(runner,"PILOTS",tmp_path/"outputs/phase2_pilots")
    scope.initialize_scope(list(scope.DATASETS),authority_note="isolated test fixture; no formal run")
    initial=json.loads(scope.PROTOCOL_PATH.read_text())
    # Support the initializer's pending-list schema and the earlier explicit
    # NOT_LOCKED per-task placeholder schema; neither is a real protocol.
    initial["SEQ_DELAY"]={"status":"NOT_LOCKED","updates":None}
    runner.write_json(scope.PROTOCOL_PATH,initial)
    assert json.loads(scope.PROTOCOL_PATH.read_text())["SEQ_DELAY"]=={"status":"NOT_LOCKED","updates":None}
    for name in runner.MODELS:
        directory=runner.PILOTS/f"SEQ_DELAY__{name}__seed17"/"attempt_001"
        directory.mkdir(parents=True)
        runner.write_json(directory/"pilot_result.json",{"dataset_id":"SEQ_DELAY","model_id":name,
            "test_accessed":False,"validation_loss":1.2,"kind":"pilot_not_formal",
            "validation_generation":{"test_accessed":False,"validation_swd_raw":.8}})
        runner.write_json(directory/"config.json",{"steps":200})
        torch.save({"step":200},directory/"checkpoint_final.pt")
    # This test isolates the lock state machine. Model/provenance validation is
    # covered independently; fixture checkpoints contain no trained weights.
    monkeypatch.setattr(runner,"assert_run_context",lambda *args,**kwargs:None)
    result=runner.lock_protocol("SEQ_DELAY",12000,rationale="test validation/cost fixture")
    assert result["updates"]==12000
    assert json.loads(scope.PROTOCOL_PATH.read_text())["SEQ_DELAY"]["updates"]==12000
    with pytest.raises(ValueError):runner.lock_protocol("SEQ_DELAY",24000,rationale="different test tier")


def test_config_reads_full_d128_from_own_dataset_spec(tmp_path,monkeypatch):
    from phase2_experiment import data
    (tmp_path/"DATA_SPEC.json").write_text(json.dumps({"dataset_id":"SEQ_DELAY","length":64,"channels":2,"state_dim":128}))
    monkeypatch.setattr(data,"data_dir",lambda dataset:tmp_path)
    cfg=runner.config("SEQ_DELAY")
    assert cfg.state_dim==128 and cfg.hidden_dim==128 and cfg.hidden_layers==3
    assert cfg.query_chunk==64 and cfg.source_chunk==32


def test_actual_pilot_validation_metrics_never_open_test_or_formal_evaluator(tmp_path,monkeypatch):
    from phase2_experiment import data,metrics
    generator=np.random.default_rng(996)
    raw=generator.standard_normal((64,64,2)).astype(np.float32)
    np.savez(tmp_path/"validation.npz",raw=raw,standardized=raw.reshape(64,128))
    np.savez(tmp_path/"stats.npz",mean=np.zeros(128),scale=np.ones(128))
    directions=generator.standard_normal((256,128));directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    joint=generator.standard_normal((256,32));joint/=np.linalg.norm(joint,axis=1,keepdims=True)
    np.savez(tmp_path/"metric_parameters.npz",directions=directions,early_indices=np.arange(16),
             late_indices=np.arange(32,48),joint_directions=joint)
    monkeypatch.setattr(data,"data_dir",lambda dataset,root=None:tmp_path)
    monkeypatch.setattr(metrics,"data_dir",lambda dataset,root=None:tmp_path)
    real_load=np.load;opened=[]
    def guarded_load(path,*args,**kwargs):
        opened.append(Path(path).name)
        if Path(path).name=="test.npz":raise AssertionError("A pilot opened formal test")
        return real_load(path,*args,**kwargs)
    monkeypatch.setattr(np,"load",guarded_load)
    def forbidden_evaluator(*args,**kwargs):raise AssertionError("A pilot called formal evaluate_run")
    monkeypatch.setattr(metrics,"evaluate_run",forbidden_evaluator)
    samples=generator.standard_normal((2,32,128)).astype(np.float32)
    evidence=runner.pilot_generation_diagnostics(samples,"SEQ_DELAY",tmp_path/"pilot_output")
    assert evidence["test_accessed"] is False and evidence["small_sample_nonformal"] is True
    assert np.isfinite(evidence["validation_swd_raw"])
    assert "generated_paired_structure_mse_raw" in evidence
    assert set(opened)=={"validation.npz","metric_parameters.npz","stats.npz"}
    saved=json.loads((tmp_path/"pilot_output/pilot_validation_diagnostics.json").read_text())
    assert saved==evidence
