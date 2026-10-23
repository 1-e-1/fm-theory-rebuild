"""Measure diagnostic cost on this project's new completed pilot, zero updates."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from theory_fm_models import ModelConfig, build_model
from experiment.data import load_split
from experiment.diagnostics import checkpoint_diagnostics

root=Path(__file__).resolve().parents[1]
model_id="FM_COMPLEX_ODE"
directory=root/"outputs/pilots"/f"SEQ_IID__{model_id}__seed17"/"attempt_001"
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
checkpoint=torch.load(directory/"checkpoint_final.pt",map_location="cuda",weights_only=False)
model=build_model(model_id,ModelConfig(**checkpoint["model_config"])).cuda()
model.load_state_dict(checkpoint["model"])
result=checkpoint_diagnostics(model,directory,load_split("validation",device="cuda"))
compact={"model_id":model_id,"kind":"pilot_checkpoint_auxiliary_cost_check","training_updates":0,
         "total_diagnostic_seconds":result["total_diagnostic_seconds"],
         "field_seconds":result["field"]["seconds"],
         "checks":{k:v["cost"] for k,v in result["checks"].items()},
         "difference_RMS":result["difference_RMS"]}
(root/"tests/model_diagnostics_pilot_benchmark_result.json").write_text(json.dumps(compact,indent=2),encoding="utf-8")
print(json.dumps(compact),flush=True)
