import sys, json, torch
from pathlib import Path
sys.path.insert(0,"/content/mini-wam-eval/src")
from mini_wam.models.action_only import ActionOnlyPolicy
from mini_wam.models.future_head import FutureHeadPolicy
from mini_wam.data import NormalizationStats
from mini_wam.studio.pusht import SceneState, random_scene, validate_scene, run_rollout
import mini_wam.studio.pusht as pusht_module

repo=Path("/content/mini-wam-eval")
drive=Path("/content/drive/MyDrive/mini-wam-runs")
action_run=drive/"action-only-seed0"
future_run=drive/"future-aware-seed0-w4"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_ck=torch.load(action_run/"checkpoints"/"best.pt",map_location=device,weights_only=True)
future_ck=torch.load(future_run/"checkpoints"/"best.pt",map_location=device,weights_only=True)
action_model=ActionOnlyPolicy(pretrained=False).to(device)
future_model=FutureHeadPolicy(pretrained=False,target_pretrained=False).to(device)
action_model.load_state_dict(action_ck["model"],strict=True)
future_model.load_state_dict(future_ck["model"],strict=True)
action_model.eval(); future_model.eval()

def load_flat_norm(path):
    v=json.loads(path.read_text())
    return NormalizationStats(torch.tensor(v["position_mean"]),torch.tensor(v["position_std"]),torch.tensor(v["action_mean"]),torch.tensor(v["action_std"]),float(v["std_floor"]))
action_norm=load_flat_norm(action_run/"normalization.json")
future_norm=load_flat_norm(future_run/"normalization.json")
same_norm=all(torch.equal(getattr(action_norm,k),getattr(future_norm,k)) for k in ("position_mean","position_std","action_mean","action_std"))

if not getattr(pusht_module.iio.imwrite,"_colab_compat",False):
    _original=pusht_module.iio.imwrite
    def _compat(uri,image,**kwargs):
        if "out_pixel_format" in kwargs and "pixelformat" not in kwargs:
            kwargs["pixelformat"]=kwargs.pop("out_pixel_format")
        return _original(uri,image,**kwargs)
    _compat._colab_compat=True
    pusht_module.iio.imwrite=_compat

seed_candidates=list(range(2026091000,2026091021))
valid_records=[]
invalid=[]
for seed in seed_candidates:
    scene=random_scene(seed)
    try:
        validate_scene(scene)
        valid_records.append({"scene_id":f"web-random-{len(valid_records):02d}","seed":seed,"state":{"agent_x":scene.agent_x,"agent_y":scene.agent_y,"block_x":scene.block_x,"block_y":scene.block_y,"block_angle":scene.block_angle}})
    except ValueError as exc:
        invalid.append({"seed":seed,"reason":str(exc)})
valid_records=valid_records[:20]
print("READY",device,"action_step",action_ck["step"],"future_step",future_ck["step"],"same_norm",same_norm)
print("VALID_SCENES",len(valid_records),"INVALID",invalid)
